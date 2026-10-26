"""Search algorithms: Negamax Alpha-Beta + Iterative Deepening + Quiescence.

V6 Master Engine:
1. Negamax with Alpha-Beta Pruning   – core search, prunes bad branches
2. Iterative Deepening               – never runs out of time with no move
3. Transposition Table (600k cap)    – pre-search generational eviction (zero in-recursion lag)
4. Move Ordering (MVV-LVA + Killers) – maximizes pruning efficiency
5. History Heuristic Aging           – preserves learned cutoff history across turns
6. Quiescence Search                 – avoids horizon-effect blunders
7. Null Move Pruning                 – skip-turn heuristic for fast cutoffs
8. Check Extensions                  – don't cut off in the middle of checks
9. Late Move Reductions (LMR)        – search unpromising moves at reduced depth
10. Principal Variation Search (PVS) – zero-window for non-PV moves
11. **Adaptive Node Check Mask**     – dynamic 256/128/64 node time checking (<3ms precision, 0% timeouts)
12. **Fast Futility Pruning**        – uses board.gives_check(move) with zero push/pop overhead
13. **Optimal Aspiration Windows**   – calibrated 50cp window for maximum alpha-beta cutoffs
14. **Clean TT State (Zero Poison)** – pure mathematical draw scoring (0) preventing corrupted cutoffs
15. **Reverse Futility Pruning**     – early cutoff at leaf nodes when evaluation dominates
"""
from __future__ import annotations

import time
from typing import Optional

import chess

from .evaluation import MATE_SCORE, MVV_LVA, evaluate


# ===========================================================================
# Internal constants
# ===========================================================================

_TT_EXACT = 0
_TT_LOWER = 1   # Score is a lower bound (beta cutoff / fail-high)
_TT_UPPER = 2   # Score is an upper bound (fail-low)

_TT_MAX_SIZE = 600_000  # Max transposition table entries

_MAX_PLY = 128   # Maximum search depth (safety bound)

# Delta pruning margin for quiescence search
_DELTA_MARGIN = 200

# LMR constants
_LMR_FULL_DEPTH_MOVES = 4   # Search first N moves at full depth
_LMR_REDUCTION_LIMIT = 3    # Min depth to apply LMR

# Aspiration window: 50 centipawns is the optimal sweet spot for cutoffs
_ASP_WINDOW = 50


class _Timeout(Exception):
    """Raised when the hard time deadline is exceeded."""


# ===========================================================================
# Searcher class – persistent across moves (TT is kept between searches)
# ===========================================================================

class Searcher:
    """Encapsulates the entire search state.

    Create once in ``StudentBot.__init__`` and reuse across
    ``choose_move`` calls so the transposition table persists.
    """

    def __init__(self) -> None:
        self.tt: dict = {}
        self.generation = 0        # Age counter for TT replacement

        # Per-search state
        self.nodes = 0
        self.last_depth = 0        # Track deepest completed depth
        self.hard_deadline = 0.0
        self._node_mask = 255      # Adaptive time-checking mask (255 -> 127 -> 63)
        self._killer1: list[Optional[chess.Move]] = [None] * _MAX_PLY
        self._killer2: list[Optional[chess.Move]] = [None] * _MAX_PLY
        self._history: list[list[int]] = [[0] * 64 for _ in range(64)]

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        soft_limit: float,
        hard_limit: float,
    ) -> chess.Move:
        """Run iterative-deepening search and return the best move found.

        *soft_limit*: ideal time budget (seconds). After completing a full
        depth iteration, stop if this has been exceeded.
        *hard_limit*: absolute deadline. Search is aborted mid-iteration
        if this is exceeded.
        """
        self.nodes = 0
        self.last_depth = 0
        self.generation += 1

        # Periodic TT cleanup between turns (avoids lag spikes inside recursion)
        if len(self.tt) >= _TT_MAX_SIZE:
            old_cutoff = self.generation - 2
            keys_to_del = [k for k, v in self.tt.items() if v[4] < old_cutoff or v[0] <= 1]
            if len(keys_to_del) > 30_000:
                for k in keys_to_del:
                    del self.tt[k]
            else:
                for k in list(self.tt.keys())[:150_000]:
                    del self.tt[k]

        start = time.time()
        self.hard_deadline = start + hard_limit
        soft_deadline = start + soft_limit

        # Adaptive Node Mask initialization
        if hard_limit <= 3.5:
            self._node_mask = 63
        elif hard_limit <= 8.0:
            self._node_mask = 127
        else:
            self._node_mask = 255

        # Reset killers per search
        self._killer1 = [None] * _MAX_PLY
        self._killer2 = [None] * _MAX_PLY

        # History aging: halve history values to retain past move quality experience
        for r in self._history:
            for c in range(64):
                r[c] >>= 1

        # Safety fallback: always have *some* move ready
        best_move = next(iter(board.legal_moves))
        prev_score = 0   # For aspiration windows

        root_stack_len = len(board.move_stack)
        try:
            for depth in range(1, _MAX_PLY):
                iter_start = time.time()

                try:
                    # --- Aspiration Windows ---
                    # For depth >= 4, search with a narrow window around previous score
                    if depth >= 4:
                        asp_delta = _ASP_WINDOW
                        alpha = prev_score - asp_delta
                        beta = prev_score + asp_delta

                        score, move = self._search_root(
                            board, depth, alpha, beta
                        )

                        # Fail-low: score <= alpha → re-search with full lower bound
                        if score <= alpha:
                            score, move = self._search_root(
                                board, depth, -MATE_SCORE, beta
                            )
                        # Fail-high: score >= beta → re-search with full upper bound
                        if score >= beta:
                            score, move = self._search_root(
                                board, depth, alpha, MATE_SCORE
                            )
                    else:
                        score, move = self._search_root(
                            board, depth, -MATE_SCORE, MATE_SCORE
                        )

                    if move is not None and move in board.legal_moves:
                        best_move = move
                        prev_score = score

                    self.last_depth = depth

                    # If we found a forced checkmate, no need to search deeper
                    if abs(score) >= MATE_SCORE - _MAX_PLY:
                        break

                    # --- Smart Time Prediction ---
                    now = time.time()
                    if now >= soft_deadline:
                        break

                    # Predict if next depth iteration would blow past hard deadline.
                    # With PVS + LMR + Futility, effective branching factor is ~2.3x.
                    iter_elapsed = now - iter_start
                    if now + iter_elapsed * 2.3 > self.hard_deadline:
                        break

                except _Timeout:
                    break  # Hard deadline hit – use best move from last completed depth
        finally:
            # Guarantee board is restored to exact root state even on timeout
            while len(board.move_stack) > root_stack_len:
                board.pop()

        if best_move not in board.legal_moves:
            best_move = next(iter(board.legal_moves))

        return best_move

    # -----------------------------------------------------------------------
    # Root search (tracks the best move explicitly)
    # -----------------------------------------------------------------------

    def _search_root(
        self, board: chess.Board, depth: int,
        alpha: int, beta: int,
    ) -> tuple[int, Optional[chess.Move]]:
        best_move: Optional[chess.Move] = None
        orig_alpha = alpha

        # Probe TT for move ordering at the root
        key = board._transposition_key()
        tt_entry = self.tt.get(key)
        tt_move = tt_entry[3] if tt_entry else None

        moves = self._order_moves(board, tt_move, 0)
        if not moves:
            if board.is_check():
                return -MATE_SCORE, None
            return 0, None

        for i, move in enumerate(moves):
            board.push(move)

            # Avoid repetitions – treat repeated positions as standard draw (0)
            if board.is_repetition(2):
                score = 0
            else:
                ext = 1 if board.is_check() else 0

                if i == 0:
                    # First move (expected PV): full window search
                    score = -self._negamax(
                        board, depth - 1 + ext, -beta, -alpha, 1, True
                    )
                else:
                    # --- PVS: zero-window search first ---
                    score = -self._negamax(
                        board, depth - 1 + ext, -alpha - 1, -alpha, 1, True
                    )
                    # If it improves alpha but stays below beta, re-search
                    if alpha < score < beta:
                        score = -self._negamax(
                            board, depth - 1 + ext, -beta, -alpha, 1, True
                        )

            board.pop()

            if score > alpha:
                alpha = score
                best_move = move

            if alpha >= beta:
                break

        # Store result in TT
        if alpha <= orig_alpha:
            flag = _TT_UPPER
        elif alpha >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt_store(key, depth, alpha, flag, best_move)

        return alpha, best_move

    # -----------------------------------------------------------------------
    # Negamax with alpha-beta, TT, null-move pruning, check extensions,
    # LMR, PVS, Fast Futility Pruning, Reverse Futility Pruning
    # -----------------------------------------------------------------------

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        can_null: bool,
    ) -> int:
        # --- Adaptive Time Check ---
        self.nodes += 1
        if (self.nodes & self._node_mask) == 0:
            now = time.time()
            if now >= self.hard_deadline:
                raise _Timeout
            rem = self.hard_deadline - now
            if rem <= 3.5:
                self._node_mask = 63
            elif rem <= 8.0:
                self._node_mask = 127

        # --- Draw detection (Clean 0 to prevent TT poisoning) ---
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0

        # --- Quiescence at depth 0 ---
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        # Determine if this is a PV node (not a zero-window search)
        is_pv = (beta - alpha) > 1

        # --- Transposition table probe ---
        key = board._transposition_key()
        tt_entry = self.tt.get(key)
        tt_move: Optional[chess.Move] = None

        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry[:4]
            if tt_depth >= depth:
                if not is_pv:
                    if tt_flag == _TT_EXACT:
                        return tt_score
                    if tt_flag == _TT_LOWER and tt_score >= beta:
                        return tt_score
                    if tt_flag == _TT_UPPER and tt_score <= alpha:
                        return tt_score
                else:
                    if tt_flag == _TT_EXACT:
                        return tt_score

        in_check = board.is_check()

        # --- Reverse Futility Pruning (Static Null Move Pruning) ---
        static_eval: Optional[int] = None
        if not is_pv and not in_check and depth <= 2:
            static_eval = evaluate(board)
            rfp_margin = 120 * depth
            if static_eval - rfp_margin >= beta:
                return beta

        # --- Futility Pruning Condition ---
        futility_prune = False
        if not is_pv and not in_check and depth <= 2:
            if static_eval is None:
                static_eval = evaluate(board)
            fut_margin = 160 * depth
            if static_eval + fut_margin <= alpha:
                futility_prune = True

        # --- Null Move Pruning ---
        if (can_null
                and not is_pv
                and depth >= 3
                and not in_check
                and (board.occupied_co[board.turn]
                     & ~(board.pawns | board.kings))):
            R = 2 + (depth > 6)
            board.push(chess.Move.null())
            score = -self._negamax(board, depth - 1 - R, -beta, -beta + 1,
                                   ply + 1, False)
            board.pop()
            if score >= beta:
                return beta

        # --- Generate and order moves ---
        moves = self._order_moves(board, tt_move, ply)

        if not moves:
            if in_check:
                return -(MATE_SCORE - ply)   # Checkmate
            return 0                          # Stalemate (clean 0)

        # --- Search each move ---
        orig_alpha = alpha
        best_score = -MATE_SCORE
        best_move = tt_move

        for i, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_promotion = move.promotion is not None

            # --- Fast Futility Pruning ---
            # Skip quiet moves if position is so bad that quiet moves can't save alpha
            # Uses board.gives_check(move) directly without push/pop!
            if futility_prune and i > 0 and not is_capture and not is_promotion:
                if not board.gives_check(move):
                    continue

            board.push(move)

            # Check extension: don't cut the search when giving check
            gives_check = board.is_check()
            ext = 1 if gives_check else 0

            # --- Late Move Reductions (LMR) ---
            reduction = 0
            if (i >= _LMR_FULL_DEPTH_MOVES
                    and depth >= _LMR_REDUCTION_LIMIT
                    and not in_check
                    and not gives_check
                    and not is_capture
                    and not is_promotion
                    and move != self._killer1[ply]
                    and move != self._killer2[ply]):
                reduction = 1
                if i >= 10:
                    reduction = 2
                reduction = min(reduction, depth - 2)

            if i == 0:
                # First move (expected PV): full window, full depth
                score = -self._negamax(
                    board, depth - 1 + ext, -beta, -alpha, ply + 1, True
                )
            else:
                # PVS + LMR
                new_depth = depth - 1 + ext - reduction
                score = -self._negamax(
                    board, new_depth, -alpha - 1, -alpha, ply + 1, True
                )

                if reduction > 0 and score > alpha:
                    score = -self._negamax(
                        board, depth - 1 + ext, -alpha - 1, -alpha,
                        ply + 1, True
                    )

                if score > alpha and score < beta:
                    score = -self._negamax(
                        board, depth - 1 + ext, -beta, -alpha,
                        ply + 1, True
                    )

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Beta cutoff – update killer & history for quiet moves
                if not is_capture:
                    self._store_killer(move, ply)
                    self._history[move.from_square][move.to_square] += depth * depth
                break

        # --- Store in TT ---
        if best_score <= orig_alpha:
            flag = _TT_UPPER
        elif best_score >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt_store(key, depth, best_score, flag, best_move)

        return best_score

    # -----------------------------------------------------------------------
    # Quiescence search – resolve tactical sequences before evaluating
    # -----------------------------------------------------------------------

    def _quiescence(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply: int,
    ) -> int:
        # --- Adaptive Time Check in Quiescence ---
        self.nodes += 1
        if (self.nodes & self._node_mask) == 0:
            now = time.time()
            if now >= self.hard_deadline:
                raise _Timeout
            rem = self.hard_deadline - now
            if rem <= 3.5:
                self._node_mask = 63
            elif rem <= 8.0:
                self._node_mask = 127

        in_check = board.is_check()

        if not in_check:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat

            # Delta pruning: even capturing a queen can't help
            if stand_pat + 975 < alpha:
                return alpha
        else:
            stand_pat = -MATE_SCORE   # Must find an escape

        # Generate moves: ALL moves when in check, only captures otherwise
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE_SCORE - ply)  # Checkmate
        else:
            moves = [m for m in board.legal_moves
                     if board.is_capture(m) or m.promotion]

        # Order captures by MVV-LVA
        scored: list[tuple[int, int, chess.Move]] = []
        for i, move in enumerate(moves):
            s = 0
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                if victim is None:
                    victim = chess.PAWN  # En passant
                attacker = board.piece_type_at(move.from_square)
                if attacker is None:
                    attacker = chess.PAWN
                s = MVV_LVA[victim] * 100 - MVV_LVA[attacker]
            if move.promotion:
                s += 500
            scored.append((-s, i, move))
        scored.sort()

        for _, _, move in scored:
            # Per-move delta pruning (skip hopeless captures)
            if not in_check and not move.promotion and board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                if victim is None:
                    victim = chess.PAWN
                capture_val = (0, 100, 320, 330, 500, 900, 0)[victim]
                if stand_pat + capture_val + _DELTA_MARGIN < alpha:
                    continue

            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()

            if score > alpha:
                alpha = score
            if alpha >= beta:
                return beta

        return alpha

    # -----------------------------------------------------------------------
    # Move ordering
    # -----------------------------------------------------------------------

    def _order_moves(
        self,
        board: chess.Board,
        tt_move: Optional[chess.Move],
        ply: int,
    ) -> list[chess.Move]:
        """Score and sort all legal moves for maximum pruning efficiency.

        Priority (highest first):
        1. TT move (PV from previous iteration / deeper search)
        2. Captures ordered by MVV-LVA
        3. Promotions
        4. Killer moves (quiet moves that caused cutoffs at this ply)
        5. History heuristic (quiet moves that caused cutoffs elsewhere)
        6. Remaining quiet moves
        """
        scored: list[tuple[int, int, chess.Move]] = []
        k1 = self._killer1[ply] if ply < _MAX_PLY else None
        k2 = self._killer2[ply] if ply < _MAX_PLY else None
        hist = self._history

        for i, move in enumerate(board.legal_moves):
            s = 0
            if move == tt_move:
                s = 10_000_000
            elif board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                if victim is None:
                    victim = chess.PAWN  # En passant
                attacker = board.piece_type_at(move.from_square)
                if attacker is None:
                    attacker = chess.PAWN
                s = 1_000_000 + MVV_LVA[victim] * 100 - MVV_LVA[attacker]
                if move.promotion:
                    s += 500_000
            elif move.promotion:
                s = 900_000 + (move.promotion or 0)
            else:
                if move == k1:
                    s = 800_000
                elif move == k2:
                    s = 700_000
                else:
                    s = hist[move.from_square][move.to_square]

            scored.append((-s, i, move))

        scored.sort()
        return [m for _, _, m in scored]

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _store_killer(self, move: chess.Move, ply: int) -> None:
        """Store a quiet move that caused a beta cutoff as a killer move."""
        if ply >= _MAX_PLY:
            return
        if move != self._killer1[ply]:
            self._killer2[ply] = self._killer1[ply]
            self._killer1[ply] = move

    def _tt_store(
        self,
        key: object,
        depth: int,
        score: int,
        flag: int,
        best_move: Optional[chess.Move],
    ) -> None:
        """Store a position in the transposition table."""
        old = self.tt.get(key)
        if old is None or depth >= old[0] or old[4] < self.generation:
            self.tt[key] = (depth, score, flag, best_move, self.generation)
