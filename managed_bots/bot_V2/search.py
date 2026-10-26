"""Search algorithms: Negamax Alpha-Beta + Iterative Deepening + Quiescence.

V2 Optimizations over V1:
1. Negamax with Alpha-Beta Pruning   – core search, prunes bad branches
2. Iterative Deepening               – never runs out of time with no move
3. Transposition Table               – caches results to avoid re-computation
4. Move Ordering (MVV-LVA + Killers) – maximizes pruning efficiency
5. Quiescence Search                 – avoids horizon-effect blunders
6. Null Move Pruning                 – skip-turn heuristic for fast cutoffs
7. Check Extensions                  – don't cut off in the middle of checks
8. **Late Move Reductions (LMR)**    – search unpromising moves at reduced depth
9. **Principal Variation Search**    – zero-window for non-PV moves
10. **Aspiration Windows**           – narrow search window from prev iteration
11. **Smart Time Prediction**        – predict if next depth is feasible
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

_TT_MAX_SIZE = 1_000_000  # Max transposition table entries

_MAX_PLY = 128   # Maximum search depth (safety bound)

# Delta pruning margin for quiescence search
_DELTA_MARGIN = 200

# LMR constants
_LMR_FULL_DEPTH_MOVES = 4   # Search first N moves at full depth
_LMR_REDUCTION_LIMIT = 3    # Min depth to apply LMR

# Aspiration window
_ASP_WINDOW = 50   # Initial aspiration window (centipawns)


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

        # Per-search state (reset each call to search())
        self.nodes = 0
        self.last_depth = 0        # Track deepest completed depth
        self.hard_deadline = 0.0
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

        V2 additions:
        - Aspiration windows: use narrow window around previous score
        - Smart time prediction: don't start depth N+1 if estimated time
          exceeds remaining budget (assumes branching factor ~4)
        """
        self.nodes = 0
        self.last_depth = 0
        start = time.time()
        self.hard_deadline = start + hard_limit
        soft_deadline = start + soft_limit

        # Reset per-search heuristics
        self._killer1 = [None] * _MAX_PLY
        self._killer2 = [None] * _MAX_PLY
        self._history = [[0] * 64 for _ in range(64)]

        # Safety fallback: always have *some* move ready
        best_move = next(iter(board.legal_moves))
        prev_score = 0   # For aspiration windows

        root_stack_len = len(board.move_stack)
        try:
            for depth in range(1, _MAX_PLY):
                iter_start = time.time()

                try:
                    # --- Aspiration Windows ---
                    # For depth >= 4, search with a narrow window around the
                    # previous iteration's score. If the score falls outside,
                    # re-search with a wider/full window.
                    if depth >= 4:
                        asp_delta = _ASP_WINDOW
                        alpha = prev_score - asp_delta
                        beta = prev_score + asp_delta

                        score, move = self._search_root(board, depth, alpha, beta)

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
                    # Check soft deadline: if we have reached soft limit, stop cleanly
                    now = time.time()
                    if now >= soft_deadline:
                        break

                    # Predict if next depth iteration would blow past hard deadline.
                    # With PVS + LMR + TT, effective branching factor is ~2.5x.
                    iter_elapsed = now - iter_start
                    if now + iter_elapsed * 2.5 > self.hard_deadline:
                        break

                except _Timeout:
                    break  # Hard deadline hit – use best move from last depth
        finally:
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
            # No legal moves at root (should not happen in normal play)
            if board.is_check():
                return -MATE_SCORE, None
            return 0, None

        for i, move in enumerate(moves):
            board.push(move)

            # Avoid repetitions – treat repeated positions as draws
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
    # LMR, PVS
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
        # --- Time check (every 2048 nodes) ---
        self.nodes += 1
        if self.nodes & 2047 == 0 and time.time() >= self.hard_deadline:
            raise _Timeout

        # --- Draw detection ---
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
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                # In non-PV nodes, we can use TT cutoffs directly
                if not is_pv:
                    if tt_flag == _TT_EXACT:
                        return tt_score
                    if tt_flag == _TT_LOWER and tt_score >= beta:
                        return tt_score
                    if tt_flag == _TT_UPPER and tt_score <= alpha:
                        return tt_score
                else:
                    # In PV nodes, only exact TT hits are reliable
                    if tt_flag == _TT_EXACT:
                        return tt_score

        in_check = board.is_check()

        # --- Null Move Pruning ---
        # Skip when: in check, in endgame (zugzwang risk), already did NMP,
        # or in PV node
        if (can_null
                and not is_pv
                and depth >= 3
                and not in_check
                and (board.occupied_co[board.turn]
                     & ~(board.pawns | board.kings))):
            R = 2 + (depth > 6)  # Adaptive reduction
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
            return 0                          # Stalemate

        # --- Search each move ---
        orig_alpha = alpha
        best_score = -MATE_SCORE
        best_move = tt_move

        for i, move in enumerate(moves):
            board.push(move)

            # Check extension: don't cut the search when giving check
            gives_check = board.is_check()
            ext = 1 if gives_check else 0
            is_capture = board.is_capture(move)
            is_promotion = move.promotion is not None

            # --- Late Move Reductions (LMR) ---
            # For late, quiet, non-check moves at sufficient depth,
            # search at reduced depth first
            reduction = 0
            if (i >= _LMR_FULL_DEPTH_MOVES
                    and depth >= _LMR_REDUCTION_LIMIT
                    and not in_check
                    and not gives_check
                    and not is_capture
                    and not is_promotion
                    and move != self._killer1[ply]
                    and move != self._killer2[ply]):
                # Reduction amount: 1 for most, 2 for very late moves
                reduction = 1
                if i >= 10:
                    reduction = 2
                # Don't reduce below depth 1
                reduction = min(reduction, depth - 2)

            if i == 0:
                # First move (expected PV): always full window, full depth
                score = -self._negamax(
                    board, depth - 1 + ext, -beta, -alpha, ply + 1, True
                )
            else:
                # --- LMR + PVS combined ---
                # Step 1: Try reduced-depth zero-window search
                new_depth = depth - 1 + ext - reduction
                score = -self._negamax(
                    board, new_depth, -alpha - 1, -alpha, ply + 1, True
                )

                # Step 2: If reduced search improved alpha, re-search at
                # full depth with zero window
                if reduction > 0 and score > alpha:
                    score = -self._negamax(
                        board, depth - 1 + ext, -alpha - 1, -alpha,
                        ply + 1, True
                    )

                # Step 3: If zero-window improved alpha, re-search with
                # full window (PVS re-search)
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
        self.nodes += 1
        if self.nodes & 2047 == 0 and time.time() >= self.hard_deadline:
            raise _Timeout

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
                # Simple piece values for delta check
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

            scored.append((-s, i, move))   # i as tiebreaker (Move not comparable)

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
        if len(self.tt) >= _TT_MAX_SIZE:
            self.tt.clear()   # Simple eviction: full reset
        old = self.tt.get(key)
        if old is None or depth >= old[0]:
            self.tt[key] = (depth, score, flag, best_move)
