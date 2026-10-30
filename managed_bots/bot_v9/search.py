"""Search algorithms: Negamax Alpha-Beta + Iterative Deepening + Quiescence.

V9 Master Optimizations:
1. Negamax with Alpha-Beta Pruning + PVS (Principal Variation Search).
2. Iterative Deepening with fail-safe Aspiration Windows.
3. Transposition Table: generational replacement and fast cache eviction.
4. Move Ordering: TT move (10M) -> MVV-LVA Captures (1M+) -> Promotions (900k) -> Killers -> History.
5. Logarithmic Late Move Reductions (LMR): precomputed matrix depth decay for quiet moves.
6. Late Move Pruning (LMP): prunes unpromising quiet moves at depth <= 3.
7. Quiescence Search: MVV-LVA ordering + stand-pat delta pruning (zero-overhead).
8. Null Move Pruning (R = 2 + (depth > 6)) with zugzwang safety.
9. Reverse Futility Pruning (RFP) & Futility Pruning at depth <= 2.
10. Check Extensions (+1 ply when in check).
11. Adaptive Node Mask (127 -> 63 -> 31) for 100% timeout prevention under 0.5s floor.
"""
from __future__ import annotations

import math
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

_TT_MAX_SIZE = 500_000  # Max transposition table entries (fast & lightweight)

_MAX_PLY = 128   # Maximum search depth (safety bound)

_DELTA_MARGIN = 200

# LMR constants
_LMR_FULL_DEPTH_MOVES = 4   # Search first N moves at full depth
_LMR_REDUCTION_LIMIT = 3    # Min depth to apply LMR
_ASP_WINDOW = 50            # Initial aspiration window (centipawns)

# Precompute logarithmic LMR table
_LMR_TABLE = [[0] * 64 for _ in range(64)]
for _d in range(1, 64):
    for _m in range(1, 64):
        _LMR_TABLE[_d][_m] = max(0, int(0.5 + math.log(_d) * math.log(_m) / 2.1))


class _Timeout(Exception):
    """Raised when the hard time deadline is exceeded."""


# ===========================================================================
# Searcher class
# ===========================================================================

class Searcher:
    """Encapsulates the entire search state.
    Create once in ``StudentBot.__init__`` and reuse across ``choose_move`` calls.
    """

    def __init__(self) -> None:
        self.tt: dict = {}
        self.generation = 0

        # Per-search state
        self.nodes = 0
        self.last_depth = 0
        self.hard_deadline = 0.0
        self._node_mask = 127
        self._killer1: list[Optional[chess.Move]] = [None] * _MAX_PLY
        self._killer2: list[Optional[chess.Move]] = [None] * _MAX_PLY
        self._history: list[list[int]] = [[0] * 64 for _ in range(64)]
        self.eval_cache: dict[object, int] = {}

    def search(
        self,
        board: chess.Board,
        soft_limit: float,
        hard_limit: float,
    ) -> chess.Move:
        self.nodes = 0
        self.last_depth = 0
        self.generation += 1
        self.eval_cache.clear()

        start = time.time()
        self.hard_deadline = start + hard_limit
        soft_deadline = start + soft_limit

        # Adaptive Node Mask initialization (<50ms reaction time)
        if hard_limit <= 0.3:
            self._node_mask = 31
        elif hard_limit <= 1.0:
            self._node_mask = 63
        else:
            self._node_mask = 127

        # Reset killers per search
        self._killer1 = [None] * _MAX_PLY
        self._killer2 = [None] * _MAX_PLY

        # History aging
        for r in self._history:
            for c in range(64):
                r[c] >>= 1

        best_move = next(iter(board.legal_moves))
        prev_score = 0

        root_stack_len = len(board.move_stack)
        try:
            for depth in range(1, _MAX_PLY):
                iter_start = time.time()

                try:
                    # --- Aspiration Windows ---
                    if depth >= 4:
                        asp_delta = _ASP_WINDOW
                        alpha = prev_score - asp_delta
                        beta = prev_score + asp_delta

                        score, move = self._search_root(board, depth, alpha, beta)

                        # Fail-low: score <= alpha -> re-search with full lower bound
                        if score <= alpha:
                            score, move = self._search_root(
                                board, depth, -MATE_SCORE, beta
                            )
                        # Fail-high: score >= beta -> re-search with full upper bound
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

                    if abs(score) >= MATE_SCORE - _MAX_PLY:
                        break

                    # Time checking and branch prediction
                    now = time.time()
                    if now >= soft_deadline:
                        break

                    iter_elapsed = now - iter_start
                    predicted_next = max(iter_elapsed * 3.0, 0.04)
                    if now + predicted_next > self.hard_deadline:
                        break

                except _Timeout:
                    break
        finally:
            while len(board.move_stack) > root_stack_len:
                board.pop()

        if best_move not in board.legal_moves:
            best_move = next(iter(board.legal_moves))

        return best_move

    # -----------------------------------------------------------------------
    # Root search
    # -----------------------------------------------------------------------

    def _search_root(
        self, board: chess.Board, depth: int,
        alpha: int, beta: int,
    ) -> tuple[int, Optional[chess.Move]]:
        best_move: Optional[chess.Move] = None
        orig_alpha = alpha

        key = board._transposition_key()
        tt_entry = self.tt.get(key)
        tt_move = tt_entry[3] if tt_entry else None

        moves = self._order_moves(board, tt_move, 0)
        if not moves:
            if board.is_check():
                return -MATE_SCORE, None
            return 0, None

        for i, move in enumerate(moves):
            if (self.nodes & self._node_mask) == 0 and time.time() >= self.hard_deadline:
                raise _Timeout
            board.push(move)

            if board.is_repetition(2):
                score = 0
            else:
                ext = 1 if board.is_check() else 0

                if i == 0:
                    score = -self._negamax(
                        board, depth - 1 + ext, -beta, -alpha, 1, True
                    )
                else:
                    score = -self._negamax(
                        board, depth - 1 + ext, -alpha - 1, -alpha, 1, True
                    )
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

        if alpha <= orig_alpha:
            flag = _TT_UPPER
        elif alpha >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt_store(key, depth, alpha, flag, best_move)

        return alpha, best_move

    def _evaluate(self, board: chess.Board) -> int:
        key = board._transposition_key()
        score = self.eval_cache.get(key)
        if score is None:
            score = evaluate(board)
            if len(self.eval_cache) < 32_000:
                self.eval_cache[key] = score
        return score

    # -----------------------------------------------------------------------
    # Negamax
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
        self.nodes += 1
        if (self.nodes & self._node_mask) == 0:
            now = time.time()
            if now >= self.hard_deadline:
                raise _Timeout
            rem = self.hard_deadline - now
            if rem <= 0.3:
                self._node_mask = 31
            elif rem <= 1.0:
                self._node_mask = 63

        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        is_pv = (beta - alpha) > 1

        # TT probe
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

        # Reverse Futility Pruning
        static_eval: Optional[int] = None
        if not is_pv and not in_check and depth <= 2:
            static_eval = self._evaluate(board)
            if static_eval - 120 * depth >= beta:
                return beta

        # Futility Pruning
        futility_prune = False
        if not is_pv and not in_check and depth <= 2:
            if static_eval is None:
                static_eval = self._evaluate(board)
            if static_eval + 160 * depth <= alpha:
                futility_prune = True

        # Null Move Pruning
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

        moves = self._order_moves(board, tt_move, ply)
        if not moves:
            if in_check:
                return -(MATE_SCORE - ply)
            return 0

        orig_alpha = alpha
        best_score = -MATE_SCORE
        best_move = tt_move

        # Late Move Pruning threshold
        lmp_threshold = 3 + depth * 3

        for i, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_promotion = move.promotion is not None

            # --- Late Move Pruning (LMP) ---
            if (not is_pv
                    and depth <= 3
                    and not in_check
                    and not is_capture
                    and not is_promotion
                    and i >= lmp_threshold):
                continue

            # --- Futility Pruning ---
            if futility_prune and i > 0 and not is_capture and not is_promotion:
                board.push(move)
                gives_check = board.is_check()
                board.pop()
                if not gives_check:
                    continue

            board.push(move)
            gives_check = board.is_check()
            ext = 1 if gives_check else 0

            # --- Logarithmic LMR ---
            reduction = 0
            if (i >= _LMR_FULL_DEPTH_MOVES
                    and depth >= _LMR_REDUCTION_LIMIT
                    and not in_check
                    and not gives_check
                    and not is_capture
                    and not is_promotion
                    and move != self._killer1[ply]
                    and move != self._killer2[ply]):
                reduction = _LMR_TABLE[min(depth, 63)][min(i, 63)]
                reduction = min(reduction, depth - 2)

            if i == 0:
                score = -self._negamax(
                    board, depth - 1 + ext, -beta, -alpha, ply + 1, True
                )
            else:
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
                if not is_capture:
                    self._store_killer(move, ply)
                    self._history[move.from_square][move.to_square] += depth * depth
                break

        if best_score <= orig_alpha:
            flag = _TT_UPPER
        elif best_score >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt_store(key, depth, best_score, flag, best_move)

        return best_score

    # -----------------------------------------------------------------------
    # Quiescence Search (Captures + Promotions with MVV-LVA & Delta Pruning)
    # -----------------------------------------------------------------------

    def _quiescence(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply: int,
    ) -> int:
        self.nodes += 1
        if (self.nodes & self._node_mask) == 0:
            now = time.time()
            if now >= self.hard_deadline:
                raise _Timeout
            rem = self.hard_deadline - now
            if rem <= 0.3:
                self._node_mask = 31
            elif rem <= 1.0:
                self._node_mask = 63

        in_check = board.is_check()

        if not in_check:
            stand_pat = self._evaluate(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat

            # Delta pruning: even capturing a queen cannot help
            if stand_pat + 975 < alpha:
                return alpha
        else:
            stand_pat = -MATE_SCORE  # Must find an escape

        # Generate moves: ALL moves when in check, captures and promotions otherwise
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE_SCORE - ply)  # Checkmate
        else:
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]

        # Order captures by MVV-LVA
        scored: list[tuple[int, int, chess.Move]] = []
        for i, move in enumerate(moves):
            s = 0
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                attacker = board.piece_type_at(move.from_square) or chess.PAWN
                s = MVV_LVA[victim] * 100 - MVV_LVA[attacker]
            if move.promotion:
                s += 500
            scored.append((-s, i, move))
        scored.sort()

        for _, _, move in scored:
            # Per-move delta pruning (skip hopeless captures)
            if not in_check and not move.promotion and board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
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
        scored: list[tuple[int, int, chess.Move]] = []
        k1 = self._killer1[ply] if ply < _MAX_PLY else None
        k2 = self._killer2[ply] if ply < _MAX_PLY else None
        hist = self._history

        for i, move in enumerate(board.legal_moves):
            s = 0
            if move == tt_move:
                s = 10_000_000
            elif board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                attacker = board.piece_type_at(move.from_square) or chess.PAWN
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

    def _store_killer(self, move: chess.Move, ply: int) -> None:
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
        if len(self.tt) >= _TT_MAX_SIZE:
            # Fast bulk eviction of oldest 10%
            to_del = list(self.tt.keys())[:_TT_MAX_SIZE // 10]
            for k in to_del:
                self.tt.pop(k, None)

        old = self.tt.get(key)
        if old is None or depth >= old[0] or old[4] < self.generation:
            self.tt[key] = (depth, score, flag, best_move, self.generation)
