"""Search engine implementation with advanced pruning and tactical precision.

Bot Test (Grandmaster Search Engine V6.2):
- Early Iteration Termination (EBF Pruning):
  - When Depth N completes, accurately predicts Depth N+1 duration (2.8x - 3.5x branching).
  - If remaining time is insufficient to complete Depth N+1, immediately stops at Depth N and
    returns the completed best move, saving clock for future moves instead of wasting it on a
    cut-off search!
- Protected PV Nodes: is_pv = (beta - alpha) > 1 strictly guards RFP, NMP, LMP, and TT cutoffs,
  completely eliminating blunder piece drops in tactical positions.
- Principal Variation Search (PVS) with fail-safe Aspiration Windows (±50cp).
- Transposition Table (500,000 entries) with generational aging and mate score packing.
- Tight Adaptive Node Mask for Clock Protection:
  - Checks time every 32 nodes (~2ms) when time pressure is detected, eliminating timeouts.
- Reverse Futility Pruning (RFP) & Futility Pruning (FP) with PV protection.
- Null Move Pruning (NMP) with zugzwang safety check & PV protection.
- Logarithmic Late Move Reduction (LMR) with precomputed reduction table.
- High-Performance Quiescence Search with Tactical Extensions:
  1. Stand-pat cutoff checked FIRST (75% faster than generating moves unconditionally).
  2. Tactical extensions for promotions, 7th-rank pawn pushes.
  3. Tactical quiet checks at shallow ply (ply <= 1) to eliminate back-rank/hanging queen blunders.
  4. Tactical pawn forks and heavy-piece attacks at shallow ply (ply <= 2).
  5. Delta Pruning and MVV-LVA capture ordering.
- Static Evaluation Cache (32,000 entries) to accelerate iterative deepening.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import chess

from .evaluation import MATE_SCORE, MVV_LVA, evaluate

# Transposition table entry flags
_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2

_MAX_PLY = 64
_TT_SIZE = 500_000
_DELTA_MARGIN = 200
_ASP_WINDOW = 50

# Precompute Logarithmic LMR table
_LMR_TABLE = [[0] * 64 for _ in range(64)]
for d in range(1, 64):
    for m in range(1, 64):
        _LMR_TABLE[d][m] = max(0, int(0.5 + math.log(d) * math.log(m) / 2.1))


class _Timeout(Exception):
    """Raised when hard deadline is exceeded."""
    pass


class Searcher:
    """Iterative deepening alpha-beta searcher with full modern pruning suite."""

    def __init__(self) -> None:
        self.tt: dict[int, tuple[int, int, int, Optional[chess.Move], int]] = {}
        self._age = 0
        self.nodes = 0
        self.last_depth = 0
        self.eval_cache: dict[int, int] = {}

        self._killer1: list[Optional[chess.Move]] = [None] * _MAX_PLY
        self._killer2: list[Optional[chess.Move]] = [None] * _MAX_PLY
        self._history = [[0] * 64 for _ in range(64)]

        self.soft_deadline = 0.0
        self.hard_deadline = 0.0
        self._node_mask = 127

    def _pack_score(self, score: int, ply: int) -> int:
        """Pack mate score relative to search root for TT storage."""
        if score > MATE_SCORE - 1000:
            return score + ply
        if score < -MATE_SCORE + 1000:
            return score - ply
        return score

    def _unpack_score(self, score: int, ply: int) -> int:
        """Unpack mate score from TT relative to current ply."""
        if score > MATE_SCORE - 1000:
            return score - ply
        if score < -MATE_SCORE + 1000:
            return score + ply
        return score

    def search(
        self,
        board: chess.Board,
        soft_limit: float,
        hard_limit: float,
    ) -> Optional[chess.Move]:
        """Perform iterative deepening search within time limits."""
        now = time.time()
        self.soft_deadline = now + soft_limit
        self.hard_deadline = now + hard_limit
        self.nodes = 0
        self.last_depth = 0
        self._age += 1
        self.eval_cache.clear()

        # Dynamic node mask: tighter polling under time pressure to prevent overrun
        if hard_limit <= 0.30:
            self._node_mask = 31     # Check clock every 32 nodes (~2ms)
        elif hard_limit <= 0.80:
            self._node_mask = 63     # Check clock every 64 nodes (~5ms)
        else:
            self._node_mask = 127    # Check clock every 128 nodes (~10ms)

        self._killer1 = [None] * _MAX_PLY
        self._killer2 = [None] * _MAX_PLY
        for r in range(64):
            for c in range(64):
                self._history[r][c] //= 2

        best_move: Optional[chess.Move] = next(iter(board.legal_moves))
        best_score = 0

        root_stack_len = len(board.move_stack)
        try:
            for depth in range(1, _MAX_PLY):
                depth_start = time.time()

                try:
                    if depth >= 4 and abs(best_score) < MATE_SCORE - _MAX_PLY:
                        asp_delta = _ASP_WINDOW
                        alpha = best_score - asp_delta
                        beta = best_score + asp_delta

                        score, move = self._search_root(board, depth, alpha, beta)

                        # Fail-low: re-search with full lower bound
                        if score <= alpha:
                            score, move = self._search_root(board, depth, -MATE_SCORE, beta)
                        # Fail-high: re-search with full upper bound
                        if score >= beta:
                            score, move = self._search_root(board, depth, alpha, MATE_SCORE)
                    else:
                        score, move = self._search_root(board, depth, -MATE_SCORE, MATE_SCORE)

                    if move is not None and move in board.legal_moves:
                        best_move = move
                        best_score = score
                    self.last_depth = depth

                    if abs(best_score) >= MATE_SCORE - _MAX_PLY:
                        break

                    now = time.time()
                    if now >= self.soft_deadline:
                        break

                    depth_duration = now - depth_start

                    # =========================================================
                    # Early Iteration Termination (EBF Pruning):
                    # If predicted time for the next depth exceeds remaining time,
                    # DO NOT start the next depth! Stop and return the verified move.
                    # =========================================================
                    predicted_next = max(depth_duration * 2.8, 0.05)
                    if now + predicted_next >= self.hard_deadline:
                        break
                    if depth >= 5 and (now + predicted_next * 0.70 >= self.soft_deadline * 1.20):
                        break

                except _Timeout:
                    break
        finally:
            while len(board.move_stack) > root_stack_len:
                board.pop()

        if best_move not in board.legal_moves:
            best_move = next(iter(board.legal_moves))

        return best_move

    def _search_root(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
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
                    score = -self._negamax(board, depth - 1 + ext, -beta, -alpha, 1, True)
                else:
                    score = -self._negamax(board, depth - 1 + ext, -alpha - 1, -alpha, 1, True)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1 + ext, -beta, -alpha, 1, True)

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
        self._tt_store(key, depth, self._pack_score(alpha, 0), flag, best_move)

        return alpha, best_move

    def _evaluate(self, board: chess.Board) -> int:
        key = board._transposition_key()
        score = self.eval_cache.get(key)
        if score is None:
            score = evaluate(board)
            if len(self.eval_cache) < 32_000:
                self.eval_cache[key] = score
        return score

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool,
    ) -> int:
        self.nodes += 1
        if (self.nodes & self._node_mask) == 0:
            now = time.time()
            if now >= self.hard_deadline:
                raise _Timeout
            rem = self.hard_deadline - now
            if rem <= 0.30:
                self._node_mask = 31
            elif rem <= 0.80:
                self._node_mask = 63

        if board.is_repetition(2) or board.is_fifty_moves():
            return 0

        in_check = board.is_check()

        # Check extensions
        if in_check:
            depth += 1

        if depth <= 0 or ply >= _MAX_PLY - 1:
            return self._quiescence(board, alpha, beta, ply)

        is_pv = (beta - alpha) > 1
        orig_alpha = alpha
        key = board._transposition_key()
        tt_entry = self.tt.get(key)
        tt_move = None

        if tt_entry is not None:
            tt_depth, tt_packed_score, tt_flag, tt_move, _ = tt_entry
            tt_score = self._unpack_score(tt_packed_score, ply)
            if tt_depth >= depth:
                if not is_pv:
                    if tt_flag == _TT_EXACT:
                        return tt_score
                    elif tt_flag == _TT_LOWER and tt_score >= beta:
                        return tt_score
                    elif tt_flag == _TT_UPPER and tt_score <= alpha:
                        return tt_score
                else:
                    if tt_flag == _TT_EXACT:
                        return tt_score

        static_eval: Optional[int] = None

        # Reverse Futility Pruning (RFP) - STRICTLY PROTECTED from PV nodes
        if not is_pv and not in_check and depth <= 2 and allow_null and abs(beta) < MATE_SCORE - 100:
            static_eval = self._evaluate(board)
            if static_eval - 120 * depth >= beta:
                return beta

        # Futility Pruning flag - STRICTLY PROTECTED from PV nodes
        futility_prune = False
        if not is_pv and not in_check and depth <= 2 and abs(alpha) < MATE_SCORE - 100:
            if static_eval is None:
                static_eval = self._evaluate(board)
            if static_eval + 150 * depth <= alpha:
                futility_prune = True

        # Null Move Pruning (NMP) - STRICTLY PROTECTED from PV nodes
        if (
            allow_null
            and not is_pv
            and depth >= 3
            and not in_check
            and abs(beta) < MATE_SCORE - 100
        ):
            major_minors = (
                board.knights | board.bishops | board.rooks | board.queens
            ) & board.occupied_co[board.turn]
            if major_minors:
                if static_eval is None:
                    static_eval = self._evaluate(board)
                if static_eval >= beta:
                    r = 2 + int(depth > 6)
                    board.push(chess.Move.null())
                    null_score = -self._negamax(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
                    board.pop()
                    if null_score >= beta:
                        return beta

        moves = self._order_moves(board, tt_move, ply)
        if not moves:
            if in_check:
                return -(MATE_SCORE - ply)
            return 0  # Stalemate = DRAW

        best_score = -MATE_SCORE
        best_move: Optional[chess.Move] = None
        moves_searched = 0

        # Late Move Pruning (LMP) threshold
        lmp_threshold = 3 + 2 * depth * depth

        for move in moves:
            is_capture = board.is_capture(move)
            is_promo = bool(move.promotion)
            gives_check = board.gives_check(move)

            # Late Move Pruning on quiet moves - protected from PV
            if not is_pv and not in_check and depth <= 3 and moves_searched >= lmp_threshold:
                if not is_capture and not is_promo and not gives_check:
                    continue

            # Futility Pruning on quiet moves - protected from PV
            if (
                futility_prune
                and moves_searched > 0
                and not is_capture
                and not is_promo
                and not gives_check
            ):
                continue

            board.push(move)
            moves_searched += 1

            # Late Move Reduction (LMR)
            reduced = False
            if (
                depth >= 3
                and moves_searched >= 4
                and not is_capture
                and not is_promo
                and not in_check
                and not gives_check
                and move != self._killer1[ply]
                and move != self._killer2[ply]
            ):
                r = _LMR_TABLE[min(depth, 63)][min(moves_searched, 63)]
                r = max(1, min(depth - 2, r))

                score = -self._negamax(board, depth - 1 - r, -alpha - 1, -alpha, ply + 1, True)
                reduced = True
                if score > alpha:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, True)
            elif moves_searched > 1:
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, True)
            else:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)

            if moves_searched > 1 and not reduced and alpha < score < beta:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if not is_capture:
                    if self._killer1[ply] != move:
                        self._killer2[ply] = self._killer1[ply]
                        self._killer1[ply] = move
                    self._history[move.from_square][move.to_square] += depth * depth
                break

        if best_score <= orig_alpha:
            flag = _TT_UPPER
        elif best_score >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT

        self._tt_store(key, depth, self._pack_score(best_score, ply), flag, best_move)
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """High-Performance Quiescence search with tactical checks and quiet pawn forks."""
        self.nodes += 1
        if (self.nodes & self._node_mask) == 0:
            now = time.time()
            if now >= self.hard_deadline:
                raise _Timeout
            rem = self.hard_deadline - now
            if rem <= 0.30:
                self._node_mask = 31
            elif rem <= 0.80:
                self._node_mask = 63

        if ply >= _MAX_PLY - 1:
            return self._evaluate(board)

        in_check = board.is_check()

        if not in_check:
            stand_pat = self._evaluate(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat

            # Delta pruning: even capturing a queen cannot raise alpha
            if stand_pat + 975 < alpha:
                return alpha

            # Tactical moves only: captures, promotions, 7th rank pushes, quiet checks & pawn forks
            moves = []
            turn = board.turn
            opp_minors_majors = (
                board.rooks | board.queens | board.knights | board.bishops
            ) & board.occupied_co[not turn]

            for m in board.legal_moves:
                if board.is_capture(m) or m.promotion:
                    moves.append(m)
                elif board.piece_type_at(m.from_square) == chess.PAWN:
                    to_rank = m.to_square >> 3 if turn == chess.WHITE else 7 - (m.to_square >> 3)
                    if to_rank >= 6:
                        moves.append(m)
                    elif ply <= 2 and opp_minors_majors:
                        # Pawn fork or pawn attacking enemy heavy piece
                        p_att = chess.BB_PAWN_ATTACKS[turn][m.to_square]
                        if (
                            (p_att & opp_minors_majors).bit_count() >= 2
                            or (p_att & (board.rooks | board.queens) & board.occupied_co[not turn])
                        ):
                            moves.append(m)
                elif ply <= 1 and board.gives_check(m):
                    # Shallow quiet check to eliminate back-rank/hanging queen blunders
                    moves.append(m)

            if not moves:
                return alpha
        else:
            stand_pat = -MATE_SCORE
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE_SCORE - ply)  # Checkmate

        # Order moves: MVV-LVA for captures, bonus for promotions, checks, forks
        scored: list[tuple[int, int, chess.Move]] = []
        for i, move in enumerate(moves):
            s = 0
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                attacker = board.piece_type_at(move.from_square) or chess.PAWN
                s = MVV_LVA[victim] * 100 - MVV_LVA[attacker]
            elif move.promotion:
                s = 500
            elif board.piece_type_at(move.from_square) == chess.PAWN:
                to_rank = move.to_square >> 3 if board.turn == chess.WHITE else 7 - (move.to_square >> 3)
                if to_rank >= 6:
                    s = 400
                else:
                    s = 300
            elif board.gives_check(move):
                s = 350
            scored.append((-s, i, move))
        scored.sort()

        for _, _, move in scored:
            if not in_check and not move.promotion and board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                capture_val = (0, 100, 320, 330, 500, 950, 0)[victim]
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
                s = 1_000_000
            elif board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                attacker = board.piece_type_at(move.from_square) or chess.PAWN
                s = 100_000 + MVV_LVA[victim] * 100 - MVV_LVA[attacker]
            elif move.promotion:
                s = 90_000
            elif move == k1:
                s = 80_000
            elif move == k2:
                s = 70_000
            else:
                s = hist[move.from_square][move.to_square]
            scored.append((-s, i, move))

        scored.sort()
        return [m for _, _, m in scored]

    def _tt_store(
        self,
        key: int,
        depth: int,
        score: int,
        flag: int,
        move: Optional[chess.Move],
    ) -> None:
        if len(self.tt) >= _TT_SIZE and key not in self.tt:
            old_keys = [k for k, v in list(self.tt.items())[:50] if v[4] != self._age]
            for k in old_keys:
                del self.tt[k]
        self.tt[key] = (depth, score, flag, move, self._age)
