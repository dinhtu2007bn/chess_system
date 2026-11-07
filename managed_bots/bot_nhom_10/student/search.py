"""Concise, high-performance alpha-beta search engine for Chess960.

Features:
  - Iterative Deepening with Gradual Aspiration Windows
  - Negamax with Alpha-Beta Pruning & Fail-Soft Principal Variation Search (PVS)
  - Transposition Table (64-bit Zobrist key + ply-normalized Mate scores) in Main & Q-Search
  - O(1) Path & Game-History Repetition Detection (zero stack-copy overhead)
  - Static Exchange Evaluation (SEE) with X-ray bitboards for Capture Ordering & Q-Pruning
  - Internal Iterative Reduction (IIR)
  - Improving-aware Reverse Futility Pruning (Static Null Move Pruning)
  - Adaptive Null Move Pruning (side-to-move Zugzwang safe + verification at high depth)
  - Shallow Futility Pruning & Late Move Pruning (LMP)
  - Logarithmic Late Move Reductions (LMR) with Killer/Countermove protection
  - Check Extensions & Mate Distance Pruning
  - Soft / Hard Responsive Time Management
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import chess

from .evaluation import evaluate

__all__ = ["SearchEngine", "search_best_move"]

INFINITY = 999_999
MATE_SCORE = 900_000
MAX_PLY = 96
MATE_BOUND = MATE_SCORE - MAX_PLY - 64
TT_MASK = (1 << 20) - 1  # 1,048,576 slots

EXACT = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

# Real centipawn piece values indexed by chess.PieceType (1..6)
_PIECE_VAL = (0, 100, 320, 330, 500, 1000, 20000)

# Precomputed logarithmic Late Move Reduction table [depth][move_index]
_LMR = [[0] * 64 for _ in range(64)]
for _d in range(1, 64):
    for _m in range(1, 64):
        _LMR[_d][_m] = int(0.75 + math.log(_d) * math.log(_m) / 2.25)


_NOT_FILE_A = ~chess.BB_FILE_A & 0xFFFFFFFFFFFFFFFF
_NOT_FILE_H = ~chess.BB_FILE_H & 0xFFFFFFFFFFFFFFFF

_FRONT_FILE_W = [0] * 64
_FRONT_FILE_B = [0] * 64
for _sq in range(64):
    _f, _r = _sq & 7, _sq >> 3
    _fbb = chess.BB_FILES[_f]
    for _ar in range(_r + 1, 8):
        _FRONT_FILE_W[_sq] |= _fbb & chess.BB_RANKS[_ar]
    for _ar in range(0, _r):
        _FRONT_FILE_B[_sq] |= _fbb & chess.BB_RANKS[_ar]


_BB_BETWEEN = tuple(
    tuple(chess.between(s1, s2) for s2 in range(64))
    for s1 in range(64)
)

_RAY_BEHIND = [[0] * 64 for _ in range(64)]
for _s in range(64):
    for _t1 in range(64):
        if _s != _t1:
            _r = chess.ray(_s, _t1)
            if _r:
                _RAY_BEHIND[_s][_t1] = _r & ~(1 << _s) & ~(1 << _t1) & ~_BB_BETWEEN[_s][_t1]

_RAY_BEHIND = tuple(tuple(_row) for _row in _RAY_BEHIND)

def _to_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


def get_early_center_response(board: chess.Board) -> Optional[chess.Move]:
    """In the first move, if the opponent advanced one of the two central pawns
    2 squares and our opposing response is safe (SEE >= 0), respond immediately.
    """
    if board.fullmove_number != 1 or len(board.move_stack) != 1:
        return None
    last = board.peek()
    if last == chess.Move.from_uci("e2e4"):
        resp = chess.Move.from_uci("e7e5")
    elif last == chess.Move.from_uci("d2d4"):
        resp = chess.Move.from_uci("d7d5")
    else:
        return None

    if resp in board.legal_moves:
        if _see(board, resp) < 0:
            return None
        if board.attackers(board.turn, last.to_square):
            return None
        return resp
    return None


def _see(board: chess.Board, move: chess.Move) -> int:
    """Fast bitboard Static Exchange Evaluation with X-rays, en passant, and promotions."""
    target = move.to_square
    attacker = board.piece_type_at(move.from_square) or 1
    victim = board.piece_type_at(target)
    occ = board.occupied ^ (1 << move.from_square)

    if victim is None:
        if board.is_en_passant(move):
            victim = 1
            ep_sq = target - 8 if board.turn else target + 8
            occ &= ~(1 << ep_sq)
            gain0 = _PIECE_VAL[1]
        else:
            gain0 = 0
    else:
        occ &= ~(1 << target)
        gain0 = _PIECE_VAL[victim]
    promo = move.promotion
    if promo is not None:
        gain0 += _PIECE_VAL[promo] - 100
        occupant_val = _PIECE_VAL[promo]
    else:
        occupant_val = _PIECE_VAL[attacker]

    side = not board.turn
    attackers = board.attackers_mask(side, target, occ) & occ
    if not attackers:
        return gain0

    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens
    kings = board.kings

    gains = [gain0]
    while attackers:
        if attackers & pawns:
            cands = attackers & pawns
            pt = 1
        elif attackers & knights:
            cands = attackers & knights
            pt = 2
        elif attackers & bishops:
            cands = attackers & bishops
            pt = 3
        elif attackers & rooks:
            cands = attackers & rooks
            pt = 4
        elif attackers & queens:
            cands = attackers & queens
            pt = 5
        else:
            cands = attackers & kings
            pt = 6

        lsb = cands & -cands
        after = occ ^ lsb
        not_side = not side
        next_attackers = board.attackers_mask(not_side, target, after) & after
        if pt == 6 and next_attackers:
            break

        gain = occupant_val - gains[-1]
        if pt == 1 and (target >> 3) in (0, 7):
            gain += 900
            occupant_val = 1000
        else:
            occupant_val = _PIECE_VAL[pt]
        gains.append(gain)
        occ = after
        side = not_side
        attackers = next_attackers

    for i in range(len(gains) - 1, 0, -1):
        if -gains[i] < gains[i - 1]:
            gains[i - 1] = -gains[i]
    return gains[0]


@dataclass(slots=True)
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int
    best_move: Optional[chess.Move]


class SearchEngine:
    def __init__(self) -> None:
        self.tt: dict[int, TTEntry] = {}
        self.killers: list[list[Optional[chess.Move]]] = [[None, None] for _ in range(MAX_PLY)]
        self.countermoves: list[Optional[chess.Move]] = [None] * 768
        self.history: list[list[int]] = [[0] * 64 for _ in range(12)]
        self.path_keys: list[int] = [0] * (MAX_PLY + 4)
        self.static_evals: list[int] = [0] * (MAX_PLY + 4)
        self.game_keys: set[int] = set()
        self.nodes = 0
        self.time_limit = 0.0
        self.hard_limit = 0.0
        self.stopped = False
        self.best_root_move: Optional[chess.Move] = None
        self.completed_depth = 0

    def _init_game_keys(self, board: chess.Board) -> None:
        self.game_keys.clear()
        limit = min(board.halfmove_clock, len(board.move_stack))
        if limit <= 0:
            return
        tmp = board.copy(stack=True)
        for _ in range(limit):
            tmp.pop()
            self.game_keys.add(hash(tmp._transposition_key()) & 0xFFFFFFFFFFFFFFFF)

    def _prev_move_index(self, board: chess.Board) -> int:
        if not board.move_stack:
            return -1
        last = board.move_stack[-1]
        if not last:
            return -1
        pt = board.piece_type_at(last.to_square) or chess.KING
        opp_offset = 6 if board.turn else 0
        return ((pt - 1) + opp_offset) * 64 + last.to_square

    def _score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        ply: int,
        tt_move: Optional[chess.Move],
        countermove: Optional[chess.Move] = None,
    ) -> int:
        if move == tt_move:
            return 20_000_000
        promo = move.promotion
        if promo is not None:
            if promo == chess.QUEEN:
                victim = board.piece_type_at(move.to_square) or 0
                return 15_000_000 + _PIECE_VAL[victim]
            return -600_000 + promo
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or 1
            attacker = board.piece_type_at(move.from_square) or 1
            mvv_lva = _PIECE_VAL[victim] * 8 - _PIECE_VAL[attacker]
            if _PIECE_VAL[attacker] > _PIECE_VAL[victim] + 50:
                see_val = _see(board, move)
                if see_val < 0:
                    return -200_000 + see_val
            return 10_000_000 + mvv_lva
        if ply < MAX_PLY:
            k0, k1 = self.killers[ply]
            if move == k0:
                return 8_000_000
            if move == k1:
                return 7_000_000
        # Boost tactical threats: Pawn forks, Knight forks, and Skewers
        pt = board.piece_type_at(move.from_square)
        to_sq = move.to_square

        if pt == chess.KNIGHT:
            natk = chess.BB_KNIGHT_ATTACKS[to_sq]
            enemy_fork_targets = (board.kings | board.queens | board.rooks | board.bishops | board.knights) & board.occupied_co[not board.turn]
            hits = natk & enemy_fork_targets
            if (hits & (hits - 1)) and _see(board, move) >= 0:
                return 7_500_000

        elif pt == chess.PAWN:
            patk = chess.BB_PAWN_ATTACKS[board.turn][to_sq]
            enemy_pawn_targets = (board.queens | board.rooks | board.bishops | board.knights) & board.occupied_co[not board.turn]
            hits = patk & enemy_pawn_targets
            if (hits & (hits - 1)) and _see(board, move) >= 0:
                return 7_600_000

        elif pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            opp = not board.turn
            opp_targets = (board.kings | board.queens | board.rooks) & board.occupied_co[opp]
            if opp_targets:
                f1, r1 = to_sq & 7, to_sq >> 3
                occ_after = (board.occupied ^ (1 << move.from_square)) | (1 << to_sq)
                opp_pieces = board.occupied_co[opp] & ~board.pawns
                for t1 in chess.scan_forward(opp_targets):
                    f2, r2 = t1 & 7, t1 >> 3
                    if pt == chess.BISHOP and abs(f1 - f2) != abs(r1 - r2):
                        continue
                    if pt == chess.ROOK and f1 != f2 and r1 != r2:
                        continue
                    if pt == chess.QUEEN and (ply <= 6 and not board.is_capture(move)):
                        continue
                    if not (occ_after & _BB_BETWEEN[to_sq][t1]):
                        behind = _RAY_BEHIND[to_sq][t1] & opp_pieces
                        if behind:
                            for t2 in chess.scan_forward(behind):
                                if not (occ_after & _BB_BETWEEN[t1][t2]) and _see(board, move) >= 0:
                                    return 7_400_000

        if countermove is not None and move == countermove:
            return 6_000_000
        if ply <= 2:
            if move.from_square in (chess.E2, chess.D2, chess.E7, chess.D7) and move.to_square in (
                chess.E4,
                chess.D4,
                chess.E5,
                chess.D5,
            ):
                if _see(board, move) >= 0:
                    return 5_000_000
                else:
                    return -300_000
            if board.is_castling(move):
                return 4_500_000
        if ply <= 4 and pt == chess.QUEEN and not board.is_capture(move):
            if to_sq in (chess.D4, chess.E4, chess.D5, chess.E5, chess.C4, chess.F4, chess.C5, chess.F5):
                return -200_000
            back_rank = chess.BB_RANK_1 if board.turn else chess.BB_RANK_8
            if ((board.knights | board.bishops) & back_rank).bit_count() >= 2:
                return -100_000
        pt = board.piece_type_at(move.from_square)
        if pt == chess.PAWN:
            to_sq = move.to_square
            if board.turn:
                new_atk = (((1 << to_sq) & _NOT_FILE_A) << 7) | (((1 << to_sq) & _NOT_FILE_H) << 9)
                if (new_atk & board.pawns & board.occupied_co[chess.BLACK] & (chess.BB_RANK_4 | chess.BB_RANK_3)) and _see(board, move) >= 0:
                    return 3_800_000
            else:
                new_atk = (((1 << to_sq) & _NOT_FILE_H) >> 7) | (((1 << to_sq) & _NOT_FILE_A) >> 9)
                if (new_atk & board.pawns & board.occupied_co[chess.WHITE] & (chess.BB_RANK_5 | chess.BB_RANK_6)) and _see(board, move) >= 0:
                    return 3_800_000

        if pt == chess.PAWN and not board.is_capture(move):
            to_sq = move.to_square
            to_r = to_sq >> 3
            to_f = to_sq & 7
            occ_after = board.occupied ^ (1 << move.from_square)
            if board.turn:
                if to_r >= 3 or (to_r >= 2 and to_f in (2, 3, 4, 5)):
                    behind = occ_after & _FRONT_FILE_B[to_sq]
                    if behind:
                        nearest = 1 << (behind.bit_length() - 1)
                        if (nearest & board.rooks & board.occupied_co[chess.WHITE]) and _see(board, move) >= 0:
                            return 3_500_000
            else:
                if to_r <= 4 or (to_r <= 5 and to_f in (2, 3, 4, 5)):
                    behind = occ_after & _FRONT_FILE_W[to_sq]
                    if behind:
                        nearest = behind & -behind
                        if (nearest & board.rooks & board.occupied_co[chess.BLACK]) and _see(board, move) >= 0:
                            return 3_500_000
        if pt is not None:
            pi = (pt - 1) + (0 if board.turn else 6)
            return self.history[pi][move.to_square]
        return 0

    def _order_moves(
        self,
        board: chess.Board,
        moves,
        ply: int,
        tt_move: Optional[chess.Move],
        countermove: Optional[chess.Move] = None,
    ) -> list[chess.Move]:
        return sorted(
            moves,
            key=lambda m: self._score_move(board, m, ply, tt_move, countermove),
            reverse=True,
        )

    def _update_quiet_heuristics(
        self,
        board: chess.Board,
        best_move: chess.Move,
        quiet_tried: list[chess.Move],
        depth: int,
        ply: int,
        prev_idx: int,
    ) -> None:
        if ply < MAX_PLY:
            if best_move != self.killers[ply][0]:
                self.killers[ply][1] = self.killers[ply][0]
                self.killers[ply][0] = best_move
        if prev_idx >= 0:
            self.countermoves[prev_idx] = best_move

        bonus = min(depth * depth, 400)
        color_off = 0 if board.turn else 6
        pt = board.piece_type_at(best_move.from_square)
        if pt is not None:
            pi = (pt - 1) + color_off
            val = self.history[pi][best_move.to_square] + bonus
            self.history[pi][best_move.to_square] = 160_000 if val > 160_000 else val

        malus = bonus >> 1
        if malus > 0:
            for qm in quiet_tried:
                qpt = board.piece_type_at(qm.from_square)
                if qpt is not None:
                    qpi = (qpt - 1) + color_off
                    qval = self.history[qpi][qm.to_square] - malus
                    self.history[qpi][qm.to_square] = -160_000 if qval < -160_000 else qval

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if (self.nodes & 127) == 0 and time.perf_counter() >= self.hard_limit:
            self.stopped = True
            return 0

        if board.halfmove_clock >= 100 or (
            board.occupied.bit_count() <= 4 and board.is_insufficient_material()
        ):
            return 0

        # Transposition table lookup in Quiescence
        key = hash(board._transposition_key()) & 0xFFFFFFFFFFFFFFFF
        tt_entry = self.tt.get(key & TT_MASK)
        tt_move = None
        if tt_entry is not None and tt_entry.key == key:
            tt_move = tt_entry.best_move
            tt_score = _from_tt(tt_entry.score, ply)
            if tt_entry.flag == EXACT:
                return tt_score
            if tt_entry.flag == LOWER_BOUND and tt_score >= beta:
                return tt_score
            if tt_entry.flag == UPPER_BOUND and tt_score <= alpha:
                return tt_score

        alpha_orig = alpha
        in_check = board.is_check()
        if not in_check:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            if ply >= MAX_PLY:
                return stand_pat

            # Fast capture + queen-promotion generation with Delta & SEE pruning
            candidates = []
            for m in board.generate_legal_captures():
                promo = m.promotion
                if promo is not None and promo != chess.QUEEN:
                    continue
                victim = board.piece_type_at(m.to_square) or 1
                gain = _PIECE_VAL[victim] + (880 if promo == chess.QUEEN else 0)
                if stand_pat + gain + 200 < alpha:
                    continue
                if promo is None:
                    attacker = board.piece_type_at(m.from_square) or 1
                    if _PIECE_VAL[attacker] > _PIECE_VAL[victim] + 50 and _see(board, m) < 0:
                        continue
                candidates.append(m)

            promo_rank = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
            own_promo_pawns = board.pawns & board.occupied_co[board.turn] & promo_rank
            if own_promo_pawns and stand_pat + 950 >= alpha:
                for m in board.generate_legal_moves(own_promo_pawns, chess.BB_BACKRANKS):
                    if m.promotion == chess.QUEEN and not board.is_capture(m):
                        candidates.append(m)

            if not candidates:
                return alpha
            best_score = stand_pat
        else:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply
            if ply >= MAX_PLY:
                return 0
            candidates = moves
            best_score = -INFINITY

        candidates = self._order_moves(board, candidates, ply, tt_move)
        best_move = candidates[0]

        for m in candidates:
            board.push(m)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()

            if self.stopped:
                return 0
            if score > best_score:
                best_score = score
                best_move = m
            if score >= beta:
                idx = key & TT_MASK
                prev = self.tt.get(idx)
                if prev is None or prev.depth <= 0 or prev.key == key:
                    self.tt[idx] = TTEntry(key, 0, _to_tt(best_score, ply), LOWER_BOUND, best_move)
                return best_score
            if score > alpha:
                alpha = score

        if not self.stopped:
            flag = EXACT if best_score > alpha_orig else UPPER_BOUND
            idx = key & TT_MASK
            prev = self.tt.get(idx)
            if prev is None or prev.depth <= 0 or prev.key == key:
                self.tt[idx] = TTEntry(key, 0, _to_tt(best_score, ply), flag, best_move)

        return best_score

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        do_null: bool,
        last_irrev: int = 0,
    ) -> int:
        self.nodes += 1
        if (self.nodes & 127) == 0 and time.perf_counter() >= self.hard_limit:
            self.stopped = True
            return 0

        is_root = ply == 0
        key = hash(board._transposition_key()) & 0xFFFFFFFFFFFFFFFF
        if ply < MAX_PLY:
            self.path_keys[ply] = key

        # Fast O(1) Draw & Repetition detection (skip at root)
        if not is_root:
            hmc = board.halfmove_clock
            if hmc >= 4:
                if hmc >= 100:
                    return 0
                if last_irrev == 0 and key in self.game_keys:
                    return 0
                p = ply - 4
                stop_p = max(last_irrev, ply - hmc)
                while p >= stop_p:
                    if self.path_keys[p] == key:
                        return 0
                    p -= 2
            if board.occupied.bit_count() <= 4 and board.is_insufficient_material():
                return 0

            # Mate distance pruning
            alpha = max(alpha, -MATE_SCORE + ply)
            beta = min(beta, MATE_SCORE - ply - 1)
            if alpha >= beta:
                return alpha

        is_pv = (beta - alpha) > 1
        in_check = board.is_check()

        # Check extension
        if in_check:
            depth += 1

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        if ply >= MAX_PLY:
            return evaluate(board)

        # Transposition table probe
        tt_entry = self.tt.get(key & TT_MASK)
        tt_move = None

        if tt_entry is not None and tt_entry.key == key:
            tt_move = tt_entry.best_move
            if not is_root and tt_entry.depth >= depth:
                tt_score = _from_tt(tt_entry.score, ply)
                if tt_entry.flag == EXACT:
                    return tt_score
                if tt_entry.flag == LOWER_BOUND:
                    alpha = max(alpha, tt_score)
                elif tt_entry.flag == UPPER_BOUND:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        if is_root and self.best_root_move is not None:
            tt_move = self.best_root_move

        # Internal Iterative Reduction (IIR) when deep node lacks a TT move
        if depth >= 4 and tt_move is None and not in_check:
            depth -= 1

        # Static evaluation & improving heuristic (non-check nodes)
        static_eval: Optional[int] = None
        improving = False
        if not in_check:
            static_eval = evaluate(board)
            self.static_evals[ply] = static_eval
            improving = ply >= 2 and static_eval > self.static_evals[ply - 2]
        else:
            self.static_evals[ply] = -INFINITY

        # Forward pruning at non-PV, non-check nodes
        if not in_check and not is_pv and not is_root and static_eval is not None:
            # Reverse Futility Pruning (Static Null Move Pruning)
            if depth <= 4 and abs(beta) < MATE_BOUND:
                margin = (85 if improving else 110) * depth
                if static_eval - margin >= beta:
                    return static_eval - margin

            # Adaptive Null Move Pruning (side-to-move Zugzwang safe + verification)
            if do_null and depth >= 3 and static_eval >= beta and abs(beta) < MATE_BOUND:
                stm_non_pawns = (
                    board.knights | board.bishops | board.rooks | board.queens
                ) & board.occupied_co[board.turn]
                if stm_non_pawns:
                    r = 3 + depth // 6 + min(2, (static_eval - beta) // 200)
                    board.push(chess.Move.null())
                    null_score = -self._negamax(
                        board, depth - 1 - r, -beta, -beta + 1, ply + 1, False, ply + 1
                    )
                    board.pop()
                    if self.stopped:
                        return 0
                    if null_score >= beta:
                        if null_score >= MATE_BOUND:
                            null_score = beta
                        if depth < 10:
                            return null_score
                        ver_score = self._negamax(
                            board, depth - r, beta - 1, beta, ply, False, last_irrev
                        )
                        if self.stopped:
                            return 0
                        if ver_score >= beta:
                            return null_score

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return -MATE_SCORE + ply if in_check else 0

        prev_idx = self._prev_move_index(board)
        cm = self.countermoves[prev_idx] if prev_idx >= 0 else None
        moves = self._order_moves(board, legal_moves, ply, tt_move, cm)

        futile = (
            not is_root
            and not is_pv
            and not in_check
            and depth <= 2
            and static_eval is not None
            and abs(alpha) < MATE_BOUND
            and static_eval + 125 * depth <= alpha
        )
        lmp_limit = (
            (5 + 5 * depth + (3 if improving else 0))
            if (not is_root and not is_pv and not in_check and depth <= 3)
            else 999
        )

        k0, k1 = self.killers[ply] if ply < MAX_PLY else (None, None)
        best_score = -INFINITY
        best_move = tt_move or moves[0]
        alpha_orig = alpha
        searched = 0
        quiet_tried: list[chess.Move] = []
        pawns_bb = board.pawns

        for m in moves:
            is_cap = board.is_capture(m)
            is_promo = m.promotion is not None
            is_quiet = not is_cap and not is_promo

            # Shallow Futility Pruning & Late Move Pruning for hopeless quiet moves
            if is_quiet and best_score > -MATE_BOUND:
                if (futile or searched >= lmp_limit) and not board.gives_check(m):
                    continue

            is_pawn_move = bool(pawns_bb & (1 << m.from_square))
            next_irrev = (ply + 1) if (is_cap or is_pawn_move) else last_irrev

            board.push(m)
            gives_check = board.is_check()

            # PVS & Logarithmic Late Move Reductions (LMR)
            if searched == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True, next_irrev)
            else:
                reduction = 0
                if depth >= 3 and searched >= 2 and is_quiet and not in_check and not gives_check:
                    reduction = _LMR[min(depth, 63)][min(searched, 63)]
                    if is_pv or m == k0 or m == k1 or m == cm:
                        reduction = max(0, reduction - 1)
                    elif not improving and searched >= 6:
                        reduction += 1
                    reduction = min(reduction, depth - 2) if depth > 2 else 0

                # Null-window search (possibly reduced)
                score = -self._negamax(
                    board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True, next_irrev
                )

                # Re-search at full depth if reduced search beat alpha
                if reduction > 0 and score > alpha and not self.stopped:
                    score = -self._negamax(
                        board, depth - 1, -alpha - 1, -alpha, ply + 1, True, next_irrev
                    )

                # Re-search with full window at PV nodes whenever null-window beat alpha
                if is_pv and score > alpha and not self.stopped:
                    score = -self._negamax(
                        board, depth - 1, -beta, -alpha, ply + 1, True, next_irrev
                    )

            board.pop()

            if self.stopped:
                return 0

            if score > best_score:
                best_score = score
                if searched == 0 or score > alpha:
                    best_move = m
                    if is_root:
                        self.best_root_move = m

            searched += 1

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if is_quiet:
                    self._update_quiet_heuristics(board, m, quiet_tried, depth, ply, prev_idx)
                break

            if is_quiet:
                quiet_tried.append(m)

        if not self.stopped:
            if best_score <= alpha_orig:
                flag = UPPER_BOUND
                store_move = tt_move or best_move
            elif best_score >= beta:
                flag = LOWER_BOUND
                store_move = best_move
            else:
                flag = EXACT
                store_move = best_move
            idx = key & TT_MASK
            prev = self.tt.get(idx)
            if prev is None or prev.depth <= depth or prev.key == key:
                self.tt[idx] = TTEntry(key, depth, _to_tt(best_score, ply), flag, store_move)

        return best_score

    def search(
        self,
        board: chess.Board,
        time_limit_ms: int,
        max_depth: int = 64,
        hard_limit_ms: Optional[int] = None,
    ) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("No legal moves")
        if len(moves) == 1 or time_limit_ms <= 0:
            return moves[0]

        reflex = get_early_center_response(board)
        if reflex is not None:
            return reflex

        start = time.perf_counter()
        if hard_limit_ms is not None:
            soft_s = max(0.005, time_limit_ms / 1000.0)
            safety = max(0.012, min(0.050, hard_limit_ms * 0.04 / 1000.0))
            hard_s = max(soft_s, (hard_limit_ms / 1000.0) - safety)
        else:
            safety = max(0.012, (time_limit_ms * 0.06) / 1000.0)
            hard_s = max(0.005, (time_limit_ms / 1000.0) - safety)
            soft_s = hard_s * 0.68

        self.hard_limit = start + hard_s
        self.time_limit = self.hard_limit
        self.stopped = False
        self.nodes = 0
        self.completed_depth = 0
        self._init_game_keys(board)

        key = hash(board._transposition_key()) & 0xFFFFFFFFFFFFFFFF
        prior = self.tt.get(key & TT_MASK)
        prior_move = (
            prior.best_move
            if (prior is not None and prior.key == key and prior.best_move in moves)
            else None
        )
        completed_move = prior_move or self._order_moves(board, moves, 0, None)[0]
        self.best_root_move = completed_move

        # Reset killers & age history for current turn
        for k in self.killers:
            k[0] = k[1] = None
        for row in self.history:
            for s in range(64):
                row[s] >>= 2

        prev_score = 0
        prev_elapsed = 0.0
        stable_count = 0

        for depth in range(1, min(max_depth, MAX_PLY) + 1):
            if self.stopped:
                break

            prev_move_before_iter = completed_move

            if depth >= 4 and abs(prev_score) < MATE_BOUND:
                delta = 35
                alpha = max(-INFINITY, prev_score - delta)
                beta = min(INFINITY, prev_score + delta)
            else:
                delta = INFINITY
                alpha, beta = -INFINITY, INFINITY

            while True:
                self.best_root_move = completed_move
                score = self._negamax(board, depth, alpha, beta, 0, True, 0)
                if self.stopped:
                    break

                if score <= alpha:
                    beta = (alpha + beta) // 2
                    delta *= 3
                    alpha = -INFINITY if delta > 600 else max(-INFINITY, score - delta)
                elif score >= beta:
                    if self.best_root_move is not None:
                        completed_move = self.best_root_move
                    delta *= 3
                    beta = INFINITY if delta > 600 else min(INFINITY, score + delta)
                else:
                    if self.best_root_move is not None:
                        completed_move = self.best_root_move
                    break

            if self.stopped:
                break

            self.completed_depth = depth

            if completed_move == prev_move_before_iter:
                stable_count += 1
            else:
                stable_count = 0

            score_drop = depth >= 4 and score < prev_score - 30
            prev_score = score

            if abs(score) >= MATE_BOUND:
                break

            elapsed = time.perf_counter() - start
            iter_duration = elapsed - prev_elapsed
            prev_elapsed = elapsed

            # Dynamic soft time termination:
            # Save clock when PV is rock-steady; extend toward hard_s on PV switch or score drop.
            if score_drop or (depth >= 5 and stable_count == 0):
                target = min(hard_s * 0.90, soft_s * 1.45)
            elif stable_count >= 4:
                target = soft_s * 0.72
            else:
                target = soft_s

            if elapsed >= target or elapsed + iter_duration * 1.85 >= hard_s:
                break

        return completed_move


_engine = SearchEngine()


def search_best_move(board: chess.Board, time_limit_ms: int, max_depth: int = 64) -> chess.Move:
    return _engine.search(board, time_limit_ms, max_depth)
