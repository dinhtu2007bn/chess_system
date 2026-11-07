from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import islice
from typing import Optional

import chess

from .evaluation import PIECE_VALUES, evaluate_board

MATE_SCORE = 100_000
MATE_BOUND = MATE_SCORE - 1_000
INF = 1_000_000
MAX_PLY = 96
EXACT, LOWER, UPPER = 0, 1, 2
_EXCHANGE_VALUES = (0, 100, 320, 335, 500, 900, 20_000)


class SearchTimeout(Exception):
    pass


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    flag: int
    best_move: Optional[chess.Move]
    age: int = 0


class AlphaBetaSearcher:

    def __init__(self, table_limit: int = 200_000) -> None:
        self.table: dict[object, TTEntry] = {}
        self.table_limit = max(1, table_limit)
        self.deadline = 0.0
        self.nodes = 0
        self.age = 0
        self.killers: dict[int, list[chess.Move]] = {}
        self.history: dict[tuple[bool, chess.Move], int] = {}
        self.completed_depth = 0
        self.last_score = 0
        self._repetitions: dict[object, int] = {}
        self._null_depth = 0
        self._poll_mask = 63
        self._perf_deadline: Optional[float] = None

    @staticmethod
    def _key(board: chess.Board) -> object:
        return board._transposition_key()

    def _tt_key(self, board: chess.Board) -> object:
        return self._key(board), board.halfmove_clock

    def _check_time(self) -> None:
        expired = (time.perf_counter() >= self._perf_deadline
                   if self._perf_deadline is not None else time.monotonic() >= self.deadline)
        if expired:
            raise SearchTimeout

    def _tick(self) -> None:
        self.nodes += 1
        if not self.nodes & self._poll_mask:
            self._check_time()

    def _init_repetitions(self, board: chess.Board) -> None:
        self._repetitions.clear()
        previous = board.copy(stack=True)
        limit = min(board.halfmove_clock, len(board.move_stack))
        for i in range(limit + 1):
            key = self._key(previous)
            self._repetitions[key] = self._repetitions.get(key, 0) + 1
            if i < limit:
                previous.pop()

    def _push(self, board: chess.Board, move: chess.Move):
        board.push(move)
        if not move:
            self._null_depth += 1
        if self._null_depth:
            return None
        key = self._key(board)
        self._repetitions[key] = self._repetitions.get(key, 0) + 1
        return key

    def _pop(self, board: chess.Board, key) -> None:
        if key is not None:
            count = self._repetitions[key] - 1
            if count:
                self._repetitions[key] = count
            else:
                del self._repetitions[key]
        if not board.peek():
            self._null_depth -= 1
        board.pop()

    def _draw(self, board: chess.Board, in_check: bool, ply: int) -> Optional[int]:
        if board.is_insufficient_material():
            return 0
        if not self._null_depth and (board.halfmove_clock >= 100
                                    or self._repetitions.get(self._key(board), 0) >= 3):
            if in_check and not any(board.generate_legal_moves()):
                return -MATE_SCORE + ply
            return 0
        return None

    @staticmethod
    def _static(board: chess.Board) -> int:
        score = evaluate_board(board)
        return score if board.turn else -score

    @staticmethod
    def _capture_value(board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        return PIECE_VALUES.get(board.piece_type_at(move.to_square), 0)

    @staticmethod
    def _see(board: chess.Board, move: chess.Move) -> int:
        target = move.to_square
        kind = board.piece_type_at(move.from_square)
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(target)
        gains = [_EXCHANGE_VALUES[victim or 0]]
        occupant = _EXCHANGE_VALUES[move.promotion or kind or 0]
        if move.promotion:
            gains[0] += occupant - _EXCHANGE_VALUES[chess.PAWN]
        occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
        if board.is_en_passant(move):
            captured = target - 8 if board.turn else target + 8
            occupied &= ~chess.BB_SQUARES[captured]
        occupied |= chess.BB_SQUARES[target]
        side = not board.turn
        while True:
            attackers = board.attackers_mask(side, target, occupied) & occupied
            if not attackers:
                break
            for pt in range(chess.PAWN, chess.KING + 1):
                candidates = attackers & board.pieces_mask(pt, side)
                if candidates:
                    break
            source = chess.lsb(candidates)
            after = occupied & ~chess.BB_SQUARES[source]
            if pt == chess.KING and board.attackers_mask(not side, target, after) & after:
                break
            gain = occupant - gains[-1]
            occupant = _EXCHANGE_VALUES[pt]
            if pt == chess.PAWN and target >> 3 in (0, 7):
                gain += _EXCHANGE_VALUES[chess.QUEEN] - _EXCHANGE_VALUES[chess.PAWN]
                occupant = _EXCHANGE_VALUES[chess.QUEEN]
            gains.append(gain)
            occupied = after
            side = not side
        for i in range(len(gains) - 1, 0, -1):
            gains[i - 1] = min(gains[i - 1], -gains[i])
        return gains[0]

    def _move_order_score(self, board: chess.Board, move: chess.Move, ply: int) -> int:
        if board.is_capture(move):
            victim = self._capture_value(board, move)
            attacker = PIECE_VALUES.get(board.piece_type_at(move.from_square), 0)
            see = self._see(board, move) if attacker > victim else 0
            score = (1_000_000 if see >= 0 else -100_000) + victim * 10 - attacker
        else:
            score = self.history.get((board.turn, move), 0)
            killers = self.killers.get(ply, ())
            if move in killers:
                score += 700_000 - killers.index(move) * 1_000
        if move.promotion:
            score += 900_000 + PIECE_VALUES[move.promotion]
        return score

    def _ordered(self, board: chess.Board, moves, ply: int, tt_move=None):
        return sorted(moves, key=lambda m: 3_000_000 if m == tt_move else
                      self._move_order_score(board, m, ply), reverse=True)

    def _iter_ordered(self, board: chess.Board, moves, ply: int, tt_move=None):
        if tt_move is not None and tt_move in moves:
            yield tt_move
        yield from self._ordered(board, (m for m in moves if m != tt_move), ply)

    def _remember_cutoff(self, board: chess.Board, move: chess.Move,
                         depth: int, ply: int) -> None:
        if board.is_capture(move) or move.promotion:
            return
        killers = self.killers.setdefault(ply, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]
        key = board.turn, move
        self.history[key] = min(100_000, self.history.get(key, 0) + depth * depth)

    def _store_tt(self, key, depth, score, alpha_orig, beta_orig, best_move, ply):
        flag = UPPER if score <= alpha_orig else LOWER if score >= beta_orig else EXACT
        stored = score + ply if score > MATE_BOUND else score - ply if score < -MATE_BOUND else score
        old = self.table.get(key)
        if old is not None and old.depth > depth and old.age == self.age:
            return
        if old is None and len(self.table) >= self.table_limit:
            count = min(2048, max(1, self.table_limit // 8))
            for stale in list(islice(self.table, count)):
                del self.table[stale]
        self.table[key] = TTEntry(depth, stored, flag, best_move, self.age)

    def _quiescence(self, board, alpha, beta, ply, qdepth=0):
        self._tick()
        in_check = board.is_check()
        draw = self._draw(board, in_check, ply)
        if draw is not None:
            return draw
        if not any(board.generate_legal_moves()):
            return -MATE_SCORE + ply if in_check else 0
        if ply >= MAX_PLY:
            raise SearchTimeout
        if in_check:
            stand_pat = -INF
            moves = list(board.generate_legal_moves())
        else:
            stand_pat = self._static(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            if qdepth >= 12:
                return alpha
            moves = list(board.generate_legal_captures())
            promotion_rank = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
            for move in board.generate_legal_moves(from_mask=board.pawns & promotion_rank):
                if move.promotion and not board.is_capture(move):
                    moves.append(move)
        for move in self._ordered(board, moves, ply):
            if (not in_check and not move.promotion and board.occupied.bit_count() > 8
                    and stand_pat + self._capture_value(board, move) + 220 < alpha
                    and not board.gives_check(move)):
                continue
            key = self._push(board, move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
            finally:
                self._pop(board, key)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _negamax(self, board, depth, alpha, beta, ply, do_null=True):
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)
        self._tick()
        in_check = board.is_check()
        draw = self._draw(board, in_check, ply)
        if draw is not None:
            return draw
        if ply >= MAX_PLY:
            raise SearchTimeout
        alpha = max(alpha, -MATE_SCORE + ply)
        beta = min(beta, MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha
        if in_check:
            depth += 1
        alpha_orig, beta_orig = alpha, beta
        pv_node = beta - alpha > 1
        key = self._tt_key(board)
        entry = self.table.get(key)
        tt_move = entry.best_move if entry else None
        use_tt = not self._null_depth and self._repetitions.get(self._key(board), 0) < 2
        if use_tt and entry and entry.depth >= depth:
            score = entry.score
            if score > MATE_BOUND:
                score -= ply
            elif score < -MATE_BOUND:
                score += ply
            if entry.flag == EXACT or (entry.flag == LOWER and score >= beta) or (
                entry.flag == UPPER and score <= alpha
            ):
                return score
        moves = list(board.generate_legal_moves())
        if not moves:
            return -MATE_SCORE + ply if in_check else 0
        non_pawns = board.occupied_co[board.turn] & ~(board.pawns | board.kings)
        static_eval = None
        if (not pv_node and not in_check and abs(beta) < MATE_BOUND
                and non_pawns.bit_count() >= 2):
            static_eval = self._static(board)
            if depth <= 3 and board.occupied.bit_count() > 10:
                margin = 150 * depth
                if static_eval - margin >= beta:
                    return static_eval - margin
        if (do_null and not self._null_depth and not pv_node and not in_check
                and depth >= 3 and abs(beta) < MATE_BOUND and board.halfmove_clock < 90
                and non_pawns.bit_count() >= 2
                and board.pawns & board.occupied_co[board.turn]
                and static_eval is not None and static_eval >= beta):
            reduction = 2 + depth // 6
            pushed = self._push(board, chess.Move.null())
            try:
                score = -self._negamax(board, depth - 1 - reduction,
                                       -beta, -beta + 1, ply + 1, False)
            finally:
                self._pop(board, pushed)
            if score >= beta:
                if depth < 7 or self._negamax(board, depth - reduction, beta - 1,
                                              beta, ply, False) >= beta:
                    return beta
        best_score, best_move = -INF, None
        for index, move in enumerate(self._iter_ordered(board, moves, ply, tt_move)):
            quiet = not board.is_capture(move) and not move.promotion
            killer = move in self.killers.get(ply, ())
            advanced_pawn = (board.piece_type_at(move.from_square) == chess.PAWN
                             and (move.to_square >> 3 if board.turn else
                                  7 - (move.to_square >> 3)) >= 5)
            pushed = self._push(board, move)
            try:
                child_depth = depth - 1
                reduction = int(depth >= 3 and index >= 4 and quiet and not killer
                                and not advanced_pawn and not in_check and not board.is_check())
                if index == 0:
                    score = -self._negamax(board, child_depth, -beta, -alpha, ply + 1)
                else:
                    score = -self._negamax(board, child_depth - reduction,
                                           -alpha - 1, -alpha, ply + 1)
                    if reduction and score > alpha:
                        score = -self._negamax(board, child_depth, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, child_depth, -beta, -alpha, ply + 1)
            finally:
                self._pop(board, pushed)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                self._remember_cutoff(board, move, depth, ply)
                break
        if use_tt:
            self._store_tt(key, depth, best_score, alpha_orig, beta_orig, best_move, ply)
        return best_score

    def _root_search(self, board, depth, alpha, beta, preferred):
        original_alpha, original_beta = alpha, beta
        best_score, best_move = -INF, preferred
        for i, move in enumerate(self._ordered(board, board.legal_moves, 0, preferred)):
            self._check_time()
            pushed = self._push(board, move)
            try:
                if i == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                self._pop(board, pushed)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        self._store_tt(self._tt_key(board), depth, best_score, original_alpha,
                       original_beta, best_move, 0)
        return best_score, best_move

    def choose_move(self, board: chess.Board, deadline: float, max_depth: int = 64) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("Không có nước hợp lệ trong thế kết thúc")
        self.deadline = deadline
        budget = max(0.0, deadline - time.monotonic())
        start = time.perf_counter()
        self._perf_deadline = start + max(0.0, budget - min(0.004, budget * 0.05))
        self.nodes = self.completed_depth = 0
        self.age += 1
        self._null_depth = 0
        self.killers.clear()
        self.history = {k: v // 2 for k, v in self.history.items() if v >= 2}
        self._init_repetitions(board)
        self._poll_mask = 7 if budget < 0.02 else 31
        prior = self.table.get(self._tt_key(board))
        preferred = prior.best_move if prior and prior.best_move in legal else None
        best_move = self._ordered(board, legal, 0, preferred)[0]
        if len(legal) == 1 or budget <= 0:
            return best_move
        last_duration = 0.0
        try:
            for depth in range(1, max_depth + 1):
                self._check_time()
                remaining = self._perf_deadline - time.perf_counter()
                if depth >= 4 and remaining < last_duration * 1.5:
                    break
                iteration_start = time.perf_counter()
                window = 35 if depth >= 4 else INF
                while True:
                    alpha = max(-INF, self.last_score - window)
                    beta = min(INF, self.last_score + window)
                    score, move = self._root_search(board, depth, alpha, beta, best_move)
                    if alpha < score < beta:
                        break
                    window = min(INF, window * 4)
                    if window == INF:
                        score, move = self._root_search(board, depth, -INF, INF, move)
                        break
                best_move, self.last_score = move, score
                self.completed_depth = depth
                last_duration = time.perf_counter() - iteration_start
                if abs(score) >= MATE_BOUND:
                    break
        except SearchTimeout:
            pass
        return best_move


def alpha_beta(board: chess.Board, deadline: float,
               searcher: Optional[AlphaBetaSearcher] = None) -> chess.Move:
    return (searcher or AlphaBetaSearcher()).choose_move(board, deadline)
