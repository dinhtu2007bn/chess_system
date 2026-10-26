"""Negamax/Alpha-Beta, bảng chuyển vị và quiescence gồm cả nước chiếu."""

import time
from dataclasses import dataclass

import chess

from .evaluation import PIECE_VALUES, evaluate, is_passed_pawn
from .time_manager import TimeBudget

MATE = 100_000
INF = 1_000_000
EXACT, LOWER, UPPER = 0, 1, 2
MAX_TABLE = 16_000


@dataclass(frozen=True)
class Entry:
    depth: int
    score: int
    bound: int
    move: chess.Move | None


class SearchTimeout(Exception):
    """Ngắt tìm kiếm; chỉ nhận kết quả của độ sâu đã hoàn thành."""


def position_key(board):
    """Khóa từ thuộc tính công khai, gồm quyền nhập thành và bắt qua đường."""
    return (board.pawns, board.knights, board.bishops, board.rooks,
            board.queens, board.kings, board.occupied_co[chess.WHITE],
            board.occupied_co[chess.BLACK], board.turn, board.castling_rights,
            board.ep_square, board.promoted, board.chess960)


class SearchEngine:
    def __init__(self) -> None:
        self.deadline = 0.0
        self.completed_depth = self.nodes = self.tt_hits = 0
        self.last_score = None
        self.table = {}
        self.move_hints = {}
        self.killers = {}
        self.history = {}
        self.eval_cache = {}
        self._counts = {}

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout
        self.nodes += 1

    def _init_history(self, board):
        # Lưu lịch sử thật để không dùng kết quả TT sai khi luật lặp khác nhau.
        self._counts = {}
        copy = board.copy(stack=True)
        for _ in range(min(board.halfmove_clock, len(board.move_stack)) + 1):
            key = position_key(copy)
            self._counts[key] = self._counts.get(key, 0) + 1
            if not copy.move_stack:
                break
            copy.pop()

    def _push(self, board, move):
        previous = self._counts
        board.push(move)
        # Sau bắt quân/đi tốt, không thể trở lại trạng thái trước đó.
        if board.halfmove_clock == 0:
            self._counts = {}
        key = position_key(board)
        self._counts[key] = self._counts.get(key, 0) + 1
        return previous if self._counts is not previous else None

    def _pop(self, board, previous):
        key = position_key(board)
        count = self._counts[key] - 1
        if count:
            self._counts[key] = count
        else:
            del self._counts[key]
        board.pop()
        if previous is not None:
            self._counts = previous

    def _terminal(self, board, ply):
        if board.is_checkmate():
            return -MATE + ply
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_fifty_moves() or self._counts.get(position_key(board), 0) >= 3):
            return 0
        return None

    def _evaluate(self, board):
        # Hàm đánh giá không phụ thuộc lịch sử; tái dùng giữa các vòng đào sâu.
        key = position_key(board)
        score = self.eval_cache.get(key)
        if score is None:
            score = evaluate(board)
            if len(self.eval_cache) < MAX_TABLE:
                self.eval_cache[key] = score
        return score

    def _ordered(self, board, moves, preferred=None, ply=0, checks=()):
        killers = self.killers.get(ply, ())
        def priority(move):
            score = INF if move == preferred else 0
            if board.is_capture(move):
                victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
                score += 20_000 + 10 * PIECE_VALUES.get(victim, 0)
                score -= PIECE_VALUES.get(board.piece_type_at(move.from_square), 0)
            if move.promotion:
                score += 18_000 + PIECE_VALUES[move.promotion]
            if move in killers:
                score += 9_000
            score += min(8_000, self.history.get((board.turn, move), 0))
            if move in checks:
                score += 12_000
            return score
        return sorted(moves, key=priority, reverse=True)

    def _quiescence(self, board, alpha, beta, ply, remaining=6, checks_left=3):
        self._check_time()
        terminal = self._terminal(board, ply)
        if terminal is not None:
            return terminal
        in_check = board.is_check()
        # Tốt địch ở hàng áp chót có thể phong cấp ngay lượt sau: phải xét
        # cả nước phòng thủ yên lặng, không giả định được bỏ qua lượt.
        promotion_danger = False
        enemy = not board.turn
        for square in board.pieces(chess.PAWN, enemy):
            advance = chess.square_rank(square) if enemy else 7 - chess.square_rank(square)
            ahead = square + (8 if enemy else -8)
            if (advance == 6 and not board.piece_at(ahead)
                    and is_passed_pawn(board, square, enemy)):
                promotion_danger = remaining > 0
                break
        if not in_check:
            stand_pat = self._evaluate(board)
            if not promotion_danger and stand_pat >= beta:
                return stand_pat
            if not promotion_danger:
                alpha = max(alpha, stand_pat)
            if remaining <= 0:
                return alpha
        # Khi bị chiếu, luôn xét nước thoát chiếu; không stand-pat trong chiếu.
        moves = []
        quiet_checks = set()
        for move in board.legal_moves:
            tactical = board.is_capture(move) or move.promotion
            pawn_push = False
            if board.piece_type_at(move.from_square) == chess.PAWN:
                advance = chess.square_rank(move.to_square) if board.turn else 7 - chess.square_rank(move.to_square)
                pawn_push = advance >= 6 and is_passed_pawn(board, move.to_square, board.turn)
            gives_check = not tactical and checks_left > 0 and board.gives_check(move)
            if gives_check:
                quiet_checks.add(move)
            if in_check or promotion_danger or tactical or gives_check or pawn_push:
                moves.append(move)
        for move in self._ordered(board, moves, ply=ply, checks=quiet_checks):
            quiet_check = move in quiet_checks
            previous = self._push(board, move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1,
                                         remaining - 1, checks_left - int(quiet_check))
            finally:
                self._pop(board, previous)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    @staticmethod
    def _pack_score(score, ply):
        if score > MATE - 1000:
            return score + ply
        if score < -MATE + 1000:
            return score - ply
        return score

    @staticmethod
    def _unpack_score(score, ply):
        if score > MATE - 1000:
            return score - ply
        if score < -MATE + 1000:
            return score + ply
        return score

    def _negamax(self, board, depth, alpha, beta, ply):
        self._check_time()
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)
        terminal = self._terminal(board, ply)
        if terminal is not None:
            return terminal
        pos = position_key(board)
        # Cùng bàn nhưng khác đồng hồ 50 nước/lịch sử lặp không dùng chung bound.
        key = (pos, board.halfmove_clock, frozenset(self._counts.items()))
        entry = self.table.get(key)
        preferred = entry.move if entry else self.move_hints.get(pos)
        if entry and entry.depth >= depth:
            score = self._unpack_score(entry.score, ply)
            if (entry.bound == EXACT or entry.bound == LOWER and score >= beta
                    or entry.bound == UPPER and score <= alpha):
                self.tt_hits += 1
                return score
        original_alpha = alpha
        best, best_move = -INF, None
        for move in self._ordered(board, board.legal_moves, preferred, ply):
            quiet = not board.is_capture(move) and not move.promotion
            previous = self._push(board, move)
            try:
                if best_move is None:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    # PVS: thử cửa sổ hẹp; tìm lại đầy đủ nếu có thể tốt hơn.
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                self._pop(board, previous)
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                if quiet:
                    killers = self.killers.get(ply, ())
                    self.killers[ply] = (move,) + tuple(m for m in killers if m != move)[:1]
                    history_key = (board.turn, move)
                    self.history[history_key] = min(8_000, self.history.get(history_key, 0) + depth * depth)
                break
        bound = UPPER if best <= original_alpha else LOWER if best >= beta else EXACT
        if len(self.table) < MAX_TABLE or key in self.table:
            self.table[key] = Entry(depth, self._pack_score(best, ply), bound, best_move)
        if len(self.move_hints) < MAX_TABLE or pos in self.move_hints:
            self.move_hints[pos] = best_move
        return best

    def choose_move(self, board: chess.Board, budget_seconds: float,
                    budget: TimeBudget | None = None) -> chess.Move:
        started = time.perf_counter()
        self.deadline = started + max(0.0, budget_seconds)
        self.completed_depth = self.nodes = self.tt_hits = 0
        self.last_score = None
        self.table.clear()
        self.move_hints.clear()
        self.killers.clear()
        self.history.clear()
        self.eval_cache.clear()
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("Không có nước đi hợp lệ.")
        best_move = moves[0]
        if len(moves) == 1 or budget_seconds <= 0:
            return best_move
        self._init_history(board)
        previous_duration = last_duration = 0.0
        for depth in range(1, 65):
            if depth > 1 and budget is not None:
                if not budget.should_start_iteration(time.perf_counter() - started,
                                                     last_duration, previous_duration):
                    break
            iteration_started = time.perf_counter()
            alpha, candidate = -INF, best_move
            try:
                for move in self._ordered(board, moves, best_move):
                    self._check_time()
                    previous = self._push(board, move)
                    try:
                        if alpha == -INF:
                            score = -self._negamax(board, depth - 1, -INF, INF, 1)
                        else:
                            score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                            if score > alpha:
                                score = -self._negamax(board, depth - 1, -INF, -alpha, 1)
                    finally:
                        self._pop(board, previous)
                    if score > alpha:
                        alpha, candidate = score, move
            except SearchTimeout:
                # Khi vòng đầu chưa hoàn tất, dùng nước đã đánh giá thay vì
                # nước đầu tiên tùy thứ tự thư viện. Vòng sau giữ kết quả cũ.
                if self.completed_depth == 0 and alpha > -INF:
                    best_move = candidate
                    self.last_score = alpha
                break
            best_move = candidate
            self.completed_depth = depth
            self.last_score = alpha
            previous_duration = last_duration
            last_duration = time.perf_counter() - iteration_started
            if alpha >= MATE - 1000:
                break
        return best_move
