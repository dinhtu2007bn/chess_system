"""Level 2 search: MVP search plus PVS, transposition table, quiescence,
null-move pruning, check extension and killer/history move ordering.

The evaluation function must return a score in centipawns from the point of
view of the side to move (positive = good for the side to move).
"""

import time

import chess

from ...evaluation import evaluate_board

MATE_SCORE = 100_000
MATE_THRESHOLD = MATE_SCORE - 1_000
DRAW_SCORE = 0
INFINITY = MATE_SCORE + 1

MAX_DEPTH = 64
MAX_PLY = 128
CHECK_EVERY_NODES = 128
MIN_TIME_MS = 10
TT_MAX_ENTRIES = 1_000_000
NULL_MOVE_REDUCTION = 2

EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2

# Only used for move ordering (MVV-LVA), not for evaluation.
ORDER_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

TT_MOVE_BONUS = 10_000_000
CAPTURE_BONUS = 1_000_000
PROMOTION_BONUS = 900_000
KILLER_BONUS = 800_000


class SearchTimeout(Exception):
    pass


def sort_moves(
    board: chess.Board,
    moves: list[chess.Move],
    tt_move: chess.Move | None = None,
    killers: list[chess.Move | None] | None = None,
    history: dict[tuple[bool, int, int], int] | None = None,
) -> list[chess.Move]:
    """Order moves: TT move, captures (MVV-LVA), promotions, killers, history."""

    def score(move: chess.Move) -> int:
        if move == tt_move:
            return TT_MOVE_BONUS
        if board.is_capture(move):
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            return CAPTURE_BONUS + 10 * ORDER_VALUES[victim] - ORDER_VALUES[attacker]
        if move.promotion:
            return PROMOTION_BONUS + ORDER_VALUES[move.promotion]
        if killers and move in killers:
            return KILLER_BONUS
        if history:
            return history.get((board.turn, move.from_square, move.to_square), 0)
        return 0

    return sorted(moves, key=score, reverse=True)


def _score_to_tt(score: float, ply: int) -> float:
    # Mate scores are stored relative to the node, not to the root.
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: float, ply: int) -> float:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


def _has_non_pawn_material(board: chess.Board) -> bool:
    us = board.occupied_co[board.turn]
    pieces = board.knights | board.bishops | board.rooks | board.queens
    return bool(us & pieces)


class Searcher:
    def __init__(self, time_limit_ms: int, eval_brain: str) -> None:
        self.start = time.perf_counter()
        self.time_limit_s = max(MIN_TIME_MS, int(time_limit_ms or 0)) / 1000
        self.deadline = self.start + self.time_limit_s
        self.nodes = 0
        self.eval_brain = eval_brain
        self.tt: dict[object, tuple[int, float, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 1)]
        self.history: dict[tuple[bool, int, int], int] = {}

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes % CHECK_EVERY_NODES == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    def _evaluate(self, board: chess.Board) -> float:
        return evaluate_board(board, self.eval_brain)

    def search(self, board: chess.Board) -> chess.Move:
        root_moves = sort_moves(board, list(board.legal_moves))
        best_move = root_moves[0]
        if len(root_moves) == 1:
            return best_move

        for depth in range(1, MAX_DEPTH + 1):
            try:
                scored = self._search_root(board, depth, root_moves)
            except SearchTimeout:
                break
            scored.sort(key=lambda item: item[1], reverse=True)
            best_move, best_score = scored[0]
            root_moves = [move for move, _ in scored]
            if abs(best_score) >= MATE_THRESHOLD:
                break
            # The next depth usually costs several times more than this one.
            if self.elapsed() >= self.time_limit_s / 2:
                break
        return best_move

    def _search_root(
        self,
        board: chess.Board,
        depth: int,
        root_moves: list[chess.Move],
    ) -> list[tuple[chess.Move, float]]:
        alpha = -INFINITY
        scored: list[tuple[chess.Move, float]] = []
        for index, move in enumerate(root_moves):
            board.push(move)
            try:
                if index == 0:
                    score = -self._negamax(board, depth - 1, -INFINITY, -alpha, 1)
                else:
                    # Principal variation search: try a null window first.
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if score > alpha:
                        score = -self._negamax(board, depth - 1, -INFINITY, -alpha, 1)
            finally:
                board.pop()
            scored.append((move, score))
            alpha = max(alpha, score)
        return scored

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        allow_null: bool = True,
    ) -> float:
        self._tick()

        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return DRAW_SCORE
        if board.is_repetition(2):
            return DRAW_SCORE
        if ply >= MAX_PLY:
            return self._evaluate(board)

        # Mate distance pruning.
        alpha = max(alpha, -MATE_SCORE + ply)
        beta = min(beta, MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha

        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        key = board._transposition_key()
        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None:
            entry_depth, entry_score, entry_flag, tt_move = entry
            if entry_depth >= depth:
                score = _score_from_tt(entry_score, ply)
                if entry_flag == EXACT:
                    return score
                if entry_flag == LOWER_BOUND and score >= beta:
                    return score
                if entry_flag == UPPER_BOUND and score <= alpha:
                    return score

        is_pv = beta - alpha > 1
        if (
            allow_null
            and not is_pv
            and not in_check
            and depth >= 3
            and abs(beta) < MATE_THRESHOLD
            and _has_non_pawn_material(board)
        ):
            board.push(chess.Move.null())
            try:
                score = -self._negamax(
                    board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1, False
                )
            finally:
                board.pop()
            if score >= beta:
                return beta

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else DRAW_SCORE

        killers = self.killers[ply]
        ordered = sort_moves(board, moves, tt_move, killers, self.history)

        alpha_orig = alpha
        best_score = -INFINITY
        best_move: chess.Move | None = None
        for index, move in enumerate(ordered):
            is_quiet = not board.is_capture(move) and not move.promotion
            board.push(move)
            try:
                if index == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if is_quiet:
                    if killers[0] != move:
                        killers[1] = killers[0]
                        killers[0] = move
                    history_key = (board.turn, move.from_square, move.to_square)
                    self.history[history_key] = self.history.get(history_key, 0) + depth * depth
                break

        if best_score <= alpha_orig:
            flag = UPPER_BOUND
        elif best_score >= beta:
            flag = LOWER_BOUND
        else:
            flag = EXACT
        if len(self.tt) >= TT_MAX_ENTRIES:
            self.tt.clear()
        self.tt[key] = (depth, _score_to_tt(best_score, ply), flag, best_move)
        return best_score

    def _quiescence(self, board: chess.Board, alpha: float, beta: float, ply: int) -> float:
        """Search captures (and check evasions) until the position is quiet."""
        self._tick()

        if ply >= MAX_PLY:
            return self._evaluate(board)

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply
            best_score = -INFINITY
        else:
            stand_pat = self._evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            best_score = stand_pat
            moves = [
                move
                for move in board.legal_moves
                if board.is_capture(move) or move.promotion == chess.QUEEN
            ]

        for move in sort_moves(board, moves):
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best_score


def find_best_move(
    board: chess.Board,
    time_limit_ms: int,
    eval_brain: str = "MVP",
) -> chess.Move:
    """Return the best move found within ``time_limit_ms`` milliseconds.

    The search runs on a copy, so ``board`` is never modified. If time runs out,
    the best move of the last fully searched depth is returned.
    """
    if not any(board.legal_moves):
        raise ValueError("No legal moves in this position")
    return Searcher(time_limit_ms, eval_brain).search(board.copy())
