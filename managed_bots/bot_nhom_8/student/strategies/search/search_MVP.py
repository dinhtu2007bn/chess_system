"""MVP search: negamax with alpha-beta pruning and iterative deepening.

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
CHECK_EVERY_NODES = 256
MIN_TIME_MS = 10

# Only used for move ordering (MVV-LVA), not for evaluation.
ORDER_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}


class SearchTimeout(Exception):
    pass


def sort_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Captures first (most valuable victim, least valuable attacker), then the rest."""

    def score(move: chess.Move) -> int:
        if not board.is_capture(move):
            return 0
        if board.is_en_passant(move):
            victim = chess.PAWN
        else:
            victim = board.piece_type_at(move.to_square) or chess.PAWN
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        return 10 * ORDER_VALUES[victim] - ORDER_VALUES[attacker] + 100_000

    return sorted(moves, key=score, reverse=True)


class Searcher:
    def __init__(self, time_limit_ms: int, eval_brain: str) -> None:
        self.start = time.perf_counter()
        self.time_limit_s = max(MIN_TIME_MS, int(time_limit_ms or 0)) / 1000
        self.deadline = self.start + self.time_limit_s
        self.nodes = 0
        self.eval_brain = eval_brain

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes % CHECK_EVERY_NODES == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

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
            # Search the best moves of this depth first at the next depth.
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
        for move in root_moves:
            board.push(move)
            try:
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
    ) -> float:
        self._tick()

        moves = list(board.legal_moves)
        if not moves:
            # Checkmate: the sooner the mate, the higher the score for the winner.
            return -MATE_SCORE + ply if board.is_check() else DRAW_SCORE
        if board.is_insufficient_material():
            return DRAW_SCORE
        if depth <= 0:
            return evaluate_board(board, self.eval_brain)

        best_score = -INFINITY
        for move in sort_moves(board, moves):
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, score)
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
