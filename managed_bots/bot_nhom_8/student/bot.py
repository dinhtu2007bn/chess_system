import chess

from chesslab_sdk import BotConfig, SearchContext
from .search import find_best_move
from .time_manager import allocate_time


class StudentBot:
    """The only class imported by the local runner and official grader."""

    def __init__(self, config: BotConfig, brain = {"eval": "v3", "search": "v2"}) -> None:
        self.config = config
        self.brain = brain

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves in this position")

        try:
            time_limit_ms = allocate_time(context)
            move = find_best_move(board.copy(stack=True), time_limit_ms, self.brain)
            if isinstance(move, chess.Move) and move in legal_moves:
                return move
            raise ValueError(f"Search returned an illegal move: {move!r}")
        except Exception as exc:
            print(
                f"StudentBot search failed; using first legal move: "
                f"{type(exc).__name__}: {exc}"
            )
            return legal_moves[0]