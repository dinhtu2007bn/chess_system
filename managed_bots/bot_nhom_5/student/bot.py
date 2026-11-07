import chess
from chesslab_sdk import BotConfig, SearchContext

from .search import Searcher, SearchConfig
from .time_manager import Deadline, allocate_time, extract_time_left


def _phase(board):
    return min(24, 1 * (chess.popcount(board.knights | board.bishops)) + 2 * chess.popcount(board.rooks) + 4 * chess.popcount(board.queens))


class StudentBot:
    """Entry point required by the grader: iterative-deepening alpha-beta search + clock-aware time manager."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.searcher = Searcher(SearchConfig())
        self.last_result = None

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        fallback = next(iter(board.legal_moves), chess.Move.null())
        try:
            work = board.copy()
            time_left = extract_time_left(context, board.turn)
            n_captures = sum(1 for _ in work.generate_legal_captures())
            soft, hard = allocate_time(
                time_left,
                work.ply(),
                _phase(work),
                work.legal_moves.count(),
                work.is_check(),
                n_captures,
            )
            result = self.searcher.search(work, Deadline(soft, hard))
            self.last_result = result
            move = result.best_move
            if move is not None and board.is_legal(move):
                return move
        except Exception:
            self.searcher = Searcher(SearchConfig())
        return fallback

