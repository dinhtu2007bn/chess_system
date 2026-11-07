"""The only class imported by the local runner and official grader."""

import time

import chess

from chesslab_sdk import BotConfig, SearchContext

from .search import Searcher
from .time_manager import allocate_time


class StudentBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.searcher = Searcher()

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        started = time.perf_counter()
        # Preserve the actual legal Chess960 move, including king-to-rook castling.
        fallback = next(iter(board.legal_moves), None)
        if fallback is None:
            raise ValueError("choose_move requires a position with a legal move")
        budget = allocate_time(self.config, context)
        if budget.hard_ms <= 0:
            return fallback
        return self.searcher.choose(board, fallback, started, budget, context.ply)
