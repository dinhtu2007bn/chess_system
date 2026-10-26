from __future__ import annotations

import chess

from chesslab_sdk import BotConfig, SearchContext
from .search import Searcher
from .time_manager import TimeManager


class StudentBot:
    """Production-grade Chess960 Bot (~1700 Elo) using classical Alpha-Beta,
    Iterative Deepening, Transposition Table, and Adaptive Time Management.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.searcher = Searcher()
        self.time_manager = TimeManager(safety_margin_ms=180)

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        target_time, hard_limit = self.time_manager.allocate_time(board, context)
        best_move, depth, nodes = self.searcher.search(
            board,
            target_time=target_time,
            hard_limit=hard_limit,
        )
        return best_move
