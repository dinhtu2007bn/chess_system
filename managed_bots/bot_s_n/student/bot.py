from __future__ import annotations

import chess
import random

from chesslab_sdk import BotConfig, SearchContext

from .search import SearchEngine
from .time_manager import TimeManager


class StudentBot:
    """The only class imported by the local runner and official grader."""

    def __init__(self, config: BotConfig) -> None:
        """Khởi tạo một lần cho mỗi ván đấu."""
        self.config = config
        self.color = config.color
        self.rng = random.Random(config.seed)
        self.search = SearchEngine()
        self.time_manager = TimeManager()

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        """Phân bổ thời gian còn lại và tìm nước bằng đào sâu lặp."""
        budget = self.time_manager.allocate(board, context)
        return self.search.choose_move(board, budget.hard_seconds, budget)
