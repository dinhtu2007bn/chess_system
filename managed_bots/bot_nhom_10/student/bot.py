from __future__ import annotations

import chess

from chesslab_sdk import BotConfig, SearchContext
from .search import SearchEngine
from .time_manager import allocate_time_bounds, max_search_depth


class StudentBot:
    """The only class imported by the local runner and official grader."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.engine = SearchEngine()

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves")
        if len(legal) == 1:
            return legal[0]

        soft_ms, hard_ms = allocate_time_bounds(
            context, int(getattr(self.config, "initial_time_ms", 0) or 0)
        )
        depth_limit = max_search_depth(context)
        try:
            move = self.engine.search(
                board, soft_ms, max_depth=depth_limit, hard_limit_ms=hard_ms
            )
            if move in board.legal_moves:
                return move
        except Exception:
            pass
        return legal[0]
