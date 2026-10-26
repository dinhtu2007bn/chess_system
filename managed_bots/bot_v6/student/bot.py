"""StudentBot – the single entry point for the grading system.

Connects evaluation, search, and time management into one class.

The grading system creates one ``StudentBot`` at the start of each game
and calls ``choose_move()`` once per turn.  The ``Searcher`` (with its
transposition table) persists across moves so that cached positions from
previous turns speed up later searches.
"""
from __future__ import annotations

import chess

from chesslab_sdk import BotConfig, SearchContext

from .search import Searcher
from .time_manager import get_time_limits


class StudentBot:
    """The only class imported by the local runner and official grader."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.searcher = Searcher()

    def choose_move(
        self,
        board: chess.Board,
        context: SearchContext,
    ) -> chess.Move:
        # --- Fast path: only one legal move → play instantly ---
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]

        # --- Compute time budget ---
        soft, hard = get_time_limits(context, board, self.config)

        # --- Defensive copy to completely protect caller's board state ---
        board_copy = board.copy()

        # --- Run iterative-deepening search ---
        best = self.searcher.search(board_copy, soft, hard)

        # Guaranteed legality safety check
        if best is not None and best in board.legal_moves:
            return best
        return moves[0]
