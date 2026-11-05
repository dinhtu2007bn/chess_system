"""StudentBot – single entry point for the ChessLab grading and runner system.

Bot Test 3 (Grandmaster SEE & Promotion Master):
- Static Exchange Evaluation (SEE) distinguishing Safe Checks from Suicidal Checks.
- Preserves tactical safe checks for perpetual check drawing sequences and king pressure.
- Capped passed pawn evaluation completely resolving the 7th rank promotion paradox.
- Attacker-to-defender superiority requirement for King danger evaluation.
- High-Performance Quiescence Search with tactical promotions and safe quiet checks.
- Dynamic tournament clock manager with 0.34s Golden Window platform floor.
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
        # Fast path: only one legal move -> play instantly
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]

        # Compute dynamic time limits
        soft, hard = get_time_limits(context, board, self.config)

        # Defensive copy to protect caller board state
        board_copy = board.copy()

        # Run search
        best = self.searcher.search(board_copy, soft, hard)

        # Safety fallback
        if best is not None and best in board.legal_moves:
            return best
        return moves[0]

