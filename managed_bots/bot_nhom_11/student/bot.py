from __future__ import annotations

import chess

from chesslab_sdk import BotConfig, SearchContext

from .search import AlphaBetaSearcher
from .time_manager import make_deadline


class StudentBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.searcher = AlphaBetaSearcher()

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("choose_move called when no legal moves exist")

        remaining = max(1, int(context.time_remaining_ms))
        ply = max(0, int(context.ply))
   
        deadline = make_deadline(
            remaining,
            ply,
            int(getattr(self.config, "initial_time_ms", remaining)),
        )
        try:
            move = self.searcher.choose_move(board, deadline)
            if move in board.legal_moves:
                return move
        except Exception:
            pass
        return legal_moves[0]
