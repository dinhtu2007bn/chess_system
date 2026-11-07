
from time import perf_counter

import chess
from chesslab_sdk import BotConfig, SearchContext

from .search import Searcher
from .time_manager import time_budget


class StudentBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.searcher = Searcher()

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        started = perf_counter()
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("Bàn cờ đã hết nước hợp lệ.")
        if len(moves) == 1:
            return moves[0]

        soft, hard = time_budget(context.time_remaining_ms, context.ply)
        return self.searcher.choose(
            board, moves, context.ply, started + soft, started + hard,
        )
