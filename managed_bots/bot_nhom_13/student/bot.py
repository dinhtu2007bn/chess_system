from __future__ import annotations
import chess
from chesslab_sdk import BotConfig, SearchContext
from .search import Searcher
from .time_manager import allocate

class StudentBot:
    """The only class imported by the local runner and official grader."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.searcher = Searcher()

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        legal = list(board.legal_moves)

        # Nếu chỉ còn 1 nước đi hợp lệ thì trả về luôn
        if len(legal) == 1:
            return legal[0]

        # Tính soft và hard limit theo hàm allocate, gọi hàm search để tìm nước đi tốt nhất
        try:
            soft, hard = allocate(context.time_remaining_ms, context.ply)
            move = self.searcher.search(board.copy(), soft, hard)
            if move in board.legal_moves:
                return move
        except Exception:
            pass

        # Nếu search không được thì trả về nước đi ăn quân đầu tiên, không có thì trả về nước hợp lệ đầu tiên
        capture = [m for m in legal if board.is_capture(m)]
        return capture[0] if capture else legal[0]