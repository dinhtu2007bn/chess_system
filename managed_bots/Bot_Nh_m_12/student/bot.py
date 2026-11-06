from __future__ import annotations

import chess

from chesslab_sdk import BotConfig, SearchContext
from .search import Searcher
from .time_manager import TimeManager, compute_budget_ms

class StudentBot:
    """The only class imported by the local runner and official grader."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.searcher = Searcher()

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        """
        Chọn nước đi tối ưu:
        - Luôn trả về chess.Move thuộc board.legal_moves.
        - Không làm thay đổi trạng thái board gốc.
        - Không bao giờ vượt quá thời gian hoặc văng ngoại lệ.
        """
        # TẦNG 1: Lấy sẵn danh sách nước đi hợp lệ
        legal = list(board.legal_moves)
        if not legal:
            return chess.Move.null()
        # Tối ưu nước đi duy nhất: đi ngay trong 0ms, không tốn tài nguyên tìm kiếm
        if len(legal) == 1:
            return legal[0]
        # TẦNG 2: Thiết lập thời gian và gọi tìm kiếm
        budget_ms = compute_budget_ms(
            time_remaining_ms=context.time_remaining_ms,
            ply=context.ply,
            initial_time_ms=self.config.initial_time_ms,
        )
        tm = TimeManager(budget_ms)
        try:
            move = self.searcher.search(board, tm)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            move = legal[0]
        # TẦNG 3: Xác minh tính hợp lệ tuyệt đối trước khi trả về
        if move not in board.legal_moves:
            move = legal[0]
        return move
