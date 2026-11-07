"""Điểm vào của bot: lớp StudentBot được hệ thống chấm khởi tạo và gọi `choose_move`.

Luồng xử lý mỗi lượt:
    1. Chỉ có 1 nước hợp lệ -> trả luôn (tiết kiệm giờ).
    2. time_manager.allocate_time  -> ngân sách giờ cho nước này.
    3. search.Searcher.search      -> IDS + alpha-beta (dùng evaluation.evaluate ở lá).
    4. Có bất kỳ lỗi nào -> trả nước hợp lệ dự phòng (không bao giờ để bot bị xử thua vì lỗi).
"""

from __future__ import annotations

import random

import chess

from chesslab_sdk import BotConfig, SearchContext

from .search import Searcher
from .time_manager import allocate_time


class StudentBot:
    """The only class imported by the local runner and official grader."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        # Nguồn ngẫu nhiên duy nhất, khởi tạo từ config.seed (yêu cầu của đề) -> chạy lại cho kết quả giống nhau.
        self.rng = random.Random(config.seed)
        self.searcher = Searcher(self.rng)

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        legal_moves = list(board.legal_moves)
        fallback = legal_moves[0]
        if len(legal_moves) == 1:
            return fallback
        try:
            budget = allocate_time(context.time_remaining_ms, context.ply)
            # Tìm kiếm trên BẢN SAO để dù có lỗi giữa chừng, bàn cờ gốc vẫn nguyên vẹn.
            move = self.searcher.search(board.copy(), budget.soft_seconds, budget.hard_seconds)
            if move in legal_moves:
                return move
        except Exception:
            pass  # lỗi bất ngờ: dùng nước dự phòng bên dưới
        return fallback
