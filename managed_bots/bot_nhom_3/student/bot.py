"""Điểm vào bắt buộc: ghép lượng giá, tìm kiếm và ngân sách thời gian.

Đây là nơi **duy nhất** trong ``student/`` import ``chesslab_sdk``. Mọi lỗi bên
trong đều bị chặn ở đây và biến thành một nước hợp lệ: một exception lọt ra
ngoài là mất ván.

``__init__`` bị tính vào đồng hồ của bot, nên chỉ lưu config và tạo
``TimeManager`` — không dựng cấu trúc nặng ở đây.
"""

import time

import chess

from chesslab_sdk import BotConfig, SearchContext

from .evaluation import evaluate
from .search import find_best_move
from .time_manager import TimeManager


class StudentBot:
    """Bot của nhóm; lớp duy nhất runner và bộ chấm import."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._time = TimeManager()

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        """Trả một nước hợp lệ khi bàn còn nước; không để lỗi search lọt ra ngoài."""
        try:
            budget = self._time.budget(
                context.time_remaining_ms,
                context.ply,
                opponent_ms=context.opponent_time_remaining_ms,
            )
            # perf_counter chứ không phải monotonic: trên Windows + Python 3.12,
            # monotonic chỉ phân giải 15,6 ms, không đủ cho ngân sách sàn 10 ms.
            result = find_best_move(
                board,
                evaluate=evaluate,
                budget=budget,
                now=time.perf_counter,
            )
            if result.move in board.legal_moves:
                return result.move
        except Exception:  # noqa: BLE001, S110 - chốt an toàn; không được log vì stdout là giao thức JSON
            # Hợp đồng: mọi lỗi bên trong phải thành một nước hợp lệ.
            pass
        return next(iter(board.legal_moves))
