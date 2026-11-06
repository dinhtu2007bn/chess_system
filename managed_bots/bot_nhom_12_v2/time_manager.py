"""TODO: implement dynamic allocation of the remaining game clock here."""
from __future__ import annotations
import time

_OVERHEAT_MS = 60          # Dự phòng overhead hệ thống
_CRITICAL_MS = 2_000       # Ngưỡng thời gian nguy cấp
_PANIC_MS = 500            # Ngưỡng thời gian hoảng loạn

def compute_budget_ms(
    time_remaining_ms: int,
    ply: int,
    initial_time_ms: int,
) -> int:
    """
    Tính ngân sách tìm kiếm (ms) cho nước đi hiện tại.
    """
    safe_remaining = max(0, time_remaining_ms - _OVERHEAT_MS)

    # Chế độ hoảng loạn: còn dưới 500ms
    if safe_remaining < _PANIC_MS:
        return max(20, safe_remaining // 5)

    # Chế độ nguy cấp: còn dưới 2s
    if safe_remaining < _CRITICAL_MS:
        return max(50, safe_remaining // 4)

    # Ước tính số nước còn lại dựa trên tiến độ ván cờ
    moves_played = ply // 2
    if moves_played < 10:
        moves_left = 35
    elif moves_played < 30:
        moves_left = 30
    elif moves_played < 50:
        moves_left = 25
    else:
        moves_left = 20

    # Ngân sách cơ sở
    budget = safe_remaining // moves_left

    # Giới hạn tối đa không vượt quá 15% quỹ thời gian còn lại
    max_allowed = max(100, safe_remaining * 15 // 100)
    budget = min(budget, max_allowed)

    # Tối thiểu 80 ms
    return max(budget, 80)

class TimeManager:
    """
    Quản lý đồng hồ và deadline cho một lượt tìm kiếm.
    """

    def __init__(self, budget_ms: int) -> None:
        self.budget_ms = budget_ms
        self._start_time: float = 0.0
        self._deadline: float = 0.0

    def start(self) -> None:
        """
        Bắt đầu đồng hồ và tính toán deadline.
        """
        self._start_time = time.time()
        self._deadline = self._start_time + self.budget_ms / 1000.0

    def should_stop(self) -> bool:
        """
        Kiểm tra xem đã đến deadline hay chưa.
        """
        return time.time() >= self._deadline

    def elapsed_ms(self) -> int:
        """Số mili-giây đã trôi qua kể từ khi start()."""
        return int((time.monotonic() - self._start_time) * 1000)

    def can_start_next_depth(self) -> bool:
        """
        Kiểm tra xem có đủ thời gian để mở độ sâu tiếp theo không.
        Nếu đã tiêu tốn hơn 55% ngân sách, từ chối đào sâu thêm tầng mới
        để tránh bị timeout giữa chừng.
        """
        used = self.elapsed_ms()
        return used < int(self.budget_ms * 0.55)