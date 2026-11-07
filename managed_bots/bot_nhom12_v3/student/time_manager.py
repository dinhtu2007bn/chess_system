"""
Module time_manager.py – Quản trị ngân sách thời gian thích ứng đa tầng (Adaptive Delay-Aware Time Manager).

Thiết kế tối ưu hóa triệt để dựa trên nguyên lý bù trừ độ trễ GameController (500ms - 1000ms):
1. Cơ chế trừ giờ của GameController:
   elapsed = max(bot_search_time_ms, random_delay_500_1000ms)
   - Nếu bot_search_time_ms <= 350ms, chi phí thời gian tìm kiếm được "hấp thụ" hoàn toàn
     bởi độ trễ tối thiểu (500ms), không làm tiêu hao thêm bất kỳ mili-giây nào của đồng hồ!
   - Nếu bot_search_time_ms vượt quá 500ms, mỗi mili-giây vượt quá sẽ trực tiếp làm hao hụt
     quỹ thời gian dự trữ cho giai đoạn tàn cuộc sâu (ply 80-120).

2. Phân chia thể thức:
   - Blitz (initial_time_ms <= 60,000, ví dụ 45s benchmark):
     + Ước lượng 50 nước cờ (100 plies).
     + Quỹ trễ tối thiểu trọng tài: turns_left * 550 ms.
     + Ngân sách tối ưu cho mỗi nước đi: soft_ms = 180 - 280ms, hard_ms = 350ms.
     + Nhờ đó, bot tìm kiếm ở depth 5-6 nhưng HOÀN TOÀN KHÔNG TỐN THÊM GIỜ ngoài độ trễ bắt buộc.
     + Khi thời gian còn lại < 8.0s: soft_ms = 30ms, hard_ms = 50ms.
     + Khi thời gian còn lại < 3.5s: soft_ms = 10ms, hard_ms = 15ms.

   - Standard (initial_time_ms > 60,000, ví dụ 180s - 360s thi đấu chính thức):
     + Quỹ thời gian dồi dào (180,000ms - 360,000ms).
     + Dự phòng trễ 60 nước cờ = 40,000ms.
     + Ngân sách trung bình: soft_ms = 2,500ms - 3,500ms, hard_ms = 4,000ms - 5,500ms.
     + Đạt độ sâu d=7-8 trên mọi nước đi trung cuộc.

3. Dự đoán phân nhánh mở tầng (Branching Factor Prediction):
   - Thay vì mở tầng tiếp theo mù quáng khi elapsed < soft_ms, ta kiểm tra:
     nếu elapsed + last_iteration_ms * 2.8 > hard_ms thì DỪNG LẠI NGAY ở độ sâu hiện tại,
     không bắt đầu tầng mới để tránh bị timeout dở dang lãng phí tài nguyên.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeBudget:
    """Cấu trúc dữ liệu chứa giới hạn mềm (soft limit) và giới hạn cứng (hard limit)."""
    soft_ms: int
    hard_ms: int


def compute_budget_ms(
    time_remaining_ms: int,
    ply: int,
    initial_time_ms: int,
) -> TimeBudget:
    """
    Tính toán ngân sách thời gian tìm kiếm thích ứng (ms) theo mô hình bù trừ độ trễ.

    Args:
        time_remaining_ms: Thời gian còn lại của bot trên đồng hồ (ms).
        ply: Số nửa nước đi (half-moves) đã diễn ra.
        initial_time_ms: Tổng thời gian ban đầu của ván đấu (ms).

    Returns:
        TimeBudget: Ngân sách gồm soft_ms và hard_ms.
    """
    safe_remaining = max(0, time_remaining_ms - 50)
    is_blitz = initial_time_ms <= 60_000

    moves_played = ply // 2
    turns_left = max(15, min(55, 52 - moves_played))

    if is_blitz:
        # Thể thức Blitz (45s benchmark):
        # Trọng tài áp trễ ngẫu nhiên 500-1000ms (trung bình 750ms).
        # Giữ ngân sách tìm kiếm <= 350ms để toàn bộ thời gian tìm kiếm bị "hấp thụ" bởi trễ trọng tài.
        delay_floor_reserve = turns_left * 550
        usable = max(0, safe_remaining - delay_floor_reserve)

        if safe_remaining <= 3_500:
            # Panic mode: nước đi khẩn cấp cực hạn
            return TimeBudget(soft_ms=10, hard_ms=15)
        elif safe_remaining <= 8_000:
            # Caution mode cao: hạn chế tìm kiếm để bảo vệ quỹ giờ
            return TimeBudget(soft_ms=30, hard_ms=50)
        elif usable <= 0:
            # Thiếu hụt quỹ trễ: duy trì tìm kiếm nhanh
            return TimeBudget(soft_ms=50, hard_ms=80)
        else:
            # Quỹ giờ an toàn: phân bổ ngân sách tối ưu nằm trong khung trễ
            nominal = usable // turns_left
            soft = max(60, min(260, nominal))
            hard = min(380, int(soft * 1.4))
            return TimeBudget(soft_ms=soft, hard_ms=hard)
    else:
        # Thể thức Standard (180s - 360s thi đấu chính thức):
        delay_floor_reserve = turns_left * 650
        usable = max(0, safe_remaining - delay_floor_reserve)

        if safe_remaining <= 10_000:
            return TimeBudget(soft_ms=25, hard_ms=40)
        elif safe_remaining <= 25_000:
            return TimeBudget(soft_ms=150, hard_ms=250)
        elif usable <= 0:
            return TimeBudget(soft_ms=100, hard_ms=180)
        else:
            nominal = usable // turns_left
            cap = max(500, safe_remaining * 6 // 100)
            soft = max(120, min(cap, nominal, 3_500))
            hard = min(5_500, int(soft * 1.5))
            return TimeBudget(soft_ms=soft, hard_ms=hard)


class TimeManager:
    """
    Bộ quản lý đồng hồ đơn điệu và deadline cho lượt tìm kiếm.
    Hỗ trợ Soft Limit, Hard Limit và Dự đoán tăng trưởng thời gian lặp sâu (Branching Factor Prediction).
    """

    def __init__(self, budget: TimeBudget | int) -> None:
        if isinstance(budget, TimeBudget):
            self.soft_ms: int = budget.soft_ms
            self.hard_ms: int = budget.hard_ms
        else:
            self.soft_ms = int(budget * 0.70)
            self.hard_ms = budget

        self._start_time: float = 0.0
        self._deadline: float = 0.0
        self.last_iteration_ms: int = 0

    def start(self) -> None:
        """Bắt đầu tính giờ và xác lập deadline cứng."""
        self._start_time = time.monotonic()
        self._deadline = self._start_time + self.hard_ms / 1000.0

    def should_stop(self) -> bool:
        """Kiểm tra Hard Limit (hết giờ tuyệt đối)."""
        return time.monotonic() >= self._deadline

    def elapsed_ms(self) -> int:
        """Số mili-giây đã trôi qua kể từ khi start()."""
        return int((time.monotonic() - self._start_time) * 1000)

    def can_start_next_depth(self) -> bool:
        """
        Kiểm tra Soft Limit và Dự đoán tăng trưởng thời gian (Branching Factor Prediction).
        Nếu thời gian đã trôi qua cộng với thời gian ước lượng cho tầng kế tiếp (x2.8)
        vượt quá Hard Limit, dừng ngay ở độ sâu hiện tại để bảo toàn thành quả.
        """
        elapsed = self.elapsed_ms()
        if elapsed >= self.soft_ms:
            return False

        if self.last_iteration_ms > 0:
            predicted_ms = int(self.last_iteration_ms * 2.8)
            if elapsed + predicted_ms >= self.hard_ms:
                return False

        return True
