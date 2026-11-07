"""
Module time_manager.py – Quản trị ngân sách thời gian thích ứng đa tầng (Tactical Delay-Aware Time Manager).

Nguyên lý quản lý thời gian:
1. Cơ chế hấp thụ độ trễ GameController (500ms - 1000ms):
   elapsed = max(bot_search_time_ms, random_delay_500_1000ms)
   - Trong giai đoạn khai cuộc và trung cuộc (nước 1 - 31): Ngân sách 150ms - 220ms được "hấp thụ" hoàn toàn
     bởi độ trễ tối thiểu (500ms), không làm tiêu hao thêm thời gian thực của đồng hồ thi đấu.
2. Kiểm soát ngưỡng thời gian tàn cuộc (Endgame Clock Safety):
   - Duy trì ngân sách 35ms - 55ms cho các nước sau 40 hoặc khi đồng hồ <= 10s.
   - Với tốc độ khoảng 85k-95k NPS, ngân sách này duyệt được 3,500 - 5,000 nodes, hoàn tất độ sâu 3-4 ply.
     Điều này đảm bảo nhận diện kịp thời các đòn chiếu hết, thoát chiếu và bắt quân.
   - Sau nước 55 (ply >= 110), ngân sách hạ nhẹ xuống 25ms - 40ms để duy trì đồng hồ sinh tồn trong các ván cờ cực dài.
   - Dưới 1.5 giây còn lại, kích hoạt chế độ khẩn cấp cực hạn (8ms - 15ms) để tránh bị rụng cờ (flag fall).
3. Đo lường chính xác nano-giây bằng time.perf_counter().
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeBudget:
    """
    Cấu trúc dữ liệu chứa giới hạn mềm (soft limit) và giới hạn cứng (hard limit).

    Thuộc tính:
        soft_ms: Ngưỡng thời gian để quyết định có mở thêm tầng độ sâu mới hay không.
        hard_ms: Ngưỡng thời gian tối đa tuyệt đối; nếu vượt quá sẽ ngắt tìm kiếm ngay lập tức.
    """
    soft_ms: int
    hard_ms: int


def compute_budget_ms(
    time_remaining_ms: int,
    ply: int,
    initial_time_ms: int = 45_000,
) -> TimeBudget:
    """
    Mục đích:
        Tính toán ngân sách thời gian tìm kiếm thích ứng (ms) theo mô hình bù trừ độ trễ hệ thống.

    Tham số:
        time_remaining_ms: Thời gian còn lại của bot trên đồng hồ thi đấu (mili-giây).
        ply: Số nửa nước đi (half-moves) tính từ đầu ván đấu.
        initial_time_ms: Tổng thời gian ban đầu được cấp cho ván đấu (mili-giây).

    Trả về:
        TimeBudget: Đối tượng chứa soft_ms và hard_ms phù hợp với giai đoạn trận đấu.
    """
    safe_remaining = max(0, time_remaining_ms - 50)
    is_blitz = initial_time_ms <= 60_000
    moves_played = ply // 2

    if is_blitz:
        # 1. Chế độ khẩn cấp cực hạn (dưới 1.5 giây còn lại):
        if safe_remaining <= 1_500:
            return TimeBudget(soft_ms=8, hard_ms=15)

        # 2. Chế độ khẩn cấp (dưới 3.5 giây còn lại):
        elif safe_remaining <= 3_500:
            return TimeBudget(soft_ms=15, hard_ms=25)

        # 3. Tàn cuộc siêu dài (sau nước 55, ply >= 110):
        # Hạ nhẹ xuống 25ms - 40ms để duy trì đồng hồ sinh tồn mà vẫn đạt trọn vẹn Depth 3
        elif moves_played >= 55:
            return TimeBudget(soft_ms=25, hard_ms=40)

        # 4. Sau nước 40 hoặc quỹ thời gian dưới 10s (Tàn cuộc sâu):
        # 35ms - 55ms với 90k NPS duyệt được 3,500 - 5,000 nodes, hoàn tất Depth 3-4,
        # hoàn toàn được hấp thụ bởi độ trễ 500ms của GameController.
        elif moves_played >= 40 or safe_remaining <= 10_000:
            return TimeBudget(soft_ms=35, hard_ms=55)

        # 5. Giai đoạn chuyển tiếp (nước 32 - 39):
        elif moves_played >= 32 or safe_remaining <= 18_000:
            return TimeBudget(soft_ms=60, hard_ms=90)

        # 6. Giai đoạn khai cuộc & trung cuộc (nước 1 - 31):
        else:
            return TimeBudget(soft_ms=150, hard_ms=220)
    else:
        # Thể thức Standard (180s - 360s thi đấu chính thức):
        safe_ply = max(0, ply)
        moves_left = max(1, (301 - safe_ply) // 2)
        usable_ms = max(0, time_remaining_ms - 1_000)

        if usable_ms <= 3_000:
            return TimeBudget(soft_ms=15, hard_ms=25)
        elif usable_ms <= 10_000:
            return TimeBudget(soft_ms=50, hard_ms=100)
        elif usable_ms <= 25_000:
            return TimeBudget(soft_ms=120, hard_ms=200)
        else:
            per_move = usable_ms // (moves_left + 10)
            hard_ms = min(per_move, int(time_remaining_ms * 0.15))
            hard_ms = max(hard_ms, min(1_500, usable_ms))
            soft_ms = int(hard_ms * 0.70)
            return TimeBudget(soft_ms=soft_ms, hard_ms=hard_ms)


class TimeManager:
    """
    Bộ quản lý đồng hồ đơn điệu và deadline cho lượt tìm kiếm.
    Hỗ trợ Soft Limit, Hard Limit và Dự đoán tăng trưởng thời gian lặp sâu (Branching Factor Prediction).
    """

    __slots__ = ("soft_ms", "hard_ms", "_start_time", "_deadline", "last_iteration_ms")

    def __init__(self, budget: TimeBudget | int) -> None:
        """
        Mục đích:
            Khởi tạo bộ quản lý thời gian với ngân sách xác định.

        Tham số:
            budget: Đối tượng TimeBudget hoặc một số nguyên biểu thị mili-giây.

        Trả về:
            None.
        """
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
        """
        Mục đích:
            Bắt đầu tính giờ và xác lập deadline cứng sử dụng time.perf_counter().

        Tham số:
            None.

        Trả về:
            None.
        """
        self._start_time = time.perf_counter()
        self._deadline = self._start_time + self.hard_ms / 1000.0

    def should_stop(self) -> bool:
        """
        Mục đích:
            Kiểm tra xem đã vượt quá Hard Limit (hạn chót tuyệt đối) hay chưa.

        Tham số:
            None.

        Trả về:
            bool: True nếu đã hết thời gian, False nếu vẫn còn thời gian.
        """
        return time.perf_counter() >= self._deadline

    def elapsed_ms(self) -> int:
        """
        Mục đích:
            Tính số mili-giây đã trôi qua kể từ khi gọi hàm start().

        Tham số:
            None.

        Trả về:
            int: Thời gian đã trôi qua (ms).
        """
        if self._start_time == 0.0:
            return 0
        return int((time.perf_counter() - self._start_time) * 1000)

    def can_start_next_depth(self) -> bool:
        """
        Mục đích:
            Kiểm tra Soft Limit và Dự đoán tăng trưởng thời gian (Branching Factor Prediction).
            Nếu thời gian đã trôi qua cộng với thời gian ước lượng cho tầng kế tiếp (x2.2)
            vượt quá Hard Limit, dừng ngay ở độ sâu hiện tại để bảo toàn thành quả.

        Tham số:
            None.

        Trả về:
            bool: True nếu an toàn để bắt đầu tầng độ sâu mới, False nếu nên dừng lại.
        """
        elapsed = self.elapsed_ms()
        if elapsed >= self.soft_ms:
            return False

        if self.last_iteration_ms > 0:
            predicted_ms = int(self.last_iteration_ms * 2.2)
            if elapsed + predicted_ms >= self.hard_ms:
                return False

        return True
