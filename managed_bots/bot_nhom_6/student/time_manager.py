"""Phân bổ thời gian cho mỗi nước đi.

Tương ứng bài giảng 04, slide "Giới hạn tài nguyên": phải cân bằng độ sâu tìm kiếm và thời gian
còn lại. Nghĩ quá lâu -> hết giờ là THUA NGAY; nghĩ quá ít -> đánh yếu.

Các sự thật cần nhớ (từ README):
    - Mỗi nước bị trừ TỐI THIỂU 0.5 giây dù bot trả lời ngay -> phải để dành 0.5 s cho mỗi nước còn lại.
    - Không được cộng giờ sau mỗi nước.
    - Ván tối đa 300 nửa nước (ply) rồi hòa.
Chiến lược: chia thời gian còn lại cho số nước bot ước tính còn phải đi.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_MOVE_COST_S = 0.75    # độ trễ trung bình hệ thống trừ cho mỗi nước (tối thiểu 0.5 s; để dư cho an toàn)
MAX_RESERVED_MOVES = 50   # dành sẵn giờ trả độ trễ cho tối đa ngần này nước còn lại (ván dài mà không dự trữ sẽ thua giờ)
OVERHEAD_S = 0.15         # dự phòng cho giao tiếp giữa các tiến trình, khởi tạo, sao chép bàn cờ...
PLY_LIMIT = 300           # sau 300 nửa nước ván hòa
MIN_MOVES_TO_GO = 30      # luôn giả định còn ít nhất ngần này nước (ván có thể kéo dài tới 300 ply, mỗi nước tốn >= 0.5 s)
PANIC_REMAINING_S = 2.0   # dưới mức này chỉ tìm cực nhanh


@dataclass(frozen=True)
class TimeBudget:
    soft_seconds: float  # thời gian dự định dùng cho nước đi này
    hard_seconds: float  # tuyệt đối không vượt (đến mốc này là dừng tìm kiếm)


def moves_to_go(ply: int) -> int:
    """Ước lượng số nước bot còn phải đi: ván càng muộn càng còn ít, và không quá số nước còn lại tới giới hạn 300 ply."""
    estimate = max(MIN_MOVES_TO_GO, 55 - ply // 4)
    until_limit = max(1, (PLY_LIMIT - ply) // 2)
    return min(estimate, until_limit)


def allocate_time(time_remaining_ms: int, ply: int) -> TimeBudget:
    """Tính ngân sách cho nước đi hiện tại."""
    remaining = time_remaining_ms / 1000.0
    if remaining <= PANIC_REMAINING_S:
        # Sắp hết giờ: chỉ cần một nước hợp lệ hợp lý, độ sâu 1-2 là đủ.
        return TimeBudget(0.02, 0.1)

    n_moves = moves_to_go(ply)
    # Ván có thể kéo dài tới 300 ply: dành sẵn giờ trả độ trễ cố định cho các nước còn lại (tới MAX_RESERVED_MOVES),
    # phần còn lại mới dùng để "nghĩ" và chia đều cho số nước ước tính.
    reserved_moves = min(max(1, (PLY_LIMIT - ply) // 2), MAX_RESERVED_MOVES)
    thinking_pool = max(0.0, remaining - MIN_MOVE_COST_S * reserved_moves)
    soft = thinking_pool / n_moves - OVERHEAD_S
    soft = max(soft, 0.05)

    # Hard: cho phép vượt soft tối đa ~2.5 lần (vòng IDS cuối chưa xong sẽ bị bỏ),
    # nhưng không bao giờ dùng quá 30% tổng thời gian còn lại cho một nước.
    hard = min(soft * 2.5, remaining * 0.3)
    hard = max(hard, 0.05)
    return TimeBudget(soft, hard)
