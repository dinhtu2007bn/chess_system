"""Phân bổ thời gian mỗi nước đi cho bot Chess960.

Module lá: không import ``student.*``, ``chesslab_sdk``, ``chess`` hay I/O; hàm thuần
và tất định — cùng ``(remaining_ms, ply)`` luôn cho cùng ``TimeBudget``.

- ``soft_ms``: hạn mềm, không bắt đầu độ sâu mới sau mốc này (iterative deepening).
- ``hard_ms``: hạn cứng, ngắt ngay độ sâu đang chạy.

Hệ thống trừ vào đồng hồ mỗi lượt theo ``max(thời gian nghĩ, 500-1000 ms)``:
nghĩ nhanh vẫn bị tính sàn, còn nghĩ lâu hơn sàn thì không bị cộng thêm.

Vì vậy công thức dưới đây:
1. Trừ ``min_margin_ms`` cho phí sàn tối thiểu.
2. Chia phần còn lại cho **số nước của mình còn lại** + một khoản đệm, nên mức
   tiêu mỗi nước gần như không đổi suốt ván và tổng chi luôn dưới đồng hồ.
3. Không bao giờ tiêu dưới 500 ms khi vẫn còn hơn 500 ms, vì phần đó là "miễn phí".
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class TimeBudget:
    """Ngân sách thời gian cho 1 nước, tính bằng ms kể từ lúc gọi ``budget()``.

    ``TimeManager.budget`` bảo đảm:
    ``0 <= soft_ms <= hard_ms <= max(0, remaining_ms)`` và ``remaining_ms > 0 => hard_ms >= 1``.
    """

    soft_ms: int
    hard_ms: int


# `search.py` đã được vá để nhận kết quả của độ sâu bị ngắt giữa chừng
# (trường `SearchResult.partial`). Khi đó hạn mềm không còn tác dụng bảo vệ:
# một độ sâu dở vẫn cho nước dùng được, nên chỉ cần hạn cứng.
USING_PARTIAL_RESULT: Final[bool] = True

DEFAULT_MIN_MARGIN_MS: Final[int] = 500  # phí sàn tối thiểu mỗi nước của runner
DEFAULT_SOFT_FRACTION: Final[float] = 1.0 if USING_PARTIAL_RESULT else 0.5
DEFAULT_MAX_FRACTION: Final[float] = 0.25  # chặn bệnh lý; thực tế không bao giờ chạm
DEFAULT_FLOOR_MS: Final[int] = 10  # ngân sách tối thiểu khi gần cạn giờ
DEFAULT_SAFETY_MOVES: Final[int] = 10  # đệm số nước, giữ lại ~6% tổng ngân sách


class TimeManager:
    """Chính sách phân bổ thời gian không trạng thái, thuần số học."""

    def __init__(
        self,
        *,
        min_margin_ms: int = DEFAULT_MIN_MARGIN_MS,
        soft_fraction: float = DEFAULT_SOFT_FRACTION,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        floor_ms: int = DEFAULT_FLOOR_MS,
        safety_moves: int = DEFAULT_SAFETY_MOVES,
    ) -> None:
        # Chuẩn hóa tham số để budget() luôn giữ được bất biến, kể cả khi cấu hình sai.
        self.min_margin_ms = max(0, int(min_margin_ms))
        self.soft_fraction = max(0.0, min(1.0, float(soft_fraction)))
        self.max_fraction = max(0.0, min(1.0, float(max_fraction)))
        self.floor_ms = max(1, int(floor_ms))
        self.safety_moves = max(0, int(safety_moves))

    def budget(
        self,
        remaining_ms: int,
        ply: int,
        *,
        opponent_ms: int | None = None,
    ) -> TimeBudget:
        """Ngân sách cho lượt hiện tại. ``opponent_ms`` dành cho mở rộng sau này."""
        if remaining_ms <= 0:
            return TimeBudget(soft_ms=0, hard_ms=0)

        safe_ply = max(0, ply)

        # Số nước CÒN LẠI CỦA MÌNH. Bot đi ở các ply cùng tính chẵn lẻ với `ply`;
        # nước cuối cùng của ván là ply 299 (ván dừng khi len(moves) >= 300).
        # Đếm: ((299 - ply) // 2) + 1 = (301 - ply) // 2.
        moves_left = max(1, (301 - safe_ply) // 2)

        usable_ms = max(0, remaining_ms - self.min_margin_ms)
        if usable_ms == 0:
            # Vùng khẩn cấp: chỉ còn sàn sinh tồn, ép search chạy đúng độ sâu 1.
            hard_ms = max(1, min(remaining_ms, self.floor_ms))
            return TimeBudget(soft_ms=0, hard_ms=hard_ms)

        # Chia đều cho (số nước còn lại + đệm). Vì `moves_left` giảm 1 sau mỗi
        # nước của mình, thương này gần như không đổi suốt ván; phần đệm giữ lại
        # khoảng `safety / (moves_left + safety)` tổng ngân sách.
        per_move_ms = usable_ms // (moves_left + self.safety_moves)
        hard_ms = min(per_move_ms, int(remaining_ms * self.max_fraction))

        # Cửa sổ "miễn phí": runner luôn trừ ít nhất 500 ms mỗi nước, nên nghĩ
        # dưới 500 ms không tiết kiệm được giây nào. Dùng `usable_ms` (không phải
        # `remaining_ms`) để bảo đảm nghĩ xong vẫn còn 500 ms cho phí sàn.
        hard_ms = max(hard_ms, min(500, usable_ms))

        # Sàn bắt buộc và trần theo đồng hồ còn lại (giữ bất biến hard >= 1).
        hard_ms = min(remaining_ms, max(min(remaining_ms, self.floor_ms), hard_ms))

        soft_ms = int(hard_ms * self.soft_fraction)
        soft_ms = max(0, min(hard_ms, soft_ms))
        return TimeBudget(soft_ms=soft_ms, hard_ms=hard_ms)
