# """TODO: implement dynamic allocation of the remaining game clock here."""

# from chesslab_sdk import SearchContext


# def allocate_time(context: SearchContext) -> int:
#     ...
"""Quản lý đồng hồ: quyết định mỗi nước đi được nghĩ tối đa bao nhiêu ms.

Ý tưởng: cả ván cờ có một quỹ giờ chung. Mỗi nước ta chỉ lấy một phần nhỏ
của quỹ giờ *còn lại* để nghĩ, nên quỹ giờ càng ít thì mỗi nước nghĩ càng ngắn
và đồng hồ không bao giờ cạn đột ngột.

Mỗi nước có hai mốc (đơn vị: mili-giây, ms):
- soft: thời gian NÊN dùng.
- hard: giới hạn TUYỆT ĐỐI, vượt là phải cắt.
"""

from chesslab_sdk import SearchContext

# ---------------------------------------------------------------------------
# Các hằng số (muốn chỉnh chiến thuật thì chỉ cần sửa ở đây)
# ---------------------------------------------------------------------------
ASSUMED_MOVES_LEFT = 30       # giả định ván còn khoảng 30 nước nữa
OVERHEAD_MS = 500             # hệ thống luôn trừ ít nhất 500 ms cho mỗi nước

NORMAL_FLOOR_MS = 400         # soft tối thiểu khi quỹ giờ còn nhiều
URGENT_FLOOR_MS = 50          # soft tối thiểu khi sắp hết giờ
URGENT_THRESHOLD_MS = 10_000  # còn trên 10 giây = "còn nhiều", dưới = "khẩn cấp"

HARD_MULTIPLIER = 3           # hard tối đa gấp 3 lần soft
HARD_FRACTION_DIVISOR = 10    # và không quá 1/10 (tức 10%) quỹ giờ còn lại


def compute_budget(remaining_ms: int) -> tuple[int, int]:
    """Nhận thời gian còn lại, trả về (soft_ms, hard_ms) cho nước đi này."""

    # Bước 1 - Chia đều: lấy quỹ giờ trừ phần hệ thống chắc chắn sẽ trừ
    # (30 nước x 500 ms), rồi chia cho 30 nước. Dùng // để ra số nguyên.
    soft = (remaining_ms - ASSUMED_MOVES_LEFT * OVERHEAD_MS) // ASSUMED_MOVES_LEFT

    # Bước 2 - Mức sàn: soft không được quá nhỏ.
    # Nghĩ dưới 500 ms gần như miễn phí vì đằng nào cũng bị trừ 500 ms.
    if remaining_ms > URGENT_THRESHOLD_MS:
        floor = NORMAL_FLOOR_MS
    else:
        floor = URGENT_FLOOR_MS
    soft = max(soft, floor)

    # Bước 3 - Giới hạn cứng: tối đa 3 lần soft, nhưng không bao giờ quá 10%
    # quỹ giờ còn lại; và hard luôn phải >= soft.
    hard = min(HARD_MULTIPLIER * soft, remaining_ms // HARD_FRACTION_DIVISOR)
    hard = max(hard, soft)

    return soft, hard


def allocate_time_window(context: SearchContext) -> tuple[int, int]:
    """Lấy thời gian còn lại của bot từ context rồi trả về (soft, hard)."""
    return compute_budget(context.time_remaining_ms)


def allocate_time(context: SearchContext) -> int:
    """Hàm mà bot.py gọi: trả về số ms bot được nghĩ cho nước này.

    search_MVP.py chỉ nhận MỘT con số và tự dùng nó theo 2 cách: dừng tăng độ
    sâu khi đã dùng quá một nửa số đó, và cắt ngang khi chạm đúng số đó.
    Nửa số đó khớp với quy tắc "quá nửa soft thì dừng" của blueprint, nên ta
    trả về soft.
    """
    soft, _hard = allocate_time_window(context)
    return soft