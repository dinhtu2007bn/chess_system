"""Hàm kinh nghiệm đánh giá thế cờ.

Quy ước: centipawn, tính theo bên đi (``board.turn``); dương = có lợi cho bên đi.
Giá trị quân: tốt 100, mã 320, tượng 330, xe 500, hậu 900.

PST viết theo kiểu "nhìn bàn cờ" (hạng 8 trước); ``_from_display()`` đảo hạng
một lần để ra ``index = rank * 8 + file``, ``a1 == 0``. Nhờ vậy bảng đối chiếu
được trực tiếp với bàn cờ đang in -- đảo tay là nguồn lỗi lệch hướng âm thầm.

Chỉ vua được taper theo giai đoạn (xem ``KING_END_TABLE``); các bảng khác
không có bản tàn cuộc. ``evaluate()`` cố ý không bắt lỗi: trả 0 sẽ biến bug
thành "thế cờ cân bằng"; việc bảo đảm nước hợp lệ thuộc về phía gọi.
"""

from typing import Callable, Final

import chess

# Kiểu hàm lượng giá dùng chung: centipawn tính theo bên đi.
EvalFn = Callable[[chess.Board], int]

PAWN_VALUE: Final[int] = 100
KNIGHT_VALUE: Final[int] = 320
BISHOP_VALUE: Final[int] = 330
ROOK_VALUE: Final[int] = 500
QUEEN_VALUE: Final[int] = 900

PIECE_VALUES: Final[dict[int, int]] = {
    chess.PAWN: PAWN_VALUE,
    chess.KNIGHT: KNIGHT_VALUE,
    chess.BISHOP: BISHOP_VALUE,
    chess.ROOK: ROOK_VALUE,
    chess.QUEEN: QUEEN_VALUE,
}


# ---- Bảng điểm theo ô (PST) ----
# 64 số nguyên, index = rank*8 + file (a1 = 0, h8 = 63), viết theo góc nhìn
# TRẮNG. Quân đen tra ô đối xứng chess.square_mirror(sq).
def _from_display(rows: tuple[int, ...]) -> tuple[int, ...]:
    """Đổi bảng viết kiểu hiển thị (hạng 8 trước) sang index a1 = 0."""
    if len(rows) != 64:
        raise ValueError("PST phải đúng 64 phần tử")
    out: list[int] = []
    for rank in range(8):  # rank 0 = hạng 1 = hàng cuối của bảng hiển thị
        start = (7 - rank) * 8
        out.extend(rows[start : start + 8])
    return tuple(out)


# Bảng được giữ nguyên dạng lưới 8x8: ruff format mặc định sẽ bung mỗi
# bảng thành 64 dòng, một số một dòng, phá mất khả năng đối chiếu trực tiếp
# với bàn cờ -- đúng thứ mà _from_display() sinh ra để bảo vệ.
# fmt: off
PAWN_TABLE: Final[tuple[int, ...]] = _from_display((
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
))

KNIGHT_TABLE: Final[tuple[int, ...]] = _from_display((
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
))

BISHOP_TABLE: Final[tuple[int, ...]] = _from_display((
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
))

ROOK_TABLE: Final[tuple[int, ...]] = _from_display((
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   0,   0,   0,   0,   0,
))

QUEEN_TABLE: Final[tuple[int, ...]] = _from_display((
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
))

KING_MIDDLE_TABLE: Final[tuple[int, ...]] = _from_display((
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
))

# Vua ở tàn cuộc: ra trung tâm là tốt (+40 ở d4/d5/e4/e5), nằm ở góc là xấu.
# Đối nghịch với KING_MIDDLE_TABLE về hướng ưu tiên -- một bảng trung cuộc duy
# nhất sẽ dạy bot giữ vua ở hạng 1 trong khi ván cờ đã hết quân nặng.
KING_END_TABLE: Final[tuple[int, ...]] = _from_display((
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
))
# fmt: on

PIECE_SQUARE_TABLES: Final[dict[int, tuple[int, ...]]] = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_MIDDLE_TABLE,
}

# Bảng đen lật sẵn về hướng trắng: đúng bằng ``table[chess.square_mirror(sq)]``,
# chỉ khác là tra trước một lần thay vì gọi hàm cho từng quân đen.
# Không có KING: điểm vua phụ thuộc giai đoạn nên phải đi qua ``_king_score()``.
_BLACK_TABLES: Final[dict[int, tuple[int, ...]]] = {
    piece_type: tuple(table[chess.square_mirror(square)] for square in chess.SQUARES)
    for piece_type, table in PIECE_SQUARE_TABLES.items()
    if piece_type != chess.KING
}

# Trọng số giai đoạn: quân nhẹ 1, xe 2, hậu 4. Không tính tốt và vua.
PHASE_WEIGHTS: Final[dict[int, int]] = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}

# Thế xuất phát: 2*(2*1 + 2*1 + 2*2 + 1*4) = 24. 0 = sạch quân nặng = tàn cuộc.
PHASE_TOTAL: Final[int] = 24


def evaluate(board: chess.Board) -> int:
    """Trả điểm theo bên đi, đơn vị centipawn (dương = tốt cho board.turn)."""
    score = _evaluate_white(board)
    return score if board.turn == chess.WHITE else -score


def _game_phase(board: chess.Board) -> int:
    """Tổng trọng số quân nặng còn lại, chặn trên ở ``PHASE_TOTAL``.

    Phong cấp có thể đẩy tổng vượt ``PHASE_TOTAL`` (7 hậu = 28); không chặn thì
    ``(PHASE_TOTAL - phase)`` âm và ``_king_score()`` nội suy ra ngoài hai bảng.
    """
    phase = 0
    for piece_type, weight in PHASE_WEIGHTS.items():
        pieces = board.pieces_mask(piece_type, chess.WHITE) | board.pieces_mask(
            piece_type, chess.BLACK
        )
        phase += weight * pieces.bit_count()
    return min(phase, PHASE_TOTAL)


def _king_score(square: int, phase: int) -> int:
    """Điểm vua trắng ở ``square``, nội suy giữa trung cuộc và tàn cuộc."""
    middle = _piece_score(chess.KING, square)
    return (middle * phase + KING_END_TABLE[square] * (PHASE_TOTAL - phase)) // PHASE_TOTAL


def _evaluate_white(board: chess.Board) -> int:
    """Điểm theo góc nhìn quân trắng (dương = tốt cho trắng).

    Duyệt bitboard thay vì ``piece_map()``: hàm nằm trong vòng lặp nóng nhất của
    search. Vua tách riêng vì điểm của nó phụ thuộc giai đoạn.
    """
    total = 0
    for piece_type, table in PIECE_SQUARE_TABLES.items():
        if piece_type == chess.KING:
            continue
        value = PIECE_VALUES[piece_type]
        black_table = _BLACK_TABLES[piece_type]
        squares = board.pieces_mask(piece_type, chess.WHITE)
        while squares:
            low = squares & -squares
            total += value + table[low.bit_length() - 1]
            squares ^= low
        squares = board.pieces_mask(piece_type, chess.BLACK)
        while squares:
            low = squares & -squares
            total -= value + black_table[low.bit_length() - 1]
            squares ^= low
    phase = _game_phase(board)
    white_king = board.king(chess.WHITE)
    if white_king is not None:
        total += _king_score(white_king, phase)
    black_king = board.king(chess.BLACK)
    if black_king is not None:
        total -= _king_score(chess.square_mirror(black_king), phase)
    return total


def _piece_score(piece_type: int, square: int) -> int:
    """Giá trị quân + PST của 1 quân trắng (đã quy về hướng trắng)."""
    return PIECE_VALUES.get(piece_type, 0) + PIECE_SQUARE_TABLES[piece_type][square]
