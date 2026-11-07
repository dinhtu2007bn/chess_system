"""
Module evaluation.py – Động cơ lượng giá vị trí cờ vua tốc độ cao (Heuristic Evaluation Core).

Các thành phần chính của hàm lượng giá:
1. Giá trị vật chất và vị trí ô (PeSTO Piece-Square Tables):
   - Đóng gói tuyến tính an toàn: val = mg + (eg << 16).
   - Giải nén số nguyên có dấu: eg = (val + 0x8000) >> 16, mg = val - (eg << 16).
   - Triệt tiêu hoàn toàn rủi ro tràn số âm 16-bit trong Python.
2. Nội suy theo giai đoạn trận đấu (Tapered Evaluation):
   - Tổng trọng số game phase từ 0 (tàn cuộc thuần túy) đến 24 (khai cuộc đầy đủ quân).
   - Điểm số nội suy: score = (mg * phase + eg * (24 - phase)) // 24.
3. Hoạt động của các quân cờ:
   - Thưởng cặp Tượng (Bishop Pair).
   - Thưởng Xe trên cột mở (Open File), cột bán mở (Semi-open File) và hàng 7 (7th Rank).
   - Thưởng Mã chiếm tiền đồn trung tâm được Tốt bảo vệ (Knight Outposts).
4. Cấu trúc Tốt:
   - Phạt Tốt chồng (Doubled Pawns) và Tốt cô lập (Isolated Pawns).
   - Thưởng Tốt thông (Passed Pawns) tăng dần theo rank tiến về phía trước.
   - Thưởng Tốt thông liên kết (Connected Passed Pawns).
5. An toàn Vua thích ứng Chess960 (Dynamic Bitwise King Ring):
   - Tính toán vành đai bảo vệ quanh Vua theo tọa độ thực tế của Vua, độc lập với vị trí ô khởi đầu.
   - Thưởng lá chắn Tốt gần/xa, phạt cột mở trước mặt Vua trong trung cuộc.
6. Dồn Vua tàn cuộc (Gated Mop-Up Heuristic):
   - Chỉ kích hoạt khi tàn cuộc áp đảo (phase <= 6 và chênh lệch >= 200 centipawns).
   - Thưởng điểm khi ép Vua đối phương ra xa tâm bàn cờ và kéo Vua ta áp sát để tạo mạng lưới chiếu hết.
7. Quy ước đối xứng gương (Mirror Symmetry):
   - Đảm bảo evaluate(b) == evaluate(b.mirror()) trên mọi thế cờ.
"""

from __future__ import annotations

import chess

# ---------------------------------------------------------------------------
# Giá trị vật chất cơ sở (PeSTO standard values)
# ---------------------------------------------------------------------------
MG_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN:   81,
    chess.KNIGHT: 324,
    chess.BISHOP: 360,
    chess.ROOK:   486,
    chess.QUEEN:  1024,
    chess.KING:   0,
}

EG_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN:   99,
    chess.KNIGHT: 288,
    chess.BISHOP: 306,
    chess.ROOK:   512,
    chess.QUEEN:  972,
    chess.KING:   0,
}

# Trọng số tính phase (tổng tối đa 24 cho cả 2 bên)
PHASE_WEIGHTS: dict[chess.PieceType, int] = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK:   2,
    chess.QUEEN:  4,
    chess.KING:   0,
}
MAX_PHASE = 24

# ---------------------------------------------------------------------------
# PeSTO Piece-Square Tables (Mảng thô 64 ô từ Rank 8 xuống Rank 1)
# ---------------------------------------------------------------------------
_PAWN_MG_RAW = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PAWN_EG_RAW = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8, -10,  -6,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

_KNIGHT_MG_RAW = [
   -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23,
]
_KNIGHT_EG_RAW = [
    -58, -38, -13, -28, -31, -27, -63,  -99,
    -25,  -8, -25,  -2,  -9, -25, -24,  -52,
    -24, -20,  10,   9,  -1,  -9, -19,  -41,
    -17,   3,  22,  22,  22,  11,   8,  -18,
    -18,  -6,  16,  25,  16,  17,   4,  -18,
    -23,  -3,  -1,  15,  10,  -3, -20,  -22,
    -42, -20, -10,  -5,  -2, -20, -23,  -44,
    -29, -51, -23, -15, -22, -18, -50,  -64,
]

_BISHOP_MG_RAW = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
_BISHOP_EG_RAW = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]

_ROOK_MG_RAW = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]
_ROOK_EG_RAW = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]

_QUEEN_MG_RAW = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
     -1, -16,   9,  18,  56,  60,  57,  35,
    -27, -27, -14,  -1,  12,  10,   6,  10,
    -16, -15,  -6,   4,   8,  14,  12,  -5,
    -22, -15,  -7,   6,   4,  10,   5, -17,
    -17,  20,  12,  -2,   9,  12,  46,  16,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]
_QUEEN_EG_RAW = [
    -9,  22,  22,  27,  27,  19,  10,  20,
   -17,  20,  32,  41,  58,  25,  30,   0,
   -20,   6,   9,  49,  47,  35,  19,   9,
     3,  22,  24,  45,  57,  40,  57,  36,
   -18,  28,  19,  47,  31,  34,  39,  18,
   -16, -27,  15,   6,   9,  17,  10,   5,
   -22, -23, -30, -16, -16, -23, -36, -32,
   -33, -28, -22, -43,  -5, -32, -20, -41,
]

_KING_MG_RAW = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]
_KING_EG_RAW = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34,  -8, -11,  -6, -13, -24, -46,
]


def _orient_table(raw: list[int]) -> list[int]:
    """
    Mục đích:
        Chuyển đổi mảng thô (xếp từ Rank 8 xuống Rank 1) sang chỉ số ô chuẩn của python-chess (0 = a1, 63 = h8).

    Tham số:
        raw: Mảng 64 số nguyên biểu diễn điểm vị trí.

    Trả về:
        list[int]: Mảng 64 số nguyên đã căn chỉnh theo thứ tự ô từ 0 đến 63.
    """
    out = [0] * 64
    for r in range(8):
        for f in range(8):
            sq = r * 8 + f
            raw_idx = (7 - r) * 8 + f
            out[sq] = raw[raw_idx]
    return out


PST_MG = {
    chess.PAWN:   _orient_table(_PAWN_MG_RAW),
    chess.KNIGHT: _orient_table(_KNIGHT_MG_RAW),
    chess.BISHOP: _orient_table(_BISHOP_MG_RAW),
    chess.ROOK:   _orient_table(_ROOK_MG_RAW),
    chess.QUEEN:  _orient_table(_QUEEN_MG_RAW),
    chess.KING:   _orient_table(_KING_MG_RAW),
}

PST_EG = {
    chess.PAWN:   _orient_table(_PAWN_EG_RAW),
    chess.KNIGHT: _orient_table(_KNIGHT_EG_RAW),
    chess.BISHOP: _orient_table(_BISHOP_EG_RAW),
    chess.ROOK:   _orient_table(_ROOK_EG_RAW),
    chess.QUEEN:  _orient_table(_QUEEN_EG_RAW),
    chess.KING:   _orient_table(_KING_EG_RAW),
}

# Điểm thưởng tốt thông theo từng rank (chỉ số 0 đến 7)
PASSED_PAWN_BONUS_MG = [0, 5, 10, 20, 35, 60, 100, 0]
PASSED_PAWN_BONUS_EG = [0, 10, 20, 40, 70, 120, 200, 0]


def _pk(mg: int, eg: int) -> int:
    """
    Mục đích:
        Đóng gói hai giá trị trung cuộc (mg) và tàn cuộc (eg) vào một số nguyên duy nhất.
        Công thức tuyến tính mg + (eg << 16) bảo toàn chính xác số nguyên có dấu trong Python.

    Tham số:
        mg: Điểm số trung cuộc (centipawn).
        eg: Điểm số tàn cuộc (centipawn).

    Trả về:
        int: Số nguyên đã đóng gói.
    """
    return mg + (eg << 16)


def _build_packed_pst(color_white: bool, pt: chess.PieceType) -> tuple[int, ...]:
    """
    Mục đích:
        Tạo bảng PST đóng gói sẵn (gồm giá trị vật chất + giá trị vị trí ô) cho từng loại quân.

    Tham số:
        color_white: True nếu là quân Trắng, False nếu là quân Đen.
        pt: Loại quân cờ (chess.PAWN, chess.KNIGHT, ...).

    Trả về:
        tuple[int, ...]: Tuple 64 phần tử chứa giá trị đã đóng gói cho 64 ô cờ.
    """
    mg_base, eg_base = MG_PIECE_VALUES[pt], EG_PIECE_VALUES[pt]
    out = []
    for sq in range(64):
        src = sq if color_white else sq ^ 56
        out.append(_pk(mg_base + PST_MG[pt][src], eg_base + PST_EG[pt][src]))
    return tuple(out)


# Khởi tạo sẵn bảng đóng gói cho quân Trắng
_WP = _build_packed_pst(True, chess.PAWN)
_WN = _build_packed_pst(True, chess.KNIGHT)
_WB = _build_packed_pst(True, chess.BISHOP)
_WR = _build_packed_pst(True, chess.ROOK)
_WQ = _build_packed_pst(True, chess.QUEEN)
_WK = _build_packed_pst(True, chess.KING)

# Khởi tạo sẵn bảng đóng gói cho quân Đen
_BP = _build_packed_pst(False, chess.PAWN)
_BN = _build_packed_pst(False, chess.KNIGHT)
_BB = _build_packed_pst(False, chess.BISHOP)
_BR = _build_packed_pst(False, chess.ROOK)
_BQ = _build_packed_pst(False, chess.QUEEN)
_BK = _build_packed_pst(False, chess.KING)

_PASSED_W = tuple(_pk(PASSED_PAWN_BONUS_MG[sq >> 3], PASSED_PAWN_BONUS_EG[sq >> 3]) for sq in range(64))
_PASSED_B = tuple(_pk(PASSED_PAWN_BONUS_MG[7 - (sq >> 3)], PASSED_PAWN_BONUS_EG[7 - (sq >> 3)]) for sq in range(64))

# Hằng số thưởng và phạt theo cấu trúc quân cờ
TEMPO_BONUS = 9
_BISHOP_PAIR = _pk(36, 54)
_ROOK_OPEN_FILE = _pk(27, 21)
_ROOK_SEMI_OPEN = _pk(15, 12)
_ROOK_7TH = _pk(27, 36)
_DOUBLED_PAWN = _pk(18, 24)
_ISOLATED_PAWN = _pk(18, 24)
_CONNECTED_PASSED = _pk(24, 48)
_KNIGHT_OUTPOST = _pk(24, 12)

# Hằng số an toàn Vua
KS_MIN_PHASE = 10
_KS_SHIELD_NEAR    = _pk(18, 0)
_KS_SHIELD_FAR     = _pk(9, 0)
_KS_OPEN_FILE      = _pk(36, 0)
_KS_HALF_OPEN_FILE = _pk(18, 0)
_CASTLE_RIGHT      = _pk(18, 0)

# Mặt nạ bit (Bitmasks)
_ALL = 0xFFFFFFFFFFFFFFFF
_NOT_FILE_A = ~chess.BB_FILE_A & _ALL
_NOT_FILE_H = ~chess.BB_FILE_H & _ALL
_RANKS_4_6 = chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6
_RANKS_3_5 = chess.BB_RANK_3 | chess.BB_RANK_4 | chess.BB_RANK_5


def _build_king_zones():
    """
    Mục đích:
        Xây dựng sẵn mặt nạ các cột lân cận và hàng lá chắn bảo vệ Vua cho toàn bộ 64 ô cờ.

    Tham số:
        None.

    Trả về:
        tuple: (files, near_w, far_w, near_b, far_b) phục vụ đánh giá an toàn Vua động.
    """
    files, near_w, far_w, near_b, far_b = [], [], [], [], []
    for sq in range(64):
        f = sq & 7
        r = sq >> 3
        fl = max(0, f - 1)
        fr = min(7, f + 1)
        fmask = 0
        for fi in range(fl, fr + 1):
            fmask |= 1 << fi
        files.append(fmask)

        def _row_mask(r_idx, fl_i, fr_i):
            if not (0 <= r_idx < 8):
                return 0
            m = 0
            for fi in range(fl_i, fr_i + 1):
                m |= 1 << (r_idx * 8 + fi)
            return m

        near_w.append(_row_mask(r + 1, fl, fr))
        far_w.append(_row_mask(r + 2, fl, fr))
        near_b.append(_row_mask(r - 1, fl, fr))
        far_b.append(_row_mask(r - 2, fl, fr))

    return (tuple(files), tuple(near_w), tuple(far_w), tuple(near_b), tuple(far_b))


_KS_FILES, _KS_NEAR_W, _KS_FAR_W, _KS_NEAR_B, _KS_FAR_B = _build_king_zones()

_POPCOUNT_TABLE = [bin(i).count("1") for i in range(256)]


def _popcount(bb: int) -> int:
    """
    Mục đích:
        Đếm số lượng bit 1 trong một số nguyên 64-bit (Bitboard).

    Tham số:
        bb: Số nguyên 64-bit.

    Trả về:
        int: Số lượng bit 1.
    """
    try:
        return bb.bit_count()
    except AttributeError:
        cnt = 0
        while bb:
            cnt += _POPCOUNT_TABLE[bb & 0xFF]
            bb >>= 8
        return cnt


def _pst_sum(bb: int, table: tuple[int, ...]) -> int:
    """
    Mục đích:
        Cộng dồn giá trị PST đóng gói cho tất cả các quân cờ đang hiện diện trên Bitboard.

    Tham số:
        bb: Mặt nạ bit vị trí các quân cờ.
        table: Bảng giá trị đóng gói tương ứng với loại quân.

    Trả về:
        int: Tổng giá trị đóng gói.
    """
    tot = 0
    while bb:
        lsb = bb & -bb
        sq = lsb.bit_length() - 1
        tot += table[sq]
        bb ^= lsb
    return tot


def _north_fill(bb: int) -> int:
    """Mở rộng mặt nạ bit về hướng Bắc (cho quân Trắng)."""
    bb |= (bb << 8)  & _ALL
    bb |= (bb << 16) & _ALL
    bb |= (bb << 32) & _ALL
    return bb


def _south_fill(bb: int) -> int:
    """Mở rộng mặt nạ bit về hướng Nam (cho quân Đen)."""
    bb |= bb >> 8
    bb |= bb >> 16
    bb |= bb >> 32
    return bb


def _spread_files(bb: int) -> int:
    """Mở rộng mặt nạ bit sang hai cột liền kề trái và phải."""
    return bb | ((bb << 1) & _NOT_FILE_A) | ((bb >> 1) & _NOT_FILE_H)


def evaluate(board: chess.Board) -> int:
    """
    Mục đích:
        Đánh giá giá trị vị trí của thế cờ hiện tại dựa trên tri thức chuyên gia.

    Quy trình đánh giá gồm 11 bước:
        1. Game Phase: Xác định giai đoạn trận đấu (0 = tàn cuộc, 24 = khai cuộc) dựa trên số lượng quân cờ lớn.
        2. Material & PST: Lượng giá vật chất và vị trí ô bằng bảng đóng gói tuyến tính.
        3. Cặp Tượng (Bishop Pair): Thưởng khi sở hữu cả 2 Tượng trên hai màu ô khác nhau.
        4. Cấu trúc Tốt: Phạt Tốt chồng, Tốt cô lập; thưởng Tốt thông và Tốt thông liên kết.
        5. Hoạt động của Xe: Thưởng Xe trên cột mở, cột bán mở và hàng 7 (hoặc hàng 2 đối với Đen).
        6. Mã tiền đồn (Knight Outposts): Thưởng Mã đứng ở các hàng 4-6 (hoặc 3-5 đối với Đen) có Tốt bảo vệ.
        7. An toàn Vua (Bitwise King Ring): Tính toán vành đai bảo vệ quanh Vua, thưởng lá chắn Tốt gần/xa.
        8. Quyền nhập thành: Thưởng điểm nếu bên đi còn quyền nhập thành hợp lệ.
        9. Tapered Evaluation: Nội suy mượt mà giữa trung cuộc và tàn cuộc theo tỷ lệ phase / 24.
        10. Mop-Up Heuristic: Ép Vua đối phương vào góc và kéo Vua ta áp sát trong tàn cuộc áp đảo (phase <= 6).
        11. Tempo Bonus & Negamax Sign: Thưởng lượt đi đầu tiên và chuẩn hóa góc nhìn theo bên đến lượt (board.turn).

    Tham số:
        board: Đối tượng chess.Board thể hiện trạng thái bàn cờ hiện tại.

    Trả về:
        int: Điểm đánh giá thế cờ tính bằng centipawn theo góc nhìn của bên đang đến lượt đi.
    """
    wp = board.pawns & board.occupied_co[chess.WHITE]
    wn = board.knights & board.occupied_co[chess.WHITE]
    wb = board.bishops & board.occupied_co[chess.WHITE]
    wr = board.rooks & board.occupied_co[chess.WHITE]
    wq = board.queens & board.occupied_co[chess.WHITE]
    wk = board.king(chess.WHITE)

    bp = board.pawns & board.occupied_co[chess.BLACK]
    bn = board.knights & board.occupied_co[chess.BLACK]
    bb = board.bishops & board.occupied_co[chess.BLACK]
    br = board.rooks & board.occupied_co[chess.BLACK]
    bq = board.queens & board.occupied_co[chess.BLACK]
    bk = board.king(chess.BLACK)

    if wk is None or bk is None:
        return 0

    # 1. Tính toán giai đoạn trận đấu (Game Phase)
    phase = (
        _popcount(wn | bn | wb | bb) * PHASE_WEIGHTS[chess.KNIGHT]
        + _popcount(wr | br) * PHASE_WEIGHTS[chess.ROOK]
        + _popcount(wq | bq) * PHASE_WEIGHTS[chess.QUEEN]
    )
    if phase > 24:
        phase = 24

    # 2. Vật chất và vị trí ô (Material & PST với Linear packing mg + (eg << 16))
    s = (
        _pst_sum(wp, _WP) + _pst_sum(wn, _WN) + _pst_sum(wb, _WB)
        + _pst_sum(wr, _WR) + _pst_sum(wq, _WQ) + _WK[wk]
        - _pst_sum(bp, _BP) - _pst_sum(bn, _BN) - _pst_sum(bb, _BB)
        - _pst_sum(br, _BR) - _pst_sum(bq, _BQ) - _BK[bk]
    )

    # 3. Đánh giá cặp Tượng
    if _popcount(wb) >= 2:
        s += _BISHOP_PAIR
    if _popcount(bb) >= 2:
        s -= _BISHOP_PAIR

    # 4. Đánh giá cấu trúc Tốt
    files_w = 0
    p = wp
    while p:
        lsb = p & -p
        files_w |= 1 << ((lsb.bit_length() - 1) & 7)
        p ^= lsb

    files_b = 0
    p = bp
    while p:
        lsb = p & -p
        files_b |= 1 << ((lsb.bit_length() - 1) & 7)
        p ^= lsb

    # Phạt Tốt chồng
    s -= _DOUBLED_PAWN * (_popcount(wp) - _popcount(files_w))
    s += _DOUBLED_PAWN * (_popcount(bp) - _popcount(files_b))

    # Phạt Tốt cô lập
    iso_w = files_w & ~((files_w << 1) | (files_w >> 1))
    iso_b = files_b & ~((files_b << 1) | (files_b >> 1))
    s -= _ISOLATED_PAWN * _popcount(iso_w)
    s += _ISOLATED_PAWN * _popcount(iso_b)

    # Thưởng Tốt thông và Tốt thông liên kết
    full_w = _north_fill(wp)
    full_b = _south_fill(bp)
    passed_w = wp & ~_spread_files(full_b)
    passed_b = bp & ~_spread_files(full_w)

    pf = 0
    while passed_w:
        lsb = passed_w & -passed_w
        sq = lsb.bit_length() - 1
        s += _PASSED_W[sq]
        pf |= 1 << (sq & 7)
        passed_w ^= lsb
    s += _CONNECTED_PASSED * _popcount(pf & (pf >> 1))

    pf = 0
    while passed_b:
        lsb = passed_b & -passed_b
        sq = lsb.bit_length() - 1
        s -= _PASSED_B[sq]
        pf |= 1 << (sq & 7)
        passed_b ^= lsb
    s -= _CONNECTED_PASSED * _popcount(pf & (pf >> 1))

    # 5. Hoạt động của Xe (Cột mở, bán mở, hàng 7)
    no_w = ~full_w
    no_b = ~full_b
    open_cols = no_w & no_b
    s += _ROOK_OPEN_FILE * (_popcount(wr & open_cols) - _popcount(br & open_cols))
    s += _ROOK_SEMI_OPEN * (
        _popcount(wr & no_w & full_b) - _popcount(br & no_b & full_w)
    )
    s += _ROOK_7TH * (
        _popcount(wr & chess.BB_RANK_7) - _popcount(br & chess.BB_RANK_2)
    )

    # 6. Mã chiếm tiền đồn trung tâm
    if wn or bn:
        w_pawn_att = ((wp << 7) & _NOT_FILE_H) | ((wp << 9) & _NOT_FILE_A)
        b_pawn_att = ((bp >> 7) & _NOT_FILE_A) | ((bp >> 9) & _NOT_FILE_H)
        s += _KNIGHT_OUTPOST * (
            _popcount(wn & w_pawn_att & _RANKS_4_6)
            - _popcount(bn & b_pawn_att & _RANKS_3_5)
        )

    # 7. An toàn Vua (Thiết kế bitwise động, an toàn với cả 960 thế cờ Chess960)
    if phase > KS_MIN_PHASE:
        if 0 <= wk < 24:
            zone = _KS_FILES[wk]
            lack = zone & ~files_w
            s += (
                _KS_SHIELD_NEAR * _popcount(wp & _KS_NEAR_W[wk])
                + _KS_SHIELD_FAR * _popcount(wp & _KS_FAR_W[wk])
                - _KS_OPEN_FILE * _popcount(lack & ~files_b)
                - _KS_HALF_OPEN_FILE * _popcount(lack & files_b)
            )
        if bk >= 40:
            zone = _KS_FILES[bk]
            lack = zone & ~files_b
            s -= (
                _KS_SHIELD_NEAR * _popcount(bp & _KS_NEAR_B[bk])
                + _KS_SHIELD_FAR * _popcount(bp & _KS_FAR_B[bk])
                - _KS_OPEN_FILE * _popcount(lack & ~files_w)
                - _KS_HALF_OPEN_FILE * _popcount(lack & files_w)
            )

    # 8. Quyền nhập thành
    cr = board.castling_rights
    if cr:
        cw = _popcount(cr & wr & chess.BB_RANK_1) if 0 <= wk < 8 else 0
        cb = _popcount(cr & br & chess.BB_RANK_8) if bk >= 56 else 0
        s += _CASTLE_RIGHT * (cw - cb)

    # 9. Tapered Evaluation: Tách mg và eg từ giá trị đóng gói
    eg = (s + 0x8000) >> 16
    mg = s - (eg << 16)
    x = mg * phase + eg * (24 - phase)
    score = x // 24 if x >= 0 else -(-x // 24)

    # 10. Mop-Up Heuristic trong tàn cuộc áp đảo (Zero-overhead ở trung cuộc)
    # cmd: Khoảng cách từ Vua đối phương đến trung tâm bàn cờ (ép Vua ra mép và góc)
    # dist: Khoảng cách giữa hai Vua (kéo Vua ta áp sát để tạo mạng lưới chiếu hết)
    if phase <= 6 and wk >= 0 and bk >= 0:
        wk_f, wk_r = wk & 7, wk >> 3
        bk_f, bk_r = bk & 7, bk >> 3
        if score >= 200:
            cmd = max(3 - bk_f, bk_f - 4) + max(3 - bk_r, bk_r - 4)
            dist = abs(wk_f - bk_f) + abs(wk_r - bk_r)
            score += ((cmd * 10) + (14 - dist) * 4) * (6 - phase) // 6
        elif score <= -200:
            cmd = max(3 - wk_f, wk_f - 4) + max(3 - wk_r, wk_r - 4)
            dist = abs(wk_f - bk_f) + abs(wk_r - bk_r)
            score -= ((cmd * 10) + (14 - dist) * 4) * (6 - phase) // 6

    # 11. Tempo Bonus & Chuẩn hóa dấu Negamax theo lượt đi
    return score + TEMPO_BONUS if board.turn else TEMPO_BONUS - score
