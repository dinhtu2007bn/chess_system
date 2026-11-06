"""
evaluation.py – Hàm đánh giá vị trí tốc độ cao cho bot Chess960 (ChessLab).

Thiết kế
--------
* Hàm đánh giá Heuristic + Tapered Evaluation (PeSTO): nội suy giữa trung cuộc
  (MG) và tàn cuộc (EG) theo game phase 0..24.
* Toàn bộ logic chạy trên bitboard số nguyên của python-chess
  (``board.pawns``, ``board.occupied_co``...) – không dùng ``SquareSet``,
  không tạo list/tuple tạm, không ``append``/``sort`` trong ``evaluate()``.
* Điểm MG và EG được **đóng gói vào một số nguyên duy nhất**
  (``packed = mg + (eg << 16)``) nên mỗi phép cộng cập nhật cả hai thành phần.
* Chuyên biệt Chess960:
    - Bảng PST của hai bên là ảnh phản chiếu theo trục ngang (``sq ^ 56``),
      vì ở Chess960 quân Đen luôn nằm cùng cột với quân Trắng nên thế xuất phát
      của mọi trong 960 thế cờ đều đối xứng tuyệt đối -> điểm = 0 (+ Tempo).
    - King Safety dựa trên vị trí Vua thực tế (cột/hàng của Vua), không giả định
      Vua ở c1/g1 hay e1.
    - Quyền nhập thành đọc từ ``board.castling_rights`` (bitboard các ô Xe còn
      quyền), đúng với cả chế độ ``chess960=True`` lẫn cờ truyền thống.
* Quy ước Negamax: điểm trả về luôn theo góc nhìn của bên đang đến lượt
  (``board.turn``). Hàm KHÔNG kiểm tra chiếu hết/hoà cờ – bộ máy tìm kiếm phải
  xử lý trạng thái kết thúc trước khi gọi ``evaluate()``.

Yêu cầu: Python >= 3.8 (dùng ``int.bit_count`` nếu có, tự động fallback).
"""

from __future__ import annotations

import chess

# ---------------------------------------------------------------------------
# Giá trị vật chất cơ sở (PeSTO standard values)
# ---------------------------------------------------------------------------
MG_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN:   82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK:   477,
    chess.QUEEN:  1025,
    chess.KING:   0,
}

EG_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN:   94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK:   512,
    chess.QUEEN:  936,
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
# PeSTO Piece-Square Tables (Viết trực quan: Rank 8 ở trên, Rank 1 ở dưới)
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
     13,   9,   8,  10,  11,  11,  15,  12,
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
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -48, -18,
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
    -53, -34, -21, -11, -28, -14, -24, -43,
]


def _orient_table(table: list[int]) -> list[int]:
    """
    Chuyển đổi bảng từ dạng trực quan (Rank 8 ở trên)
    sang dạng chỉ số ô chuẩn của python-chess (Rank 1 ở index 0..7).
    """
    res: list[int] = []
    for r in range(7, -1, -1):
        res.extend(table[r * 8 : (r + 1) * 8])
    return res


# Các bảng sau khi orient (index 0 = A1, index 63 = H8)
PST_MG: dict[chess.PieceType, list[int]] = {
    chess.PAWN:   _orient_table(_PAWN_MG_RAW),
    chess.KNIGHT: _orient_table(_KNIGHT_MG_RAW),
    chess.BISHOP: _orient_table(_BISHOP_MG_RAW),
    chess.ROOK:   _orient_table(_ROOK_MG_RAW),
    chess.QUEEN:  _orient_table(_QUEEN_MG_RAW),
    chess.KING:   _orient_table(_KING_MG_RAW),
}

PST_EG: dict[chess.PieceType, list[int]] = {
    chess.PAWN:   _orient_table(_PAWN_EG_RAW),
    chess.KNIGHT: _orient_table(_KNIGHT_EG_RAW),
    chess.BISHOP: _orient_table(_BISHOP_EG_RAW),
    chess.ROOK:   _orient_table(_ROOK_EG_RAW),
    chess.QUEEN:  _orient_table(_QUEEN_EG_RAW),
    chess.KING:   _orient_table(_KING_EG_RAW),
}

# Điểm tốt thông theo rank (index = rank 0..7)
PASSED_PAWN_BONUS_MG = [0, 5, 10, 20, 35, 60, 100, 0]
PASSED_PAWN_BONUS_EG = [0, 10, 20, 40, 70, 120, 200, 0]


# ===========================================================================
# PHẦN 2 – TIỀN XỬ LÝ (chạy MỘT LẦN lúc import, không tốn chi phí khi evaluate)
# ===========================================================================

try:  # Python >= 3.10: popcount viết bằng C
    _popcount = int.bit_count
except AttributeError:  # pragma: no cover - fallback cho Python cũ
    def _popcount(bb: int) -> int:
        return bin(bb).count("1")

_ALL = chess.BB_ALL
_NOT_FILE_A = ~chess.BB_FILE_A & _ALL
_NOT_FILE_H = ~chess.BB_FILE_H & _ALL
_FILE_MUL = chess.BB_FILE_A          # 0x0101..01: nhân byte-cột -> mặt nạ cột đầy đủ
_RANKS_4_6 = chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6   # ổ chặn Mã Trắng
_RANKS_3_5 = chess.BB_RANK_3 | chess.BB_RANK_4 | chess.BB_RANK_5   # ổ chặn Mã Đen (đối xứng)


def _pk(mg: int, eg: int) -> int:
    """Đóng gói (mg, eg) thành một số nguyên: ``mg + eg * 65536`` (tuyến tính)."""
    return mg + (eg << 16)


def _build_packed_pst(
    color_white: bool, pt: chess.PieceType
) -> tuple[int, ...]:
    """
    Bảng 64 ô đã đóng gói [vật chất + PST] cho một loại quân.
    Với quân Đen, bảng được phản chiếu sẵn (``sq ^ 56``) để evaluate() tra
    thẳng bằng chỉ số ô thật, không cần XOR/lật lúc chạy.
    """
    mg_base, eg_base = MG_PIECE_VALUES[pt], EG_PIECE_VALUES[pt]
    out = []
    for sq in range(64):
        src = sq if color_white else sq ^ 56
        out.append(_pk(mg_base + PST_MG[pt][src], eg_base + PST_EG[pt][src]))
    return tuple(out)


_WP, _WN, _WB, _WR, _WQ, _WK = (_build_packed_pst(True, pt) for pt in range(1, 7))
_BP, _BN, _BB, _BR, _BQ, _BK = (_build_packed_pst(False, pt) for pt in range(1, 7))

# Điểm tốt thông tra theo ô (đã gồm lật hàng cho quân Đen).
_PASSED_W = tuple(
    _pk(PASSED_PAWN_BONUS_MG[sq >> 3], PASSED_PAWN_BONUS_EG[sq >> 3]) for sq in range(64)
)
_PASSED_B = tuple(
    _pk(PASSED_PAWN_BONUS_MG[7 - (sq >> 3)], PASSED_PAWN_BONUS_EG[7 - (sq >> 3)])
    for sq in range(64)
)


def _build_king_zones() -> tuple[tuple[int, ...], ...]:
    """
    Với mỗi ô Vua (0..63) dựng:
      * file_byte : byte 8-bit các cột [f-1, f, f+1] quanh Vua,
      * near_w/far_w : 3 ô ngay trước / cách 2 hàng phía trước Vua (hướng Trắng),
      * near_b/far_b : tương tự theo hướng Đen.
    Dựa hoàn toàn vào ô Vua thực tế -> đúng cho mọi vị trí xuất phát Chess960.
    """
    file_byte, near_w, far_w, near_b, far_b = [], [], [], [], []
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        fb = nw = fw = nb = fb2 = 0
        for ff in (f - 1, f, f + 1):
            if not 0 <= ff <= 7:
                continue
            fb |= 1 << ff
            if r + 1 <= 7:
                nw |= 1 << ((r + 1) * 8 + ff)
            if r + 2 <= 7:
                fw |= 1 << ((r + 2) * 8 + ff)
            if r - 1 >= 0:
                nb |= 1 << ((r - 1) * 8 + ff)
            if r - 2 >= 0:
                fb2 |= 1 << ((r - 2) * 8 + ff)
        file_byte.append(fb)
        near_w.append(nw)
        far_w.append(fw)
        near_b.append(nb)
        far_b.append(fb2)
    return tuple(file_byte), tuple(near_w), tuple(far_w), tuple(near_b), tuple(far_b)


_KS_FILES, _KS_NEAR_W, _KS_FAR_W, _KS_NEAR_B, _KS_FAR_B = _build_king_zones()

# ---------------------------------------------------------------------------
# Hằng số tinh chỉnh (đã đóng gói: _pk(mg, eg)). Chỉnh ở đây khi tuning.
# ---------------------------------------------------------------------------
TEMPO_BONUS = 10

_BISHOP_PAIR = _pk(35, 50)
_ROOK_OPEN_FILE = _pk(25, 20)
_ROOK_SEMI_OPEN = _pk(15, 10)
_ROOK_7TH = _pk(25, 35)
_DOUBLED_PAWN = _pk(15, 20)       # trừ cho mỗi Tốt nằm trong cột có >= 2 Tốt
_ISOLATED_PAWN = _pk(15, 20)
_CONNECTED_PASSED = _pk(20, 40)   # mỗi cặp cột kề nhau cùng có Tốt thông
_KNIGHT_OUTPOST = _pk(20, 10)

# King Safety (chỉ trung cuộc => eg = 0)
KS_MIN_PHASE = 10                 # chỉ tính khi phase > KS_MIN_PHASE
_KS_SHIELD_NEAR = 8               # Tốt ngay trước Vua (3 ô: f-1, f, f+1)
_KS_SHIELD_FAR = 4                # Tốt cách Vua 2 hàng
_KS_OPEN_FILE = 20                # cột gần Vua hoàn toàn không có Tốt
_KS_HALF_OPEN_FILE = 10           # cột gần Vua không có Tốt nhà, chỉ có Tốt địch

# Quyền nhập thành (chỉ trung cuộc): mỗi quyền còn lại = một "tuỳ chọn an toàn".
_CASTLE_RIGHT = _pk(10, 0)


# ===========================================================================
# PHẦN 3 – HÀM TIỆN ÍCH BITBOARD
# ===========================================================================

def _pst_sum(bb: int, table: tuple[int, ...]) -> int:
    """Tổng điểm đóng gói của mọi quân trong ``bb`` (tách LSB bằng ``bb & -bb``)."""
    total = 0
    while bb:
        lsb = bb & -bb
        total += table[lsb.bit_length() - 1]
        bb ^= lsb
    return total


def _north_fill(bb: int) -> int:
    """Tô dọc lên phía hàng 8 (Kogge-Stone), bao gồm cả các bit gốc."""
    bb |= (bb << 8) & _ALL
    bb |= (bb << 16) & _ALL
    bb |= (bb << 32) & _ALL
    return bb


def _south_fill(bb: int) -> int:
    """Tô dọc xuống phía hàng 1 (Kogge-Stone), bao gồm cả các bit gốc."""
    bb |= bb >> 8
    bb |= bb >> 16
    bb |= bb >> 32
    return bb


def _spread_files(bb: int) -> int:
    """Mở rộng mỗi bit sang hai cột kề bên (không tràn vòng qua mép bàn cờ)."""
    return bb | ((bb << 1) & _NOT_FILE_A) | ((bb >> 1) & _NOT_FILE_H)


# ===========================================================================
# PHẦN 4 – HÀM ĐÁNH GIÁ
# ===========================================================================

def evaluate(board: chess.Board) -> int:
    """
    Đánh giá tĩnh vị trí theo quy ước Negamax.

    Args:
        board: thế cờ (cờ truyền thống hoặc Chess960). Giả định mỗi bên có
            đúng một Vua; hàm không kiểm tra chiếu hết/hoà cờ.

    Returns:
        Điểm (centipawn) theo góc nhìn của bên đang đến lượt ``board.turn``;
        dương = có lợi cho bên đang đi. Thế xuất phát của mọi vị trí Chess960
        cho đúng ``+TEMPO_BONUS``.
    """
    occ_b, occ_w = board.occupied_co          # chess.BLACK = 0, chess.WHITE = 1
    pawns = board.pawns
    kings = board.kings

    wp = pawns & occ_w
    bp = pawns & occ_b
    rooks = board.rooks
    wr = rooks & occ_w
    br = rooks & occ_b
    bishops = board.bishops
    wb = bishops & occ_w
    bb_ = bishops & occ_b
    knights = board.knights
    wn = knights & occ_w
    bn = knights & occ_b

    # --- 1. Game phase: chỉ cần popcount, không duyệt quân ------------------
    phase = (
        _popcount(knights | bishops)
        + 2 * _popcount(rooks)
        + 4 * _popcount(board.queens)
    )
    if phase > 24:
        phase = 24

    # --- 2. Vật chất + PST (đóng gói mg|eg) ----------------------------------
    queens = board.queens
    s = (
        _pst_sum(wp, _WP) + _pst_sum(wn, _WN) + _pst_sum(wb, _WB)
        + _pst_sum(wr, _WR) + _pst_sum(queens & occ_w, _WQ)
        - _pst_sum(bp, _BP) - _pst_sum(bn, _BN) - _pst_sum(bb_, _BB)
        - _pst_sum(br, _BR) - _pst_sum(queens & occ_b, _BQ)
    )
    wk_bb = kings & occ_w
    bk_bb = kings & occ_b
    wk = wk_bb.bit_length() - 1               # -1 nếu thiếu Vua (thế cờ không hợp lệ)
    bk = bk_bb.bit_length() - 1
    if wk_bb:
        s += _WK[wk]
    if bk_bb:
        s -= _BK[bk]

    # --- 3. Cặp Tượng: "bb & (bb-1)" != 0  <=>  có từ 2 bit trở lên -----------
    if wb & (wb - 1):
        s += _BISHOP_PAIR
    if bb_ & (bb_ - 1):
        s -= _BISHOP_PAIR

    # --- 4. Cấu trúc cột Tốt bằng fill (thay vòng lặp từng Tốt) ---------------
    w_front = _north_fill((wp << 8) & _ALL)   # ô phía trước các Tốt Trắng
    w_rear = _south_fill(wp >> 8)             # ô phía sau các Tốt Trắng
    b_front = _south_fill(bp >> 8)            # ô phía trước các Tốt Đen
    b_rear = _north_fill((bp << 8) & _ALL)    # ô phía sau các Tốt Đen

    files_w = (w_rear | wp) & 0xFF            # byte: bit f = cột f có Tốt Trắng
    files_b = ((b_rear | bp) >> 56) & 0xFF
    full_w = files_w * _FILE_MUL              # mặt nạ cột đầy đủ 64-bit
    full_b = files_b * _FILE_MUL

    # Tốt chồng: Tốt có Tốt cùng màu cùng cột (phía trước hoặc sau)
    s -= _DOUBLED_PAWN * (
        _popcount(wp & (w_front | w_rear)) - _popcount(bp & (b_front | b_rear))
    )

    # Tốt cô lập: cột của nó không có Tốt nhà ở hai cột kề
    adj_w = ((files_w << 1) & 0xFF) | (files_w >> 1)
    adj_b = ((files_b << 1) & 0xFF) | (files_b >> 1)
    s -= _ISOLATED_PAWN * (
        _popcount(wp & ((files_w & ~adj_w) * _FILE_MUL))
        - _popcount(bp & ((files_b & ~adj_b) * _FILE_MUL))
    )

    # Tốt thông: không có Tốt địch phía trước trên cột mình và hai cột kề
    passed_w = wp & ~_spread_files(b_front)
    passed_b = bp & ~_spread_files(w_front)

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

    # --- 5. Xe: cột mở / nửa mở / hàng 7 -------------------------------------
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

    # --- 6. Mã chiếm tiền đồn: bản đồ ô bị Tốt tấn công tính bằng shift -------
    if wn or bn:
        w_pawn_att = ((wp << 7) & _NOT_FILE_H) | ((wp << 9) & _NOT_FILE_A)
        b_pawn_att = ((bp >> 7) & _NOT_FILE_A) | ((bp >> 9) & _NOT_FILE_H)
        s += _KNIGHT_OUTPOST * (
            _popcount(wn & w_pawn_att & _RANKS_4_6)
            - _popcount(bn & b_pawn_att & _RANKS_3_5)
        )

    # --- 7. King Safety (Chess960-safe: dựa trên ô Vua thực tế) ---------------
    if phase > KS_MIN_PHASE:
        if 0 <= wk < 24:                       # Vua còn ở 3 hàng nhà (hàng 1-3)
            zone = _KS_FILES[wk]
            lack = zone & ~files_w             # cột gần Vua không có Tốt nhà
            s += (
                _KS_SHIELD_NEAR * _popcount(wp & _KS_NEAR_W[wk])
                + _KS_SHIELD_FAR * _popcount(wp & _KS_FAR_W[wk])
                - _KS_OPEN_FILE * _popcount(lack & ~files_b)
                - _KS_HALF_OPEN_FILE * _popcount(lack & files_b)
            )
        if bk >= 40:                           # Vua Đen còn ở hàng 6-8
            zone = _KS_FILES[bk]
            lack = zone & ~files_b
            s -= (
                _KS_SHIELD_NEAR * _popcount(bp & _KS_NEAR_B[bk])
                + _KS_SHIELD_FAR * _popcount(bp & _KS_FAR_B[bk])
                - _KS_OPEN_FILE * _popcount(lack & ~files_w)
                - _KS_HALF_OPEN_FILE * _popcount(lack & files_w)
            )

    # --- 8. Quyền nhập thành (Chess960): castling_rights = bitboard ô Xe ------
    cr = board.castling_rights
    if cr:
        # Chỉ tính khi Xe còn đó và Vua còn ở hàng nhà tương ứng.
        cw = _popcount(cr & wr & chess.BB_RANK_1) if 0 <= wk < 8 else 0
        cb = _popcount(cr & br & chess.BB_RANK_8) if bk >= 56 else 0
        s += _CASTLE_RIGHT * (cw - cb)

    # --- 9. Tapered Evaluation: tách mg/eg khỏi số nguyên đóng gói ------------
    eg = (s + 0x8000) >> 16                   # sàn chia -> eg chính xác
    mg = s - (eg << 16)
    x = mg * phase + eg * (24 - phase)
    score = x // 24 if x >= 0 else -(-x // 24)  # chia cắt về 0 => đối xứng dấu tuyệt đối

    # --- 10. Tempo + đổi dấu theo Negamax -------------------------------------
    return score + TEMPO_BONUS if board.turn else TEMPO_BONUS - score


# ===========================================================================
# Tự kiểm tra nhanh: python evaluation.py
# ===========================================================================
if __name__ == "__main__":
    import random
    import time

    # (1) Đối xứng 100% ở cả 960 thế xuất phát: điểm == +TEMPO_BONUS cho bên đi.
    for n in range(960):
        for turn in (chess.WHITE, chess.BLACK):
            b = chess.Board.from_chess960_pos(n)
            b.turn = turn
            assert evaluate(b) == TEMPO_BONUS, f"Chess960 #{n} mất đối xứng"
    print("OK: 960/960 thế xuất phát đều = +%d" % TEMPO_BONUS)

    # (2) Đối xứng Negamax trên các thế ngẫu nhiên: evaluate(b) == evaluate(b.mirror()).
    rng = random.Random(0)
    sample: list[chess.Board] = []
    for _ in range(300):
        b = chess.Board.from_chess960_pos(rng.randrange(960))
        for _ in range(rng.randint(1, 100)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
            if rng.random() < 0.2:
                sample.append(b.copy())
    assert all(evaluate(p) == evaluate(p.mirror()) for p in sample)
    print("OK: mirror-symmetry trên %d thế ngẫu nhiên" % len(sample))

    # (3) Tốc độ.
    t0 = time.perf_counter()
    for _ in range(5):
        for p in sample:
            evaluate(p)
    nps = 5 * len(sample) / (time.perf_counter() - t0)
    print("NPS: %d evaluations/giây" % nps)