"""Compact, high-speed, Grandmaster-level evaluation for Chess960 and standard chess.

Features:
  - Bảng giá trị vị trí quân PeSTO Piece-Square Tables (PST) cộng gộp giá trị chất quân
  - Đánh giá cấu trúc tốt lưu bộ nhớ đệm (LRU cache): tốt thông, liên kết, song hành, cô lập, chồng, lạc hậu
  - Khả năng cơ động an toàn (Safe Mobility) của Mã, Tượng, Xe, Hậu
  - Điểm thưởng căn cứ điểm tiền đồn (Outpost) cho Mã và Tượng
  - Nhận diện Tượng xấu (Bad Bishop) bị cản trở bởi tốt bị khóa cùng màu ô
  - Xe trên cột mở, cột bán mở, chồng Xe trên cột, chiếm hàng 7/2 và quy tắc Tarrasch
  - Tầm nhìn Tượng (Bishop Vision) và phối hợp Xe hỗ trợ tốt từ phía sau
  - Các đòn chiến thuật: Chĩa đôi Tốt (Pawn Fork), Chĩa đôi Mã (Knight Fork), Đòn xiên (Skewer)
  - Kiểm soát trung tâm, cặp tốt trung tâm song hành và phạt tốt trung tâm bị chặn
  - Phạt xuất Hậu quá sớm ra giữa bàn cờ khi quân nhẹ chưa kịp phát triển
  - An toàn Vua: Nhập thành, nhận diện Xe bị kẹt ở góc (Chess960 Trapped Rook), lá chắn tốt, áp lực tấn công Vua phối hợp
  - Làm dịu các thế cờ hòa thiếu quân (Drawish material dampening)
  - Dồn Vua tàn cuộc mượt mà (Smooth Endgame Mop-Up)
  - Nội suy chuyển tiếp mượt mà giữa Trung cuộc và Tàn cuộc (Symmetric Tapered Game-Phase Interpolation)
"""

from __future__ import annotations

from functools import lru_cache

import chess

# Điểm số quy ước cho trạng thái Chiếu hết (Mate)
MATE_SCORE = 999_999

# Trọng số giai đoạn ván cờ theo từng loại quân (None, Pawn, Knight, Bishop, Rook, Queen, King)
# Mã=1, Tượng=1, Xe=2, Hậu=4. Tổng trọng số khởi đầu của cả 2 bên: 4*1 + 4*1 + 4*2 + 2*4 = 24
PHASE = (0, 0, 1, 1, 2, 4, 0)


def build_piece_table(t: list[int], v: int) -> tuple[int, ...]:
    """Chuyển đổi bảng PeSTO Piece-Square Table về hệ tọa độ chuẩn của quân Trắng (a1=0 .. h8=63),
    đồng thời cộng gộp giá trị cơ bản của quân cờ v vào từng ô để tăng tốc độ tính toán.
    """
    r = []
    for rank in reversed(range(8)):
        r.extend(t[rank * 8 : (rank + 1) * 8])
    return tuple(x + v for x in r)


# =============================================================================
# BẢNG GIÁ TRỊ VỊ TRÍ QUÂN CỜ (PESTO PIECE-SQUARE TABLES)
# Mỗi loại quân có 2 bảng:
#   - MG (Middlegame): Trọng số vị trí trong giai đoạn Khai cuộc / Trung cuộc
#   - EG (Endgame): Trọng số vị trí trong giai đoạn Tàn cuộc
# =============================================================================

# --- TỐT (PAWN) ---
MG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
EG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
       4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

# --- MÃ (KNIGHT) ---
MG_KNIGHT = [
   -167, -89, -34, -49,  61, -97, -15,-107,
    -73, -41,  72,  36,  23,  62,   7, -17,
    -47,  60,  37,  65,  84, 129,  73,  44,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -13,   4,  16,  13,  28,  19,  21,  -8,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
   -105, -21, -58, -33, -17, -28, -19, -23,
]
EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

# --- TƯỢNG (BISHOP) ---
MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
EG_BISHOP = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
]

# --- XE (ROOK) ---
MG_ROOK = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]
EG_ROOK = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]

# --- HẬU (QUEEN) ---
MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]
EG_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  18,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

# --- VUA (KING) ---
MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]
EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

# Bảng PST đã cộng sẵn giá trị chất quân cơ bản (Material Value + PST)
# Tốt: MG 82, EG 94 | Mã: MG 337, EG 281 | Tượng: MG 365, EG 297 | Xe: MG 477, EG 512 | Hậu: MG 1025, EG 936
MG_PAWN_T = build_piece_table(MG_PAWN, 82)
EG_PAWN_T = build_piece_table(EG_PAWN, 94)
MG_KNIGHT_T = build_piece_table(MG_KNIGHT, 337)
EG_KNIGHT_T = build_piece_table(EG_KNIGHT, 281)
MG_BISHOP_T = build_piece_table(MG_BISHOP, 365)
EG_BISHOP_T = build_piece_table(EG_BISHOP, 297)
MG_ROOK_T = build_piece_table(MG_ROOK, 477)
EG_ROOK_T = build_piece_table(EG_ROOK, 512)
MG_QUEEN_T = build_piece_table(MG_QUEEN, 1025)
EG_QUEEN_T = build_piece_table(EG_QUEEN, 936)
MG_KING_T = build_piece_table(MG_KING, 0)
EG_KING_T = build_piece_table(EG_KING, 0)

# =============================================================================
# MẶT NẠ BITBOARD & BẢNG TRA TRƯỚC HÌNH HỌC (PRECOMPUTED BITBOARDS & GEOMETRY)
# =============================================================================
BB_FILES = chess.BB_FILES
BB_LIGHT = chess.BB_LIGHT_SQUARES
BB_DARK = chess.BB_DARK_SQUARES
BB_RANK_1 = chess.BB_RANK_1
BB_RANK_2 = chess.BB_RANK_2
BB_RANK_3 = chess.BB_RANK_3
BB_RANK_6 = chess.BB_RANK_6
BB_RANK_7 = chess.BB_RANK_7
BB_RANK_8 = chess.BB_RANK_8
NOT_FILE_A = ~chess.BB_FILE_A & 0xFFFFFFFFFFFFFFFF
NOT_FILE_H = ~chess.BB_FILE_H & 0xFFFFFFFFFFFFFFFF

# Các ô trung tâm chính (d4, e4, d5, e5) và trung tâm mở rộng (c4, f4, c5, f5)
BB_CENTER_W = chess.BB_D4 | chess.BB_E4
BB_CENTER_B = chess.BB_D5 | chess.BB_E5
BB_EXTENDED_CENTER_W = chess.BB_C4 | chess.BB_F4
BB_EXTENDED_CENTER_B = chess.BB_C5 | chess.BB_F5
BB_CENTER_ALL = BB_CENTER_W | BB_CENTER_B

# Các hàng tiền đồn (Outpost Ranks): hàng 4, 5, 6 cho Trắng; hàng 5, 4, 3 cho Đen
OUTPOST_RANKS_W = chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6
OUTPOST_RANKS_B = chess.BB_RANK_5 | chess.BB_RANK_4 | chess.BB_RANK_3

# ADJACENT_FILES[f]: Mặt nạ 2 cột liền kề cột f (cột f-1 và f+1)
ADJACENT_FILES = tuple(
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
)

# KING_ZONE_FILES[f]: Mặt nạ 3 cột bao quanh Vua (f-1, f, f+1)
KING_ZONE_FILES = tuple(
    chess.BB_FILES[max(0, f - 1)] | chess.BB_FILES[f] | chess.BB_FILES[min(7, f + 1)]
    for f in range(8)
)

# Các bảng bitboard tra cứu theo chỉ số ô (0..63):
PASSED_SPAN_W = [0] * 64         # Vùng kiểm tra tốt thông Trắng (3 cột f-1, f, f+1 trên tất cả các hàng phía trước)
PASSED_SPAN_B = [0] * 64         # Vùng kiểm tra tốt thông Đen (3 cột f-1, f, f+1 trên tất cả các hàng phía trước)
FRONT_FILE_W = [0] * 64          # Các ô cùng cột phía trước hướng Trắng tiến
FRONT_FILE_B = [0] * 64          # Các ô cùng cột phía trước hướng Đen tiến
SUPPORT_SPAN_W = [0] * 64        # Vùng tốt đồng minh phía sau có thể dâng lên bảo vệ (Trắng)
SUPPORT_SPAN_B = [0] * 64        # Vùng tốt đồng minh phía sau có thể dâng lên bảo vệ (Đen)
OUTPOST_SPAN_W = [0] * 64        # Vùng tốt địch phía trước có thể dâng lên đuổi quân ở outpost (Trắng)
OUTPOST_SPAN_B = [0] * 64        # Vùng tốt địch phía trước có thể dâng lên đuổi quân ở outpost (Đen)
ADJACENT_PASSER_MASK = [0] * 64    # Mặt nạ các ô lân cận để nhận diện cặp tốt thông liên kết
KING_ATTACK_ZONE_W = [0] * 64    # Vùng xung quanh và phía trước Vua Trắng cần phòng thủ
KING_ATTACK_ZONE_B = [0] * 64    # Vùng xung quanh và phía trước Vua Đen cần phòng thủ
MOP_CENTER = [0] * 64            # Điểm khoảng cách từ ô tới trung tâm (phục vụ dồn Vua tàn cuộc)
CHEBYSHEV_DISTANCE = [[0] * 64 for _ in range(64)]  # Khoảng cách Chebyshev giữa 2 ô bất kỳ: max(|f1-f2|, |r1-r2|)

for _sq in range(64):
    _f, _r = _sq & 7, _sq >> 3
    # Điểm phạt khoảng cách xa trung tâm bàn cờ (tính từ tâm 3.5, 3.5)
    MOP_CENTER[_sq] = int(max(abs(_f - 3.5), abs(_r - 3.5)) * 16)
    for _af in range(max(0, _f - 1), min(8, _f + 2)):
        _fbb = chess.BB_FILES[_af]
        for _ar in range(_r + 1, 8):
            PASSED_SPAN_W[_sq] |= _fbb & chess.BB_RANKS[_ar]
        for _ar in range(0, _r):
            PASSED_SPAN_B[_sq] |= _fbb & chess.BB_RANKS[_ar]
    FRONT_FILE_W[_sq] = PASSED_SPAN_W[_sq] & chess.BB_FILES[_f]
    FRONT_FILE_B[_sq] = PASSED_SPAN_B[_sq] & chess.BB_FILES[_f]
    OUTPOST_SPAN_W[_sq] = PASSED_SPAN_W[_sq] & ADJACENT_FILES[_f]
    OUTPOST_SPAN_B[_sq] = PASSED_SPAN_B[_sq] & ADJACENT_FILES[_f]
    SUPPORT_SPAN_W[_sq] = ADJACENT_FILES[_f] & ~PASSED_SPAN_W[_sq]
    SUPPORT_SPAN_B[_sq] = ADJACENT_FILES[_f] & ~PASSED_SPAN_B[_sq]
    _adj_ranks = chess.BB_RANKS[_r]
    if _r > 0:
        _adj_ranks |= chess.BB_RANKS[_r - 1]
    if _r < 7:
        _adj_ranks |= chess.BB_RANKS[_r + 1]
    ADJACENT_PASSER_MASK[_sq] = ADJACENT_FILES[_f] & _adj_ranks
    _ka = chess.BB_KING_ATTACKS[_sq] | (1 << _sq)
    KING_ATTACK_ZONE_W[_sq] = _ka | ((_ka << 8) & 0xFFFFFFFFFFFFFFFF)
    KING_ATTACK_ZONE_B[_sq] = _ka | (_ka >> 8)
    for _sq2 in range(64):
        CHEBYSHEV_DISTANCE[_sq][_sq2] = max(abs(_f - (_sq2 & 7)), abs(_r - (_sq2 >> 3)))

# Điểm thưởng lũy tiến theo hàng cho tốt thông (Passed Pawns) theo hàng rank 0..7
PASSED_MG = (0, 8, 14, 24, 45, 75, 120, 0)
PASSED_EG = (0, 12, 22, 38, 68, 110, 165, 0)
PASSED_UNBLOCKED_EG = (0, 4, 8, 14, 24, 40, 60, 0)  # Thưởng thêm nếu ô phía trước tốt thông hoàn toàn không bị cản


def _precompute_bishop_rays() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Tính trước 4 tia đường chéo của Tượng từ mỗi ô (dùng để đánh giá Tầm nhìn Tượng - Bishop Vision)."""
    rays = [[] for _ in range(64)]
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        for df, dr in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
            ray = []
            cf, cr = f + df, r + dr
            while 0 <= cf < 8 and 0 <= cr < 8:
                ray.append(cr * 8 + cf)
                cf += df
                cr += dr
            rays[sq].append(tuple(ray))
    return tuple(tuple(r) for r in rays)


BISHOP_RAYS = _precompute_bishop_rays()

# BB_BETWEEN[s1][s2]: Mặt nạ các ô nằm giữa ô s1 và s2
BB_BETWEEN = tuple(
    tuple(chess.between(s1, s2) for s2 in range(64))
    for s1 in range(64)
)

# RAY_BEHIND[s][t]: Tia chiếu tiếp tục mở rộng phía sau ô t khi bị chiếu/tấn công từ ô s
RAY_BEHIND = [[0] * 64 for _ in range(64)]
for _s in range(64):
    for _t1 in range(64):
        if _s != _t1:
            _r = chess.ray(_s, _t1)
            if _r:
                RAY_BEHIND[_s][_t1] = _r & ~(1 << _s) & ~(1 << _t1) & ~BB_BETWEEN[_s][_t1]

RAY_BEHIND = tuple(tuple(_row) for _row in RAY_BEHIND)

# Hàm scan_forward của python-chess duyệt qua các sq_bit 1 theo thứ tự từ thấp đến cao (LSB)
scan_forward = chess.scan_forward

# Bộ nhớ đệm bảng băm ánh xạ trực tiếp (Direct-mapped cache) gồm 16,384 slots lưu kết quả hàm evaluate
EVAL_CACHE_MASK = 16383
EVAL_CACHE: list[tuple[tuple[int, ...], int, int] | None] = [None] * (EVAL_CACHE_MASK + 1)


# =============================================================================
# HÀM ĐÁNH GIÁ CẤU TRÚC TỐT (CACHED PAWN STRUCTURE EVALUATION)
# =============================================================================
@lru_cache(maxsize=32768)
def eval_pawns(w_pawns: int, b_pawns: int) -> tuple[int, int, int, int, int, int]:
    """Đánh giá toàn diện cấu trúc tốt (PST, tốt liên kết, tốt song hành, tốt cô lập,
    tốt chồng, tốt lạc hậu, tốt thông). Sử dụng LRU cache để tăng tốc tối đa.

    Trả về:
      - mg: Điểm trung cuộc cấu trúc tốt
      - eg: Điểm tàn cuộc cấu trúc tốt
      - w_pawn_atk: Mặt nạ các ô bị tốt Trắng tấn công
      - b_pawn_atk: Mặt nạ các ô bị tốt Đen tấn công
      - w_passed: Bitboard các tốt thông của Trắng
      - b_passed: Bitboard các tốt thông của Đen
    """
    # 1. Tính mặt nạ các ô bị tốt kiểm soát/tấn công
    # Trắng tấn công chéo lên (dịch trái 7 và dịch trái 9)
    w_pawn_atk = ((w_pawns & NOT_FILE_A) << 7) | ((w_pawns & NOT_FILE_H) << 9)
    # Đen tấn công chéo xuống (dịch phải 7 và dịch phải 9)
    b_pawn_atk = ((b_pawns & NOT_FILE_H) >> 7) | ((b_pawns & NOT_FILE_A) >> 9)

    # 2. Cặp tốt đứng song song ngang hàng (Phalanx): đứng cạnh nhau trên cùng một rank
    w_phalanx = ((w_pawns & NOT_FILE_H) << 1) | ((w_pawns & NOT_FILE_A) >> 1)
    b_phalanx = ((b_pawns & NOT_FILE_H) << 1) | ((b_pawns & NOT_FILE_A) >> 1)

    # Mặt nạ tốt liên kết: được tốt khác bảo vệ từ phía sau HOẶC đứng cạnh nhau thành hàng
    w_connected_mask = w_pawn_atk | w_phalanx
    b_connected_mask = b_pawn_atk | b_phalanx
    connected_diff = (w_pawns & w_connected_mask).bit_count() - (b_pawns & b_connected_mask).bit_count()
    mg = connected_diff * 9
    eg = connected_diff * 12

    w_passed = 0
    b_passed = 0

    # 3. Đánh giá từng tốt Trắng
    for sq in scan_forward(w_pawns):
        mg += MG_PAWN_T[sq]
        eg += EG_PAWN_T[sq]
        file = sq & 7
        sq_bit = 1 << sq

        # Tốt thông (Passed Pawn): không có tốt Đen cản đường ở cả 3 cột phía trước
        # và không có tốt Trắng chắn ngay phía trước cùng cột
        if not (b_pawns & PASSED_SPAN_W[sq]) and not (w_pawns & FRONT_FILE_W[sq]):
            w_passed |= sq_bit
            rank = sq >> 3
            mg += PASSED_MG[rank]
            eg += PASSED_EG[rank]
            # Thưởng thêm nếu tốt thông được liên kết với các tốt khác
            if sq_bit & w_connected_mask:
                mg += 6
                eg += 12

        # Tốt chồng (Doubled Pawn): có tốt đồng minh cùng cột đứng phía trước
        if w_pawns & FRONT_FILE_W[sq]:
            mg -= 12
            eg -= 20

        # Tốt cô lập (Isolated Pawn): không có bất kỳ tốt đồng minh nào ở 2 cột liền kề
        if not (w_pawns & ADJACENT_FILES[file]):
            mg -= 10
            eg -= 18
        # Tốt lạc hậu (Backward Pawn): không có tốt đồng minh phía sau có thể tiến lên bảo vệ,
        # trong khi ô ngay phía trước đã bị tốt đối phương kiểm soát hoặc chiếm giữ
        elif not (w_pawns & SUPPORT_SPAN_W[sq]) and ((b_pawn_atk | b_pawns) & (1 << (sq + 8))):
            mg -= 8
            eg -= 8

    # 4. Đánh giá từng tốt Đen (đối xứng, dùng sq ^ 56 để lật đối xứng bàn cờ)
    for sq in scan_forward(b_pawns):
        msq = sq ^ 56
        mg -= MG_PAWN_T[msq]
        eg -= EG_PAWN_T[msq]
        file = sq & 7
        sq_bit = 1 << sq

        # Tốt thông Đen
        if not (w_pawns & PASSED_SPAN_B[sq]) and not (b_pawns & FRONT_FILE_B[sq]):
            b_passed |= sq_bit
            rank = 7 - (sq >> 3)
            mg -= PASSED_MG[rank]
            eg -= PASSED_EG[rank]
            if sq_bit & b_connected_mask:
                mg -= 6
                eg -= 12

        # Tốt chồng Đen
        if b_pawns & FRONT_FILE_B[sq]:
            mg += 12
            eg += 20

        # Tốt cô lập Đen
        if not (b_pawns & ADJACENT_FILES[file]):
            mg += 10
            eg += 18
        # Tốt lạc hậu Đen
        elif not (b_pawns & SUPPORT_SPAN_B[sq]) and ((w_pawn_atk | w_pawns) & (1 << (sq - 8))):
            mg += 8
            eg += 8

    # 5. Cặp tốt thông liên kết cạnh nhau (Connected Passed Pawns)
    if w_passed & (w_passed - 1):
        for sq in scan_forward(w_passed):
            if w_passed & ADJACENT_PASSER_MASK[sq]:
                mg += 4
                eg += 9
    if b_passed & (b_passed - 1):
        for sq in scan_forward(b_passed):
            if b_passed & ADJACENT_PASSER_MASK[sq]:
                mg -= 4
                eg -= 9

    return mg, eg, w_pawn_atk, b_pawn_atk, w_passed, b_passed


# =============================================================================
# HÀM ĐÁNH GIÁ CHÍNH (EVALUATE FUNCTION)
# =============================================================================
def evaluate(board: chess.Board) -> int:
    """Trả về điểm đánh giá thế cờ theo đơn vị centipawn (cp), có dấu tương đối
    theo lượt đi của bên hiện tại (board.turn): dương là có lợi, âm là bất lợi.
    """
    occupied_w = board.occupied_co[chess.WHITE]
    occupied_b = board.occupied_co[chess.BLACK]
    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens
    kings = board.kings
    castling_rights = board.castling_rights

    # --- TRA CỨU BỘ NHỚ ĐỆM (EVAL CACHE PROBE) ---
    cache_key = (pawns, knights, bishops, rooks, queens, kings, occupied_w, occupied_b, castling_rights)
    cache_idx = hash(cache_key) & EVAL_CACHE_MASK
    cached = EVAL_CACHE[cache_idx]
    if cached is not None and cached[0] == cache_key:
        score, tempo = cached[1], cached[2]
        return (score + tempo) if board.turn == chess.WHITE else (-score + tempo)

    occupied = occupied_w | occupied_b

    # --- NHẬN DIỆN CỜ HÒA KHI KHÔNG CÒN TỐT (INSUFFICIENT MATERIAL DETECTION) ---
    if not pawns:
        total_pieces = occupied.bit_count()
        if not (rooks | queens):
            # Vua + Mã vs Vua hoặc Vua + Tượng vs Vua (<= 3 quân) -> Hòa tuyệt đối
            if total_pieces <= 3:
                return 0
            # Vua + 2 Mã vs Vua hoặc Vua + Mã vs Vua + Mã (4 quân không có Tượng) -> Hòa lý thuyết
            if total_pieces == 4 and not bishops:
                return 0

    w_pawns, b_pawns = pawns & occupied_w, pawns & occupied_b
    w_knights, b_knights = knights & occupied_w, knights & occupied_b
    w_bishops, b_bishops = bishops & occupied_w, bishops & occupied_b
    w_rooks, b_rooks = rooks & occupied_w, rooks & occupied_b
    w_queens, b_queens = queens & occupied_w, queens & occupied_b
    w_kings, b_kings = kings & occupied_w, kings & occupied_b

    # --- TÍNH TOÁN GIAI ĐOẠN VÁN CỜ (GAME PHASE) ---
    # Phase = 24: Khai cuộc đầy đủ quân | Phase = 0: Tàn cuộc chỉ còn Vua và Tốt
    phase = (
        w_knights.bit_count() + b_knights.bit_count()
        + w_bishops.bit_count() + b_bishops.bit_count()
        + 2 * (w_rooks.bit_count() + b_rooks.bit_count())
        + 4 * (w_queens.bit_count() + b_queens.bit_count())
    )

    # Lấy điểm cấu trúc tốt ban đầu từ hàm đệm eval_pawns
    mg, eg, w_pawn_atk, b_pawn_atk, w_passed, b_passed = eval_pawns(w_pawns, b_pawns)

    # --- ĐE DỌA CỦA TỐT LÊN CÁC QUÂN NẶNG/NHẸ ĐỐI PHƯƠNG (PAWN THREATS) ---
    # Tốt đuổi hoặc đe dọa ăn quân đắt hơn (Mã, Tượng, Xe, Hậu)
    threat_diff = (w_pawn_atk & (b_knights | b_bishops | b_rooks | b_queens)).bit_count() - (b_pawn_atk & (w_knights | w_bishops | w_rooks | w_queens)).bit_count()
    mg += threat_diff * 35
    eg += threat_diff * 25

    # Mặt nạ các ô an toàn để quân nhẹ/nặng di chuyển (không bị quân ta cản và không bị tốt địch kiểm soát)
    w_safe = ~(occupied_w | b_pawn_atk)
    b_safe = ~(occupied_b | w_pawn_atk)
    attacks_mask = board.attacks_mask

    # Vị trí Vua và Vùng an toàn quanh Vua (King Attack Zones)
    w_king = w_kings.bit_length() - 1 if w_kings else None
    b_king = b_kings.bit_length() - 1 if b_kings else None
    w_king_zone = KING_ATTACK_ZONE_W[w_king] if w_king is not None else 0
    b_king_zone = KING_ATTACK_ZONE_B[b_king] if b_king is not None else 0

    # Biến tích lũy sức ép tấn công Vua: số lượng quân tấn công (cnt) và tổng trọng số đe dọa (wt)
    w_atker_count = w_atk_weight = 0
    b_atker_count = b_atk_weight = 0

    # -------------------------------------------------------------------------
    # 1. ĐÁNH GIÁ TỐT THÔNG ĐỘNG (DYNAMIC PASSED PAWN EVALUATION)
    # -------------------------------------------------------------------------
    if w_passed:
        for sq in scan_forward(w_passed):
            rank = sq >> 3
            push_sq = sq + 8
            push_bit = 1 << push_sq
            # Ô phía trước trống: thưởng điểm tốt thông không bị cản
            if not (occupied & push_bit):
                eg += PASSED_UNBLOCKED_EG[rank]
                # Nếu ô phía trước cũng không bị tốt đối phương kiểm soát
                if not (b_pawn_atk & push_bit):
                    eg += rank * 3
            # Vua hộ tống tốt thông (King Escort): So sánh khoảng cách Vua Đen và Vua Trắng tới ô tiến
            if rank >= 3 and w_king is not None and b_king is not None:
                eg += (CHEBYSHEV_DISTANCE[b_king][push_sq] * 5 - CHEBYSHEV_DISTANCE[w_king][push_sq] * 2) * (rank - 2)
            # Quy tắc Tarrasch (Tarrasch Rule): Xe đứng phía sau tốt thông cùng cột
            behind_occupied = occupied & FRONT_FILE_B[sq]
            if behind_occupied:
                nearest_bit = 1 << (behind_occupied.bit_length() - 1)
                if nearest_bit & w_rooks:
                    mg += 10
                    eg += 20  # Xe ta phía sau hỗ trợ tốt dâng lên
                elif nearest_bit & b_rooks:
                    mg -= 10
                    eg -= 20  # Xe địch phía sau kiềm chế tốt thông

    if b_passed:
        for sq in scan_forward(b_passed):
            rank = 7 - (sq >> 3)
            push_sq = sq - 8
            push_bit = 1 << push_sq
            if not (occupied & push_bit):
                eg -= PASSED_UNBLOCKED_EG[rank]
                if not (w_pawn_atk & push_bit):
                    eg -= rank * 3
            if rank >= 3 and w_king is not None and b_king is not None:
                eg -= (CHEBYSHEV_DISTANCE[w_king][push_sq] * 5 - CHEBYSHEV_DISTANCE[b_king][push_sq] * 2) * (rank - 2)
            behind_occupied = occupied & FRONT_FILE_W[sq]
            if behind_occupied:
                nearest_bit = behind_occupied & -behind_occupied
                if nearest_bit & b_rooks:
                    mg -= 10
                    eg -= 20
                elif nearest_bit & w_rooks:
                    mg += 10
                    eg += 20

    # -------------------------------------------------------------------------
    # 2. ĐÁNH GIÁ QUÂN MÃ (KNIGHTS)
    # -------------------------------------------------------------------------
    w_outpost_sqs = OUTPOST_RANKS_W & w_pawn_atk
    b_outpost_sqs = OUTPOST_RANKS_B & b_pawn_atk

    for sq in scan_forward(w_knights):
        mg += MG_KNIGHT_T[sq]
        eg += EG_KNIGHT_T[sq]
        atk = attacks_mask(sq)
        # Khả năng cơ động an toàn (Safe Mobility): số ô an toàn trừ mốc cơ sở 4
        safe_mobility = (atk & w_safe).bit_count()
        mobility = safe_mobility - 4
        mg += mobility * 4
        eg += mobility * 4
        sq_bit = 1 << sq
        # Căn cứ điểm (Outpost): Mã ở hàng 4, 5, 6 được tốt bảo vệ và không bị tốt địch đuổi
        if (w_outpost_sqs & sq_bit) and not (b_pawns & OUTPOST_SPAN_W[sq]):
            mg += 24
            eg += 16
        # Cảnh báo quân bị bẫy/đe dọa: Mã bị tốt địch tấn công mà ít ô chạy an toàn
        if (sq_bit & b_pawn_atk) and safe_mobility <= 1:
            penalty = 20 if safe_mobility == 0 else 10
            mg -= penalty
            eg -= penalty
        # Tấn công vào vùng an toàn của Vua đối phương
        king_zone_hits = atk & b_king_zone
        if king_zone_hits:
            w_atker_count += 1
            w_atk_weight += 2 * king_zone_hits.bit_count()

    for sq in scan_forward(b_knights):
        msq = sq ^ 56
        mg -= MG_KNIGHT_T[msq]
        eg -= EG_KNIGHT_T[msq]
        atk = attacks_mask(sq)
        safe_mobility = (atk & b_safe).bit_count()
        mobility = safe_mobility - 4
        mg -= mobility * 4
        eg -= mobility * 4
        sq_bit = 1 << sq
        if (b_outpost_sqs & sq_bit) and not (w_pawns & OUTPOST_SPAN_B[sq]):
            mg -= 24
            eg -= 16
        if (sq_bit & w_pawn_atk) and safe_mobility <= 1:
            penalty = 20 if safe_mobility == 0 else 10
            mg += penalty
            eg += penalty
        king_zone_hits = atk & w_king_zone
        if king_zone_hits:
            b_atker_count += 1
            b_atk_weight += 2 * king_zone_hits.bit_count()

    # -------------------------------------------------------------------------
    # 3. ĐÁNH GIÁ QUÂN TƯỢNG (BISHOPS)
    # -------------------------------------------------------------------------
    # Thưởng sở hữu Cặp Tượng (Bishop Pair: 1 Tượng ô trắng + 1 Tượng ô đen)
    if (w_bishops & BB_LIGHT) and (w_bishops & BB_DARK):
        mg += 30
        eg += 55
    if (b_bishops & BB_LIGHT) and (b_bishops & BB_DARK):
        mg -= 30
        eg -= 55

    # Tốt bị khóa đối đầu (Blocked Pawns): tốt hai bên đứng đối mặt trực tiếp
    w_blocked_pawns = w_pawns & (b_pawns >> 8)
    b_blocked_pawns = b_pawns & (w_pawns << 8)

    for sq in scan_forward(w_bishops):
        mg += MG_BISHOP_T[sq]
        eg += EG_BISHOP_T[sq]
        atk = attacks_mask(sq)
        # Khả năng cơ động an toàn của Tượng (mốc cơ sở 6)
        safe_mobility = (atk & w_safe).bit_count()
        mobility = safe_mobility - 6
        mg += mobility * 4
        eg += mobility * 4
        sq_bit = 1 << sq
        # Căn cứ điểm Tượng (Outpost)
        if (w_outpost_sqs & sq_bit) and not (b_pawns & OUTPOST_SPAN_W[sq]):
            mg += 14
            eg += 10
        # Tượng xấu (Bad Bishop): Tượng bị cản trở bởi nhiều tốt đồng minh bị khóa cùng màu ô
        sq_color = BB_LIGHT if (sq_bit & BB_LIGHT) else BB_DARK
        locked = (w_blocked_pawns & sq_color).bit_count()
        if locked:
            mg -= locked * 3
            eg -= locked * 4
        # Tượng bị bẫy/đe dọa bởi tốt
        if (sq_bit & b_pawn_atk) and safe_mobility <= 1:
            penalty = 28 if safe_mobility == 0 else 14
            mg -= penalty
            eg -= penalty
        # Tấn công vùng Vua đối phương
        king_zone_hits = atk & b_king_zone
        if king_zone_hits:
            w_atker_count += 1
            w_atk_weight += 2 * king_zone_hits.bit_count()

        # Tầm nhìn Tượng (Bishop Vision): kiểm soát các đường chéo trống và ngắm quân nặng
        w_bishop_vision = 0
        for ray in BISHOP_RAYS[sq]:
            for target_sq in ray:
                target_bit = 1 << target_sq
                if not (occupied & target_bit):
                    w_bishop_vision += 1
                    continue
                if target_bit & occupied_w:
                    break
                if target_bit & b_pawns:
                    break
                w_bishop_vision += 2
                break
        mg += w_bishop_vision
        eg += w_bishop_vision // 2

    for sq in scan_forward(b_bishops):
        msq = sq ^ 56
        mg -= MG_BISHOP_T[msq]
        eg -= EG_BISHOP_T[msq]
        atk = attacks_mask(sq)
        safe_mobility = (atk & b_safe).bit_count()
        mobility = safe_mobility - 6
        mg -= mobility * 4
        eg -= mobility * 4
        sq_bit = 1 << sq
        if (b_outpost_sqs & sq_bit) and not (w_pawns & OUTPOST_SPAN_B[sq]):
            mg -= 14
            eg -= 10
        sq_color = BB_LIGHT if (sq_bit & BB_LIGHT) else BB_DARK
        locked = (b_blocked_pawns & sq_color).bit_count()
        if locked:
            mg += locked * 3
            eg += locked * 4
        if (sq_bit & w_pawn_atk) and safe_mobility <= 1:
            penalty = 28 if safe_mobility == 0 else 14
            mg += penalty
            eg += penalty
        king_zone_hits = atk & w_king_zone
        if king_zone_hits:
            b_atker_count += 1
            b_atk_weight += 2 * king_zone_hits.bit_count()

        b_bishop_vision = 0
        for ray in BISHOP_RAYS[sq]:
            for target_sq in ray:
                target_bit = 1 << target_sq
                if not (occupied & target_bit):
                    b_bishop_vision += 1
                    continue
                if target_bit & occupied_b:
                    break
                if target_bit & w_pawns:
                    break
                b_bishop_vision += 2
                break
        mg -= b_bishop_vision
        eg -= b_bishop_vision // 2

    # -------------------------------------------------------------------------
    # 4. ĐÁNH GIÁ QUÂN XE (ROOKS)
    # -------------------------------------------------------------------------
    w_rook_files = 0
    for sq in scan_forward(w_rooks):
        mg += MG_ROOK_T[sq]
        eg += EG_ROOK_T[sq]
        atk = attacks_mask(sq)
        # Khả năng cơ động an toàn của Xe (mốc cơ sở 7)
        safe_mobility = (atk & w_safe).bit_count()
        mobility = safe_mobility - 7
        mg += mobility * 2
        eg += mobility * 3
        # Cặp Xe liên kết bảo vệ lẫn nhau (Connected Rooks)
        if atk & w_rooks:
            mg += 5
            eg += 6
        # Cột mở (Open file) và Cột bán mở (Semi-open file)
        file = sq & 7
        file_bb = BB_FILES[file]
        if not (w_pawns & file_bb):
            if not (b_pawns & file_bb):
                mg += 22  # Cột mở hoàn toàn (không có tốt cả hai bên)
                eg += 14
            else:
                mg += 11  # Cột bán mở (chỉ có tốt đối phương)
                eg += 8
        # Chồng 2 Xe trên cùng một cột (Doubled Rooks on file)
        file_bit = 1 << file
        if w_rook_files & file_bit:
            mg += 14
            eg += 18
        w_rook_files |= file_bit
        # Xe bị tốt đe dọa
        sq_bit = 1 << sq
        if (sq_bit & b_pawn_atk) and safe_mobility <= 1:
            penalty = 24 if safe_mobility == 0 else 12
            mg -= penalty
            eg -= penalty
        # Tấn công vùng Vua
        king_zone_hits = atk & b_king_zone
        if king_zone_hits:
            w_atker_count += 1
            w_atk_weight += 3 * king_zone_hits.bit_count()

        # Xe đứng phía sau hỗ trợ tốt đồng minh dâng cao
        front_occupied = occupied & FRONT_FILE_W[sq]
        if front_occupied:
            nearest_bit = front_occupied & -front_occupied
            if nearest_bit & w_pawns:
                pawn_sq = nearest_bit.bit_length() - 1
                pawn_rank = pawn_sq >> 3
                pawn_file = pawn_sq & 7
                if pawn_rank >= 3 or (pawn_rank >= 2 and pawn_file in (2, 3, 4, 5)):
                    advance_bonus = 8 + (pawn_rank - 2) * 8
                    if pawn_file in (3, 4):  # Thưởng thêm trên các cột trung tâm d, e
                        advance_bonus += 12
                    mg += advance_bonus
                    eg += advance_bonus + 6

    b_rook_files = 0
    for sq in scan_forward(b_rooks):
        msq = sq ^ 56
        mg -= MG_ROOK_T[msq]
        eg -= EG_ROOK_T[msq]
        atk = attacks_mask(sq)
        safe_mobility = (atk & b_safe).bit_count()
        mobility = safe_mobility - 7
        mg -= mobility * 2
        eg -= mobility * 3
        if atk & b_rooks:
            mg -= 5
            eg -= 6
        file = sq & 7
        file_bb = BB_FILES[file]
        if not (b_pawns & file_bb):
            if not (w_pawns & file_bb):
                mg -= 22
                eg -= 14
            else:
                mg -= 11
                eg -= 8
        file_bit = 1 << file
        if b_rook_files & file_bit:
            mg -= 14
            eg -= 18
        b_rook_files |= file_bit
        sq_bit = 1 << sq
        if (sq_bit & w_pawn_atk) and safe_mobility <= 1:
            penalty = 24 if safe_mobility == 0 else 12
            mg += penalty
            eg += penalty
        king_zone_hits = atk & w_king_zone
        if king_zone_hits:
            b_atker_count += 1
            b_atk_weight += 3 * king_zone_hits.bit_count()

        front_occupied = occupied & FRONT_FILE_B[sq]
        if front_occupied:
            nearest_bit = 1 << (front_occupied.bit_length() - 1)
            if nearest_bit & b_pawns:
                pawn_sq = nearest_bit.bit_length() - 1
                pawn_rank = 7 - (pawn_sq >> 3)
                pawn_file = pawn_sq & 7
                if pawn_rank >= 3 or (pawn_rank >= 2 and pawn_file in (2, 3, 4, 5)):
                    advance_bonus = 8 + (pawn_rank - 2) * 8
                    if pawn_file in (3, 4):
                        advance_bonus += 12
                    mg -= advance_bonus
                    eg -= advance_bonus + 6

    # Xe xâm nhập hàng 7 (hoặc hàng 2 của Đen): đè bẹp tốt đối phương và áp sát Vua
    if (b_pawns & BB_RANK_7) or (b_kings & BB_RANK_8) or b_pawns.bit_count() >= 3:
        w_rook_rank_7 = (w_rooks & BB_RANK_7).bit_count()
        if w_rook_rank_7 == 1:
            mg += 20
            eg += 35
        elif w_rook_rank_7 >= 2:  # Cặp Xe hàng 7 ("Blind pigs") cực kỳ nguy hiểm
            mg += 45
            eg += 75

    if (w_pawns & BB_RANK_2) or (w_kings & BB_RANK_1) or w_pawns.bit_count() >= 3:
        b_rook_rank_2 = (b_rooks & BB_RANK_2).bit_count()
        if b_rook_rank_2 == 1:
            mg -= 20
            eg -= 35
        elif b_rook_rank_2 >= 2:
            mg -= 45
            eg -= 75

    # -------------------------------------------------------------------------
    # 5. ĐÁNH GIÁ QUÂN HẬU (QUEENS)
    # -------------------------------------------------------------------------
    for sq in scan_forward(w_queens):
        mg += MG_QUEEN_T[sq]
        eg += EG_QUEEN_T[sq]
        atk = attacks_mask(sq)
        safe_mobility = (atk & w_safe).bit_count()
        mobility = safe_mobility - 12  # Mốc cơ sở cơ động của Hậu là 12
        # Giới hạn cơ động nếu khai cuộc mà quân nhẹ chưa phát triển
        if phase >= 18 and ((w_knights | w_bishops) & BB_RANK_1).bit_count() >= 2:
            mobility = min(0, mobility)
        mg += mobility * 1
        eg += mobility * 2
        # Phạt nặng nếu Hậu bị tốt đối phương đe dọa
        sq_bit = 1 << sq
        if sq_bit & b_pawn_atk:
            penalty = 70 if safe_mobility <= 1 else 35
            mg -= penalty
            eg -= penalty // 2
        # Phạt nếu Hậu bị quân nhẹ đối phương đe dọa trong khai cuộc
        elif phase >= 12 and (board.attackers_mask(chess.BLACK, sq) & (b_knights | b_bishops)):
            mg -= 30
            eg -= 15
        king_zone_hits = atk & b_king_zone
        if king_zone_hits:
            w_atker_count += 1
            w_atk_weight += 5 * king_zone_hits.bit_count()

    for sq in scan_forward(b_queens):
        msq = sq ^ 56
        mg -= MG_QUEEN_T[msq]
        eg -= EG_QUEEN_T[msq]
        atk = attacks_mask(sq)
        safe_mobility = (atk & b_safe).bit_count()
        mobility = safe_mobility - 12
        if phase >= 18 and ((b_knights | b_bishops) & BB_RANK_8).bit_count() >= 2:
            mobility = min(0, mobility)
        mg -= mobility * 1
        eg -= mobility * 2
        sq_bit = 1 << sq
        if sq_bit & w_pawn_atk:
            penalty = 70 if safe_mobility <= 1 else 35
            mg += penalty
            eg += penalty // 2
        elif phase >= 12 and (board.attackers_mask(chess.WHITE, sq) & (w_knights | w_bishops)):
            mg += 30
            eg += 15
        king_zone_hits = atk & w_king_zone
        if king_zone_hits:
            b_atker_count += 1
            b_atk_weight += 5 * king_zone_hits.bit_count()

    # -------------------------------------------------------------------------
    # 5.5 ĐÒN CHIẾN THUẬT: CHĨA ĐÔI & ĐÒN XIÊN (TACTICAL MOTIFS)
    # -------------------------------------------------------------------------
    # Tốt chĩa đôi (Pawn Forks)
    w_pawn_targets = b_knights | b_bishops | b_rooks | b_queens
    if w_pawns and w_pawn_targets:
        for sq in scan_forward(w_pawns):
            hits = chess.BB_PAWN_ATTACKS[chess.WHITE][sq] & w_pawn_targets
            if hits & (hits - 1):  # Đánh trúng >= 2 quân
                mg += 70
                eg += 60

    b_pawn_targets = w_knights | w_bishops | w_rooks | w_queens
    if b_pawns and b_pawn_targets:
        for sq in scan_forward(b_pawns):
            hits = chess.BB_PAWN_ATTACKS[chess.BLACK][sq] & b_pawn_targets
            if hits & (hits - 1):
                mg -= 70
                eg -= 60

    # Mã chĩa đôi (Knight Forks)
    w_fork_targets = w_pawn_targets | b_kings
    if w_knights and w_fork_targets:
        for sq in scan_forward(w_knights):
            hits = chess.BB_KNIGHT_ATTACKS[sq] & w_fork_targets
            if hits & (hits - 1):
                bonus = 75 if (hits & (b_kings | b_queens)) else 45
                mg += bonus
                eg += bonus - 10

    b_fork_targets = b_pawn_targets | w_kings
    if b_knights and b_fork_targets:
        for sq in scan_forward(b_knights):
            hits = chess.BB_KNIGHT_ATTACKS[sq] & b_fork_targets
            if hits & (hits - 1):
                bonus = 75 if (hits & (w_kings | w_queens)) else 45
                mg -= bonus
                eg -= bonus - 10

    # Đòn xiên (Skewers): Quân tấn công chiếu Vua hoặc đe dọa quân lớn, quân phía sau bị xiên
    w_sliders = w_bishops | w_rooks | w_queens
    b_targets = b_kings | b_queens | b_rooks
    b_pieces = occupied_b & ~b_pawns
    if w_sliders and b_targets:
        for target_1 in scan_forward(b_targets):
            is_king = bool((1 << target_1) & b_kings)
            attackers = board.attackers_mask(chess.WHITE, target_1) & w_sliders
            if attackers:
                for s in scan_forward(attackers):
                    behind = RAY_BEHIND[s][target_1] & b_pieces
                    if behind:
                        for target_2 in scan_forward(behind):
                            if not (occupied & BB_BETWEEN[target_1][target_2]):
                                bonus = 60 if is_king else 35
                                mg += bonus
                                eg += bonus
                                break

    b_sliders = b_bishops | b_rooks | b_queens
    w_targets = w_kings | w_queens | w_rooks
    w_pieces = occupied_w & ~w_pawns
    if b_sliders and w_targets:
        for target_1 in scan_forward(w_targets):
            is_king = bool((1 << target_1) & w_kings)
            attackers = board.attackers_mask(chess.BLACK, target_1) & b_sliders
            if attackers:
                for s in scan_forward(attackers):
                    behind = RAY_BEHIND[s][target_1] & w_pieces
                    if behind:
                        for target_2 in scan_forward(behind):
                            if not (occupied & BB_BETWEEN[target_1][target_2]):
                                bonus = 60 if is_king else 35
                                mg -= bonus
                                eg -= bonus
                                break

    # -------------------------------------------------------------------------
    # CHIẾM GIỮ TRUNG TÂM & PHẠT TỐT BỊ CẢN TRỞ (CENTER CONTROL)
    # -------------------------------------------------------------------------
    if phase >= 14:
        w_center_score = 0
        for sq in scan_forward(w_pawns & BB_CENTER_W):
            sq_bit = 1 << sq
            if sq_bit & b_pawn_atk:
                w_center_score += 16 if (sq_bit & w_pawn_atk) else -15
            else:
                w_center_score += 28

        b_center_score = 0
        for sq in scan_forward(b_pawns & BB_CENTER_B):
            sq_bit = 1 << sq
            if sq_bit & w_pawn_atk:
                b_center_score += 16 if (sq_bit & b_pawn_atk) else -15
            else:
                b_center_score += 28

        mg += (w_center_score - b_center_score)

        # Cặp tốt trung tâm song hành (Center Pawn Duo d4 + e4 / d5 + e5)
        if (w_pawns & BB_CENTER_W) == BB_CENTER_W:
            mg += 16
        if (b_pawns & BB_CENTER_B) == BB_CENTER_B:
            mg -= 16

        # Kiểm soát các ô trung tâm mở rộng (c4, f4, c5, f5)
        w_extended_center = (w_pawns & BB_EXTENDED_CENTER_W).bit_count()
        b_extended_center = (b_pawns & BB_EXTENDED_CENTER_B).bit_count()
        mg += (w_extended_center - b_extended_center) * 8

        # Kiểm soát hỏa lực chéo của tốt vào trung tâm
        w_center_atk = (w_pawn_atk & BB_CENTER_ALL).bit_count()
        b_center_atk = (b_pawn_atk & BB_CENTER_ALL).bit_count()
        mg += (w_center_atk - b_center_atk) * 6

        # Phạt tốt trung tâm bị chặn đứng ở d2/e2 (làm kẹt Tượng/Hậu phía sau)
        if (w_pawns & chess.BB_D2) and (occupied & chess.BB_D3):
            mg -= 25
        if (w_pawns & chess.BB_E2) and (occupied & chess.BB_E3):
            mg -= 25
        if (b_pawns & chess.BB_D7) and (occupied & chess.BB_D6):
            mg += 25
        if (b_pawns & chess.BB_E7) and (occupied & chess.BB_E6):
            mg += 25

    # -------------------------------------------------------------------------
    # PHẠT XUẤT HẬU SỚM RA GIỮA BÀN CỜ (PREMATURE QUEEN DEVELOPMENT)
    # -------------------------------------------------------------------------
    if phase >= 14:
        if w_queens:
            w_undev = ((w_knights | w_bishops) & BB_RANK_1).bit_count()
            # Hậu rời hàng 1 khi quân nhẹ chưa phát triển
            if not (w_queens & BB_RANK_1):
                if w_undev >= 3:
                    mg -= 60
                elif w_undev == 2:
                    mg -= 40
                elif w_undev == 1:
                    mg -= 20
            # Phạt Hậu ra hẳn giữa bàn cờ (ô trung tâm và ngoại vi)
            w_q_center = (w_queens & BB_CENTER_ALL).bit_count()
            if w_q_center:
                mg -= 50 * w_q_center
            w_q_extended = (w_queens & (BB_EXTENDED_CENTER_W | BB_EXTENDED_CENTER_B)).bit_count()
            if w_q_extended:
                mg -= 30 * w_q_extended
            # Phạt Hậu lang thang ở các hàng 4, 5, 6 khi quân nhà chưa phát triển
            if w_undev >= 2 and (w_queens & (chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6)):
                mg -= 25

        if b_queens:
            b_undev = ((b_knights | b_bishops) & BB_RANK_8).bit_count()
            if not (b_queens & BB_RANK_8):
                if b_undev >= 3:
                    mg += 60
                elif b_undev == 2:
                    mg += 40
                elif b_undev == 1:
                    mg += 20
            b_q_center = (b_queens & BB_CENTER_ALL).bit_count()
            if b_q_center:
                mg += 50 * b_q_center
            b_q_extended = (b_queens & (BB_EXTENDED_CENTER_W | BB_EXTENDED_CENTER_B)).bit_count()
            if b_q_extended:
                mg += 30 * b_q_extended
            if b_undev >= 2 and (b_queens & (chess.BB_RANK_5 | chess.BB_RANK_4 | chess.BB_RANK_3)):
                mg += 25

    # -------------------------------------------------------------------------
    # 5.1 TỐT TIẾN CAO GÂY ÁP BỨC (CRAMP PAWNS) & ĐÒN BẨY TỐT (PAWN LEVERS)
    # -------------------------------------------------------------------------
    w_advanced_p = w_pawns & (chess.BB_RANK_5 | chess.BB_RANK_6)
    if w_advanced_p:
        for sq in scan_forward(w_advanced_p):
            rank = sq >> 3
            file = sq & 7
            if not (b_pawns & (1 << (sq + 8))):
                cramp = 14 if rank == 4 else 28
                if b_king is not None and abs(file - (b_king & 7)) <= 2:
                    cramp += 22 if rank == 4 else 35
                mg += cramp
                eg += cramp // 2
            else:
                mg -= 10

        # Phạt nếu tốt nhô cao bị tốt đối phương dùng đòn bẩy phản công
        lever_hits = b_pawn_atk & w_advanced_p
        if lever_hits:
            for sq in scan_forward(lever_hits):
                file = sq & 7
                bonus = 20
                if b_king is not None and abs(file - (b_king & 7)) <= 2:
                    bonus += 15
                mg -= bonus
                eg -= bonus // 2

    b_advanced_p = b_pawns & (chess.BB_RANK_4 | chess.BB_RANK_3)
    if b_advanced_p:
        for sq in scan_forward(b_advanced_p):
            rank = 7 - (sq >> 3)
            file = sq & 7
            if not (w_pawns & (1 << (sq - 8))):
                cramp = 14 if rank == 4 else 28
                if w_king is not None and abs(file - (w_king & 7)) <= 2:
                    cramp += 22 if rank == 4 else 35
                mg -= cramp
                eg -= cramp // 2
            else:
                mg += 10

        lever_hits = w_pawn_atk & b_advanced_p
        if lever_hits:
            for sq in scan_forward(lever_hits):
                file = sq & 7
                bonus = 20
                if w_king is not None and abs(file - (w_king & 7)) <= 2:
                    bonus += 15
                mg += bonus
                eg += bonus // 2

    # -------------------------------------------------------------------------
    # 6. AN TOÀN VUA, NHẬP THÀNH & XE BỊ KẸT (CHESS960 TRAPPED ROOK)
    # -------------------------------------------------------------------------
    # Phát triển quân nhẹ (Development Lead): đếm số quân nhẹ đã rời hàng xuất phát
    dev_diff = ((w_knights | w_bishops) & ~BB_RANK_1).bit_count() - (
        (b_knights | b_bishops) & ~BB_RANK_8
    ).bit_count()
    mg += dev_diff * 12

    # Đánh giá trạng thái Nhập thành (Castling Status)
    w_castled = False
    if w_king is not None:
        w_king_file, w_king_rank = w_king & 7, w_king >> 3
        if w_king_rank == 0 and w_king_file in (1, 2, 6):
            if not (castling_rights & BB_RANK_1):  # Đã hoàn thành nhập thành (mất quyền nhập thành và Vua ở ô nhập thành)
                w_castled = True
                mg += 30
    if not w_castled:
        if castling_rights & BB_RANK_1:
            mg += 12  # Vẫn còn quyền nhập thành
        elif phase >= 18:
            mg -= 25  # Mất quyền nhập thành khi chưa nhập thành

    b_castled = False
    if b_king is not None:
        b_king_file, b_king_rank = b_king & 7, b_king >> 3
        if b_king_rank == 7 and b_king_file in (1, 2, 6):
            if not (castling_rights & BB_RANK_8):
                b_castled = True
                mg -= 30
    if not b_castled:
        if castling_rights & BB_RANK_8:
            mg -= 12
        elif phase >= 18:
            mg += 25

    if w_king is not None:
        mg += MG_KING_T[w_king]
        eg += EG_KING_T[w_king]
        w_king_file, w_king_rank = w_king & 7, w_king >> 3

        # Nhận diện Xe bị Vua kẹt ở góc trong Chess960 (Trapped Rook)
        if w_king_rank == 0:
            if w_king_file in (1, 2) and (w_rooks & ((1 << w_king) - 1) & ~castling_rights):
                mg -= 40
            elif w_king_file in (5, 6) and (w_rooks & (0xFF ^ ((1 << (w_king + 1)) - 1)) & ~castling_rights):
                mg -= 40

        # Lá chắn tốt bảo vệ Vua (Pawn Shield)
        king_zone = KING_ZONE_FILES[w_king_file]
        if w_king_rank <= 1 and (w_king_file <= 2 or w_king_file >= 5):
            mg += (w_pawns & king_zone & BB_RANK_2).bit_count() * 11
            mg += (w_pawns & king_zone & BB_RANK_3).bit_count() * 5

        # Phạt các cột mở hoặc bán mở trước mặt Vua khi đối phương còn Hậu hoặc nhiều Xe
        if b_queens or (b_rooks & (b_rooks - 1)):
            for file in range(max(0, w_king_file - 1), min(8, w_king_file + 2)):
                file_bb = BB_FILES[file]
                if not (w_pawns & file_bb):
                    mg -= 22 if not (b_pawns & file_bb) else 12

        # Sức ép tấn công Vua phối hợp (Coordinated King Attack)
        if b_atker_count >= 2:
            danger = min(180, b_atk_weight * min(4, b_atker_count) * 2)
            if not b_queens:
                danger //= 2
            mg -= danger

    if b_king is not None:
        mbk = b_king ^ 56
        mg -= MG_KING_T[mbk]
        eg -= EG_KING_T[mbk]
        b_king_file, b_king_rank = b_king & 7, b_king >> 3

        if b_king_rank == 7:
            if b_king_file in (1, 2) and (b_rooks & (((1 << (b_king - 56)) - 1) << 56) & ~castling_rights):
                mg += 40
            elif b_king_file in (5, 6) and (b_rooks & ((0xFF ^ ((1 << (b_king - 55)) - 1)) << 56) & ~castling_rights):
                mg += 40

        king_zone = KING_ZONE_FILES[b_king_file]
        if b_king_rank >= 6 and (b_king_file <= 2 or b_king_file >= 5):
            mg -= (b_pawns & king_zone & BB_RANK_7).bit_count() * 11
            mg -= (b_pawns & king_zone & BB_RANK_6).bit_count() * 5

        if w_queens or (w_rooks & (w_rooks - 1)):
            for file in range(max(0, b_king_file - 1), min(8, b_king_file + 2)):
                file_bb = BB_FILES[file]
                if not (b_pawns & file_bb):
                    mg += 22 if not (w_pawns & file_bb) else 12

        if w_atker_count >= 2:
            danger = min(180, w_atk_weight * min(4, w_atker_count) * 2)
            if not w_queens:
                danger //= 2
            mg += danger

    # --- LÀM DỊU ĐIỂM SỐ Ở CÁC THẾ CỜ HÒA THIẾU QUÂN (DRAWISH MATERIAL DAMPENING) ---
    if not pawns and occupied.bit_count() == 4:
        if not (rooks | queens) and (w_knights | w_bishops) and (b_knights | b_bishops):
            # 1 quân nhẹ vs 1 quân nhẹ (hoặc 2 Tượng cùng màu ô)
            if (
                ((w_bishops & BB_LIGHT) and (b_bishops & BB_LIGHT))
                or ((w_bishops & BB_DARK) and (b_bishops & BB_DARK))
            ):
                return 0
            mg //= 8
            eg //= 8
        elif not queens and (
            (w_rooks and (b_knights | b_bishops) and not b_rooks)
            or (b_rooks and (w_knights | w_bishops) and not w_rooks)
        ):
            eg //= 2

    # -------------------------------------------------------------------------
    # 7. DỒN VUA TÀN CUỘC MƯỢT MÀ (SMOOTH ENDGAME MOP-UP)
    # -------------------------------------------------------------------------
    # Áp dụng khi ván cờ bước vào tàn cuộc sâu (phase < 10) và có một bên chiếm ưu thế rõ rệt (|eg| > 200)
    if phase < 10 and w_king is not None and b_king is not None:
        mop_weight = 10 - phase
        if eg > 200:
            # Đẩy Vua địch ra góc/biên và kéo Vua ta lại gần: mop = MOP_CENTER[b_king] + (14 - dist)*4
            mop = MOP_CENTER[b_king] + (14 - CHEBYSHEV_DISTANCE[w_king][b_king]) * 4
            eg += (mop * mop_weight) // 10
        elif eg < -200:
            mop = MOP_CENTER[w_king] + (14 - CHEBYSHEV_DISTANCE[w_king][b_king]) * 4
            eg -= (mop * mop_weight) // 10

    # -------------------------------------------------------------------------
    # 8. NỘI SUY CHUYỂN TIẾP GIAI ĐOẠN VÁN CỜ (TAPERED PHASE INTERPOLATION)
    # -------------------------------------------------------------------------
    if phase > 24:
        phase = 24
    # Nội suy tuyến tính theo phase giữa điểm trung cuộc (mg) và điểm tàn cuộc (eg)
    score = int((mg * phase + eg * (24 - phase)) / 24)
    # Điểm cộng ưu thế đi trước (Tempo bonus), giảm dần khi tiến về tàn cuộc
    tempo = (10 * phase) // 24

    # Lưu kết quả vào bộ nhớ đệm
    EVAL_CACHE[cache_idx] = (cache_key, score, tempo)
    return (score + tempo) if board.turn == chess.WHITE else (-score + tempo)
