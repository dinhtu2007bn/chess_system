"""Module đánh giá cấu trúc Tốt (Pawn Structure) sử dụng Mặt nạ Bitboard tính trước & Cache.

Đáp ứng đầy đủ yêu cầu:
1. Tốt thông (Passed Pawns): Thưởng tăng dần theo hàng (10 -> 150), mạnh hơn ở tàn cuộc.
2. Tốt chồng (Doubled Pawns): Phạt -10 MG / -20 EG cho mỗi tốt thừa trên cùng cột.
3. Tốt cô lập (Isolated Pawns): Phạt -15 MG / -20 EG nếu 2 cột kề không có tốt đồng minh.
4. Tốt liên kết (Connected / Phalanx): Thưởng +5 -> +15 tăng theo hàng nếu được bảo vệ chéo hoặc đi cạnh nhau.
5. Bitboard Masks: Toàn bộ mặt nạ 64-bit được tính trước ở mức module (O(1)).
6. Cache: Bảng băm lưu trữ kết quả cấu trúc tốt tránh tính toán lại.
"""

import chess

# ==============================================================================
# 1. BẢNG THƯỞNG PHẠT ĐIỂM CẤU TRÚC TỐT 
# ==============================================================================
DOUBLED_PENALTY_MG = 10
DOUBLED_PENALTY_EG = 20

ISOLATED_PENALTY_MG = 15
ISOLATED_PENALTY_EG = 20

# Điểm thưởng Tốt thông theo hàng (Rank 0..7)
PASSED_BONUS_MG = [0, 10, 15, 25, 45, 75, 110, 0]
PASSED_BONUS_EG = [0, 15, 25, 45, 75, 115, 150, 0]

# Điểm thưởng Tốt liên kết (hàng tốt mạnh) theo hàng
CONNECTED_BONUS_MG = [0, 5, 6, 8, 10, 12, 15, 0]
CONNECTED_BONUS_EG = [0, 5, 7, 9, 11, 13, 15, 0]


# ==============================================================================
# 2. TÍNH TRƯỚC MẶT NẠ BITBOARD (PRECOMPUTED MASKS)
# ==============================================================================
# Mặt nạ 8 cột (Files a..h)
FILE_MASKS: list[int] = [0] * 8
for f in range(8):
    mask = 0
    for r in range(8):
        mask |= 1 << (r * 8 + f)    # Bật bit 1 ở các ô: r=0..7 trên cùng cột f
    FILE_MASKS[f] = mask

# Mặt nạ 2 cột liền kề (Adjacent files) dùng để kiểm tra Tốt cô lập
ADJACENT_FILES_MASK: list[int] = [0] * 8
for f in range(8):
    adj = 0
    if f > 0:
        adj |= FILE_MASKS[f - 1]
    if f < 7:
        adj |= FILE_MASKS[f + 1]
    ADJACENT_FILES_MASK[f] = adj

# Mặt nạ Tốt thông (Passed pawn masks): Kiểm tra cột hiện tại và 2 cột kề phía trước
WHITE_PASSED_MASKS: list[int] = [0] * 64
BLACK_PASSED_MASKS: list[int] = [0] * 64

for sq in range(64):
    file_idx = sq % 8
    rank_idx = sq // 8

    w_mask = 0
    b_mask = 0
    for adj_f in range(max(0, file_idx - 1), min(8, file_idx + 2)):
        for r in range(8):
            target_sq = r * 8 + adj_f
            if r > rank_idx:
                w_mask |= 1 << target_sq
            if r < rank_idx:
                b_mask |= 1 << target_sq

    WHITE_PASSED_MASKS[sq] = w_mask
    BLACK_PASSED_MASKS[sq] = b_mask


# ==============================================================================
# 3. BẢNG BĂM LƯU TẠM (PAWN HASH TABLE / CACHE)
# ==============================================================================
# Key: (white_pawns_bitboard, black_pawns_bitboard) -> Value: (mg_score_diff, eg_score_diff)
_PAWN_CACHE: dict[tuple[int, int], tuple[int, int]] = {}
_MAX_CACHE_SIZE = 16384


def _eval_side_pawns(
    friendly_pawns: int,    #tốt ta
    enemy_pawns: int,       # tốt địch 
    is_white: bool,
) -> tuple[int, int]:
    """Đánh giá toàn bộ cấu trúc tốt của một phe bằng phép toán Bitboard."""
    mg_score = 0
    eg_score = 0

    # 1. Tốt chồng (Doubled Pawns) & Tốt cô lập (Isolated Pawns) quét theo 8 cột
    for f in range(8):
        f_mask = FILE_MASKS[f]
        pawns_in_file = (friendly_pawns & f_mask).bit_count()

        if pawns_in_file > 0:
            # Tốt chồng: Cứ mỗi quân tốt thừa trên cùng 1 cột thì phạt
            if pawns_in_file > 1:
                excess = pawns_in_file - 1
                mg_score -= excess * DOUBLED_PENALTY_MG
                eg_score -= excess * DOUBLED_PENALTY_EG

            # Tốt cô lập: Không có bất kỳ tốt đồng minh nào ở 2 cột liền kề
            if (friendly_pawns & ADJACENT_FILES_MASK[f]) == 0:
                mg_score -= pawns_in_file * ISOLATED_PENALTY_MG
                eg_score -= pawns_in_file * ISOLATED_PENALTY_EG

    # 2. Tốt thông & Tốt liên kết (Quét từng con tốt qua bitboard)
    passed_masks = WHITE_PASSED_MASKS if is_white else BLACK_PASSED_MASKS
    temp_pawns = friendly_pawns

    while temp_pawns:
        # Lấy chỉ số bit 1 thấp nhất (Lowest Set Bit) trong O(1)
        lsb = temp_pawns & -temp_pawns
        sq = lsb.bit_length() - 1
        temp_pawns ^= lsb  # Xóa bit vừa xét

        rank = sq // 8
        rel_rank = rank if is_white else (7 - rank)

        # A. Kiểm tra Tốt thông: Không có tốt địch nào nằm trong passed_mask
        if (enemy_pawns & passed_masks[sq]) == 0:
            mg_score += PASSED_BONUS_MG[rel_rank]
            eg_score += PASSED_BONUS_EG[rel_rank]

        # B. Kiểm tra Hàng tốt mạnh (Tốt liên kết: được bảo vệ chéo HOẶC đứng cạnh nhau cùng hàng)
        f = sq % 8
        is_connected = False

        # Kiểm tra được bảo vệ chéo từ phía sau
        if is_white:
            if (f > 0 and (friendly_pawns & (1 << (sq - 9)))) or (
                f < 7 and (friendly_pawns & (1 << (sq - 7)))
            ):
                is_connected = True
        else:
            if (f > 0 and (friendly_pawns & (1 << (sq + 7)))) or (
                f < 7 and (friendly_pawns & (1 << (sq + 9)))
            ):
                is_connected = True

        # Kiểm tra đứng cạnh tốt đồng minh cùng hàng (Phalanx)
        if not is_connected:
            if (f > 0 and (friendly_pawns & (1 << (sq - 1)))) or (
                f < 7 and (friendly_pawns & (1 << (sq + 1)))
            ):
                is_connected = True

        if is_connected:
            mg_score += CONNECTED_BONUS_MG[rel_rank]
            eg_score += CONNECTED_BONUS_EG[rel_rank]

    return mg_score, eg_score


def evaluate_pawn_structure(board: chess.Board) -> tuple[int, int]:
    """Lượng giá cấu trúc tốt cho cả 2 bên.

    Trả về: (mg_diff, eg_diff) với quy ước (Điểm Trắng - Điểm Đen).
    """
    global _PAWN_CACHE

    white_pawns = int(board.pieces_mask(chess.PAWN, chess.WHITE))
    black_pawns = int(board.pieces_mask(chess.PAWN, chess.BLACK))
    cache_key = (white_pawns, black_pawns)

    # 1. Tra cứu Cache
    cached = _PAWN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # 2. Tính toán nếu chưa có trong Cache
    w_mg, w_eg = _eval_side_pawns(white_pawns, black_pawns, is_white=True)
    b_mg, b_eg = _eval_side_pawns(black_pawns, white_pawns, is_white=False)

    diff = (w_mg - b_mg, w_eg - b_eg)

    # 3. Lưu vào Cache (giới hạn dung lượng để tránh tràn RAM)
    if len(_PAWN_CACHE) >= _MAX_CACHE_SIZE:
        _PAWN_CACHE.clear()
    _PAWN_CACHE[cache_key] = diff

    return diff
