import chess

from .pawn_structure import FILE_MASKS, evaluate_pawn_structure
from .pst import (
    KING_EG,
    KING_MG,
    PHASE_WEIGHTS,
    PIECE_VALUES_EG,
    PIECE_VALUES_MG,
    TABLES_EG,
    TABLES_MG,
    TOTAL_PHASE,
)

# ==============================================================================
# HẰNG SỐ ĐẶC THÙ: KHAI CUỘC 960, CẶP TƯỢNG, XE CỘT MỞ, LÁ CHẮN VUA & MOP-UP
# ==============================================================================
CASTLING_RIGHT_BONUS = 25       # Thưởng cho mỗi cánh còn quyền nhập thành
MINOR_DEV_BONUS = 15            # Thưởng mỗi Mã/Tượng rời hàng đáy tham chiến
EARLY_QUEEN_PENALTY = 25        # Phạt xuất Hậu khi các quân nhẹ chưa phát triển

BISHOP_PAIR_BONUS_MG = 35       # Thưởng cặp Tượng ở trung cuộc
BISHOP_PAIR_BONUS_EG = 45       # Thưởng cặp Tượng ở tàn cuộc (bàn cờ thoáng)
ROOK_OPEN_FILE_BONUS = 25       # Thưởng Xe chiếm cột mở hoàn toàn (không còn tốt)
ROOK_SEMI_OPEN_BONUS = 15       # Thưởng Xe chiếm cột nửa mở (chỉ có tốt địch)
KING_SHIELD_PENALTY = 15        # Phạt cho mỗi cột thiếu tốt che chắn trước mặt Vua


def _mop_up_eval(winning_king: int, losing_king: int) -> int:
    """Tính điểm dồn ép Vua ở cờ tàn:
    1. Đẩy Vua địch ra xa trung tâm (sát mép hoặc góc bàn cờ).
    2. Kéo Vua mình lại gần Vua địch để che chắn và chiếu hết.
    """
    los_f = chess.square_file(losing_king)
    los_r = chess.square_rank(losing_king)
    center_dist = max(3 - los_f, los_f - 4) + max(3 - los_r, los_r - 4)

    win_f = chess.square_file(winning_king)
    win_r = chess.square_rank(winning_king)
    kings_dist = max(abs(win_f - los_f), abs(win_r - los_r))

    return center_dist * 15 + (7 - kings_dist) * 10


def evaluate_board(board: chess.Board) -> int:
    """Lượng giá thế cờ toàn diện v3:
    - Tapered + PST thủ công
    - Bitboard Cached Pawn Structure (Tốt thông, chồng, cô lập, liên kết)
    - Cặp Tượng & Xe chiếm cột mở / nửa mở
    - Lá chắn Tốt bảo vệ Vua sau khi nhập thành
    - Đặc thù khai cuộc Chess960 (Nhập thành, phát triển quân nhẹ, tránh xuất Hậu sớm)
    - Mop-up tàn cuộc và nhận diện hòa lý thuyết
    - Chuẩn Negamax (trả về số nguyên int cho cửa sổ hẹp PVS)
    """
    # 1. NHẬN DIỆN HÒA LÝ THUYẾT (Thiếu quân chiếu hết: K vs K, K+N vs K, K+B vs K)
    if board.is_insufficient_material():
        return 0

    mg_white, mg_black = 0, 0
    eg_white, eg_black = 0, 0
    game_phase = 0

    # Lấy sẵn bitboard tốt của 2 bên để dùng cho Xe cột mở & Lá chắn Vua
    white_pawns = int(board.pieces_mask(chess.PAWN, chess.WHITE))
    black_pawns = int(board.pieces_mask(chess.PAWN, chess.BLACK))
    all_pawns = white_pawns | black_pawns

    # 2. QUÉT 5 LOẠI QUÂN THÔNG THƯỜNG
    white_minors_on_back_rank = 0
    black_minors_on_back_rank = 0

    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        val_mg = PIECE_VALUES_MG[piece_type]
        val_eg = PIECE_VALUES_EG[piece_type]
        tbl_mg = TABLES_MG[piece_type]
        tbl_eg = TABLES_EG[piece_type]

        if piece_type in PHASE_WEIGHTS:
            count = len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            game_phase += count * PHASE_WEIGHTS[piece_type]

        # --- Quân Trắng ---
        for sq in board.pieces(piece_type, chess.WHITE):
            mg_white += val_mg + tbl_mg[sq]
            eg_white += val_eg + tbl_eg[sq]

            # Khai cuộc 960: kiểm tra phát triển Mã, Tượng rời hàng 1
            if piece_type in (chess.KNIGHT, chess.BISHOP):
                if chess.square_rank(sq) == 0:
                    white_minors_on_back_rank += 1
                else:
                    mg_white += MINOR_DEV_BONUS

            # Xe chiếm cột mở / nửa mở
            elif piece_type == chess.ROOK:
                f_mask = FILE_MASKS[chess.square_file(sq)]
                if (all_pawns & f_mask) == 0:
                    mg_white += ROOK_OPEN_FILE_BONUS
                    eg_white += ROOK_OPEN_FILE_BONUS
                elif (white_pawns & f_mask) == 0:
                    mg_white += ROOK_SEMI_OPEN_BONUS
                    eg_white += ROOK_SEMI_OPEN_BONUS

        # --- Quân Đen (lật gương ô cờ qua trục ngang) ---
        for sq in board.pieces(piece_type, chess.BLACK):
            mirror_sq = chess.square_mirror(sq)
            mg_black += val_mg + tbl_mg[mirror_sq]
            eg_black += val_eg + tbl_eg[mirror_sq]

            # Khai cuộc 960: kiểm tra phát triển Mã, Tượng rời hàng 8
            if piece_type in (chess.KNIGHT, chess.BISHOP):
                if chess.square_rank(sq) == 7:
                    black_minors_on_back_rank += 1
                else:
                    mg_black += MINOR_DEV_BONUS

            # Xe chiếm cột mở / nửa mở
            elif piece_type == chess.ROOK:
                f_mask = FILE_MASKS[chess.square_file(sq)]
                if (all_pawns & f_mask) == 0:
                    mg_black += ROOK_OPEN_FILE_BONUS
                    eg_black += ROOK_OPEN_FILE_BONUS
                elif (black_pawns & f_mask) == 0:
                    mg_black += ROOK_SEMI_OPEN_BONUS
                    eg_black += ROOK_SEMI_OPEN_BONUS

    # 3. THƯỞNG CẶP TƯỢNG (BISHOP PAIR)
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        mg_white += BISHOP_PAIR_BONUS_MG
        eg_white += BISHOP_PAIR_BONUS_EG

    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        mg_black += BISHOP_PAIR_BONUS_MG
        eg_black += BISHOP_PAIR_BONUS_EG

    # 4. QUÂN VUA & LÁ CHẮN TỐT (KING PAWN SHIELD)
    king_white = board.king(chess.WHITE)
    if king_white is not None:
        mg_white += KING_MG[king_white]
        eg_white += KING_EG[king_white]

        # Kiểm tra lá chắn tốt khi Vua Trắng nhập thành ở cánh Vua (f, g, h) hoặc cánh Hậu (a, b, c)
        if game_phase >= 14:
            kw_f = chess.square_file(king_white)
            if kw_f in (5, 6, 7):  # Cánh Vua
                for f in (5, 6, 7):
                    if (white_pawns & FILE_MASKS[f]) == 0:
                        mg_white -= KING_SHIELD_PENALTY
            elif kw_f in (0, 1, 2):  # Cánh Hậu
                for f in (0, 1, 2):
                    if (white_pawns & FILE_MASKS[f]) == 0:
                        mg_white -= KING_SHIELD_PENALTY

    king_black = board.king(chess.BLACK)
    if king_black is not None:
        mirror_kb = chess.square_mirror(king_black)
        mg_black += KING_MG[mirror_kb]
        eg_black += KING_EG[mirror_kb]

        # Kiểm tra lá chắn tốt Vua Đen
        if game_phase >= 14:
            kb_f = chess.square_file(king_black)
            if kb_f in (5, 6, 7):  # Cánh Vua
                for f in (5, 6, 7):
                    if (black_pawns & FILE_MASKS[f]) == 0:
                        mg_black -= KING_SHIELD_PENALTY
            elif kb_f in (0, 1, 2):  # Cánh Hậu
                for f in (0, 1, 2):
                    if (black_pawns & FILE_MASKS[f]) == 0:
                        mg_black -= KING_SHIELD_PENALTY

    # 5. ĐẶC THÙ KHAI CUỘC CHESS960 (Chỉ áp dụng khi game_phase >= 18)
    if game_phase >= 18:
        # A. Quyền nhập thành
        if board.has_kingside_castling_rights(chess.WHITE):
            mg_white += CASTLING_RIGHT_BONUS
        if board.has_queenside_castling_rights(chess.WHITE):
            mg_white += CASTLING_RIGHT_BONUS

        if board.has_kingside_castling_rights(chess.BLACK):
            mg_black += CASTLING_RIGHT_BONUS
        if board.has_queenside_castling_rights(chess.BLACK):
            mg_black += CASTLING_RIGHT_BONUS

        # B. Phạt xuất Hậu quá sớm nếu quân nhẹ còn ùn tắc ở hàng đáy
        for sq in board.pieces(chess.QUEEN, chess.WHITE):
            if chess.square_rank(sq) > 0 and white_minors_on_back_rank >= 3:
                mg_white -= EARLY_QUEEN_PENALTY

        for sq in board.pieces(chess.QUEEN, chess.BLACK):
            if chess.square_rank(sq) < 7 and black_minors_on_back_rank >= 3:
                mg_black -= EARLY_QUEEN_PENALTY

    # 6. CẤU TRÚC TỐT (BITBOARD MASKS & CACHE)
    pawn_mg_diff, pawn_eg_diff = evaluate_pawn_structure(board)
    mg_score = (mg_white - mg_black) + pawn_mg_diff
    eg_score = (eg_white - eg_black) + pawn_eg_diff

    # 7. MOP-UP TÀN CUỘC (Chỉ kích hoạt khi phase <= 4 và một bên có ưu thế vật chất >= 200cp)
    phase = min(game_phase, TOTAL_PHASE)
    if phase <= 4 and king_white is not None and king_black is not None:
        if eg_score >= 200:
            eg_score += _mop_up_eval(winning_king=king_white, losing_king=king_black)
        elif eg_score <= -200:
            eg_score -= _mop_up_eval(winning_king=king_black, losing_king=king_white)

    # 8. NỘI SUY TAPERED THEO GIAI ĐOẠN VÁN CỜ (CHIA SỐ NGUYÊN //)
    total_score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE

    # 9. CHUẨN NEGAMAX (SỐ NGUYÊN INT)
    return int(total_score) if board.turn == chess.WHITE else int(-total_score)
