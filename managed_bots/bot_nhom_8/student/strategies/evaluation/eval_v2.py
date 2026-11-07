import chess

from .pawn_structure import evaluate_pawn_structure
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


def evaluate_board(board: chess.Board) -> int:
    """Lượng giá thế cờ toàn diện kết hợp:

    1. Giá trị vật chất 2 giai đoạn (Trung cuộc & Tàn cuộc).
    2. Bảng điểm vị trí (PST) thủ công theo nguyên lý cờ vua.
    3. Cấu trúc Tốt bằng Bitboard Masks & Cache (Tốt thông, Tốt chồng, Tốt cô lập, Tốt liên kết).
    4. Nội suy Tapered theo giai đoạn ván cờ (game_phase).
    5. Chuẩn hóa góc nhìn Negamax theo bên đang tới lượt đi.
    """
    mg_white = 0
    mg_black = 0
    eg_white = 0
    eg_black = 0
    game_phase = 0

    # ==========================================================================
    # 1. QUÉT CÁC QUÂN CỜ THÔNG THƯỜNG (TỐT, MÃ, TƯỢNG, XE, HẬU)
    # ==========================================================================
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        val_mg = PIECE_VALUES_MG[piece_type]
        val_eg = PIECE_VALUES_EG[piece_type]
        tbl_mg = TABLES_MG[piece_type]
        tbl_eg = TABLES_EG[piece_type]

        # Tích lũy Game Phase (chỉ tính quân nặng và quân nhẹ)
        if piece_type in PHASE_WEIGHTS:
            count = len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            game_phase += count * PHASE_WEIGHTS[piece_type]

        # Quân Trắng
        for sq in board.pieces(piece_type, chess.WHITE):
            mg_white += val_mg + tbl_mg[sq]
            eg_white += val_eg + tbl_eg[sq]

        # Quân Đen (lật gương ô cờ để dùng chung bảng điểm với Trắng)
        for sq in board.pieces(piece_type, chess.BLACK):
            mirror_sq = chess.square_mirror(sq)
            mg_black += val_mg + tbl_mg[mirror_sq]
            eg_black += val_eg + tbl_eg[mirror_sq]

    # ==========================================================================
    # 2. VỊ TRÍ QUÂN VUA (CHỈ TÍNH PST, KHÔNG CỘNG VẬT CHẤT)
    # ==========================================================================
    king_white = board.king(chess.WHITE)
    if king_white is not None:
        mg_white += KING_MG[king_white]
        eg_white += KING_EG[king_white]

    king_black = board.king(chess.BLACK)
    if king_black is not None:
        mirror_kb = chess.square_mirror(king_black)
        mg_black += KING_MG[mirror_kb]
        eg_black += KING_EG[mirror_kb]

    # ==========================================================================
    # 3. CẤU TRÚC TỐT (BITBOARD MASKS & CACHE)
    # ==========================================================================
    pawn_mg_diff, pawn_eg_diff = evaluate_pawn_structure(board)

    # ==========================================================================
    # 4. TÍNH ĐIỂM CHÊNH LỆCH VÀ NỘI SUY TAPERED
    # ==========================================================================
    mg_score = (mg_white - mg_black) + pawn_mg_diff
    eg_score = (eg_white - eg_black) + pawn_eg_diff

    phase = min(game_phase, TOTAL_PHASE)
    total_score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE

    # ==========================================================================
    # 5. CHUẨN NEGAMAX: TRẢ VỀ THEO GÓC NHÌN BÊN ĐANG ĐI (SỐ NGUYÊN)
    # ==========================================================================
    return int(total_score) if board.turn == chess.WHITE else int(-total_score)
