import chess

# ==============================================================================
# 1. GIÁ TRỊ VẬT CHẤT (CENTIPAWNS: 1 TỐT = 100)
# ==============================================================================
# Trung cuộc: Ưu tiên quân nhẹ cơ động, Hậu kiểm soát
PIECE_VALUES_MG: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# Tàn cuộc: Tốt tăng giá vì đe dọa hóa Hậu; Mã giảm giá vì thế cờ mở; Tượng & Xe tăng giá
PIECE_VALUES_EG: dict[chess.PieceType, int] = {
    chess.PAWN: 130,
    chess.KNIGHT: 280,
    chess.BISHOP: 340,
    chess.ROOK: 550,
    chess.QUEEN: 950,
}

# Trọng số tính Phase ván cờ (Tổng = 24: 4 Mã*1 + 4 Tượng*1 + 4 Xe*2 + 2 Hậu*4)
PHASE_WEIGHTS: dict[chess.PieceType, int] = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
TOTAL_PHASE = 24


def _flip(table: list[int]) -> list[int]:
    """Lật bảng từ thứ tự nhìn (hàng 8 xuống 1) sang chỉ số nội bộ của python-chess (a1 -> h8)."""
    res: list[int] = []
    for rank in range(7, -1, -1):
        res.extend(table[rank * 8 : (rank + 1) * 8])
    return res


# ==============================================================================
# 2. BẢNG ĐIỂM VỊ TRÍ (PST THỦ CÔNG - TỰ XÂY DỰNG THEO NGUYÊN LÝ CỜ VUA)
# ==============================================================================

# --- TỐT (Pawn) ---
# MG: Thưởng tốt chiếm trung tâm (e4, d4); phạt tốt dâng biên vô nghĩa.
PAWN_MG = _flip([
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,  # Sát hàng phong cấp
    10, 10, 20, 30, 30, 20, 10, 10,  # Kiểm soát hàng 6
     5,  5, 15, 25, 25, 15,  5,  5,  # Chiếm trung tâm e4/d4
     0,  0, 10, 20, 20, 10,  0,  0,  # Hàng 4 ổn định
     5, -5,-10,  0,  0,-10, -5,  5,  # Hạn chế tốt lùi/bị chặn
     5, 10, 10,-20,-20, 10, 10,  5,  # Bảo vệ khiên tốt quanh vua
     0,  0,  0,  0,  0,  0,  0,  0,
])

# EG: Tốt dâng càng cao càng thưởng mạnh vì đe dọa hóa Hậu.
PAWN_EG = _flip([
      0,   0,   0,   0,   0,   0,   0,   0,
    120, 120, 120, 120, 120, 120, 120, 120,  # Hàng 7: Chỉ cách 1 ô là hóa Hậu (+120)
     80,  80,  80,  80,  80,  80,  80,  80,  # Hàng 6: Rất nguy hiểm (+80)
     50,  50,  50,  50,  50,  50,  50,  50,  # Hàng 5 (+50)
     30,  30,  30,  30,  30,  30,  30,  30,  # Hàng 4 (+30)
     15,  15,  15,  15,  15,  15,  15,  15,  # Hàng 3 (+15)
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
])

# --- MÃ (Knight) ---
# MG: Mã ở góc/rìa chỉ đi được 2 ô nên phạt nặng (-40, -50); ở trung tâm kiểm soát 8 ô nên thưởng (+20).
KNIGHT_MG = _flip([
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,  # Ô trung tâm e4, d4, e5, d5 là lý tưởng (+20)
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,  # Các ô c3, f3 phát triển quân sớm (+10, +15)
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,  # Hàng đáy và góc: Phạt nặng để đẩy Mã ra giữa
])
KNIGHT_EG = KNIGHT_MG

# --- TƯỢNG (Bishop) ---
# MG: Tượng kiểm soát các đường chéo mở, tránh bị kẹt ở góc bàn cờ.
BISHOP_MG = _flip([
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,  10,  15,  15,  10,   5, -10,  # Bắn chéo qua trung tâm
    -10,   0,  10,  15,  15,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
])
BISHOP_EG = BISHOP_MG

# --- XE (Rook) ---
# MG & EG: Xe ưu tiên hàng 7 áp sát vua địch (+15), cột trung tâm (+10); tránh đứng yên góc đáy.
ROOK_MG = _flip([
      0,   0,   0,   0,   0,   0,   0,   0,
     15,  15,  15,  15,  15,  15,  15,  15,  # Hàng 7: Hàng ác mộng của cờ vua (+15)
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,  # Cột d, e kiểm soát trung tâm (+10)
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   5,  10,  10,   5,   0,   0,
])
ROOK_EG = ROOK_MG

# --- HẬU (Queen) ---
# MG: Hậu không ra sớm quá đà, nhưng khi ra phải kiểm soát ô trung tâm an toàn.
QUEEN_MG = _flip([
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,  10,  10,   5,   0,  -5,
      0,   0,   5,  10,  10,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
])
QUEEN_EG = _flip([
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   5,  10,  10,  10,  10,   5, -10,
     -5,   5,  10,  15,  15,  10,   5,  -5,  # Hậu tàn cuộc thống trị trung tâm (+15)
     -5,   5,  10,  15,  15,  10,   5,  -5,
    -10,   5,  10,  10,  10,  10,   5, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
])

# --- VUA (King) - ĐỐI LẬP HOÀN TOÀN GIỮA MG VÀ EG ---
# MG: Vua nấp sau khiên tốt ở góc (g1, c1 sau nhập thành) (+30); ra giữa bàn bị chiếu hết (-50).
KING_MG = _flip([
    -40, -40, -40, -40, -40, -40, -40, -40,
    -40, -40, -40, -40, -40, -40, -40, -40,
    -40, -40, -40, -40, -40, -40, -40, -40,
    -40, -40, -40, -40, -40, -40, -40, -40,
    -30, -30, -30, -30, -30, -30, -30, -30,
    -20, -20, -20, -20, -20, -20, -20, -20,
     15,  15,  -5,  -5,  -5,  -5,  15,  15,
     20,  30,  10, -20, -20,  10,  30,  20,  # g1, c1: Điểm vàng an toàn cho Vua (+30)
])

# EG: Bàn cờ hết quân nặng, Vua phải xông vào trung tâm e4/d4 để ép tốt (+40); đứng góc bị phạt (-40).
KING_EG = _flip([
    -40, -30, -20, -10, -10, -20, -30, -40,
    -30, -10,   0,  10,  10,   0, -10, -30,
    -20,   0,  20,  30,  30,  20,   0, -20,
    -10,  10,  30,  40,  40,  30,  10, -10,  # Vua ở trung tâm tàn cuộc (+40)
    -10,  10,  30,  40,  40,  30,  10, -10,
    -20,   0,  20,  30,  30,  20,   0, -20,
    -30, -10,   0,  10,  10,   0, -10, -30,
    -40, -30, -20, -10, -10, -20, -30, -40,  # Đứng im ở góc bị phạt (-40)
])

TABLES_MG: dict[chess.PieceType, list[int]] = {
    chess.PAWN: PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_MG,
}

TABLES_EG: dict[chess.PieceType, list[int]] = {
    chess.PAWN: PAWN_EG,
    chess.KNIGHT: KNIGHT_EG,
    chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG,
    chess.QUEEN: QUEEN_EG,
    chess.KING: KING_EG,
}
