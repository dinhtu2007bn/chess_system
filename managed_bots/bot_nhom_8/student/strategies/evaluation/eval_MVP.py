import chess

# Bảng giá trị vật chất chuẩn theo centipawn (1 Tốt = 100 centipawns)
PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def evaluate_board(board: chess.Board) -> float:
    white_material = sum(
        len(board.pieces(piece_type, chess.WHITE)) * val  # tổng quân trắng
        for piece_type, val in PIECE_VALUES.items()
    )

    black_material = sum(
        len(board.pieces(piece_type, chess.BLACK)) * val  # tổng quân đen
        for piece_type, val in PIECE_VALUES.items()
    )
    score = float(white_material - black_material)
    return score if board.turn == chess.WHITE else -score  # trả về score nếu bên trắng đi, -score nếu bên đen đi
