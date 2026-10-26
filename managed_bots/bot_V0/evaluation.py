from __future__ import annotations

import chess

# Material Values (Centipawns)
PIECE_VALUES_MG = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 930,
    chess.KING: 20000,
}

PIECE_VALUES_EG = {
    chess.PAWN: 110,
    chess.KNIGHT: 300,
    chess.BISHOP: 340,
    chess.ROOK: 520,
    chess.QUEEN: 960,
    chess.KING: 20000,
}

# Piece square tables (White perspective, square 0 is a1, 63 is h8)
PAWN_PST_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

PAWN_PST_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    90, 90, 90, 90, 90, 90, 90, 90,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_PST_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

KING_PST_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

# Phase Weights for Tapered Evaluation
PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
MAX_PHASE = 24  # 4*1 + 4*1 + 4*2 + 2*4 = 24


def evaluate_board(board: chess.Board) -> int:
    """Evaluates the position from White's perspective in centipawns.
    
    Positive score means White is better, negative means Black is better.
    """
    if board.is_checkmate():
        return -30000 if board.turn == chess.WHITE else 30000
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0

    mg_white = 0
    mg_black = 0
    eg_white = 0
    eg_black = 0
    game_phase = 0

    # Piece counts for special evaluations
    white_bishops = 0
    black_bishops = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        piece_type = piece.piece_type
        color = piece.color
        sq_index = square if color == chess.WHITE else chess.square_mirror(square)

        # Game phase accumulation
        if piece_type in PHASE_WEIGHTS:
            game_phase += PHASE_WEIGHTS[piece_type]

        # Material & PST
        if color == chess.WHITE:
            mg_white += PIECE_VALUES_MG[piece_type]
            eg_white += PIECE_VALUES_EG[piece_type]
            if piece_type == chess.PAWN:
                mg_white += PAWN_PST_MG[sq_index]
                eg_white += PAWN_PST_EG[sq_index]
            elif piece_type == chess.KNIGHT:
                mg_white += KNIGHT_PST[sq_index]
                eg_white += KNIGHT_PST[sq_index]
            elif piece_type == chess.BISHOP:
                white_bishops += 1
                mg_white += BISHOP_PST[sq_index]
                eg_white += BISHOP_PST[sq_index]
            elif piece_type == chess.ROOK:
                mg_white += ROOK_PST[sq_index]
                eg_white += ROOK_PST[sq_index]
                # Open or semi-open file
                file_idx = chess.square_file(square)
                pawns_on_file = len(board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[file_idx])
                enemy_pawns = len(board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[file_idx])
                if pawns_on_file == 0:
                    bonus = 25 if enemy_pawns == 0 else 15
                    mg_white += bonus
                    eg_white += bonus
            elif piece_type == chess.QUEEN:
                mg_white += QUEEN_PST[sq_index]
                eg_white += QUEEN_PST[sq_index]
            elif piece_type == chess.KING:
                mg_white += KING_PST_MG[sq_index]
                eg_white += KING_PST_EG[sq_index]
        else:
            mg_black += PIECE_VALUES_MG[piece_type]
            eg_black += PIECE_VALUES_EG[piece_type]
            if piece_type == chess.PAWN:
                mg_black += PAWN_PST_MG[sq_index]
                eg_black += PAWN_PST_EG[sq_index]
            elif piece_type == chess.KNIGHT:
                mg_black += KNIGHT_PST[sq_index]
                eg_black += KNIGHT_PST[sq_index]
            elif piece_type == chess.BISHOP:
                black_bishops += 1
                mg_black += BISHOP_PST[sq_index]
                eg_black += BISHOP_PST[sq_index]
            elif piece_type == chess.ROOK:
                mg_black += ROOK_PST[sq_index]
                eg_black += ROOK_PST[sq_index]
                file_idx = chess.square_file(square)
                pawns_on_file = len(board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[file_idx])
                enemy_pawns = len(board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[file_idx])
                if pawns_on_file == 0:
                    bonus = 25 if enemy_pawns == 0 else 15
                    mg_black += bonus
                    eg_black += bonus
            elif piece_type == chess.QUEEN:
                mg_black += QUEEN_PST[sq_index]
                eg_black += QUEEN_PST[sq_index]
            elif piece_type == chess.KING:
                mg_black += KING_PST_MG[sq_index]
                eg_black += KING_PST_EG[sq_index]

    # Bishop pair bonus
    if white_bishops >= 2:
        mg_white += 35
        eg_white += 40
    if black_bishops >= 2:
        mg_black += 35
        eg_black += 40

    # Tapered evaluation interpolation
    phase = min(MAX_PHASE, game_phase)
    mg_score = mg_white - mg_black
    eg_score = eg_white - eg_black
    total_score = ((mg_score * phase) + (eg_score * (MAX_PHASE - phase))) // MAX_PHASE

    return total_score
