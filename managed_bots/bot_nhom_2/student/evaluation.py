"""Original, deterministic handcrafted evaluation; no learned weights or books."""

import chess


VALUES = (0, 100, 320, 335, 500, 950, 0)
END_VALUES = (0, 120, 305, 325, 525, 940, 0)
PHASE = (0, 0, 1, 1, 2, 4, 0)


def _tables():
    middlegame, endgame = [], []
    for piece in range(7):
        mg, eg = [], []
        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            center = 7 - abs(2 * file - 7) - abs(2 * rank - 7)
            if piece == chess.PAWN:
                a = 7 * rank + 3 * center - (12 if file in (0, 7) else 0)
                b = 12 * rank + 2 * center
            elif piece == chess.KNIGHT:
                a, b = 9 * center, 6 * center
            elif piece == chess.BISHOP:
                a, b = 5 * center + 3 * rank, 4 * center
            elif piece == chess.ROOK:
                a, b = (18 if rank == 6 else 0) + 2 * rank, 3 * center
            elif piece == chess.QUEEN:
                a, b = 2 * center - 2 * rank, 4 * center
            elif piece == chess.KING:
                a = -7 * center - 12 * rank
                b = 10 * center
            else:
                a = b = 0
            mg.append(VALUES[piece] + a)
            eg.append(END_VALUES[piece] + b)
        middlegame.append(tuple(mg))
        endgame.append(tuple(eg))
    return tuple(middlegame), tuple(endgame)


MG_TABLE, EG_TABLE = _tables()
FILE_NEIGHBORS = tuple(
    (chess.BB_FILES[f - 1] if f else 0)
    | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
)
PASSED_MASKS = tuple(
    tuple(
        sum(
            chess.BB_SQUARES[chess.square(f, r)]
            for f in range(max(0, chess.square_file(sq) - 1), min(7, chess.square_file(sq) + 1) + 1)
            for r in (range(chess.square_rank(sq) + 1, 8) if color else range(chess.square_rank(sq)))
        )
        for sq in chess.SQUARES
    )
    for color in (chess.BLACK, chess.WHITE)
)


def evaluate(board: chess.Board) -> int:
    """Centipawns from the side to move's perspective (terminals handled in search)."""
    mg = eg = phase = 0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color else -1
        own = board.occupied_co[color]
        pawns = board.pawns & own
        enemy_pawns = board.pawns & board.occupied_co[not color]
        for piece in range(chess.PAWN, chess.KING + 1):
            squares = board.pieces_mask(piece, color)
            phase += PHASE[piece] * squares.bit_count()
            for square in chess.scan_forward(squares):
                relative = square if color else chess.square_mirror(square)
                mg += sign * MG_TABLE[piece][relative]
                eg += sign * EG_TABLE[piece][relative]
                if piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    mobility = (board.attacks_mask(square) & ~own).bit_count()
                    weight = 4 if piece in (chess.KNIGHT, chess.BISHOP) else 2
                    mg += sign * weight * mobility
                    eg += sign * weight * mobility
                if piece == chess.ROOK:
                    file_mask = chess.BB_FILES[chess.square_file(square)]
                    if not pawns & file_mask:
                        bonus = 12 if enemy_pawns & file_mask else 22
                        mg += sign * bonus
                        eg += sign * bonus

        if (board.bishops & own).bit_count() >= 2:
            mg += sign * 28
            eg += sign * 40
        for file in range(8):
            extra = max(0, (pawns & chess.BB_FILES[file]).bit_count() - 1)
            mg -= sign * 13 * extra
            eg -= sign * 20 * extra
        for square in chess.scan_forward(pawns):
            file = chess.square_file(square)
            rank = chess.square_rank(square) if color else 7 - chess.square_rank(square)
            if not pawns & FILE_NEIGHBORS[file]:
                mg -= sign * 11
                eg -= sign * 16
            if not enemy_pawns & PASSED_MASKS[color][square]:
                mg += sign * (8 + rank * rank * 2)
                eg += sign * (12 + rank * rank * 5)

        king = board.king(color)
        if king is not None:
            shield_rank = chess.square_rank(king) + (1 if color else -1)
            if 0 <= shield_rank < 8:
                shield = sum(
                    bool(pawns & chess.BB_SQUARES[chess.square(f, shield_rank)])
                    for f in range(max(0, chess.square_file(king) - 1), min(7, chess.square_file(king) + 1) + 1)
                )
                mg += sign * 13 * shield
            # Attacked king-ring squares matter in middlegames, not bare-king endings.
            danger = sum(
                board.is_attacked_by(not color, sq)
                for sq in chess.scan_forward(chess.BB_KING_ATTACKS[king])
            )
            mg -= sign * 10 * danger

    phase = min(24, phase)
    score = (mg * phase + eg * (24 - phase)) // 24
    return (score if board.turn else -score) + 8
