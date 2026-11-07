
from functools import lru_cache

import chess

VALUES = (0, 100, 325, 345, 510, 965, 0)
END_VALUES = (0, 118, 320, 347, 535, 1010, 0)
MOBILITY = (0, 0, 4, 4, 2, 1, 0)
ATTACK_WEIGHT = (0, 0, 2, 2, 3, 4, 0)
PASSED_BONUS = (0, 0, 9, 20, 40, 72, 125, 0)
EDGE = (0, 1, 2, 3, 3, 2, 1, 0)
FILES = (0, *chess.BB_FILES, 0)
NEIGHBORS = tuple(left | right for left, right in zip(FILES, FILES[2:]))
KING_FILES = tuple(FILES[file:file + 3] for file in range(8))


def square_points(square):
    file, rank = chess.square_file(square), chess.square_rank(square)
    center_file = EDGE[file]
    center = center_file + EDGE[rank]
    pawn = 4 * rank + 4 * center_file * (rank - 1)
    knight = 12 * center - 42
    bishop = 6 * center + 2 * rank - 18
    rook = 3 * center_file + 18 * (rank == 6)
    queen = 2 * center - 3 * rank
    king = -7 * center_file - 13 * rank
    middle = (0, pawn, knight, bishop, rook, queen, king)
    end = (0, 8 * rank, knight, bishop, rook, 3 * center, 11 * center)
    return tuple((VALUES[kind] + middle[kind], END_VALUES[kind] + end[kind])
                 for kind in range(chess.KING + 1))


WHITE_POINTS = tuple(map(square_points, chess.SQUARES))
PIECE_POINTS = (
    tuple(WHITE_POINTS[chess.square_mirror(sq)] for sq in chess.SQUARES),
    WHITE_POINTS,
)
AHEAD = (tuple(sum(chess.BB_RANKS[:rank]) for rank in range(8)),
         tuple(sum(chess.BB_RANKS[rank + 1:]) for rank in range(8)))
PASSED_SPANS = tuple(
    tuple(AHEAD[color][sq // 8] & (chess.BB_FILES[sq % 8] | NEIGHBORS[sq % 8])
          for sq in chess.SQUARES)
    for color in (chess.BLACK, chess.WHITE)
)


@lru_cache(maxsize=49_152)
def placement(color, kind, squares):
    mid = end = 0
    points = PIECE_POINTS[color]
    for square in chess.scan_forward(squares):
        mg, eg = points[square][kind]
        mid += mg
        end += eg
    return mid, end


def pawn_attacks(pawns, color):
    if color == chess.WHITE:
        return ((pawns & ~chess.BB_FILE_A) << 7) | ((pawns & ~chess.BB_FILE_H) << 9)
    return ((pawns & ~chess.BB_FILE_H) >> 7) | ((pawns & ~chess.BB_FILE_A) >> 9)


@lru_cache(maxsize=24_576)
def pawn_structure(color, pawns, enemy_pawns):
    attacks = pawn_attacks(pawns, color)
    supported = (pawns & attacks).bit_count()
    mid, end = 6 * supported, 9 * supported
    passed = 0
    for file, adjacent in zip(chess.BB_FILES, NEIGHBORS):
        count = (pawns & file).bit_count()
        if count:
            isolated = count if not pawns & adjacent else 0
            mid -= 14 * (count - 1) + 10 * isolated
            end -= 18 * (count - 1) + 11 * isolated
    for sq in chess.scan_forward(pawns):
        if not enemy_pawns & PASSED_SPANS[color][sq]:
            rank = sq // 8 if color else 7 - sq // 8
            bonus = PASSED_BONUS[rank]
            mid += bonus // 2
            end += bonus
            passed |= chess.BB_SQUARES[sq]
    return mid, end, passed, attacks


def king_shelter(color, king, pawns):
    files = KING_FILES[chess.square_file(king)]
    zone = files[0] | files[1] | files[2]
    rank = chess.square_rank(king)
    step = 1 if color else -1
    first = rank + step
    second = rank + 2 * step
    near = (pawns & zone & chess.BB_RANKS[first]).bit_count() if 0 <= first < 8 else 0
    far = (pawns & zone & chess.BB_RANKS[second]).bit_count() if 0 <= second < 8 else 0
    open_files = sum(1 for file in files if file and not pawns & file)
    return 13 * near + 7 * far - 12 * open_files


def endgame_bonus(board, color, own, enemy, king, enemy_king, passed):
    score = 0
    step = 8 if color else -8
    for sq in chess.scan_forward(passed):
        front = sq + step
        if 0 <= front < 64 and not board.occupied & chess.BB_SQUARES[front]:
            rank = sq // 8 if color else 7 - sq // 8
            support = chess.square_distance(enemy_king, front) - chess.square_distance(king, front)
            score += max(0, 4 * rank + 4 * support)

    if enemy == board.kings & enemy and own & (board.rooks | board.queens):
        edge_distance = EDGE[enemy_king % 8] + EDGE[enemy_king // 8]
        score += 160 - 17 * edge_distance - 14 * chess.square_distance(king, enemy_king)
    return score


def evaluate(board: chess.Board) -> int:
    occupied = board.occupied_co
    black, white = board.pawns & occupied[chess.BLACK], board.pawns & occupied[chess.WHITE]
    pawn_terms = (pawn_structure(chess.BLACK, black, white),
                  pawn_structure(chess.WHITE, white, black))
    kings = board.king(chess.BLACK), board.king(chess.WHITE)
    masks = (0, board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)
    phase = min(24, (board.knights | board.bishops).bit_count()
                + 2 * board.rooks.bit_count() + 4 * board.queens.bit_count())
    middle = end = 0

    for color in chess.COLORS:
        own, enemy = occupied[color], occupied[not color]
        king, enemy_king = kings[color], kings[not color]
        mid, late, passed, _ = pawn_terms[color]
        allowed = ~(own | pawn_terms[not color][3])
        ring = chess.BB_KING_ATTACKS[enemy_king]
        pressure = attackers = 0

        for kind in chess.PIECE_TYPES:
            squares = masks[kind] & own
            if not squares:
                continue
            mg, eg = placement(color, kind, squares)
            mid += mg
            late += eg
            mobility_weight = MOBILITY[kind]
            if not mobility_weight:
                continue
            for sq in chess.scan_forward(squares):
                attacks = board.attacks_mask(sq)
                mobility = (attacks & allowed).bit_count()
                mid += mobility_weight * mobility
                late += 2 * mobility
                hits = (attacks & ring & ~own).bit_count()
                if hits:
                    pressure += ATTACK_WEIGHT[kind] * hits
                    attackers += 1

        if attackers >= 2:
            mid += min(160, pressure * pressure // 4)
        if (board.bishops & own).bit_count() >= 2:
            mid += 29
            late += 39
        pawns = board.pawns & own
        for sq in chess.scan_forward(board.rooks & own):
            file = chess.BB_FILES[sq % 8]
            if not pawns & file:
                mid += 10 + 10 * (not board.pawns & file)
                late += 7
        if enemy & (board.queens | board.rooks):
            shelter = king_shelter(color, king, pawns)
            mid += shelter if enemy & board.queens else shelter // 2
        if board.castling_rights & own:
            mid += 9
        if phase <= 10:
            late += endgame_bonus(board, color, own, enemy, king, enemy_king, passed)

        sign = 1 if color == board.turn else -1
        middle += sign * mid
        end += sign * late

    return (middle * phase + end * (24 - phase)) // 24 + 8
