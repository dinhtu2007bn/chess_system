from __future__ import annotations

from functools import lru_cache

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

_RAW_PAWN_PST = (
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
)

_RAW_KNIGHT_PST = (
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
)

_RAW_BISHOP_PST = (
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
)

_RAW_ROOK_PST = (
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
)

_RAW_QUEEN_PST = (
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
)

_RAW_KING_MID_PST = (
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
)

_RAW_KING_END_PST = (
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
)

PAWN_PST = tuple(_RAW_PAWN_PST[sq ^ 56] for sq in range(64))
KNIGHT_PST = tuple(_RAW_KNIGHT_PST[sq ^ 56] for sq in range(64))
BISHOP_PST = tuple(_RAW_BISHOP_PST[sq ^ 56] for sq in range(64))
ROOK_PST = tuple(_RAW_ROOK_PST[sq ^ 56] for sq in range(64))
QUEEN_PST = tuple(_RAW_QUEEN_PST[sq ^ 56] for sq in range(64))
KING_MID_PST = tuple(_RAW_KING_MID_PST[sq ^ 56] for sq in range(64))
KING_END_PST = tuple(_RAW_KING_END_PST[sq ^ 56] for sq in range(64))

PST_MAP = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}

PHASE_WEIGHTS = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE = 24
TEMPO_BONUS = 8
PASSED_PAWN_BONUS = (0, 5, 10, 20, 35, 60, 100, 0)
MOBILITY_WEIGHTS = (0, 0, 3, 3, 2, 1, 0)
MOBILITY_BASE = (0, 0, 4, 6, 7, 12, 0)
ATTACK_WEIGHTS = (0, 0, 2, 2, 3, 5, 0)
CENTRAL_FILES = chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F
TRAPPED_PIECE_PENALTY = (0, 0, 24, 40, 32, 48, 0)
SQUARE_DISTANCE = tuple(tuple(chess.square_distance(a, b) for b in range(64))
                        for a in range(64))

ENDGAME_VALUES = (0, 110, 310, 335, 520, 940, 0)
MG_TABLE = ((),) + tuple(
    tuple(PIECE_VALUES[kind] + (KING_MID_PST[sq] if kind == chess.KING
                              else PST_MAP[kind][sq]) for sq in range(64))
    for kind in range(chess.PAWN, chess.KING + 1)
)
EG_TABLE = ((),) + tuple(
    tuple(ENDGAME_VALUES[kind] + (KING_END_PST[sq] if kind == chess.KING
                                 else PST_MAP[kind][sq]) for sq in range(64))
    for kind in range(chess.PAWN, chess.KING + 1)
)


def _make_pawn_masks() -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    forward = [[], []]
    adjacent = []
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        side_files = ((chess.BB_FILES[f - 1] if f else 0)
                      | (chess.BB_FILES[f + 1] if f < 7 else 0))
        adjacent.append(side_files)
        for color in (chess.BLACK, chess.WHITE):
            ranks = range(r + 1, 8) if color else range(r)
            forward[color].append(sum(chess.BB_RANKS[rank] for rank in ranks))
    return tuple(map(tuple, forward)), tuple(adjacent)


FORWARD_RANKS, ADJACENT_FILES = _make_pawn_masks()


def _pawn_attacks(pawns: int, color: chess.Color) -> int:
    if color:
        return (((pawns << 7) & ~chess.BB_FILE_H)
                | ((pawns << 9) & ~chess.BB_FILE_A)) & chess.BB_ALL
    return ((pawns >> 7) & ~chess.BB_FILE_A) | ((pawns >> 9) & ~chess.BB_FILE_H)


@lru_cache(maxsize=16384)
def _pawn_structure(
    white: int, black: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    pawns = (black, white)
    mg, eg, passed = [0, 0], [0, 0], [0, 0]
    for color in (chess.BLACK, chess.WHITE):
        own, enemy = pawns[color], pawns[not color]
        attacks = _pawn_attacks(own, color)
        enemy_attacks = _pawn_attacks(enemy, not color)
        for f in range(8):
            count = (own & chess.BB_FILES[f]).bit_count()
            mg[color] -= 12 * max(0, count - 1)
            eg[color] -= 18 * max(0, count - 1)
        for sq in chess.scan_forward(own):
            f, r = sq & 7, sq >> 3
            pst_sq = sq if color else sq ^ 56
            mg[color] += MG_TABLE[chess.PAWN][pst_sq]
            eg[color] += EG_TABLE[chess.PAWN][pst_sq]
            rank = r if color else 7 - r
            sides = ADJACENT_FILES[sq]
            ahead = FORWARD_RANKS[color][sq]
            supported = bool(attacks & chess.BB_SQUARES[sq])
            if not own & sides:
                mg[color] -= 12
                eg[color] -= 10
            else:
                front = sq + (8 if color else -8)
                can_support = own & sides & ~ahead
                if (not supported and not can_support and 0 <= front < 64
                        and enemy_attacks & chess.BB_SQUARES[front]):
                    mg[color] -= 9
                    eg[color] -= 6
            if supported:
                mg[color] += 4
                eg[color] += 6
            if (enemy & ahead & (sides | chess.BB_FILES[f])
                    or own & ahead & chess.BB_FILES[f]):
                continue
            passed[color] |= chess.BB_SQUARES[sq]
            bonus = PASSED_PAWN_BONUS[rank]
            mg[color] += bonus // 2
            eg[color] += bonus
            if supported:
                mg[color] += 4
                eg[color] += 8
        for sq in chess.scan_forward(passed[color]):
            f, r = sq & 7, sq >> 3
            if f < 7:
                ranks = chess.BB_RANKS[r]
                if r:
                    ranks |= chess.BB_RANKS[r - 1]
                if r < 7:
                    ranks |= chess.BB_RANKS[r + 1]
                pairs = (passed[color] & chess.BB_FILES[f + 1] & ranks).bit_count()
                mg[color] += 5 * pairs
                eg[color] += 12 * pairs
    return tuple(mg), tuple(eg), tuple(passed)


@lru_cache(maxsize=16384)
def _pawn_shelter(color: chess.Color, king: int, own: int, enemy: int) -> int:
    f, r = king & 7, king >> 3
    danger = 0
    for file in range(max(0, f - 1), min(8, f + 2)):
        shield = own & chess.BB_FILES[file] & FORWARD_RANKS[color][king]
        if shield:
            nearest = (chess.lsb(shield) >> 3) - r if color else r - (chess.msb(shield) >> 3)
            danger += 0 if nearest == 1 else (5 if nearest == 2 else 12)
        else:
            danger += 14
        if not own & chess.BB_FILES[file]:
            danger += 8 if enemy & chess.BB_FILES[file] else 14
    return danger


def _king_safety(
    board: chess.Board, color: chess.Color, king: int | None,
    pawns: tuple[int, int], pressure: int, attackers: int, enemy_phase: int,
) -> int:
    if king is None:
        return 0
    danger = _pawn_shelter(color, king, pawns[color], pawns[not color])
    danger += min(180, pressure * min(4, attackers) * 2)
    scale = min(12, enemy_phase)
    if not board.queens & board.occupied_co[not color]:
        scale = scale * 2 // 3
    return danger * scale // 12


def _piece_trap_penalty(
    board: chess.Board, square: int, kind: int, color: chess.Color,
    attacks: int, enemy_pawn_attacks: int,
) -> int:
    if not enemy_pawn_attacks & chess.BB_SQUARES[square]:
        return 0
    exits = attacks & ~board.occupied_co[color] & ~enemy_pawn_attacks
    exits &= board.pin_mask(color, square)
    count = exits.bit_count()
    if count > 1:
        return 0
    penalty = TRAPPED_PIECE_PENALTY[kind]
    return penalty if count == 0 else penalty // 2


def _evaluate_position(board: chess.Board) -> int:
    if board.is_insufficient_material():
        return 0
    pawns = (board.pieces_mask(chess.PAWN, chess.BLACK),
             board.pieces_mask(chess.PAWN, chess.WHITE))
    pawn_attacks = (_pawn_attacks(pawns[0], False), _pawn_attacks(pawns[1], True))
    pawn_mg, pawn_eg, passed = _pawn_structure(pawns[1], pawns[0])
    mg, eg = list(pawn_mg), list(pawn_eg)
    kings = (board.king(chess.BLACK), board.king(chess.WHITE))
    zones = tuple(chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k]
                  if k is not None else 0 for k in kings)
    phase, pressure, attackers = [0, 0], [0, 0], [0, 0]
    controlled = [pawn_attacks[0], pawn_attacks[1]]
    for color in (chess.BLACK, chess.WHITE):
        if kings[color] is not None:
            controlled[color] |= chess.BB_KING_ATTACKS[kings[color]]
    blocked = (pawns[0] & (pawns[1] << 8), pawns[1] & (pawns[0] >> 8))

    for color in (chess.BLACK, chess.WHITE):
        own = board.occupied_co[color]
        bishops = board.bishops & own
        if bishops & chess.BB_LIGHT_SQUARES and bishops & chess.BB_DARK_SQUARES:
            mg[color] += 30
            eg[color] += 40
        for kind in range(chess.KNIGHT, chess.KING + 1):
            squares = board.pieces_mask(kind, color)
            phase[color] += PHASE_WEIGHTS[kind] * squares.bit_count()
            mid_table, end_table = MG_TABLE[kind], EG_TABLE[kind]
            for sq in chess.scan_forward(squares):
                pst_sq = sq if color else sq ^ 56
                mg[color] += mid_table[pst_sq]
                eg[color] += end_table[pst_sq]
                if kind == chess.KING:
                    continue
                attacks = board.attacks_mask(sq)
                controlled[color] |= attacks
                mobility = (attacks & ~own & ~pawn_attacks[not color]).bit_count()
                activity = (mobility - MOBILITY_BASE[kind]) * MOBILITY_WEIGHTS[kind]
                mg[color] += activity
                eg[color] += activity
                if pawn_attacks[not color] & chess.BB_SQUARES[sq]:
                    trap = _piece_trap_penalty(board, sq, kind, color, attacks,
                                              pawn_attacks[not color])
                    mg[color] -= trap
                    eg[color] -= trap
                hits = (attacks & zones[not color]).bit_count()
                if hits:
                    pressure[not color] += ATTACK_WEIGHTS[kind] * hits
                    attackers[not color] += 1
                f, r = sq & 7, sq >> 3
                rank = r if color else 7 - r
                if kind == chess.KNIGHT:
                    if (3 <= rank <= 5 and pawn_attacks[color] & chess.BB_SQUARES[sq]
                            and not pawns[not color] & ADJACENT_FILES[sq] & FORWARD_RANKS[color][sq]):
                        mg[color] += 18
                        eg[color] += 12
                elif kind == chess.BISHOP:
                    square_color = (chess.BB_LIGHT_SQUARES if
                                    chess.BB_SQUARES[sq] & chess.BB_LIGHT_SQUARES
                                    else chess.BB_DARK_SQUARES)
                    locked = blocked[color] & square_color
                    penalty = 2 * locked.bit_count() + 3 * (locked & CENTRAL_FILES).bit_count()
                    mg[color] -= penalty
                    eg[color] -= penalty
                elif kind == chess.ROOK:
                    connected = (attacks & board.rooks & own).bit_count()
                    mg[color] += connected * 4
                    eg[color] += connected * 6
                    if not pawns[color] & chess.BB_FILES[f]:
                        bonus = 10 if pawns[not color] & chess.BB_FILES[f] else 20
                        mg[color] += bonus
                        eg[color] += bonus // 2
                    enemy_back = 7 if color else 0
                    enemy_seventh = 6 if color else 1
                    if rank == 6 and (pawns[not color] & chess.BB_RANKS[enemy_seventh]
                                      or (kings[not color] is not None
                                          and kings[not color] >> 3 == enemy_back)):
                        mg[color] += 20
                        eg[color] += 25

    for color in (chess.BLACK, chess.WHITE):
        enemy_color = not color
        victims = (board.knights | board.bishops | board.rooks | board.queens)
        victims &= board.occupied_co[enemy_color]
        candidates = pawns[color] & ((victims >> 7) & (victims >> 9) if color
                                      else (victims << 7) & (victims << 9))
        for sq in chess.scan_forward(candidates):
            targets = chess.BB_PAWN_ATTACKS[color][sq] & victims
            if targets.bit_count() < 2:
                continue
            if (not controlled[color] & chess.BB_SQUARES[sq]
                    and board.attackers_mask(enemy_color, sq) & targets):
                continue
            least_value = min(PIECE_VALUES[board.piece_type_at(target)]
                              for target in chess.scan_forward(targets))
            bonus = min(80, least_value // 4)
            mg[color] += bonus
            eg[color] += bonus
        for sq in chess.scan_forward(passed[color]):
            f, r = sq & 7, sq >> 3
            rank = r if color else 7 - r
            front = sq + (8 if color else -8)
            if not 0 <= front < 64:
                continue
            bonus = PASSED_PAWN_BONUS[rank]
            if board.occupied & chess.BB_SQUARES[front]:
                mg[color] -= bonus // 4
                eg[color] -= bonus // 3
            elif not pawn_attacks[not color] & chess.BB_SQUARES[front]:
                eg[color] += rank * 2
            if kings[color] is not None and kings[not color] is not None:
                distance = (SQUARE_DISTANCE[kings[not color]][front]
                            - SQUARE_DISTANCE[kings[color]][front])
                eg[color] += 4 * distance
            for rook in chess.scan_forward(board.rooks & chess.BB_FILES[f]):
                behind = rook < sq if color else rook > sq
                if behind and not chess.between(rook, sq) & board.occupied:
                    direction = 1 if board.occupied_co[color] & chess.BB_SQUARES[rook] else -1
                    mg[color] += direction * 8
                    eg[color] += direction * 16
        mg[color] -= _king_safety(board, color, kings[color], pawns,
                                 pressure[color], attackers[color], phase[not color])
        king = kings[color]
        if king is not None and attackers[color] >= 2:
            flights = (chess.BB_KING_ATTACKS[king] & ~board.occupied_co[color]
                       & ~controlled[not color]).bit_count()
            if flights <= 1:
                danger = (2 - flights) * min(4, attackers[color]) * 6
                mg[color] -= danger

    if not board.pawns and kings[0] is not None and kings[1] is not None:
        for color in (chess.BLACK, chess.WHITE):
            enemy = board.occupied_co[not color]
            if enemy == chess.BB_SQUARES[kings[not color]]:
                if (board.rooks | board.queens) & board.occupied_co[color]:
                    king = kings[not color]
                    f, r = king & 7, king >> 3
                    edge = 3 - min(f, 7 - f, r, 7 - r)
                    distance = SQUARE_DISTANCE[kings[color]][king]
                    eg[color] += 12 * edge + 8 * (7 - distance)

    game_phase = min(MAX_PHASE, sum(phase))
    blended = ((mg[chess.WHITE] - mg[chess.BLACK]) * game_phase
               + (eg[chess.WHITE] - eg[chess.BLACK]) * (MAX_PHASE - game_phase))
    score = (abs(blended) // MAX_PHASE) * (1 if blended >= 0 else -1)
    return score + (TEMPO_BONUS if board.turn == chess.WHITE else -TEMPO_BONUS)


_EVAL_CACHE_SIZE = 8192
_EVAL_CACHE: list[tuple[tuple[int, ...], int] | None] = [None] * _EVAL_CACHE_SIZE


def evaluate_board(board: chess.Board) -> int:
    key = (board.pawns, board.knights, board.bishops, board.rooks,
           board.queens, board.kings, board.occupied_co[chess.BLACK],
           board.occupied_co[chess.WHITE], board.promoted, int(board.turn))
    index = hash(key) & (_EVAL_CACHE_SIZE - 1)
    entry = _EVAL_CACHE[index]
    if entry is not None and entry[0] == key:
        return entry[1]
    score = _evaluate_position(board)
    _EVAL_CACHE[index] = (key, score)
    return score
