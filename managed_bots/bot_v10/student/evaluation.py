"""Board evaluation heuristics with Tapered Evaluation (Middlegame + Endgame).

Bot Test (Ultimate Hybrid Engine):
1. Material counting + 64-square Piece-Square Tables (PST) per game phase.
2. Pawn structure: Doubled and Isolated pawn penalties.
3. Minor Piece Safe Mobility: Rewards safe squares not controlled by enemy pawns.
4. Non-linear King Safety & Shelter (Quadratic Danger Formula):
   - Non-linear penalty on coordinated enemy piece attacks around King zone.
   - Friendly pawn shelter distance penalties when pawn cover is breached/pushed.
   - Open and semi-open file penalties facing the King.
5. Chess960 Corner Bishop development & anti-trap bitmasks.
6. Passed Pawns with Blocker Detection & Rule of the Square in Endgame.
7. Mop-Up Evaluation in winning endgames: Drives lone enemy King to edge/corner and brings own King closer.
8. Rook open and semi-open file bonuses.
"""
from __future__ import annotations

import chess

MATE_SCORE = 100_000

# Base piece values
PIECE_VALUES = (0, 100, 320, 330, 500, 900, 0)
PHASE_WEIGHTS = (0, 0, 1, 1, 2, 4, 0)
ATTACK_WEIGHTS = (0, 1, 2, 2, 3, 5, 0)
TOTAL_PHASE = 24

MVV_LVA = (0, 10, 30, 30, 50, 90, 100)

_DOUBLED_PENALTY = -15
_ISOLATED_PENALTY = -15
_BISHOP_PAIR_BONUS = 35

_PASSED_BONUS_MG = (0, 5, 10, 20, 35, 60, 100)
_PASSED_BONUS_EG = (0, 10, 20, 40, 65, 100, 150)

_ROOK_OPEN_FILE = 20
_ROOK_SEMI_OPEN_FILE = 10

# Chess960 Corner Bishop Trap Masks
_W_BISHOP_TRAP = (
    1 << 9, (1 << 8) | (1 << 10), (1 << 9) | (1 << 11), (1 << 10) | (1 << 12),
    (1 << 11) | (1 << 13), (1 << 12) | (1 << 14), (1 << 13) | (1 << 15), 1 << 14
)
_B_BISHOP_TRAP = (
    1 << 49, (1 << 48) | (1 << 50), (1 << 49) | (1 << 51), (1 << 50) | (1 << 52),
    (1 << 51) | (1 << 53), (1 << 52) | (1 << 54), (1 << 53) | (1 << 55), 1 << 54
)

_BISHOP_TRAPPED_PENALTY = -25
_BISHOP_OPEN_BONUS = 15

# Piece-Square Tables (White perspective)
_PAWN_MG = (
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_KNIGHT_MG = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)
_BISHOP_MG = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)
_ROOK_MG = (
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_QUEEN_MG = (
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
)
_KING_MG = (
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
)

_PAWN_EG = (
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
     10,  10,  10,  10,  10,  10,  10,  10,
     20,  20,  20,  25,  25,  20,  20,  20,
     30,  30,  30,  35,  35,  30,  30,  30,
     50,  50,  50,  55,  55,  50,  50,  50,
     80,  80,  80,  80,  80,  80,  80,  80,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_KNIGHT_EG = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,  -5,   0,   0,   0,   0,  -5, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,  -5,   0,   0,   0,   0,  -5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)
_BISHOP_EG = (
    -10,  -5,  -5,  -5,  -5,  -5,  -5, -10,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
    -10,  -5,  -5,  -5,  -5,  -5,  -5, -10,
)
_ROOK_EG = (
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_QUEEN_EG = (
    -10,  -5,  -5,  -5,  -5,  -5,  -5, -10,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
    -10,  -5,  -5,  -5,  -5,  -5,  -5, -10,
)
_KING_EG = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   5,  10,  15,  15,  10,   5, -10,
    -10,   5,  15,  20,  20,  15,   5, -10,
    -10,   5,  15,  20,  20,  15,   5, -10,
    -10,   5,  10,  15,  15,  10,   5, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)

_PST_MG = (None, _PAWN_MG, _KNIGHT_MG, _BISHOP_MG, _ROOK_MG, _QUEEN_MG, _KING_MG)
_PST_EG = (None, _PAWN_EG, _KNIGHT_EG, _BISHOP_EG, _ROOK_EG, _QUEEN_EG, _KING_EG)

_MOBILITY_WEIGHT = (0, 0, 4, 3, 2, 1, 0)


def evaluate(board: chess.Board) -> int:
    """Static evaluation of the board from the active player's perspective."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
        return 0

    score_mg = 0
    score_eg = 0
    phase = 0

    w_occ = board.occupied_co[chess.WHITE]
    b_occ = board.occupied_co[chess.BLACK]
    all_occ = w_occ | b_occ

    w_pawn_sqs = []
    b_pawn_sqs = []
    wp_files = [0] * 8
    bp_files = [0] * 8

    w_rooks = []
    b_rooks = []

    w_minors = []
    b_minors = []

    w_bishops = 0
    b_bishops = 0

    w_king_sq = None
    b_king_sq = None

    # Pawn attacks bitmasks for safe mobility
    w_pawn_attacks = 0
    b_pawn_attacks = 0

    # 1. Piece scanning and material + PST accumulation
    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        c = piece.color

        val = PIECE_VALUES[pt]
        pw = PHASE_WEIGHTS[pt]

        if c == chess.WHITE:
            score_mg += val + _PST_MG[pt][sq]
            score_eg += val + _PST_EG[pt][sq]
            phase += pw
            if pt == chess.PAWN:
                w_pawn_sqs.append(sq)
                wp_files[sq & 7] += 1
                w_pawn_attacks |= board.attacks_mask(sq)
            elif pt == chess.BISHOP:
                w_bishops += 1
                w_minors.append(sq)
                if (1 << sq) & 0xFF:
                    f = sq & 7
                    mask = _W_BISHOP_TRAP[f]
                    if (mask & w_occ) == mask:
                        score_mg += _BISHOP_TRAPPED_PENALTY
                    elif not (mask & all_occ):
                        score_mg += _BISHOP_OPEN_BONUS
            elif pt == chess.KNIGHT:
                w_minors.append(sq)
            elif pt == chess.ROOK:
                w_rooks.append(sq)
            elif pt == chess.KING:
                w_king_sq = sq
        else:
            sq_flip = sq ^ 56
            score_mg -= val + _PST_MG[pt][sq_flip]
            score_eg -= val + _PST_EG[pt][sq_flip]
            phase += pw
            if pt == chess.PAWN:
                b_pawn_sqs.append(sq)
                bp_files[sq & 7] += 1
                b_pawn_attacks |= board.attacks_mask(sq)
            elif pt == chess.BISHOP:
                b_bishops += 1
                b_minors.append(sq)
                if (1 << sq) & (0xFF << 56):
                    f = sq & 7
                    mask = _B_BISHOP_TRAP[f]
                    if (mask & b_occ) == mask:
                        score_mg -= _BISHOP_TRAPPED_PENALTY
                    elif not (mask & all_occ):
                        score_mg -= _BISHOP_OPEN_BONUS
            elif pt == chess.KNIGHT:
                b_minors.append(sq)
            elif pt == chess.ROOK:
                b_rooks.append(sq)
            elif pt == chess.KING:
                b_king_sq = sq

    # 2. Bishop Pair Bonus
    if w_bishops >= 2:
        score_mg += _BISHOP_PAIR_BONUS
        score_eg += _BISHOP_PAIR_BONUS
    if b_bishops >= 2:
        score_mg -= _BISHOP_PAIR_BONUS
        score_eg -= _BISHOP_PAIR_BONUS

    # 3. Pawn Structure: Doubled & Isolated pawns
    pawn_adj = 0
    for f in range(8):
        wc = wp_files[f]
        if wc > 1:
            pawn_adj += _DOUBLED_PENALTY * (wc - 1)
        if wc > 0:
            left = wp_files[f - 1] if f > 0 else 0
            right = wp_files[f + 1] if f < 7 else 0
            if left == 0 and right == 0:
                pawn_adj += _ISOLATED_PENALTY * wc

        bc = bp_files[f]
        if bc > 1:
            pawn_adj -= _DOUBLED_PENALTY * (bc - 1)
        if bc > 0:
            left = bp_files[f - 1] if f > 0 else 0
            right = bp_files[f + 1] if f < 7 else 0
            if left == 0 and right == 0:
                pawn_adj -= _ISOLATED_PENALTY * bc

    # 4. Safe Minor Mobility (only squares not attacked by enemy pawns)
    w_safe = ~(w_occ | b_pawn_attacks)
    b_safe = ~(b_occ | w_pawn_attacks)

    w_mob = 0
    for sq in w_minors:
        w_mob += (board.attacks_mask(sq) & w_safe).bit_count() * 3
    for sq in w_rooks:
        w_mob += (board.attacks_mask(sq) & w_safe).bit_count() * 2

    b_mob = 0
    for sq in b_minors:
        b_mob += (board.attacks_mask(sq) & b_safe).bit_count() * 3
    for sq in b_rooks:
        b_mob += (board.attacks_mask(sq) & b_safe).bit_count() * 2

    score_mg += (w_mob - b_mob)

    # 5. Quadratic King Safety & Shelter (Middlegame)
    if phase >= 8:
        if w_king_sq is not None:
            kw_f, kw_r = w_king_sq & 7, w_king_sq >> 3
            kw_zone = board.attacks_mask(w_king_sq) | (1 << w_king_sq)
            if kw_r < 7:
                kw_zone |= (kw_zone << 8)

            attackers = 0
            pressure = 0
            for sq in b_minors:
                att = board.attacks_mask(sq) & kw_zone
                if att:
                    attackers += 1; pressure += att.bit_count() * 2
            for sq in b_rooks:
                att = board.attacks_mask(sq) & kw_zone
                if att:
                    attackers += 1; pressure += att.bit_count() * 3
            for sq in board.pieces(chess.QUEEN, chess.BLACK):
                att = board.attacks_mask(sq) & kw_zone
                if att:
                    attackers += 1; pressure += att.bit_count() * 5

            danger = 0
            if attackers >= 2:
                danger = (pressure * pressure) // 3 + (attackers - 1) * 12
            elif attackers == 1:
                danger = (pressure * 2)

            # King wedge pawn protection
            if kw_r <= 2 and (b_pawn_attacks & (1 << w_king_sq)):
                danger += 40

            if not board.pieces(chess.QUEEN, chess.BLACK):
                danger = (danger * 2) // 3

            # Shelter evaluation
            shelter = 0
            for f in range(max(0, kw_f - 1), min(8, kw_f + 2)):
                nearest = min(((sq >> 3) - kw_r for sq in w_pawn_sqs
                               if (sq & 7) == f and (sq >> 3) > kw_r), default=8)
                shelter += {1: 20, 2: 8}.get(nearest, -20)
                if nearest == 8:
                    if bp_files[f] == 0:
                        shelter -= 12
                    if (b_occ & (board.rooks | board.queens) & chess.BB_FILES[f]):
                        shelter -= 15

            score_mg += (shelter - danger) * phase // 24

        if b_king_sq is not None:
            kb_f, kb_r = b_king_sq & 7, b_king_sq >> 3
            kb_zone = board.attacks_mask(b_king_sq) | (1 << b_king_sq)
            if kb_r > 0:
                kb_zone |= (kb_zone >> 8)

            attackers = 0
            pressure = 0
            for sq in w_minors:
                att = board.attacks_mask(sq) & kb_zone
                if att:
                    attackers += 1; pressure += att.bit_count() * 2
            for sq in w_rooks:
                att = board.attacks_mask(sq) & kb_zone
                if att:
                    attackers += 1; pressure += att.bit_count() * 3
            for sq in board.pieces(chess.QUEEN, chess.WHITE):
                att = board.attacks_mask(sq) & kb_zone
                if att:
                    attackers += 1; pressure += att.bit_count() * 5

            danger = 0
            if attackers >= 2:
                danger = (pressure * pressure) // 3 + (attackers - 1) * 12
            elif attackers == 1:
                danger = (pressure * 2)

            if kb_r >= 5 and (w_pawn_attacks & (1 << b_king_sq)):
                danger += 40

            if not board.pieces(chess.QUEEN, chess.WHITE):
                danger = (danger * 2) // 3

            shelter = 0
            for f in range(max(0, kb_f - 1), min(8, kb_f + 2)):
                nearest = min((kb_r - (sq >> 3) for sq in b_pawn_sqs
                               if (sq & 7) == f and (sq >> 3) < kb_r), default=8)
                shelter += {1: 20, 2: 8}.get(nearest, -20)
                if nearest == 8:
                    if wp_files[f] == 0:
                        shelter -= 12
                    if (w_occ & (board.rooks | board.queens) & chess.BB_FILES[f]):
                        shelter -= 15

            score_mg -= (shelter - danger) * phase // 24

    # 6. Passed Pawns & Rule of the Square in Endgame
    pp_mg = 0
    pp_eg = 0
    enemy_nonpawns_w = b_occ & ~(board.pawns | board.kings)
    enemy_nonpawns_b = w_occ & ~(board.pawns | board.kings)

    for sq in w_pawn_sqs:
        f, r = sq & 7, sq >> 3
        is_passed = True
        for bsq in b_pawn_sqs:
            bf, br = bsq & 7, bsq >> 3
            if abs(bf - f) <= 1 and br > r:
                is_passed = False
                break
        if is_passed:
            bonus_mg = _PASSED_BONUS_MG[r]
            bonus_eg = _PASSED_BONUS_EG[r]
            ahead = sq + 8
            if ahead < 64 and (all_occ & (1 << ahead)):
                bonus_mg //= 3; bonus_eg //= 3

            # King proximity in endgame
            if b_king_sq is not None and w_king_sq is not None:
                prom_sq = (7 << 3) | f
                d_enemy = chess.square_distance(b_king_sq, prom_sq)
                d_own = chess.square_distance(w_king_sq, prom_sq)
                bonus_eg += (d_enemy - d_own) * r * 2

                # Rule of the square in pure pawn endgame
                if not enemy_nonpawns_w and r >= 3:
                    steps = 7 - r
                    tempo = int(board.turn != chess.WHITE)
                    if d_enemy > steps + tempo:
                        bonus_eg += 180  # Unstoppable promotion

            pp_mg += bonus_mg
            pp_eg += bonus_eg

    for sq in b_pawn_sqs:
        f, r = sq & 7, sq >> 3
        is_passed = True
        for wsq in w_pawn_sqs:
            wf, wr = wsq & 7, wsq >> 3
            if abs(wf - f) <= 1 and wr < r:
                is_passed = False
                break
        if is_passed:
            rr = 7 - r
            bonus_mg = _PASSED_BONUS_MG[rr]
            bonus_eg = _PASSED_BONUS_EG[rr]
            ahead = sq - 8
            if ahead >= 0 and (all_occ & (1 << ahead)):
                bonus_mg //= 3; bonus_eg //= 3

            if w_king_sq is not None and b_king_sq is not None:
                prom_sq = f
                d_enemy = chess.square_distance(w_king_sq, prom_sq)
                d_own = chess.square_distance(b_king_sq, prom_sq)
                bonus_eg += (d_enemy - d_own) * rr * 2

                if not enemy_nonpawns_b and rr >= 3:
                    steps = 7 - rr
                    tempo = int(board.turn != chess.BLACK)
                    if d_enemy > steps + tempo:
                        bonus_eg += 180  # Unstoppable promotion

            pp_mg -= bonus_mg
            pp_eg -= bonus_eg

    # 7. Rook open/semi-open files
    rook_adj = 0
    for sq in w_rooks:
        f = sq & 7
        if wp_files[f] == 0:
            rook_adj += _ROOK_OPEN_FILE if bp_files[f] == 0 else _ROOK_SEMI_OPEN_FILE
    for sq in b_rooks:
        f = sq & 7
        if bp_files[f] == 0:
            rook_adj -= _ROOK_OPEN_FILE if wp_files[f] == 0 else _ROOK_SEMI_OPEN_FILE

    # 8. Mop-Up Evaluation in winning endgames
    # If one side has significant material advantage and the opponent has a lone king
    # (or no heavy pieces), drive enemy king to the nearest edge/corner and bring own king close.
    mopup_eg = 0
    if w_king_sq is not None and b_king_sq is not None:
        if score_eg > 350 and not enemy_nonpawns_w:
            bk_f, bk_r = b_king_sq & 7, b_king_sq >> 3
            center_dist = max(3 - bk_f, bk_f - 4) + max(3 - bk_r, bk_r - 4)
            king_dist = chess.square_distance(w_king_sq, b_king_sq)
            mopup_eg += center_dist * 10 + (7 - king_dist) * 6
        elif score_eg < -350 and not enemy_nonpawns_b:
            wk_f, wk_r = w_king_sq & 7, w_king_sq >> 3
            center_dist = max(3 - wk_f, wk_f - 4) + max(3 - wk_r, wk_r - 4)
            king_dist = chess.square_distance(w_king_sq, b_king_sq)
            mopup_eg -= center_dist * 10 + (7 - king_dist) * 6

    # 9. Tapered evaluation blending
    phase = min(phase, TOTAL_PHASE)
    score_mg += pp_mg
    score_eg += pp_eg + mopup_eg
    score = (score_mg * phase + score_eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    score += pawn_adj + rook_adj

    return score if board.turn == chess.WHITE else -score

