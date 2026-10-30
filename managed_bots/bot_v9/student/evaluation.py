"""Board evaluation heuristics with Tapered Evaluation (Middlegame + Endgame).

V9 Grandmaster Engine:
1. Material counting + 64-square Piece-Square Tables (PST) per game phase.
2. Pawn structure: Doubled and Isolated pawn penalties.
3. Minor Piece Safe Mobility: Rewards safe squares not controlled by enemy pawns.
4. Non-linear King Safety & Shelter (Quadratic Danger Formula):
   - Non-linear penalty on coordinated enemy piece attacks around King zone.
   - Friendly pawn shelter distance penalties when pawn cover is breached/pushed.
   - Open and semi-open file penalties facing the King.
5. Early Queen development penalty in opening when minors are undeveloped.
6. Chess960 Corner Bishop development & anti-trap bitmasks.
7. Passed Pawns with Blocker Detection & Rule of the Square in Endgame.
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
_KING_EG = (
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
)


def _build():
    mg_pst = {1: _PAWN_MG, 2: _KNIGHT_MG, 3: _BISHOP_MG, 4: _ROOK_MG, 5: _QUEEN_MG, 6: _KING_MG}
    eg_pst = {1: _PAWN_MG, 2: _KNIGHT_MG, 3: _BISHOP_MG, 4: _ROOK_MG, 5: _QUEEN_MG, 6: _KING_EG}
    mg_w, eg_w, mg_b, eg_b = [None]*7, [None]*7, [None]*7, [None]*7
    for pt in range(1, 7):
        mg_w[pt] = tuple(PIECE_VALUES[pt] + mg_pst[pt][sq] for sq in range(64))
        eg_w[pt] = tuple(PIECE_VALUES[pt] + eg_pst[pt][sq] for sq in range(64))
        mg_b[pt] = tuple(PIECE_VALUES[pt] + mg_pst[pt][sq ^ 56] for sq in range(64))
        eg_b[pt] = tuple(PIECE_VALUES[pt] + eg_pst[pt][sq ^ 56] for sq in range(64))
    return tuple(mg_w), tuple(eg_w), tuple(mg_b), tuple(eg_b)


MG_W, EG_W, MG_B, EG_B = _build()


def evaluate(board: chess.Board) -> int:
    score_mg = 0
    score_eg = 0
    phase = 0

    wp_bb = int(board.pawns & board.occupied_co[chess.WHITE])
    bp_bb = int(board.pawns & board.occupied_co[chess.BLACK])

    w_occ = int(board.occupied_co[chess.WHITE])
    b_occ = int(board.occupied_co[chess.BLACK])
    all_occ = w_occ | b_occ

    wp_files = [0] * 8
    bp_files = [0] * 8
    w_pawn_sqs: list[int] = []
    b_pawn_sqs: list[int] = []
    w_minors: list[tuple[int, int]] = []
    b_minors: list[tuple[int, int]] = []
    w_rooks: list[int] = []
    b_rooks: list[int] = []
    w_bishops_count = 0
    b_bishops_count = 0
    w_king_sq: int | None = None
    b_king_sq: int | None = None

    # Pawn attack bitboards (efficient bitwise shifts)
    w_pawn_attacks = ((wp_bb & ~0x0101010101010101) << 7) | ((wp_bb & ~0x8080808080808080) << 9)
    b_pawn_attacks = ((bp_bb & ~0x0101010101010101) >> 9) | ((bp_bb & ~0x8080808080808080) >> 7)

    # 1. Material + PST
    for pt in range(1, 7):
        pw = PHASE_WEIGHTS[pt]
        mg_wt, eg_wt = MG_W[pt], EG_W[pt]
        mg_bt, eg_bt = MG_B[pt], EG_B[pt]

        w_sqs = board.pieces(pt, chess.WHITE)
        b_sqs = board.pieces(pt, chess.BLACK)

        for sq in w_sqs:
            score_mg += mg_wt[sq]
            score_eg += eg_wt[sq]
        for sq in b_sqs:
            score_mg -= mg_bt[sq]
            score_eg -= eg_bt[sq]

        phase += (len(w_sqs) + len(b_sqs)) * pw

        if pt == 1:
            w_pawn_sqs = list(w_sqs)
            b_pawn_sqs = list(b_sqs)
            for sq in w_pawn_sqs:
                wp_files[sq & 7] += 1
            for sq in b_pawn_sqs:
                bp_files[sq & 7] += 1
        elif pt in (2, 3):
            if pt == 3:
                w_bishops_count = len(w_sqs)
                b_bishops_count = len(b_sqs)
            for sq in w_sqs:
                w_minors.append((sq, pt))
            for sq in b_sqs:
                b_minors.append((sq, pt))
        elif pt == 4:
            w_rooks = list(w_sqs)
            b_rooks = list(b_sqs)
        elif pt == 6:
            for sq in w_sqs:
                w_king_sq = sq
            for sq in b_sqs:
                b_king_sq = sq

    # 2. Pawn structure
    pawn_adj = 0
    for f in range(8):
        if wp_files[f] > 1:
            pawn_adj += _DOUBLED_PENALTY * (wp_files[f] - 1)
        if bp_files[f] > 1:
            pawn_adj -= _DOUBLED_PENALTY * (bp_files[f] - 1)
        # Isolated
        l_wp = wp_files[f-1] if f > 0 else 0
        r_wp = wp_files[f+1] if f < 7 else 0
        if wp_files[f] > 0 and l_wp == 0 and r_wp == 0:
            pawn_adj += _ISOLATED_PENALTY
        l_bp = bp_files[f-1] if f > 0 else 0
        r_bp = bp_files[f+1] if f < 7 else 0
        if bp_files[f] > 0 and l_bp == 0 and r_bp == 0:
            pawn_adj -= _ISOLATED_PENALTY

    # Bishop Pair Bonus
    if w_bishops_count >= 2:
        pawn_adj += _BISHOP_PAIR_BONUS
    if b_bishops_count >= 2:
        pawn_adj -= _BISHOP_PAIR_BONUS

    # 3. Minor Piece Safe Mobility
    w_undeveloped = 0
    for sq, pt in w_minors:
        if (sq >> 3) == 0:
            w_undeveloped += 1
        attacks = chess.BB_KNIGHT_ATTACKS[sq] if pt == 2 else board.attacks_mask(sq)
        safe_mob = (attacks & ~w_occ & ~b_pawn_attacks).bit_count()
        score_mg += (safe_mob - 4) * 4

    b_undeveloped = 0
    for sq, pt in b_minors:
        if (sq >> 3) == 7:
            b_undeveloped += 1
        attacks = chess.BB_KNIGHT_ATTACKS[sq] if pt == 2 else board.attacks_mask(sq)
        safe_mob = (attacks & ~b_occ & ~w_pawn_attacks).bit_count()
        score_mg -= (safe_mob - 4) * 4

    # Early Queen Development penalty (if 2+ minors undeveloped in opening)
    if phase >= 18:
        if w_undeveloped >= 2:
            for q_sq in board.pieces(chess.QUEEN, chess.WHITE):
                if (q_sq >> 3) > 1:
                    score_mg -= 10 * w_undeveloped
        if b_undeveloped >= 2:
            for q_sq in board.pieces(chess.QUEEN, chess.BLACK):
                if (q_sq >> 3) < 6:
                    score_mg += 10 * b_undeveloped

    # 4. Chess960 Corner Bishop anti-trap
    if phase > 4:
        for sq, pt in w_minors:
            if pt == 3 and sq < 8:
                mask = _W_BISHOP_TRAP[sq]
                if (wp_bb & mask) == mask:
                    score_mg += _BISHOP_TRAPPED_PENALTY
                elif not (wp_bb & mask):
                    score_mg += _BISHOP_OPEN_BONUS
        for sq, pt in b_minors:
            if pt == 3 and sq >= 56:
                mask = _B_BISHOP_TRAP[sq - 56]
                if (bp_bb & mask) == mask:
                    score_mg -= _BISHOP_TRAPPED_PENALTY
                elif not (bp_bb & mask):
                    score_mg -= _BISHOP_OPEN_BONUS

    # 5. Non-linear King Safety & Shelter (Quadratic Danger Formula)
    if phase > 4:
        # White King
        if w_king_sq is not None:
            kw_f = w_king_sq & 7
            kw_r = w_king_sq >> 3
            kw_zone = chess.BB_KING_ATTACKS[w_king_sq] | chess.BB_SQUARES[w_king_sq]
            pressure = 0
            attackers = 0
            for sq in board.pieces(chess.KNIGHT, chess.BLACK):
                hits = (chess.BB_KNIGHT_ATTACKS[sq] & kw_zone).bit_count()
                if hits:
                    attackers += 1; pressure += hits * 2
            for sq in board.pieces(chess.BISHOP, chess.BLACK):
                hits = (board.attacks_mask(sq) & kw_zone).bit_count()
                if hits:
                    attackers += 1; pressure += hits * 2
            for sq in board.pieces(chess.ROOK, chess.BLACK):
                hits = (board.attacks_mask(sq) & kw_zone).bit_count()
                if hits:
                    attackers += 1; pressure += hits * 3
            for sq in board.pieces(chess.QUEEN, chess.BLACK):
                hits = (board.attacks_mask(sq) & kw_zone).bit_count()
                if hits:
                    attackers += 1; pressure += hits * 5

            danger = min(350, (pressure * pressure) // 3 + max(0, attackers - 1) * 12)
            if not (board.queens & board.occupied_co[chess.BLACK]):
                danger = (danger * 2) // 3

            # Friendly pawn shelter & open files near King
            shelter = 0
            for f in range(max(0, kw_f - 1), min(8, kw_f + 2)):
                nearest = min(((sq >> 3) - kw_r for sq in w_pawn_sqs
                               if (sq & 7) == f and (sq >> 3) > kw_r), default=8)
                shelter += {1: 20, 2: 8}.get(nearest, -20)
                if nearest == 8:
                    if bp_files[f] == 0:
                        shelter -= 12 # Open file near king
                    if (b_occ & (board.rooks | board.queens) & chess.BB_FILES[f]):
                        shelter -= 15 # Enemy heavy piece facing king

            score_mg += (shelter - danger) * phase // 24

        # Black King
        if b_king_sq is not None:
            kb_f = b_king_sq & 7
            kb_r = b_king_sq >> 3
            kb_zone = chess.BB_KING_ATTACKS[b_king_sq] | chess.BB_SQUARES[b_king_sq]
            pressure = 0
            attackers = 0
            for sq in board.pieces(chess.KNIGHT, chess.WHITE):
                hits = (chess.BB_KNIGHT_ATTACKS[sq] & kb_zone).bit_count()
                if hits:
                    attackers += 1; pressure += hits * 2
            for sq in board.pieces(chess.BISHOP, chess.WHITE):
                hits = (board.attacks_mask(sq) & kb_zone).bit_count()
                if hits:
                    attackers += 1; pressure += hits * 2
            for sq in board.pieces(chess.ROOK, chess.WHITE):
                hits = (board.attacks_mask(sq) & kb_zone).bit_count()
                if hits:
                    attackers += 1; pressure += hits * 3
            for sq in board.pieces(chess.QUEEN, chess.WHITE):
                hits = (board.attacks_mask(sq) & kb_zone).bit_count()
                if hits:
                    attackers += 1; pressure += hits * 5

            danger = min(350, (pressure * pressure) // 3 + max(0, attackers - 1) * 12)
            if not (board.queens & board.occupied_co[chess.WHITE]):
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

    # 6. Passed Pawns & Rule of the square in endgame
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
                        bonus_eg += 180  # Unstoppable queen

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
                        bonus_eg += 180

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

    # 8. Tapered evaluation blending
    phase = min(phase, TOTAL_PHASE)
    score_mg += pp_mg
    score_eg += pp_eg
    score = (score_mg * phase + score_eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    score += pawn_adj + rook_adj

    return score if board.turn else -score
