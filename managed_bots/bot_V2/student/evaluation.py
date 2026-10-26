"""Board evaluation heuristics with Tapered Evaluation (Middlegame + Endgame).

Evaluates a chess position by combining:
1. Material counting (piece values)
2. Piece-Square Tables (positional bonuses per square, per game phase)
3. Pawn structure (doubled & isolated pawn penalties)
4. Bishop pair bonus

The score is tapered between middlegame and endgame tables based on
remaining material (phase), so the King naturally hides in the corner
during the middlegame but advances to the center in the endgame.
"""
from __future__ import annotations

import chess

# ===========================================================================
# Constants
# ===========================================================================

MATE_SCORE = 100_000

# Piece base values in centipawns, indexed by piece_type (1=PAWN .. 6=KING)
_VAL_MG = (0, 100, 320, 330, 500, 900, 0)
_VAL_EG = (0, 120, 310, 340, 520, 950, 0)

# Phase weights – how much each piece contributes to "midgame-ness"
_PHASE_W = (0, 0, 1, 1, 2, 4, 0)
TOTAL_PHASE = 24  # 2 sides × (2×Knight + 2×Bishop + 2×Rook + 1×Queen)

# MVV-LVA ordering values (exported for use by search.py)
MVV_LVA = (0, 10, 30, 30, 50, 90, 100)

# Pawn structure constants
_DOUBLED_PENALTY = -15
_ISOLATED_PENALTY = -20
_BISHOP_PAIR_BONUS = 50

# Passed pawn bonuses by rank (index 0 = rank 1, index 6 = rank 7, from White's view)
_PASSED_BONUS_MG = (0, 5, 10, 20, 35, 60, 100)
_PASSED_BONUS_EG = (0, 10, 20, 40, 65, 100, 150)

# Rook on open / semi-open file bonuses (centipawns)
_ROOK_OPEN_FILE = 20
_ROOK_SEMI_OPEN_FILE = 10

# ===========================================================================
# Piece-Square Tables
# From White's perspective. Index 0 = a1, index 63 = h8.
# For Black pieces, the square is mirrored vertically (sq ^ 56).
# ===========================================================================
# fmt: off

_PAWN_MG = (
      0,   0,   0,   0,   0,   0,   0,   0,  # Rank 1
      5,  10,  10, -20, -20,  10,  10,   5,  # Rank 2: discourage blocking center pawns
      5,  -5, -10,   0,   0, -10,  -5,   5,  # Rank 3
      0,   0,   0,  20,  20,   0,   0,   0,  # Rank 4: reward center control
      5,   5,  10,  25,  25,  10,   5,   5,  # Rank 5
     10,  10,  20,  30,  30,  20,  10,  10,  # Rank 6
     50,  50,  50,  50,  50,  50,  50,  50,  # Rank 7: about to promote
      0,   0,   0,   0,   0,   0,   0,   0,  # Rank 8
)
_PAWN_EG = (
      0,   0,   0,   0,   0,   0,   0,   0,
     10,  10,  10,  10,  10,  10,  10,  10,
     10,  10,  10,  10,  10,  10,  10,  10,
     20,  20,  20,  20,  20,  20,  20,  20,
     30,  30,  30,  30,  30,  30,  30,  30,
     50,  50,  50,  50,  50,  50,  50,  50,
     80,  80,  80,  80,  80,  80,  80,  80,
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
      5,  10,  10,  10,  10,  10,  10,   5,  # Rank 7: rook on 7th is powerful
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
     20,  30,  10,   0,   0,  10,  30,  20,  # Rank 1: reward castled positions
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
)
_KING_EG = (
    -50, -30, -30, -30, -30, -30, -30, -50,  # Rank 1: corner is bad in endgame
    -30, -30,   0,   0,   0,   0, -30, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,  # Center is ideal
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
)

# fmt: on

# ===========================================================================
# Pre-computed lookup tables: MATERIAL + PST combined
# MG_W[piece_type][square] = material_value + positional_bonus (White)
# MG_B[piece_type][square] = material_value + positional_bonus (Black)
# Black uses mirrored square: sq ^ 56  (vertical flip)
# ===========================================================================

def _build():
    """Build pre-computed evaluation tables at import time."""
    mg_pst = {1: _PAWN_MG, 2: _KNIGHT_MG, 3: _BISHOP_MG,
              4: _ROOK_MG, 5: _QUEEN_MG, 6: _KING_MG}
    # Endgame: only Pawn and King have different tables
    eg_pst = {1: _PAWN_EG, 2: _KNIGHT_MG, 3: _BISHOP_MG,
              4: _ROOK_MG, 5: _QUEEN_MG, 6: _KING_EG}

    mg_w = [None] * 7  # index 0 unused
    eg_w = [None] * 7
    mg_b = [None] * 7
    eg_b = [None] * 7

    for pt in range(1, 7):
        mg_w[pt] = tuple(_VAL_MG[pt] + mg_pst[pt][sq] for sq in range(64))
        eg_w[pt] = tuple(_VAL_EG[pt] + eg_pst[pt][sq] for sq in range(64))
        mg_b[pt] = tuple(_VAL_MG[pt] + mg_pst[pt][sq ^ 56] for sq in range(64))
        eg_b[pt] = tuple(_VAL_EG[pt] + eg_pst[pt][sq ^ 56] for sq in range(64))

    return tuple(mg_w), tuple(eg_w), tuple(mg_b), tuple(eg_b)


MG_W, EG_W, MG_B, EG_B = _build()


# ===========================================================================
# Main evaluation function
# ===========================================================================

def evaluate(board: chess.Board) -> int:
    """Evaluate the position from the perspective of the side to move.

    Returns a positive score if the side to move has an advantage.
    Does NOT detect checkmate/stalemate (handled by search).
    """
    score_mg = 0
    score_eg = 0
    phase = 0

    wp = [0] * 8   # White pawn count per file
    bp = [0] * 8   # Black pawn count per file
    wb = 0          # White bishop count
    bb = 0          # Black bishop count

    # --- Main piece loop: Material + PST ---
    for pt in range(chess.PAWN, chess.KING + 1):
        pw = _PHASE_W[pt]
        mg_wt = MG_W[pt]; eg_wt = EG_W[pt]
        mg_bt = MG_B[pt]; eg_bt = EG_B[pt]

        w_set = board.pieces(pt, chess.WHITE)
        b_set = board.pieces(pt, chess.BLACK)

        for sq in w_set:
            score_mg += mg_wt[sq]
            score_eg += eg_wt[sq]
        for sq in b_set:
            score_mg -= mg_bt[sq]
            score_eg -= eg_bt[sq]

        phase += (len(w_set) + len(b_set)) * pw

        # Track pawns per file for structure analysis
        if pt == chess.PAWN:
            for sq in w_set:
                wp[sq & 7] += 1
            for sq in b_set:
                bp[sq & 7] += 1
        elif pt == chess.BISHOP:
            wb = len(w_set)
            bb = len(b_set)

    # --- Pawn structure ---
    pawn_adj = 0
    for f in range(8):
        # Doubled pawns: multiple pawns on the same file block each other
        if wp[f] > 1:
            pawn_adj += _DOUBLED_PENALTY * (wp[f] - 1)
        if bp[f] > 1:
            pawn_adj -= _DOUBLED_PENALTY * (bp[f] - 1)

        # Isolated pawns: no friendly pawn on adjacent files to protect
        adj_l = wp[f - 1] if f > 0 else 0
        adj_r = wp[f + 1] if f < 7 else 0
        if wp[f] > 0 and adj_l == 0 and adj_r == 0:
            pawn_adj += _ISOLATED_PENALTY

        adj_l = bp[f - 1] if f > 0 else 0
        adj_r = bp[f + 1] if f < 7 else 0
        if bp[f] > 0 and adj_l == 0 and adj_r == 0:
            pawn_adj -= _ISOLATED_PENALTY

    # --- Bishop pair: two bishops cover both color complexes ---
    if wb >= 2:
        pawn_adj += _BISHOP_PAIR_BONUS
    if bb >= 2:
        pawn_adj -= _BISHOP_PAIR_BONUS

    # --- Passed pawns ---
    pp_mg = 0
    pp_eg = 0

    # Build pawn position lists for O(Pw * Pb) passed-pawn check
    black_pawn_pos = [(sq & 7, sq >> 3) for sq in board.pieces(chess.PAWN, chess.BLACK)]
    white_pawn_pos = [(sq & 7, sq >> 3) for sq in board.pieces(chess.PAWN, chess.WHITE)]

    for f, r in white_pawn_pos:
        is_passed = True
        for bf, br in black_pawn_pos:
            if abs(bf - f) <= 1 and br > r:
                is_passed = False
                break
        if is_passed:
            pp_mg += _PASSED_BONUS_MG[r]
            pp_eg += _PASSED_BONUS_EG[r]

    for f, r in black_pawn_pos:
        is_passed = True
        for wf, wr in white_pawn_pos:
            if abs(wf - f) <= 1 and wr < r:
                is_passed = False
                break
        if is_passed:
            rr = 7 - r  # flip rank for bonus lookup
            pp_mg -= _PASSED_BONUS_MG[rr]
            pp_eg -= _PASSED_BONUS_EG[rr]

    # --- Rook file bonuses ---
    rook_adj = 0
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        f = sq & 7
        if wp[f] == 0:
            if bp[f] == 0:
                rook_adj += _ROOK_OPEN_FILE      # Open file
            else:
                rook_adj += _ROOK_SEMI_OPEN_FILE  # Semi-open

    for sq in board.pieces(chess.ROOK, chess.BLACK):
        f = sq & 7
        if bp[f] == 0:
            if wp[f] == 0:
                rook_adj -= _ROOK_OPEN_FILE
            else:
                rook_adj -= _ROOK_SEMI_OPEN_FILE

    # --- Tapered evaluation: blend MG and EG scores by game phase ---
    phase = min(phase, TOTAL_PHASE)
    score_mg += pp_mg
    score_eg += pp_eg
    score = (score_mg * phase + score_eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    score += pawn_adj + rook_adj

    # Return from the perspective of the side to move
    return score if board.turn else -score
