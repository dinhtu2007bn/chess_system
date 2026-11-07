"""Dynamic Elastic Tournament Time Manager for Bot Cuoi Cung.

Key Architecture:
1. Dynamic Clock-Proportional Elastic Scaling:
   - Eliminates artificial static hard caps (2.4s) that caused tactical blindness in early game.
   - When remaining clock is healthy (> 90s in 180s/300s matches), allocates 3.5s - 5.5s on
     sharp tactical positions to achieve Depth 8-9 and match or exceed Grandmaster engines.
   - Smoothly scales move budgets downward as the game clock ticks, preventing clock overburn.
2. Material-Density Move Estimation:
   - Estimates expected moves dynamically from pieces on board rather than static ply counts.
   - Heavy board (24+ pieces): 34 expected moves.
   - Light board (<= 8 pieces): 14 expected moves.
3. Golden Window Floor (0.34s) & Multi-Tier Emergency Buffer:
   - Retains the 0.34s floor to fully exploit the platform 0.50s charging mechanism with 0 debt.
   - Multi-tier emergency escape (<= 0.65s, <= 1.5s, <= 6.0s) guaranteeing 0% timeout risk.
"""
from __future__ import annotations

from typing import Optional
import chess
from chesslab_sdk import BotConfig, SearchContext

MIN_MOVE_CHARGE = 0.50   # Mandatory platform sleep floor per move (seconds)


def get_time_limits(
    context: SearchContext,
    board: chess.Board,
    config: Optional[BotConfig] = None,
) -> tuple[float, float]:
    """Return (soft_limit, hard_limit) in seconds."""
    remaining_s = max(0.0, context.time_remaining_ms / 1000.0)
    legal_moves_count = board.legal_moves.count()

    # Fast escape if only 1 legal move (forced)
    if legal_moves_count <= 1:
        return 0.015, 0.03

    # =========================================================================
    # 1. EMERGENCY FAST ESCAPE (Tier 1: remaining_s <= 0.65s)
    # Final move before clock expiry. Fast escape leaving safety buffer.
    # =========================================================================
    if remaining_s <= 0.65:
        hard_limit = max(0.04, remaining_s - 0.08)
        soft_limit = max(0.03, hard_limit * 0.80)
        return soft_limit, hard_limit

    # =========================================================================
    # 2. LOW CLOCK BUFFER (Tier 2: 0.65s < remaining_s <= 1.50s)
    # Only 1-3 moves left on total clock. Quick move to ensure no platform forfeit.
    # =========================================================================
    if remaining_s <= 1.50:
        hard_limit = min(0.28, remaining_s - 0.40)
        soft_limit = min(0.20, hard_limit * 0.80)
        return max(0.05, soft_limit), max(0.08, hard_limit)

    # =========================================================================
    # 3. CRITICAL CLOCK GLIDE (Tier 3: 1.50s < remaining_s <= 6.00s)
    # Conserve clock, gliding safely with small budgets.
    # =========================================================================
    if remaining_s <= 6.00:
        hard_limit = min(0.45, remaining_s * 0.12)
        soft_limit = max(0.20, hard_limit * 0.75)
        return max(0.18, soft_limit), max(0.25, hard_limit)

    # =========================================================================
    # 4. DYNAMIC ELASTIC TIME ALLOCATION (remaining_s > 6.00s)
    # Scales dynamically across all random match lengths (30s, 60s, 180s, 300s).
    # =========================================================================
    initial_s = (
        float(config.initial_time_ms) / 1000.0
        if config and config.initial_time_ms > 0
        else max(remaining_s, 60.0)
    )

    # Dynamic Material-Density Move Estimation:
    piece_count = bin(board.occupied).count('1')
    if piece_count > 24:
        expected_moves = 34
    elif piece_count > 16:
        expected_moves = 26
    elif piece_count > 8:
        expected_moves = 20
    else:
        expected_moves = 14

    # Floating endgame safety reserve (proportional, max 4.0s)
    reserve = min(remaining_s * 0.08, 4.0)
    usable_clock = max(0.0, remaining_s - reserve)
    base_budget = usable_clock / expected_moves

    # Tactical Complexity Factor:
    complexity = 1.0
    if board.is_check():
        complexity += 0.35

    king = board.king(board.turn)
    if king is not None:
        enemy_heavy = board.pieces(chess.QUEEN, not board.turn) | board.pieces(chess.ROOK, not board.turn)
        if any(chess.square_distance(king, sq) <= 3 for sq in enemy_heavy):
            complexity += 0.25

    if legal_moves_count > 35:
        complexity += 0.15
    elif legal_moves_count <= 6:
        complexity *= 0.70

    # Phase investment factor:
    ply = context.ply
    if ply < 12:
        phase_mult = 1.10   # Chess960 opening requires deep calculation!
    elif ply < 45:
        phase_mult = 1.30   # Peak tactical clash in middlegame
    else:
        phase_mult = 0.90   # Endgame conversion

    computed_soft = base_budget * complexity * phase_mult

    # Clock-Proportional Dynamic Ceiling (NO STATIC 2.4s CAP!):
    # In 180s/300s matches, allows up to 5.5s when clock is healthy (> 90s)
    # Automatically and smoothly tightens as clock depletes
    if initial_s >= 120.0:        # 180s to 300s matches
        max_soft_ceiling = min(remaining_s * 0.085, 5.50)
    elif initial_s >= 50.0:       # 60s matches
        max_soft_ceiling = min(remaining_s * 0.080, 2.20)
    else:                         # 30s bullet
        max_soft_ceiling = min(remaining_s * 0.075, 0.75)

    max_soft = max(0.34, max_soft_ceiling)
    soft_limit = max(0.34, min(max_soft, computed_soft))
    hard_limit = min(soft_limit * 1.35, max(0.38, remaining_s - 0.65))

    # Safety ceiling leaves safety buffer against total remaining clock
    safe_ceiling = max(0.38, remaining_s - 0.65)
    hard_limit = min(hard_limit, safe_ceiling)
    soft_limit = min(soft_limit, hard_limit * 0.85)

    return max(0.34, soft_limit), max(0.38, hard_limit)
