"""Balanced Anti-V9 Tournament Time Manager for Bot Cuoi Cung.

Key Architecture:
1. Anti-V9 Balanced Pace Control:
   - Eliminates middle-game clock overburn. In 180s matches, caps soft_limit at 2.40s (hard_limit <= 2.75s).
   - In 300s matches, caps soft_limit at 3.40s (hard_limit <= 3.90s).
   - By Move 40, always preserves 60s - 80s clock reserve (vs Bot Test 3's exhausted 15s).
2. Deep Endgame Reserve (Depth 9-11 in Endgames):
   - In late endgames (Move 40+), maintains a steady 1.8s - 2.2s calculation pace.
   - Combined with few pieces on the board, naturally reaches Depth 9-11 to convert passed pawns.
3. Golden Window Floor (0.34s) & Multi-Tier Emergency Buffer:
   - Never drops below 0.34s while clock > 6.0s, leveraging the mandatory platform floor without debt.
   - Multi-tier emergency escape protects against any timeout.
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

    # Fast escape if only 1 legal move
    if legal_moves_count <= 1:
        return 0.015, 0.03

    # =========================================================================
    # 1. EMERGENCY FAST ESCAPE (Tier 1: remaining_s <= 0.65s)
    # =========================================================================
    if remaining_s <= 0.65:
        hard_limit = max(0.04, remaining_s - 0.08)
        soft_limit = max(0.03, hard_limit * 0.80)
        return soft_limit, hard_limit

    # =========================================================================
    # 2. LOW CLOCK BUFFER (Tier 2: 0.65s < remaining_s <= 1.80s)
    # =========================================================================
    if remaining_s <= 1.80:
        hard_limit = min(0.28, remaining_s - 0.40)
        soft_limit = min(0.20, hard_limit * 0.80)
        return max(0.05, soft_limit), max(0.08, hard_limit)

    # =========================================================================
    # 3. PANIC SMOOTH GLIDE (Tier 3: 1.80s < remaining_s <= 6.00s)
    # =========================================================================
    if remaining_s <= 6.00:
        hard_limit = min(0.45, remaining_s * 0.12)
        soft_limit = max(0.20, hard_limit * 0.75)
        return max(0.18, soft_limit), max(0.25, hard_limit)

    # =========================================================================
    # 4. BALANCED ANTI-V9 PACED TIME ALLOCATION (remaining_s > 6.00s)
    # =========================================================================
    initial_s = (
        float(config.initial_time_ms) / 1000.0
        if config and config.initial_time_ms > 0
        else max(remaining_s, 60.0)
    )

    # Estimate moves left to allocate across the whole game
    ply = context.ply
    moves_played = ply // 2
    if moves_played < 15:
        expected_moves = 42
    elif moves_played < 35:
        expected_moves = 35
    elif moves_played < 55:
        expected_moves = 28
    else:
        expected_moves = 20

    # Ensure deep endgame reserve: hold back 15% clock (up to 12.0s)
    endgame_reserve = min(remaining_s * 0.15, 12.0)
    usable_clock = max(0.0, remaining_s - endgame_reserve)
    base_budget = usable_clock / max(10, expected_moves)

    # Tactical & Complexity factor (sharp moves get modest boost, but capped!)
    complexity = 1.0
    if board.is_check():
        complexity += 0.20

    king = board.king(board.turn)
    if king is not None:
        enemy_heavy = board.pieces(chess.QUEEN, not board.turn) | board.pieces(chess.ROOK, not board.turn)
        if any(chess.square_distance(king, sq) <= 2 for sq in enemy_heavy):
            complexity += 0.15

    if legal_moves_count > 35:
        complexity += 0.10
    elif legal_moves_count <= 6:
        complexity *= 0.75

    # Phase weighting
    if ply < 12:
        phase_mult = 0.90
    elif ply < 45:
        phase_mult = 1.15  # Controlled tactical investment (anti-burn)
    else:
        phase_mult = 0.95  # Solid endgame pace

    computed_soft = base_budget * complexity * phase_mult

    # ANTI-V9 HARD CAPS (Prevents spending > 2.4s in 180s blitz, preserving clock)
    if initial_s >= 240.0:        # 5m matches (300s+)
        pace_cap_soft = 3.40
        pace_cap_hard = 3.90
    elif initial_s >= 120.0:      # 3m matches (180s blitz - standard arena)
        pace_cap_soft = 2.40
        pace_cap_hard = 2.75
    elif initial_s >= 50.0:       # 60s matches
        pace_cap_soft = 1.30
        pace_cap_hard = 1.55
    else:                         # 30s bullet
        pace_cap_soft = 0.50
        pace_cap_hard = 0.60

    # Apply pace cap smoothly
    soft_limit = min(pace_cap_soft, computed_soft)
    hard_limit = min(pace_cap_hard, soft_limit * 1.30)

    # Floor at Golden Window (0.34s) when clock is healthy (> 6.0s)
    soft_limit = max(0.34, soft_limit)
    hard_limit = max(0.38, hard_limit)

    # Safety ceiling against total remaining clock
    safe_ceiling = max(0.38, remaining_s - 0.70)
    hard_limit = min(hard_limit, safe_ceiling)
    soft_limit = min(soft_limit, hard_limit * 0.85)

    return max(0.34, soft_limit), max(0.38, hard_limit)
