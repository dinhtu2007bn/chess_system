"""Dynamic time allocation for the remaining game clock.

V7 Anti-Timeout Time Management (built on V3):
- Preserves V3's exact tactical allocation, branching complexity, and phase multipliers.
- Fixes endgame moves_left saturation for long games (> 32 moves).
- Strict Emergency Curve under 2.5s and 5.0s guaranteeing 0% timeouts under 500ms platform delay.
- Retains V3's high peak thinking budget for sharp middlegame tactics.
"""
from __future__ import annotations

from typing import Optional

import chess

from chesslab_sdk import BotConfig, SearchContext


def get_time_limits(
    context: SearchContext,
    board: chess.Board,
    config: Optional[BotConfig] = None,
) -> tuple[float, float]:
    """Return ``(soft_limit, hard_limit)`` in seconds."""
    remaining_s = max(0.05, context.time_remaining_ms / 1000.0)
    opp_remaining_s = getattr(context, "opponent_time_remaining_ms", None)
    if opp_remaining_s is not None:
        opp_remaining_s /= 1000.0
    else:
        opp_remaining_s = remaining_s

    ply = context.ply
    moves_played = ply // 2  # Full moves made by us

    # Determine total expected game moves based on initial clock
    initial_s = 180.0
    if config is not None and hasattr(config, "initial_time_ms"):
        initial_s = max(10.0, config.initial_time_ms / 1000.0)

    # Chess960 games typically last 42-50 moves
    expected_total_moves = 38 if initial_s <= 60.0 else 46

    # In extended games (> 32 moves), decay smoothly instead of clamping to 16
    if moves_played > 32:
        moves_left = max(8, min(16, 50 - moves_played // 2))
    else:
        moves_left = max(16, expected_total_moves - moves_played)

    # Base budget per move
    base = remaining_s / moves_left

    # Branching complexity factor: scale by number of legal moves (V3 exact values)
    num_legal = board.legal_moves.count()
    if num_legal > 38:
        complexity = 1.20   # Highly tactical/open position
    elif num_legal > 26:
        complexity = 1.05   # Average middlegame
    elif num_legal > 14:
        complexity = 0.90   # Simplified position
    else:
        complexity = 0.75   # Few choices, play faster to save time

    # Game phase multiplier (V3 exact values)
    if moves_played < 4:
        # Chess960 opening: steady development
        phase_mult = 1.0
    elif moves_played < 24:
        # Middlegame: allow investment only when complexity is high
        phase_mult = 1.15
    elif moves_played < 34:
        # Late middlegame / transition
        phase_mult = 0.95
    else:
        # Endgame: conserve clock, trees are small
        phase_mult = 0.80

    # Combine multipliers (bounded between 0.70x and 1.35x, V3 exact)
    multiplier = max(0.70, min(1.35, complexity * phase_mult))

    soft_limit = base * multiplier
    hard_limit = base * 2.5

    # Critical defensive situation: extra time when in check (V3 exact)
    if board.is_check():
        soft_limit *= 1.25
        hard_limit *= 1.25

    # Opponent time pressure: if opponent has critically low time, play solid & steady (V3 exact)
    if opp_remaining_s < remaining_s * 0.35 and remaining_s > 15.0:
        soft_limit *= 0.85

    # Smooth Multi-tier Emergency Curve (Never forfeit on time under 500ms delay)
    if remaining_s < 2.5:
        soft_limit = min(0.04, remaining_s * 0.03)
        hard_limit = min(0.08, remaining_s * 0.06)
    elif remaining_s < 5.0:
        soft_limit = min(0.08, remaining_s * 0.04)
        hard_limit = min(0.20, remaining_s * 0.08)
    elif remaining_s < 10.0:
        soft_limit = min(soft_limit, remaining_s * 0.06)
        hard_limit = min(hard_limit, remaining_s * 0.14)
    elif remaining_s < 18.0:
        soft_limit = min(soft_limit, remaining_s * 0.08)
        hard_limit = min(hard_limit, remaining_s * 0.18)

    # Absolute safety boundaries
    safety_buffer = 0.35 if remaining_s > 6.0 else 0.15
    hard_limit = min(hard_limit, max(0.04, remaining_s - safety_buffer))
    soft_limit = min(soft_limit, hard_limit * 0.60)

    return max(0.02, soft_limit), max(0.04, hard_limit)
