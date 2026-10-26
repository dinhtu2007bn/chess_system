"""Dynamic time allocation for the remaining game clock (No-Increment format).

V5 Adaptive Time Management:
- Dynamic safety buffer: light deduction when clock > 25s, full 400ms buffer in late game.
- Enhanced middlegame tactical budget: phase_mult = 1.22 with max 1.35x complexity boost.
- Dynamic moves_left decaying model for extended games (moves_played > 35).
- 3 Emergency Tiers with strict upper caps to guarantee zero timeouts under any
  clock condition (down to < 1.0s).
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

    # Reserve safety buffer: scaled by clock urgency
    buffer = 0.25 if remaining_s > 25.0 else 0.40
    safe_remaining_s = max(0.05, remaining_s - buffer)

    # Determine total expected game moves based on initial clock
    initial_s = 180.0
    if config is not None and hasattr(config, "initial_time_ms"):
        initial_s = max(10.0, config.initial_time_ms / 1000.0)

    # Dynamic moves-left curve:
    # In extended games (> 35 moves), use decaying model so clock isn't artificially choked
    if moves_played > 35:
        moves_left = max(8, min(20, 60 - moves_played // 2))
    else:
        expected_total_moves = 38 if initial_s <= 60.0 else 46
        moves_left = max(14, expected_total_moves - moves_played)

    # Base budget per move
    base = safe_remaining_s / moves_left

    # Branching complexity factor: scale by number of legal moves
    num_legal = board.legal_moves.count()
    if num_legal > 36:
        complexity = 1.22   # Highly tactical/open position
    elif num_legal > 24:
        complexity = 1.08   # Average middlegame
    elif num_legal > 14:
        complexity = 0.92   # Simplified position
    else:
        complexity = 0.75   # Few choices, play faster to save time

    # Game phase multiplier
    if moves_played < 4:
        # Chess960 opening: steady development
        phase_mult = 0.95
    elif moves_played < 26:
        # Middlegame: allow investment when complexity is high
        phase_mult = 1.22
    elif moves_played < 36:
        # Late middlegame / transition
        phase_mult = 0.98
    else:
        # Endgame: conserve clock, trees are small
        phase_mult = 0.82

    # Combine multipliers (bounded between 0.70x and 1.35x)
    multiplier = max(0.70, min(1.35, complexity * phase_mult))

    soft_limit = base * multiplier
    hard_limit = base * 2.45

    # Critical defensive situation: extra time when in check
    if board.is_check():
        soft_limit *= 1.25
        hard_limit *= 1.25

    # Opponent time pressure: if opponent has critically low time, play solid & steady
    if opp_remaining_s < remaining_s * 0.35 and remaining_s > 15.0:
        soft_limit *= 0.85

    # --- 3 Emergency Tiers (Never forfeit on time, guaranteed safe bounds) ---
    if remaining_s < 3.0:
        hard_limit = min(hard_limit, remaining_s * 0.15)
        soft_limit = hard_limit * 0.50
    elif remaining_s < 8.0:
        hard_limit = min(hard_limit, remaining_s * 0.20)
        soft_limit = hard_limit * 0.55
    elif remaining_s < 18.0:
        hard_limit = min(hard_limit, remaining_s * 0.25)
        soft_limit = hard_limit * 0.60

    # Absolute safety boundaries
    safety_buffer = 0.35 if remaining_s > 6.0 else 0.12
    hard_limit = min(hard_limit, max(0.035, remaining_s - safety_buffer))
    soft_limit = min(soft_limit, hard_limit * 0.55)

    return max(0.015, soft_limit), max(0.035, hard_limit)
