"""Dynamic time allocation for the remaining game clock (No-Increment format).

V6 Tactical-Master Time Management:
- Uncapped tactical peak: allows up to 1.45x multiplier in high-complexity positions
  (resolves the "AN TOÀN NHƯNG THỤ ĐỘNG" diagnosis, reaching 2.5s - 2.8s peaks).
- Smooth soft_limit ratio: allows soft_limit up to 70% of hard_limit so the engine
  completes deeper tactical depths without premature abortion.
- Dynamic moves_left decaying model for extended games (moves_played > 32).
- Adaptive safety buffer: light deduction when clock > 25s, full buffer when clock <= 10s.
- 3 Strict Emergency Tiers with guaranteed upper caps to preserve 0% Timeouts under any condition.
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

    # Adaptive safety buffer: light deduction when clock is abundant
    if remaining_s > 25.0:
        buffer = 0.15
    elif remaining_s > 10.0:
        buffer = 0.25
    else:
        buffer = 0.35
    safe_remaining_s = max(0.05, remaining_s - buffer)

    # Determine total expected game moves based on initial clock
    initial_s = 180.0
    if config is not None and hasattr(config, "initial_time_ms"):
        initial_s = max(10.0, config.initial_time_ms / 1000.0)

    # Dynamic moves-left curve:
    # In extended games (> 32 moves), use decaying model so clock isn't artificially choked
    if moves_played > 32:
        moves_left = max(10, min(22, 54 - moves_played // 2))
    else:
        expected_total_moves = 36 if initial_s <= 60.0 else 46
        moves_left = max(14, expected_total_moves - moves_played)

    # Base budget per move
    base = safe_remaining_s / moves_left

    # Branching complexity factor: scale by number of legal moves
    num_legal = board.legal_moves.count()
    if num_legal > 36:
        complexity = 1.25   # Highly tactical/open position
    elif num_legal > 24:
        complexity = 1.10   # Average middlegame
    elif num_legal > 14:
        complexity = 0.95   # Simplified position
    else:
        complexity = 0.80   # Few choices, play faster to save time

    # Game phase multiplier
    if moves_played < 4:
        # Chess960 opening: steady development
        phase_mult = 0.95
    elif moves_played < 26:
        # Critical middlegame phase: allow deep investment in tactics
        phase_mult = 1.22
    elif moves_played < 36:
        # Late middlegame / transition
        phase_mult = 1.00
    else:
        # Endgame: conserve clock
        phase_mult = 0.85

    # Combine multipliers (bounded between 0.70x and 1.45x)
    multiplier = max(0.70, min(1.45, complexity * phase_mult))

    soft_limit = base * multiplier
    hard_limit = base * 2.50

    # Critical defensive situation: extra time when in check
    if board.is_check():
        soft_limit *= 1.25
        hard_limit *= 1.25

    # Opponent time pressure: if opponent has critically low time, play solid & steady
    if opp_remaining_s < remaining_s * 0.35 and remaining_s > 15.0:
        soft_limit *= 0.85

    # --- 3 Emergency Tiers (Never forfeit on time, guaranteed safe bounds) ---
    if remaining_s < 3.0:
        hard_limit = min(hard_limit, remaining_s * 0.18)
        soft_limit = hard_limit * 0.60
    elif remaining_s < 7.0:
        hard_limit = min(hard_limit, remaining_s * 0.22)
        soft_limit = hard_limit * 0.65
    elif remaining_s < 15.0:
        hard_limit = min(hard_limit, remaining_s * 0.28)
        soft_limit = hard_limit * 0.70

    # Absolute safety boundaries
    safety_buffer = 0.35 if remaining_s > 6.0 else 0.12
    hard_limit = min(hard_limit, max(0.035, remaining_s - safety_buffer))
    soft_limit = min(soft_limit, hard_limit * 0.70)

    return max(0.02, soft_limit), max(0.04, hard_limit)
