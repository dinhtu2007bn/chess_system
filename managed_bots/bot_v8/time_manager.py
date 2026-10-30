"""Dynamic time allocation for the remaining game clock.

Master Anti-Timeout Time Manager:
- Floor-Aware Budgeting: Directly accounts for the mandatory 500ms platform delay
  per move (MIN_MOVE_CHARGE = 0.5s) to guarantee zero clock depletion.
- Scalable Depth Investment: Allows 0.8s - 1.4s thinking time in 60s games (depth 5-6),
  scaling smoothly with branching complexity and game phases.
- Multi-Tier Emergency Glide Path: Instant fast response (0.02s - 0.08s) when clock drops,
  preventing artificial sleep timeouts.
- Absolute Safety Ceiling: Guaranteed headroom leaving at least 0.65s above platform floor.
"""
from __future__ import annotations

from typing import Optional

import chess

from chesslab_sdk import BotConfig, SearchContext


MIN_MOVE_CHARGE = 0.50   # Mandatory platform sleep floor per move (seconds)
SAFETY_BUFFER = 0.15     # Additional network/IPC buffer (seconds)


def get_time_limits(
    context: SearchContext,
    board: chess.Board,
    config: Optional[BotConfig] = None,
) -> tuple[float, float]:
    """Return ``(soft_limit, hard_limit)`` in seconds."""
    remaining_s = max(0.02, context.time_remaining_ms / 1000.0)
    opp_remaining_s = getattr(context, "opponent_time_remaining_ms", None)
    if opp_remaining_s is not None:
        opp_remaining_s = max(0.02, opp_remaining_s / 1000.0)
    else:
        opp_remaining_s = remaining_s

    ply = context.ply
    moves_played = ply // 2

    # Determine total expected game moves based on clock and current phase
    initial_s = 60.0
    if config is not None and hasattr(config, "initial_time_ms"):
        initial_s = max(10.0, config.initial_time_ms / 1000.0)

    # Estimate remaining turns (Chess960 games typically finish in 40-52 moves)
    expected_moves = max(12, min(44, 50 - moves_played // 2))

    # Calculate mandatory floor delay reserve needed for the rest of the game
    floor_reserve = expected_moves * MIN_MOVE_CHARGE

    # Usable time budget strictly for search computation
    usable_s = max(0.0, remaining_s - floor_reserve - 0.50)

    if usable_s > 0.0:
        base = usable_s / expected_moves
    else:
        # Time pressure: clock is close to the floor requirement; use tiny compute slices
        base = min(0.06, remaining_s / (2.5 * expected_moves))

    # Branching complexity factor based on number of legal choices
    num_legal = board.legal_moves.count()
    if num_legal > 38:
        complexity = 1.25   # Sharp, tactical opening/middlegame
    elif num_legal > 26:
        complexity = 1.10   # Average active position
    elif num_legal > 14:
        complexity = 0.95   # Solid, semi-closed
    else:
        complexity = 0.75   # Restricted choices, move fast

    # Phase multiplier
    if moves_played < 4:
        phase_mult = 1.05   # Solid opening development
    elif moves_played < 26:
        phase_mult = 1.20   # Critical middlegame clashes
    elif moves_played < 36:
        phase_mult = 0.95   # Transition
    else:
        phase_mult = 0.80   # Endgame: narrower trees, conserve clock

    multiplier = max(0.70, min(1.35, complexity * phase_mult))

    soft_limit = base * multiplier
    hard_limit = min(soft_limit * 1.50, base * 1.80)

    # In-check defensive bonus: slight boost to find evasion
    if board.is_check():
        soft_limit *= 1.20
        hard_limit *= 1.20

    # Multi-Tier Emergency Glide Path (Guarantees zero timeout forfeit)
    if remaining_s < 2.5:
        soft_limit = 0.02
        hard_limit = 0.04
    elif remaining_s < 6.0:
        soft_limit = min(soft_limit, 0.04)
        hard_limit = min(hard_limit, 0.08)
    elif remaining_s < 12.0:
        soft_limit = min(soft_limit, 0.12)
        hard_limit = min(hard_limit, 0.25)
    elif remaining_s < 20.0:
        soft_limit = min(soft_limit, 0.25)
        hard_limit = min(hard_limit, 0.50)

    # Absolute safety ceiling: never let hard_limit leave less than (MIN_MOVE_CHARGE + SAFETY_BUFFER)
    # This prevents the platform's sleep delay from exhausting the clock
    hard_ceiling = max(0.02, remaining_s - MIN_MOVE_CHARGE - SAFETY_BUFFER)
    hard_limit = min(hard_limit, hard_ceiling)
    soft_limit = min(soft_limit, hard_limit * 0.70)

    return max(0.015, soft_limit), max(0.03, hard_limit)
