"""Dynamic time allocation for the remaining game clock.

Determines how long the bot should think on each move by computing two
limits:

* **soft_limit** – ideal budget. After finishing a full depth iteration,
  stop if this has been exceeded. This prevents starting a much deeper
  (and much longer) iteration that would waste time.
* **hard_limit** – absolute deadline. If the search is still running
  when this is reached, it is aborted immediately and the best move
  found so far is returned.

Features:
- Chess960 opening awareness: careful early setup without burning clock
- Peak middlegame allocation for complex tactical branching
- Streamlined endgame budgeting (fewer pieces, faster cutoffs)
- Opponent clock awareness
- Multi-tier emergency mode to guarantee zero time-forfeits under any control (50s to 360s)
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
    remaining_s = max(0.1, context.time_remaining_ms / 1000.0)
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

    expected_total_moves = 36 if initial_s <= 60.0 else 42
    moves_left = max(12, expected_total_moves - moves_played)

    # Base allocation per move
    base = remaining_s / moves_left

    # Phase-based multiplier tailored for Chess960
    if moves_played < 4:
        # Opening in Chess960: positions are novel, develop pieces safely
        multiplier = 1.05
    elif moves_played < 22:
        # Middlegame: tactical complications, invest maximum calculation
        multiplier = 1.25
    elif moves_played < 32:
        # Late middlegame / transition
        multiplier = 1.0
    else:
        # Endgame: simpler trees, preserve time
        multiplier = 0.8

    soft_limit = base * multiplier
    hard_limit = base * 2.8

    # Extra time when in check (defensive precision is critical)
    if board.is_check():
        soft_limit *= 1.25
        hard_limit *= 1.25

    # Opponent time pressure: if opponent has low time, play solid & steady
    if opp_remaining_s < remaining_s * 0.4 and remaining_s > 15.0:
        soft_limit *= 0.88

    # Multi-tier Emergency Mode (Absolute protection against flag-fall)
    if remaining_s < 3.0:
        soft_limit = min(0.06, remaining_s * 0.04)
        hard_limit = min(0.18, remaining_s * 0.08)
    elif remaining_s < 8.0:
        soft_limit = min(soft_limit, remaining_s * 0.05)
        hard_limit = min(hard_limit, remaining_s * 0.12)
    elif remaining_s < 18.0:
        soft_limit = min(soft_limit, remaining_s * 0.07)
        hard_limit = min(hard_limit, remaining_s * 0.18)

    # Hard boundary checks
    safety_buffer = 0.3 if remaining_s > 5.0 else 0.1
    hard_limit = min(hard_limit, remaining_s - safety_buffer)
    soft_limit = min(soft_limit, hard_limit * 0.55)

    return max(0.02, soft_limit), max(0.05, hard_limit)
