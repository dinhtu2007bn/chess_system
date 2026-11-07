"""Dynamic soft/hard time & search depth manager for Chess960 tournament play.

Runner clock model (app/game_controller.py):
  - Every move is charged max(response_time, U[500, 1000] ms).
    => Thinking up to ~850 ms costs almost nothing extra beyond the mandatory
       minimum delay, while emergency moves still consume ~750-800 ms of clock.
  - No increment, 300-ply cap. We keep a ~27 s reserve (~32 rapid moves) so we
    outlast aggressive opponents on the clock while investing full search power
    into the opening and middlegame where games are decided.
"""

from __future__ import annotations

from chesslab_sdk import SearchContext

_MAX_PLIES = 300
_SAFETY_MS = 1_500
_MIN_DELAY_AVG_MS = 800
_RESERVE_MOVES = 32
_FREE_WINDOW_MS = 850
_EMERGENCY_MS = 120


def _own_moves_left(ply: int) -> int:
    return max(1, (_MAX_PLIES - max(0, ply) + 1) // 2)


def allocate_time_bounds(
    context: SearchContext, initial_time_ms: int | None = None
) -> tuple[int, int]:
    """Return (soft_ms, hard_ms) search budgets for the current move."""
    remaining = max(0, int(context.time_remaining_ms))
    ply = max(0, int(context.ply))
    moves_left = _own_moves_left(ply)

    if remaining <= _SAFETY_MS:
        tiny = max(1, min(80, remaining // 8))
        return tiny, tiny

    # Short / bullet game guard (e.g. 5s - 35s test games): stay inside 500ms floor
    init = int(initial_time_ms) if initial_time_ms and initial_time_ms > 0 else remaining
    if init <= 35_000 or remaining <= 12_000:
        soft = max(60, min(300, (remaining - 400) // max(8, min(moves_left, 25))))
        hard = max(soft, min(460, (remaining - 300) // 5))
        return int(soft), int(hard)

    reserve_horizon = min(moves_left - 1, _RESERVE_MOVES)
    reserve = _SAFETY_MS + reserve_horizon * _MIN_DELAY_AVG_MS

    # Low-clock survival zone: stay inside the runner's free minimum-delay window
    if remaining <= reserve:
        avail_per_move = (remaining - _SAFETY_MS) // max(1, min(moves_left, _RESERVE_MOVES))
        soft = max(_EMERGENCY_MS, min(avail_per_move, 420))
        hard = max(soft, min(700, (remaining - _SAFETY_MS) // 4))
        return int(soft), int(hard)

    surplus = remaining - reserve

    # Distribute surplus across decisive game phases
    if ply < 18:
        expected_moves, phase_weight = 34, 1.15
    elif ply < 64:
        expected_moves, phase_weight = 28, 1.25
    elif ply < 110:
        expected_moves, phase_weight = 24, 1.05
    else:
        expected_moves, phase_weight = 20, 0.90

    expected_moves = max(4, min(expected_moves, moves_left))
    base = (surplus / expected_moves) * phase_weight

    # Modulate slightly based on clock advantage vs. opponent
    opp = max(1, int(context.opponent_time_remaining_ms))
    ratio = remaining / opp
    if ratio > 1.30:
        base *= 1.12
    elif ratio < 0.75:
        base *= 0.85

    # Take advantage of the ~850 ms minimum charge floor when surplus allows
    soft = max(_FREE_WINDOW_MS, base)
    soft = min(soft, surplus * 0.22 + _FREE_WINDOW_MS * 0.5, remaining * 0.16)
    hard = min(soft * 2.25, surplus * 0.42 + _FREE_WINDOW_MS, remaining - _SAFETY_MS)
    hard = max(soft, hard)
    return int(soft), int(hard)


def allocate_time_ms(context: SearchContext) -> int:
    """Allocate target time in ms (backwards-compatible single-value API)."""
    soft_ms, _ = allocate_time_bounds(context)
    return soft_ms


def max_search_depth(context: SearchContext) -> int:
    """Return the search depth ceiling."""
    remaining = max(0, int(context.time_remaining_ms))
    if remaining < 800:
        return 3
    if remaining < 2_000:
        return 5
    return 64
