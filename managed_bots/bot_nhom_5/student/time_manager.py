"""Clock allocation, calibrated to the REAL arena rule.

Arena (app/game_controller.py): for every move a target T is drawn uniformly from [500, 1000] ms and the move is
charged max(think_time, T).  So:
  * thinking up to ~0.6 s is almost free (it hides inside T; mean T = 0.75 s),
  * every future move costs about 0.75 s even if we think nothing,
  * the floors of all remaining own moves up to the 300-ply cap must stay reserved, otherwise a long game flags.
Reserve per move = MOVE_FLOOR (0.76 s, slightly above the mean 0.75 s) plus a fixed SAFETY margin that covers the
random spread of the floors (sd of the sum over 150 moves is ~1.8 s) and process overhead.  Only the surplus is
shared out, so the invariant  time_left >= SAFETY + MOVE_FLOOR * moves_left  is kept (checked by Monte Carlo
simulation: no flag in 300-ply games even when every move uses its hard limit).
"""
import time


class SearchTimeout(Exception):
    """Raised inside the search when the hard deadline is reached."""


MOVE_FLOOR = 0.76       # reserved per remaining move (arena floor is U(0.5, 1.0) s, mean 0.75 s)
FREE_THINK = 0.60       # think time that costs (almost) nothing extra: hides inside the random floor
SAFETY = 8.0           # fixed slack: spread of the random floors + per-move process overhead + import time (s)
MAX_PLY = 300
DEFAULT_TIME_LEFT = 30.0
MIN_HORIZON = 12        # expected own moves still to come, lower bound
BASE_HORIZON = 48
SOFT_SHARE = 1.0        # share of per-move surplus used as soft target
HARD_MULT = 2.5         # hard limit = free + HARD_MULT * soft surplus
HARD_CAP_FRACTION = 0.20  # never spend more than this share of avail on one move
NEW_ITERATION_FRACTION = 0.55
ITER_GROWTH = 3.0       # measured time ratio between consecutive iterations (see bench)


def allocate_time(time_left: float, ply: int, phase: int = 24, n_legal: int = 30,
                  in_check: bool = False, n_captures: int = 0) -> tuple[float, float]:
    """Return (soft_budget, hard_limit) in seconds for the current move.

    phase: material phase 0..24 (24 = full middlegame material).
    in_check / n_captures: small "complexity" multiplier (idea from the Thien Tuat bot). It only re-distributes
    the SURPLUS `avail`; the reserve for the floors of all remaining moves is untouched, so the no-flag
    invariant  time_left >= SAFETY + MOVE_FLOOR * moves_left  still holds.
    """
    if n_legal <= 1:
        return 0.0, 0.0
    own_left = max(1, (MAX_PLY - ply + 1) // 2)
    avail = time_left - SAFETY - MOVE_FLOOR * own_left
    free = FREE_THINK
    if time_left < SAFETY + MOVE_FLOOR:      # critical: stay far below the floor
        free = max(0.0, min(free, (time_left - 0.2) / 4))
        avail = 0.0
    horizon = min(own_left, max(MIN_HORIZON, BASE_HORIZON - ply / 2.0))
    # more time where decisions are hard (many pieces, many moves), less when clear
    mult = 0.6 + 0.9 * min(phase, 24) / 24.0
    if ply < 12:
        mult = 0.6
    if n_legal <= 5:
        mult *= 0.5
    if in_check:
        mult *= 1.25
    elif n_captures >= 3:
        mult *= 1.12
    extra = max(0.0, avail) / horizon * mult * SOFT_SHARE
    soft = free + extra
    hard = free + min(HARD_MULT * extra, max(0.0, avail) * HARD_CAP_FRACTION)
    hard = max(hard, soft)
    return soft, hard


class Deadline:
    __slots__ = ("start", "soft", "hard")

    def __init__(self, soft: float, hard: float) -> None:
        self.start = time.perf_counter()
        self.soft = soft
        self.hard = hard

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def expired(self) -> bool:
        return time.perf_counter() - self.start >= self.hard

    def can_start_iteration(self, last_iter=None) -> bool:
        """Start the next depth only if it is predicted to FINISH before the hard limit.

        The next iteration costs about GROWTH x the last one; the old rule (stop after 55% of the soft
        budget) threw away time that was still available between soft and hard.
        """
        e = self.elapsed()
        if last_iter is None or last_iter <= 0:
            return e < self.soft * NEW_ITERATION_FRACTION
        return e < self.soft and e + ITER_GROWTH * last_iter <= self.hard * 0.92


def extract_time_left(context, color: bool) -> float:
    """SearchContext.time_remaining_ms is the documented field (README 4.2)."""
    try:
        t = float(context.time_remaining_ms)
        if t == t and t >= 0:
            return t / 1000.0
    except Exception:
        pass
    return DEFAULT_TIME_LEFT
