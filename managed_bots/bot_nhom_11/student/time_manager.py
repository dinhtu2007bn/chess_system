from __future__ import annotations

import time


def allocate_time_ms(time_remaining_ms: int, ply: int = 0,
                     initial_time_ms: int | None = None) -> int:
    remaining = max(0, int(time_remaining_ms))
    if remaining <= 0:
        return 1

    initial = max(1, int(remaining if initial_time_ms is None else initial_time_ms))
    expected_moves = 42 if ply < 30 else (30 if ply < 70 else 18)
    nominal = remaining / expected_moves
    reserve = max(250, min(2_000, int(initial * 0.025)))
    usable = max(1, remaining - reserve)
    budget = min(nominal * 1.25, usable * 0.08)

    budget = max(25, budget)
    budget = min(budget, max(1, remaining - min(100, remaining // 5)))
    return max(1, int(budget))


def make_deadline(time_remaining_ms: int, ply: int = 0,
                  initial_time_ms: int | None = None) -> float:
    return time.monotonic() + allocate_time_ms(
        time_remaining_ms, ply, initial_time_ms
    ) / 1000
