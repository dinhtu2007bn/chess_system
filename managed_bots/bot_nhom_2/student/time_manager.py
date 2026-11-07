"""Clock allocation for a no-increment game with a 300-ply limit.

The controller charges max(search time, a random 500--1000 ms delay),
not the sum. Never sleep in the bot to simulate that delay.
"""

from dataclasses import dataclass

from chesslab_sdk import BotConfig, SearchContext


@dataclass(frozen=True, slots=True)
class TimeBudget:
    soft_ms: float
    hard_ms: float


def allocate_time(config: BotConfig, context: SearchContext) -> TimeBudget:
    remaining = max(0, context.time_remaining_ms)
    # Below this threshold, return an already legal fallback immediately.
    if remaining <= 3000:
        return TimeBudget(0.0, 0.0)

    moves_left = max(1, (300 - max(0, context.ply) + 1) // 2)
    expected_moves = min(moves_left, max(22, 60 - context.ply // 2))
    reserve = max(1200.0, min(remaining * 0.18, moves_left * 1000.0))
    spendable = max(0.0, remaining - reserve)
    soft = min(6000.0, spendable / expected_moves * 0.75)
    if remaining <= 15000:
        soft = min(soft, 120.0)
    elif remaining <= 30000:
        soft = min(soft, 450.0)

    # Leave room for serialization, scheduling and the controller's ticker.
    margin = max(200.0, min(1000.0, config.initial_time_ms * 0.002))
    hard = min(9000.0, soft * 1.6, remaining * 0.08, remaining - margin)
    return TimeBudget(max(0.0, min(soft, hard)), max(0.0, hard))
