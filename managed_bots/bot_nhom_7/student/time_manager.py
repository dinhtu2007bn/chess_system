
GAME_PLY_LIMIT = 300
MOVE_COST = 0.80
FAST_THINK = 0.44


def time_budget(remaining_ms: int, ply: int) -> tuple[float, float]:
    remaining = max(0.0, remaining_ms / 1000)
    turns_left = max(1, (GAME_PLY_LIMIT + 1 - ply) // 2)
    reserve = min(MOVE_COST * turns_left + 3.0, 0.80 * remaining)
    horizon = min(turns_left, max(13, 17 - ply // 38))
    think = FAST_THINK + (remaining - reserve) / horizon

    soft = min(7.0, 0.14 * remaining, 0.80 * think)
    hard = min(9.0, 0.18 * remaining, 1.30 * think)
    hard = max(0.0, hard - 0.028)
    return min(soft, hard), hard
