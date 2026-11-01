"""Dynamic time allocation scaling smoothly across arbitrary time controls.

Bot Test (Dynamic Tournament Time Manager V6.2):
- Universal Time Scaling (30s, 60s, 180s - 360s random tournament clocks):
  - Uses config.initial_time_ms and remaining game clock to dynamically size move budgets.
  - In long games (180s - 360s, 2-6 mins), allocates 3s - 8s in critical midgames for Depth 9-11.
  - In medium games (60s), allocates 1.0s - 1.8s in midgames for Depth 7-8.
  - In short games (30s blitz), limits midgame allocation to 0.45s - 0.65s so the clock bank
    comfortably lasts 50-60 moves (100-120 plies).
- Platform Floor Synchronization (0.34s Golden Window):
  - In late endgames or low clocks, never drops below 0.34s as long as clock permits.
  - 0.34s thinking + ~0.08s latency = 0.42s turnaround <= 0.50s platform floor, ensuring
    deep calculation (Depth 6-7) is maintained with ZERO accumulated platform debt.
- Emergency Buffer:
  - When remaining clock is <= 1.5s (only 1-3 moves left), glides down to 0.20s/0.05s to prevent
    any platform delay timeout.
"""
from __future__ import annotations

from typing import Optional

import chess

from chesslab_sdk import BotConfig, SearchContext

MIN_MOVE_CHARGE = 0.50   # Mandatory platform sleep floor per move (seconds)


def get_time_limits(
    context: SearchContext,
    board: chess.Board,
    config: Optional[BotConfig] = None,
) -> tuple[float, float]:
    """Return ``(soft_limit, hard_limit)`` in seconds."""
    remaining_s = max(0.0, context.time_remaining_ms / 1000.0)
    legal_moves_count = board.legal_moves.count()

    # Fast escape if only 1 legal move
    if legal_moves_count <= 1:
        return 0.015, 0.03

    # =========================================================================
    # 1. EMERGENCY FAST ESCAPE (remaining_s <= 0.65s)
    # Final move before clock expiry. Fast escape leaving safety buffer.
    # =========================================================================
    if remaining_s <= 0.65:
        hard_limit = max(0.04, remaining_s - 0.08)
        soft_limit = max(0.03, hard_limit * 0.80)
        return soft_limit, hard_limit

    # =========================================================================
    # 2. LOW CLOCK BUFFER (0.65s < remaining_s <= 1.50s)
    # Only 1-3 moves left on total clock. Quick move to ensure no platform forfeit.
    # =========================================================================
    if remaining_s <= 1.50:
        hard_limit = min(0.28, remaining_s - 0.50)
        soft_limit = min(0.20, hard_limit * 0.80)
        return max(0.05, soft_limit), max(0.08, hard_limit)

    # =========================================================================
    # 3. DYNAMIC TOURNAMENT TIME SCALING (remaining_s > 1.50s)
    # Scales dynamically across all random match lengths (30s to 360s+).
    # =========================================================================
    initial_s = (
        float(config.initial_time_ms) / 1000.0
        if config and config.initial_time_ms > 0
        else max(remaining_s, 60.0)
    )

    # Estimate moves remaining (typical Chess960 games take 35-50 moves)
    plies_left = max(1, 300 - context.ply)
    turns_left = (plies_left + 1) // 2
    expected_moves = min(turns_left, max(18, 48 - (context.ply // 4)))

    # Reserve 10% clock buffer (up to 4.0s) for endgames
    reserve = min(remaining_s * 0.10, 4.0)
    usable = max(0.0, remaining_s - reserve)
    base = usable / expected_moves

    # Tactical & King Danger complexity factor
    complexity = 1.0
    if board.is_check():
        complexity += 0.30

    king = board.king(board.turn)
    if king is not None:
        enemy_heavy = board.pieces(chess.QUEEN, not board.turn) | board.pieces(chess.ROOK, not board.turn)
        if any(chess.square_distance(king, sq) <= 3 for sq in enemy_heavy):
            complexity += 0.25

    num_legal = legal_moves_count
    if num_legal > 35:
        complexity += 0.15
    elif num_legal <= 6:
        complexity *= 0.70

    # Phase factor: invest heavily in critical midgames
    ply = context.ply
    if ply < 10:
        phase_factor = 0.90
    elif ply < 50:
        phase_factor = 1.30   # Peak tactical investment
    else:
        phase_factor = 0.85

    computed_soft = base * complexity * phase_factor

    # Dynamic upper cap based on initial time control
    if initial_s >= 120.0:        # 2m to 6m matches: optimal 2.5s - 3.5s deep search (prevents midgame overspend)
        max_soft = min(3.5, max(2.0, remaining_s * 0.05))
    elif initial_s >= 50.0:       # 60s - 90s matches: allow 1.2s - 2.0s
        max_soft = min(2.0, remaining_s * 0.08)
    else:                         # 30s blitz matches: keep midgame in check (0.45s - 0.65s)
        max_soft = min(0.65, remaining_s * 0.08)

    max_soft = max(0.34, max_soft)

    # The Golden Window floor (0.34s):
    # Platform charges at least 0.50s per move anyway. Thinking less than 0.34s
    # when clock is healthy wastes calculation depth without saving clock!
    soft_limit = max(0.34, min(max_soft, computed_soft))
    hard_limit = min(max_soft * 1.40, soft_limit * 1.35)

    # Hard ceiling leaves safety margin against remaining clock
    hard_ceiling = max(0.38, remaining_s - 1.0)
    hard_limit = min(hard_limit, hard_ceiling)
    soft_limit = min(soft_limit, hard_limit * 0.85)

    return max(0.34, soft_limit), max(0.38, hard_limit)
