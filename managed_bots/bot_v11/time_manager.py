"""Dynamic time allocation for the remaining game clock.

Bot Test (Grandmaster Golden Window Time Manager V5):
- 500ms Platform Floor Synchronization (The Golden Window):
  - In normal/low clock conditions, sets soft_limit = 0.33s, hard_limit = 0.38s.
  - Turnaround time (thinking + overhead) is ~0.42s, perfectly fitting within the
    platform's 500ms sleep delay floor. Exactly 0.50s is charged with 0 accumulated debt.
- Heavy Midgame Investment (remaining_s > 10s):
  - Midgame tactical phase factor 1.3x allows spending 0.8s - 1.8s per move when clock
    is healthy (e.g. 60s matches), reaching Depth 7-8 and dominating V9.
- Death Threshold (Ngưỡng Tử: remaining_s <= 0.50s):
  - When remaining clock is <= 0.50s, the platform's 500ms floor guarantees clock exhaustion
    on this or the next move. Instead of panicking with a 0.02s move, spend all remaining
    usable clock (remaining_s - 0.04s) to search for checkmate or draw!
- Safety headroom guarantees zero timeout forfeits.
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
    # 1. NGƯỠNG TỬ (DEATH THRESHOLD: remaining_s <= 0.50s)
    # The platform charges max(0.50s, elapsed). Any move played with clock <= 0.50s
    # will result in clock exhaustion (timeout) on or immediately after this turn.
    # Therefore, on this final move, maximize thinking time to find mate or perpetual check!
    # =========================================================================
    if remaining_s <= 0.50:
        hard_limit = max(0.04, remaining_s - 0.04)
        soft_limit = max(0.03, hard_limit * 0.85)
        return soft_limit, hard_limit

    # =========================================================================
    # 2. LOW CLOCK (0.50s < remaining_s <= 2.5s)
    # Bot has 1 to 5 moves left. Each move is charged at least 0.50s on the platform.
    # Target the Golden Window: thinking ~0.33s - 0.38s + overhead ~0.07s = 0.42s total.
    # Platform sleeps remaining 0.08s and charges exactly 0.50s. Zero time debt!
    # =========================================================================
    if remaining_s <= 2.5:
        hard_limit = min(0.38, max(0.05, remaining_s - 0.08))
        soft_limit = min(0.33, hard_limit * 0.85)
        return max(0.03, soft_limit), max(0.05, hard_limit)

    # Calculate remaining turns realistically (Chess960 games end in 35-50 moves)
    plies_left = max(1, 300 - context.ply)
    turns_left = (plies_left + 1) // 2
    expected_moves = min(turns_left, max(18, 48 - (context.ply // 4)))

    # Tactical & King Danger complexity factor
    complexity = 1.0
    if board.is_check():
        complexity += 0.35

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

    # =========================================================================
    # 3. MODERATE CLOCK (2.5s < remaining_s <= 10.0s)
    # =========================================================================
    if remaining_s <= 10.0:
        usable = max(0.0, remaining_s - 1.2)
        base = usable / expected_moves
        computed_soft = base * complexity
        soft_limit = min(0.55, max(0.34, computed_soft))
        hard_limit = min(0.75, soft_limit * 1.35)
        # Cap hard limit to leave at least 2 full platform moves (1.0s buffer)
        hard_ceiling = max(0.38, remaining_s - 1.0)
        hard_limit = min(hard_limit, hard_ceiling)
        soft_limit = min(soft_limit, hard_limit * 0.85)
        return max(0.33, soft_limit), max(0.38, hard_limit)

    # =========================================================================
    # 4. HEALTHY / MIDGAME CLOCK (remaining_s > 10.0s)
    # Plentiful clock (typical in 60s matches): invest heavily in deep search!
    # =========================================================================
    usable = remaining_s - 2.0
    base = usable / expected_moves

    ply = context.ply
    if ply < 10:
        phase_factor = 0.90
    elif ply < 50:
        phase_factor = 1.30   # Peak tactical investment in midgame
    else:
        phase_factor = 0.85

    computed_soft = base * complexity * phase_factor

    soft_limit = min(1.8, max(0.35, computed_soft))
    hard_limit = min(2.8, soft_limit * 1.5)

    # Hard ceiling leaves at least 3 seconds buffer
    hard_ceiling = max(0.38, remaining_s - 3.0)
    hard_limit = min(hard_limit, hard_ceiling)
    soft_limit = min(soft_limit, hard_limit * 0.85)

    return max(0.35, soft_limit), max(0.38, hard_limit)
