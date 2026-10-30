"""Dynamic time allocation for the remaining game clock.

Bot Test (Grandmaster Calibrated Time Manager V3):
- Platform Floor Aware: Protects and utilizes the mandatory 500ms platform delay per move.
- Midgame-Heavy Allocation: Grants 1.5s - 2.5s in complex midgame (plies 10-50).
- Platform Floor Alignment: Enforces soft_limit >= 0.42s whenever remaining_s > 0.5s to fully
  exploit the 500ms platform sleep window without wasting clock.
- Death Threshold (Ngưỡng Tử) Handling: When remaining_s <= 0.50s, timeout on the subsequent
  move is guaranteed by the platform's 500ms floor. Instead of wasting this final move with
  a 0.02s panic move, use all remaining clock (remaining_s - 0.02s) to search for checkmate or draw!
- No Over-Throttling: When remaining_s > 0.5s, never throttle below 0.42s.
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
        hard_limit = max(0.03, remaining_s - 0.02)
        soft_limit = max(0.02, remaining_s - 0.04)
        return soft_limit, hard_limit

    # =========================================================================
    # 2. LOW CLOCK (0.50s < remaining_s <= 2.0s)
    # Bot has 1 to 4 moves left. Each move is charged at least 0.50s.
    # Allow thinking ~0.42s (within the 0.50s floor) with hard_limit capped before deadline.
    # =========================================================================
    if remaining_s <= 2.0:
        hard_limit = max(0.44, remaining_s - 0.05)
        soft_limit = min(0.42, hard_limit * 0.85)
        return max(0.02, soft_limit), max(0.03, hard_limit)

    # =========================================================================
    # 3. NORMAL / MIDGAME ALLOCATION (remaining_s > 2.0s)
    # =========================================================================
    plies_left = max(1, 300 - context.ply)
    turns_left = (plies_left + 1) // 2
    expected_moves = min(turns_left, max(18, 48 - (context.ply // 4)))

    usable = max(0.0, remaining_s - 1.0)
    base = usable / expected_moves

    # Phase-Aware weighting (like bot_v10):
    ply = context.ply
    if ply < 10:
        phase_factor = 0.85
    elif ply < 50:
        phase_factor = 1.25   # Peak tactical investment in midgame
    else:
        phase_factor = 0.85

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

    computed_soft = base * complexity * phase_factor

    # Floor-Aware: Always at least 0.42s (since platform charges 0.50s anyway),
    # up to 2.5s in complex midgame!
    if remaining_s > 10.0:
        soft_limit = min(2.5, max(0.42, computed_soft))
        hard_limit = min(3.8, soft_limit * 1.5)
    elif remaining_s > 5.0:
        soft_limit = min(1.2, max(0.42, computed_soft * 0.8))
        hard_limit = min(1.8, soft_limit * 1.5)
    else:  # 2.0 < remaining_s <= 5.0
        soft_limit = 0.42
        hard_limit = min(0.65, remaining_s - 0.15)

    # Hard ceiling to guarantee no overrun
    hard_ceiling = max(0.44, remaining_s - MIN_MOVE_CHARGE - 0.10)
    hard_limit = min(hard_limit, hard_ceiling)
    soft_limit = min(soft_limit, hard_limit * 0.80)

    return max(0.42, soft_limit), max(0.44, hard_limit)
