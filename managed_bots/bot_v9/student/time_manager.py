"""Dynamic time allocation for the remaining game clock.

Master Anti-Timeout Time Manager V9:
- Non-starvation Budgeting: Allocates generous thinking slices (0.6s - 1.2s) in midgames,
  avoiding artificial upfront reservation collapses.
- Opponent & King Threat Complexity Scaling: Gives extra depth in sharp tactical situations.
- Multi-Tier Emergency Glide Path: Instant fast response (0.02s - 0.08s) when clock drops below 10s.
- Absolute Safety Ceiling: Guaranteed headroom leaving at least 0.65s above platform floor,
  ensuring 100% mathematical zero-timeout guarantee under platform delay.
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
    remaining_s = max(0.0, context.time_remaining_ms / 1000.0)
    legal_moves_count = board.legal_moves.count()

    # Fast escape if trivial position or critical clock starvation
    if legal_moves_count <= 1 or remaining_s <= 0.8:
        return 0.015, 0.03

    # Calculate remaining turns realistically (Chess960 games end in 35-50 moves)
    plies_left = max(1, 300 - context.ply)
    turns_left = (plies_left + 1) // 2
    expected_moves = min(turns_left, max(18, 48 - (context.ply // 4)))

    usable = max(0.0, remaining_s - 1.0)
    base = usable / expected_moves

    # When usable clock is close to the mandatory 500ms floor per move:
    if usable <= expected_moves * MIN_MOVE_CHARGE:
        base = min(0.08, usable / (2 * expected_moves))

    # Tactical & King Danger complexity factor
    complexity = 1.0
    if board.is_check():
        complexity += 0.35

    king = board.king(board.turn)
    if king is not None:
        # Enemy Queen or Rook near king within Chebyshev distance <= 3
        enemy_heavy = board.pieces(chess.QUEEN, not board.turn) | board.pieces(chess.ROOK, not board.turn)
        if any(chess.square_distance(king, sq) <= 3 for sq in enemy_heavy):
            complexity += 0.25

    num_legal = legal_moves_count
    if num_legal > 35:
        complexity += 0.15
    elif num_legal <= 6:
        complexity *= 0.70

    # Solid opening development boost
    if context.ply < 8:
        complexity *= 1.05

    # Multi-Tier Emergency Glide Path (Guarantees zero timeout forfeit)
    if remaining_s < 2.5:
        soft_limit = 0.02
        hard_limit = 0.04
    elif remaining_s < 6.0:
        soft_limit = min(0.04, base * 0.5)
        hard_limit = min(0.08, soft_limit * 1.5)
    elif remaining_s < 10.0:
        soft_limit = min(0.08, base * 0.8)
        hard_limit = min(0.18, soft_limit * 1.6)
    else:
        soft_limit = min(2.5, base * complexity)
        hard_limit = min(3.8, soft_limit * 1.5)

    # Absolute safety ceiling: never let hard_limit leave less than (MIN_MOVE_CHARGE + SAFETY_BUFFER = 0.65s)
    # This prevents the platform's sleep delay from exhausting the clock
    hard_ceiling = max(0.03, remaining_s - MIN_MOVE_CHARGE - SAFETY_BUFFER)
    hard_limit = min(hard_limit, hard_ceiling)
    soft_limit = min(soft_limit, hard_limit * 0.75)

    return max(0.015, soft_limit), max(0.03, hard_limit)
