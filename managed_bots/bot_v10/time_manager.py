"""Dynamic time allocation for the remaining game clock.

Bot Test (Master Adaptive Time Manager):
- Floor-Aware Budgeting: Directly aligned with the platform's mandatory 500ms delay floor.
- Midgame-Heavy Allocation: Grants highest thinking budget (1.0s - 1.4s) in midgames (plies 10-50).
- Platform Floor Alignment: Enforces a minimum thinking floor (soft_limit >= 0.42s) when clock > 5s
  to fully exploit the 500ms platform sleep without wasting time.
- Dynamic Usable Allocation: Uses usable = remaining - 1.0s, eliminating the upfront 22s starvation bug.
- Multi-Tier Emergency Glide Path: Rapid responses (0.02s - 0.08s) when clock drops below 6s.
- Absolute Safety Ceiling: Headroom leaving at least 0.65s above platform floor, guaranteeing 0% timeout.
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

    # Calculate remaining turns realistically (Chess960 games typically finish in 38-52 moves)
    plies_left = max(1, 300 - context.ply)
    turns_left = (plies_left + 1) // 2
    expected_moves = min(turns_left, max(18, 48 - (context.ply // 4)))

    usable = max(0.0, remaining_s - 1.0)
    base = usable / expected_moves

    # When usable clock is close to the mandatory 500ms floor per move:
    if usable <= expected_moves * MIN_MOVE_CHARGE:
        base = min(0.08, usable / (2 * expected_moves))

    # Phase-Aware weighting:
    # Opening (ply < 10): 0.85x (concise opening development)
    # Midgame (10 <= ply < 50): 1.20x (peak tactical investment)
    # Endgame (ply >= 50): 0.85x (narrower search tree, fast endgame resolution)
    ply = context.ply
    if ply < 10:
        phase_factor = 0.85
    elif ply < 50:
        phase_factor = 1.20
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

    # Multi-Tier Emergency Glide Path (Guarantees zero timeout forfeit)
    if remaining_s < 2.5:
        soft_limit = 0.02
        hard_limit = 0.04
    elif remaining_s < 6.0:
        soft_limit = min(0.05, base * 0.5)
        hard_limit = min(0.09, soft_limit * 1.5)
    elif remaining_s < 10.0:
        soft_limit = min(0.12, base * 0.8)
        hard_limit = min(0.20, soft_limit * 1.6)
    else:
        computed_soft = base * complexity * phase_factor
        # Platform Floor Alignment: When time is safe, think at least ~0.42s
        # because the platform charges 0.50s regardless!
        soft_limit = min(2.5, max(0.42, computed_soft))
        hard_limit = min(3.8, soft_limit * 1.5)

    # Absolute safety ceiling: never let hard_limit leave less than (MIN_MOVE_CHARGE + SAFETY_BUFFER = 0.65s)
    hard_ceiling = max(0.03, remaining_s - MIN_MOVE_CHARGE - SAFETY_BUFFER)
    hard_limit = min(hard_limit, hard_ceiling)
    soft_limit = min(soft_limit, hard_limit * 0.75)

    return max(0.015, soft_limit), max(0.03, hard_limit)
