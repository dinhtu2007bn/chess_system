from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chess
    from chesslab_sdk import SearchContext


class TimeManager:
    """Intelligent dynamic time manager for Chess960 games.
    
    Prevents timeouts by calculating optimal move time budgets with safety margins,
    taking into account remaining game clock, game phase, and tactical complexity.
    """

    def __init__(self, safety_margin_ms: int = 150) -> None:
        self.safety_margin_ms = safety_margin_ms

    def allocate_time(self, board: chess.Board, context: SearchContext) -> tuple[float, float]:
        """Calculates (target_time_seconds, hard_limit_seconds) for the current move.
        
        target_time: Expected duration of iterative deepening.
        hard_limit: Hard stop threshold where search MUST abort to avoid timeout.
        """
        remaining_ms = max(50, context.time_remaining_ms - self.safety_margin_ms)
        ply = context.ply

        # Estimate remaining moves in the game
        # Typical chess game lasts ~40 moves (80 ply), tapering towards end
        moves_played = ply // 2
        if moves_played < 10:
            expected_moves_left = 35
        elif moves_played < 30:
            expected_moves_left = 30
        elif moves_played < 50:
            expected_moves_left = 20
        else:
            expected_moves_left = max(10, 60 - moves_played)

        # Base time per move
        base_ms = remaining_ms / expected_moves_left

        # Adjust for tactical complexity (number of legal moves, captures available)
        legal_moves_count = board.legal_moves.count()
        captures_count = sum(1 for m in board.legal_moves if board.is_capture(m))
        is_in_check = board.is_check()

        factor = 1.0
        if is_in_check:
            factor *= 1.25
        if captures_count >= 4:
            factor *= 1.2
        if legal_moves_count > 35:
            factor *= 1.15
        elif legal_moves_count < 10:
            factor *= 0.85

        # In extreme time trouble (< 3s left), allocate small fraction to survive
        if remaining_ms < 3000:
            allocated_ms = min(remaining_ms * 0.35, max(80.0, remaining_ms / 10.0))
            hard_limit_ms = remaining_ms * 0.75
        elif remaining_ms < 8000:
            allocated_ms = min(remaining_ms * 0.25, base_ms * factor)
            hard_limit_ms = min(remaining_ms * 0.6, allocated_ms * 1.8)
        else:
            allocated_ms = min(remaining_ms * 0.15, max(150.0, base_ms * factor))
            hard_limit_ms = min(remaining_ms * 0.4, allocated_ms * 2.2)

        # Ensure bounds
        allocated_ms = max(50.0, min(allocated_ms, remaining_ms * 0.8))
        hard_limit_ms = max(allocated_ms + 40.0, min(hard_limit_ms, remaining_ms * 0.95))

        return allocated_ms / 1000.0, hard_limit_ms / 1000.0
