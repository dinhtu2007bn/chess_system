from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chess
    from chesslab_sdk import SearchContext


class TimeManager:
    """Fast fixed-slice time manager for aggressive bullet / blitz play."""

    def __init__(self, safety_margin_ms: int = 100) -> None:
        self.safety_margin_ms = safety_margin_ms

    def allocate_time(self, board: chess.Board, context: SearchContext) -> tuple[float, float]:
        remaining_ms = max(50, context.time_remaining_ms - self.safety_margin_ms)
        target = min(0.35, remaining_ms / 30000.0)
        hard = min(0.7, target * 1.5)
        return max(0.08, target), max(0.15, hard)
