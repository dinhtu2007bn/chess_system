"""Chia quỹ thời gian không có increment; mọi giá trị nội bộ tính bằng giây."""

from dataclasses import dataclass

import chess
from chesslab_sdk import SearchContext


@dataclass(frozen=True)
class TimeBudget:
    soft_seconds: float
    hard_seconds: float

    def should_start_iteration(self, elapsed: float, last_duration: float,
                               previous_duration: float) -> bool:
        """Ước lượng vòng kế tiếp; deadline cứng vẫn bảo vệ khi dự đoán sai."""
        growth = 3.0
        if previous_duration > 0:
            growth = max(2.0, min(6.0, last_duration / previous_duration))
        predicted = last_duration * growth
        return (elapsed < self.soft_seconds
                and elapsed + predicted < self.hard_seconds)


class TimeManager:
    MIN_MOVE_CHARGE = 0.5
    SAFETY_RESERVE = 0.15

    def allocate(self, board: chess.Board, context: SearchContext) -> TimeBudget:
        remaining = max(0.0, context.time_remaining_ms / 1000.0)
        legal_moves = list(board.legal_moves)
        if len(legal_moves) <= 1 or remaining <= 1.0:
            return TimeBudget(0.0, 0.0)

        # Không vượt số lượt của mình còn lại trước giới hạn 300 ply.
        plies_left = max(1, 300 - context.ply)
        turns_left = (plies_left + 1) // 2
        expected_moves = min(turns_left, max(20, 50 - context.ply // 2))
        usable = max(0.0, remaining - 1.0)
        base = usable / expected_moves

        # Mỗi lượt dù trả ngay vẫn mất ít nhất 500 ms. Khi không đủ quỹ
        # cho số lượt dự kiến, dùng ngân sách nhỏ để giữ nước dự phòng.
        if usable <= expected_moves * self.MIN_MOVE_CHARGE:
            base = min(0.08, usable / (2 * expected_moves))

        complexity = 1.0
        if board.is_check():
            complexity += 0.35
        king = board.king(board.turn)
        if king is not None:
            # Hậu áp sát vua có thể báo đòn chiếu ngay cả khi chưa bị chiếu.
            if any(chess.square_distance(king, queen) <= 3
                   for queen in board.pieces(chess.QUEEN, not board.turn)):
                complexity += 0.3
        tactical_moves = sum(board.is_capture(move) or bool(move.promotion)
                             for move in legal_moves)
        if tactical_moves >= 4:
            complexity += 0.2
        if len(legal_moves) <= 5:
            complexity *= 0.7

        # Quỹ nhỏ: giảm thời gian nghĩ, không cố dùng hết phần được chia.
        if remaining < 10.0:
            base = min(base, 0.05)
        soft = min(3.0, base * complexity)
        hard = min(4.5, soft * 1.5,
                   max(0.0, remaining - self.MIN_MOVE_CHARGE - self.SAFETY_RESERVE))
        return TimeBudget(min(soft, hard), hard)
