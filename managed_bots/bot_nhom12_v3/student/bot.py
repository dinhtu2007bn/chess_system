from __future__ import annotations

import random
import chess

from chesslab_sdk import BotConfig, SearchContext
from .search import Searcher
from .time_manager import TimeManager, compute_budget_ms


class StudentBot:
    """
    Lớp điều khiển chính của bot cờ vua (Giao diện bắt buộc theo README của ChessLab).

    Lớp này được hệ thống chấm (grader) và môi trường thử nghiệm local (runner)
    khởi tạo một lần duy nhất ở đầu ván đấu thông qua cấu hình BotConfig, sau đó
    gọi hàm choose_move() tại mỗi lượt đi của bot.
    """

    def __init__(self, config: BotConfig) -> None:
        """
        Khởi tạo bot với cấu hình ván đấu.

        Args:
            config: Đối tượng BotConfig chứa:
                - color: Màu quân của bot (chess.WHITE hoặc chess.BLACK).
                - chess960_id: Mã định danh thế cờ Chess960 (0 đến 959).
                - initial_time_ms: Tổng thời gian ban đầu của bot (mili-giây).
                - seed: Hạt giống ngẫu nhiên cố định cho ván đấu.
        """
        self.config = config
        self.searcher = Searcher()
        if hasattr(config, "seed") and config.seed is not None:
            random.seed(config.seed)

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        """
        Lựa chọn nước đi tối ưu cho bot trong giới hạn thời gian cho phép.

        Quy trình xử lý theo kiến trúc phòng thủ 3 tầng (3-layer defense):
        1. Tầng 1 (Kiểm tra biên): Xử lý nhanh các trường hợp đặc biệt:
           - Nếu không có nước đi hợp lệ: Trả về chess.Move.null().
           - Nếu chỉ có đúng 1 nước đi hợp lệ (forced move): Đi ngay lập tức
             trong 0ms để tiết kiệm tối đa ngân sách thời gian cho các nước sau.
        2. Tầng 2 (Tìm kiếm & Quản trị thời gian):
           - Tính toán ngân sách thời gian (ms) dựa trên thời gian còn lại và độ trễ hệ thống.
           - Khởi tạo TimeManager và thực thi thuật toán tìm kiếm Iterative Deepening.
           - Bọc khối try...except chống sập: nếu xảy ra bất kỳ ngoại lệ nào,
             ngay lập tức fallback về nước đi hợp lệ đầu tiên (legal[0]).
        3. Tầng 3 (Hậu kiểm tính hợp lệ):
           - Xác minh chắc chắn nước đi trả về nằm trong board.legal_moves
             để không bao giờ bị trọng tài xử thua do trả nước đi sai luật.

        Args:
            board: Trạng thái bàn cờ hiện tại (đối tượng chess.Board hỗ trợ Chess960).
            context: Ngữ cảnh tìm kiếm gồm:
                - time_remaining_ms: Thời gian còn lại của bot (ms).
                - opponent_time_remaining_ms: Thời gian còn lại của đối thủ (ms).
                - ply: Số nửa nước đi (half-moves) tính từ đầu ván đấu.

        Returns:
            chess.Move: Nước đi hợp lệ được chọn cho thế cờ hiện tại.
        """
        # TẦNG 1: Lấy sẵn danh sách nước đi hợp lệ và tối ưu nước đi bắt buộc
        legal = list(board.legal_moves)
        if not legal:
            return chess.Move.null()

        # Tối ưu nước đi duy nhất: đi ngay trong 0ms, không tiêu hao tài nguyên tính toán
        if len(legal) == 1:
            return legal[0]

        # TẦNG 2: Thiết lập thời gian và gọi bộ máy tìm kiếm
        budget = compute_budget_ms(
            time_remaining_ms=context.time_remaining_ms,
            ply=context.ply,
            initial_time_ms=self.config.initial_time_ms,
        )
        tm = TimeManager(budget)

        try:
            move = self.searcher.search(board, tm)
        except Exception:
            # Cơ chế tự phục hồi (Fallback an toàn): nếu thuật toán search gặp sự cố,
            # lấy ngay nước đi hợp lệ đầu tiên để tránh bị xử thua do crash hoặc văng ngoại lệ
            move = legal[0]

        # TẦNG 3: Xác minh tính hợp lệ tuyệt đối trước khi bàn giao cho hệ thống
        if move not in board.legal_moves:
            move = legal[0]

        return move
