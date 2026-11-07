import time
import chess
from chesslab_sdk import BotConfig, SearchContext
from .search import SearchEngine, SearchTimeout, MATE_THRESHOLD
from .time_manager import allocate_move_time
from .move_ordering import order_moves


class StudentBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config

        # Bảng chuyển vị được giữ lại giữa các lượt trong cùng ván.
        self.table = {}

        # History lưu những nước yên tĩnh từng giúp Alpha-Beta cắt nhánh.
        self.history = {}

        # History được giảm dần sau mỗi tám ply.
        self.last_history_decay_ply = -8

        # Thông tin dùng để theo dõi kết quả tìm kiếm.
        self.last_depth = 0
        self.last_nodes = 0
        self.last_score = 0
        self.last_tt_hits = 0

    def choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        """Chọn nước đi; nếu quá trình tìm kiếm gặp lỗi thì trả nước hợp lệ dự phòng."""

        legal_moves = list(board.legal_moves)

        if not legal_moves:
            raise ValueError("No legal moves available")

        try:
            move = self._choose_move(board, context)

            if move in legal_moves:
                return move

            print("Search returned an illegal move; using fallback.")

        except Exception as exc:
            # Không im lặng: lỗi phải hiện ra để còn sửa nguyên nhân gốc.
            print(f"Search error: {type(exc).__name__}: {exc}")

        return legal_moves[0]

    def _choose_move(self, board: chess.Board, context: SearchContext) -> chess.Move:
        """Chọn một nước hợp lệ trong thời gian được cấp."""

        started = time.monotonic()
        moves = list(board.legal_moves)

        if not moves:
            raise ValueError("No legal moves available")

        # Đặt lại thông tin thống kê cho lượt hiện tại.
        self.last_depth = 0
        self.last_nodes = 0
        self.last_score = 0
        self.last_tt_hits = 0

        # Luôn giữ sẵn một nước hợp lệ để trả về khi hết thời gian.
        best = moves[0]

        if len(moves) == 1:
            return best

        remaining = max(0.0, context.time_remaining_ms / 1000)

        # Khi đồng hồ còn quá ít, trả ngay nước dự phòng.
        if remaining <= 0.75:
            return best

        # Sử dụng move ordering để chọn nước dự phòng tốt hơn.
        best = order_moves(board)[0]

        # Ngân sách mềm do time manager cung cấp.
        budget = allocate_move_time(context)

        if budget < 0.015:
            # Gần hết giờ: vẫn search ngắn thay vì chỉ lấy nước ăn quân đầu tiên.
            # Hệ thống luôn trễ tối thiểu 0,5 giây mỗi nước nên 0,3 giây search này không tốn thêm giờ.
            if remaining <= 1.5:
                return best

            budget = 0.3

        # Hard deadline là giới hạn search tuyệt đối không được vượt qua.
        hard = min(budget * 1.35, max(0.0, remaining - 0.75))

        if hard <= 0:
            return best

        deadline = started + hard

        # Giới hạn kích thước bảng chuyển vị.
        if len(self.table) > 50000:
            self.table.clear()

        # Giảm dần history sau mỗi tám ply để dữ liệu cũ không giữ điểm ưu tiên quá lâu.
        if (context.ply - self.last_history_decay_ply >= 8):
            for key in list(self.history):
                value = self.history[key] * 3 // 4

                if value < 4:
                    del self.history[key]
                else:
                    self.history[key] = value

            self.last_history_decay_ply = context.ply

        # Truyền bảng chuyển vị và history dùng chung cho search.
        engine = SearchEngine(board, deadline, self.table, context.ply, self.history)

        previous_elapsed = 0.0
        previous_iteration = None
        previous_move = None
        previous_score = None
        stable_depths = 0

        # Iterative Deepening Search: tìm lần lượt từ độ sâu nhỏ đến độ sâu lớn.
        for depth in range(1, 65):
            if depth > 1:
                elapsed = time.monotonic() - started

                # Khi kết quả ổn định qua nhiều độ sâu, bot có thể dừng sớm để tiết kiệm thời gian.
                if stable_depths >= 2:
                    soft_limit = budget * 0.80
                else:
                    # Khi kết quả chưa ổn định, bot được dùng thêm một phần ngân sách dự phòng.
                    soft_limit = budget * 1.15

                if elapsed >= min(soft_limit, hard):
                    break

                # Không bắt đầu tầng tiếp theo nếu dự đoán rằng tầng đó có thể vượt hard deadline.
                if (previous_iteration is not None and elapsed + 2.0 * previous_iteration >= hard):
                    break

            try:
                move, score = engine.search(board, depth, best)
            except SearchTimeout:
                break

            # Chỉ lưu kết quả của độ sâu đã hoàn thành.
            best = move
            self.last_depth = depth
            self.last_score = score

            # Khi đã tìm thấy kết quả chiếu hết thì không cần tìm sâu hơn.
            if abs(score) >= MATE_THRESHOLD:
                break

            elapsed = time.monotonic() - started

            # Ước lượng thời gian của lần tìm kiếm vừa hoàn thành.
            previous_iteration = max(0.0001, elapsed - previous_elapsed)

            previous_elapsed = elapsed

            # Một kết quả được coi là ổn định khi cùng chọn một nước và điểm đánh giá không thay đổi quá 30 centipawn.
            if (previous_move == move and previous_score is not None and abs(score - previous_score) <= 30):
                stable_depths += 1
            else:
                stable_depths = 0

            previous_move = move
            previous_score = score

        self.last_nodes = engine.nodes
        self.last_tt_hits = engine.tt_hits

        # best luôn là nước thuộc board.legal_moves.
        return best