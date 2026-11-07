"""Thuật toán tìm kiếm có đối thủ cho cờ vua.

Ghép từ các kỹ thuật đã học:
    - Minimax + cắt nhánh Alpha-Beta   (04_adversarial search, slide "Cài đặt Alpha-Beta")
    - Giới hạn độ sâu + hàm ước lượng   (04, slide "Giới hạn tài nguyên")
    - Tìm kiếm độ sâu tăng dần (IDS)    (02_uninformed search, slide "Tìm kiếm độ sâu tăng dần";
                                         04 nói "dùng IDS để thoát khi cần")
    - Sắp xếp nước đi                   (04: "thứ tự duyệt quyết định mức cắt nhánh")
    - Bảng chuyển vị = tập đóng         (03_informed search, slide "Tìm kiếm trên đồ thị":
                                         "không mở rộng một trạng thái 2 lần")
    - Tìm kiếm tĩnh (quiescence)        (mở rộng độ sâu ở thế chưa "yên tĩnh" để giảm sai số
                                         của hàm ước lượng khi đang giằng co ăn quân)

Ba cờ USE_* bên dưới cho phép TẮT từng kỹ thuật để so sánh (xem phần benchmark khi vấn đáp).

NEGAMAX LÀ GÌ? Trong trò chơi tổng bằng 0: giá trị MIN của một nút = -(giá trị MAX của đối thủ)
    min(a, b) = -max(-a, -b)
Nên thay vì viết hai hàm max-value / min-value như slide, ta viết MỘT hàm `negamax`, điểm luôn
tính theo góc nhìn của bên đang đi, rồi đổi dấu khi đệ quy xuống nút con. Kết quả giống hệt minimax.
"""

from __future__ import annotations

import random
import time

import chess

from .evaluation import PIECE_VALUE, evaluate

# --------------------------------------------------------------------------------------
# Công tắc để bật/tắt từng kỹ thuật (phục vụ đánh giá tác dụng của chúng).
# --------------------------------------------------------------------------------------
USE_TT = True           # bảng chuyển vị
USE_ORDERING = True     # sắp xếp nước đi (nước tốt xét trước -> cắt nhánh nhiều hơn)
USE_QUIESCENCE = True   # tìm kiếm tĩnh ở lá
USE_CHECK_EXTENSION = True  # bị chiếu thì tìm sâu thêm 1 nước
USE_HISTORY = True      # ưu tiên nước yên lặng từng gây cắt nhánh ở những nơi khác (history heuristic)
USE_DELTA_PRUNING = True  # tìm kiếm tĩnh: bỏ qua nước ăn quân mà ăn xong vẫn không thể vượt alpha
USE_ADAPTIVE_TIME = True  # thế bất ổn thì nghĩ lâu hơn, thế rõ ràng thì dừng sớm
DELTA_MARGIN = 200      # biên an toàn (centipawn) của cắt delta: ăn quân xong còn phải vượt alpha thêm ngần này mới đáng xét

MATE = 100_000          # điểm chiếu hết; chiếu hết càng sớm điểm càng cao (MATE - ply)
MATE_BOUND = MATE - 1_000  # điểm lớn hơn mức này nghĩa là "đã thấy chiếu hết"
INF = 10**9
MAX_PLY = 64            # chặn an toàn độ sâu đệ quy
TT_MAX_ENTRIES = 200_000
TIME_CHECK_INTERVAL = 512  # cứ 512 nút mới kiểm tra đồng hồ một lần (đọc đồng hồ cũng tốn thời gian)

UNSTABLE_DROP = 40  # điểm tụt quá mức này giữa hai vòng IDS thì coi là thế bất ổn

# Cờ loại giá trị lưu trong bảng chuyển vị.
EXACT, LOWER, UPPER = 0, 1, 2


class SearchTimeout(Exception):
    """Ném ra khi hết thời gian cứng để thoát nhanh khỏi đệ quy."""


class Searcher:
    """Giữ trạng thái của quá trình tìm kiếm (bảng chuyển vị, bộ đếm nút, đồng hồ)."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random(0)
        self.tt: dict = {}                 # khóa thế cờ -> (độ sâu, giá trị, cờ, nước tốt nhất)
        self.killers: list[list] = []      # killer[ply] = vài nước "yên lặng" đã gây cắt nhánh
        self.history = [[0] * 64 for _ in range(64)]  # history[từ][đến] = mức độ "hay gây cắt nhánh"
        self.nodes = 0
        self.hard_deadline = float("inf")
        self.check_time = False
        self.last_depth = 0                # độ sâu hoàn tất của lần tìm kiếm gần nhất
        self.last_score = 0

    # ----------------------------------------------------------------------------------
    # Điểm vào: tìm kiếm độ sâu tăng dần
    # ----------------------------------------------------------------------------------
    def search(
        self,
        board: chess.Board,
        soft_seconds: float,
        hard_seconds: float,
        max_depth: int = MAX_PLY,
    ) -> chess.Move:
        """Tìm nước đi tốt nhất.

        Tìm kiếm độ sâu tăng dần (IDS): chạy độ sâu 1, rồi 2, 3, ... Mỗi vòng dùng kết quả vòng
        trước để sắp xếp nước đi (nước tốt nhất đứng đầu) nên các vòng sau cắt nhánh mạnh hơn.
        Hết giờ lúc nào thì dùng kết quả của vòng ĐÃ XONG gần nhất -> luôn có nước đi hợp lệ.

        soft_seconds: không bắt đầu vòng mới nếu đã dùng quá 25-70% mốc này (xem `_start_fraction`;
                      vòng sau thường tốn gấp 3-5 lần vòng trước).
        hard_seconds: hết mốc này thì dừng ngay, bỏ vòng đang dở.
        """
        start = time.perf_counter()
        self.hard_deadline = start + hard_seconds
        self.nodes = 0
        self.killers = [[] for _ in range(MAX_PLY + 2)]
        # Giảm dần điểm lịch sử của các nước trước để thông tin cũ không lấn át thông tin mới.
        self.history = [[score // 2 for score in row] for row in self.history]

        root_moves = list(board.legal_moves)
        self.rng.shuffle(root_moves)  # phá thế hòa khi sắp xếp: các nước ngang nhau được chọn ngẫu nhiên (seed cố định)
        best_move = root_moves[0]
        if len(root_moves) == 1:
            return best_move

        stable_iterations = 0  # số vòng liên tiếp nước tốt nhất không đổi
        unstable = False       # vòng trước nước tốt nhất đổi hoặc điểm tụt mạnh
        for depth in range(1, max_depth + 1):
            if depth > 1:
                time_limit = self._start_fraction(stable_iterations, unstable) * soft_seconds
                if time.perf_counter() - start > time_limit:
                    break
            self.check_time = depth > 1  # độ sâu 1 luôn phải hoàn tất để chắc chắn có nước đi
            try:
                score, move, root_moves = self._search_root(board, root_moves, depth)
            except SearchTimeout:
                break
            if depth > 1:
                stable_iterations = stable_iterations + 1 if move == best_move else 0
                unstable = move != best_move or score < self.last_score - UNSTABLE_DROP
            best_move = move
            self.last_depth = depth
            self.last_score = score
            if abs(score) >= MATE_BOUND:  # đã thấy chiếu hết (thắng hoặc thua): đào sâu thêm vô ích
                break
        return best_move

    @staticmethod
    def _start_fraction(stable_iterations: int, unstable: bool) -> float:
        """Đã dùng quá tỉ lệ này của `soft_seconds` thì không mở vòng IDS mới.

        Thế bất ổn (nước tốt nhất vừa đổi hoặc điểm tụt) -> đáng nghĩ thêm; nước tốt nhất đã giữ nguyên
        nhiều vòng -> dừng sớm để dành giờ cho các nước khó hơn.
        """
        if not USE_ADAPTIVE_TIME:
            return 0.4
        if unstable:
            return 0.7
        if stable_iterations >= 3:
            return 0.25
        return 0.4

    def _search_root(self, board: chess.Board, root_moves: list, depth: int):
        """Một vòng IDS ở nút gốc; trả về (điểm, nước tốt nhất, danh sách nước đã sắp lại)."""
        # Ở vòng sau, danh sách `root_moves` đã được sắp theo điểm vòng trước nên nước tốt nhất đứng đầu.
        scored: list[tuple[int, chess.Move]] = []
        alpha, beta = -INF, INF
        best_score, best_move = -INF, root_moves[0]
        for move in root_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            scored.append((score, move))
            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score
        # Sắp xếp ổn định: điểm cao trước. Nước fail-low chỉ có cận trên nên thứ tự giữa chúng không quan trọng.
        scored.sort(key=lambda item: -item[0])
        ordered = [m for _, m in scored]
        # Đưa nước tốt nhất (PV) lên đầu để vòng sau xét trước.
        ordered.remove(best_move)
        ordered.insert(0, best_move)
        return best_score, best_move, ordered

    # ----------------------------------------------------------------------------------
    # Negamax + Alpha-Beta
    # ----------------------------------------------------------------------------------
    def _tick(self) -> None:
        """Đếm nút và kiểm tra đồng hồ định kỳ."""
        self.nodes += 1
        if self.check_time and self.nodes % TIME_CHECK_INTERVAL == 0:
            if time.perf_counter() >= self.hard_deadline:
                raise SearchTimeout

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        """Giá trị minimax của thế cờ (góc nhìn bên đi) với cửa sổ (alpha, beta).

        alpha: kết quả tốt nhất mà bên đi đã chắc chắn đạt được trên đường từ gốc (như α trên slide).
        beta : kết quả tốt nhất mà đối thủ đã chắc chắn đạt được; nếu giá trị >= beta thì đối thủ
               sẽ không bao giờ cho đi vào nhánh này -> cắt (như điều kiện `v >= β` trên slide).
        """
        self._tick()

        # --- Các kết thúc hòa theo luật: lặp thế, luật 50 nước, thiếu quân ---
        if board.halfmove_clock >= 100:
            return 0
        if board.halfmove_clock >= 4 and board.is_repetition(2):
            return 0  # coi lặp thế lần 2 như hòa (chuẩn trong cờ máy: nếu có thể lặp lại thì cũng lặp được lần 3)
        if chess.popcount(board.occupied) <= 8 and board.is_insufficient_material():
            return 0  # chỉ có thể xảy ra khi còn rất ít quân nên không cần kiểm tra ở mọi nút

        # --- Mở rộng khi bị chiếu: nước chiếu thường quyết định ván cờ nên không dừng ở đây ---
        if USE_CHECK_EXTENSION and ply < MAX_PLY and board.is_check():
            depth += 1

        if depth <= 0:
            if USE_QUIESCENCE:
                return self._quiescence(board, alpha, beta, ply)
            return evaluate(board)
        if ply >= MAX_PLY:
            return evaluate(board)

        # --- Bảng chuyển vị: thế đã xét rồi thì dùng lại (giống tập đóng của tìm kiếm trên đồ thị) ---
        original_alpha = alpha
        tt_move = None
        key = None
        if USE_TT:
            key = board._transposition_key()
            entry = self.tt.get(key)
            if entry is not None:
                entry_depth, entry_value, entry_flag, tt_move = entry
                if entry_depth >= depth:  # kết quả đã tính ở độ sâu >= độ sâu cần
                    entry_value = _from_tt(entry_value, ply)
                    if entry_flag == EXACT:
                        return entry_value
                    if entry_flag == LOWER and entry_value > alpha:
                        alpha = entry_value
                    elif entry_flag == UPPER and entry_value < beta:
                        beta = entry_value
                    if alpha >= beta:
                        return entry_value

        moves = self._ordered_moves(board, tt_move, ply)
        if not moves:
            # Hết nước đi: bị chiếu hết thì thua (điểm rất thấp, thua càng muộn càng đỡ), không thì hòa (bí).
            return -MATE + ply if board.is_check() else 0

        best_value = -INF
        best_move = None
        for move in moves:
            board.push(move)
            value = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value
            if alpha >= beta:  # cắt nhánh beta
                if not board.is_capture(move):
                    self._remember_killer(ply, move)
                    self.history[move.from_square][move.to_square] += depth * depth
                break

        if USE_TT:
            if best_value <= original_alpha:
                flag = UPPER   # không có nước nào vượt alpha: chỉ biết giá trị thật <= best_value
            elif best_value >= beta:
                flag = LOWER   # bị cắt: giá trị thật >= best_value
            else:
                flag = EXACT
            if len(self.tt) >= TT_MAX_ENTRIES:
                self.tt.clear()  # đơn giản: đầy thì xóa hết (tránh tốn RAM)
            self.tt[key] = (depth, _to_tt(best_value, ply), flag, best_move)
        return best_value

    # ----------------------------------------------------------------------------------
    # Tìm kiếm tĩnh (quiescence)
    # ----------------------------------------------------------------------------------
    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Ở độ sâu giới hạn, tiếp tục xét các nước ĂN QUÂN cho tới khi thế cờ "yên tĩnh".

        Lý do: nếu dừng ngay giữa chuỗi ăn - ăn lại, hàm ước lượng sẽ sai nặng (ví dụ vừa ăn
        hậu nhưng lượt sau mất xe vẫn chưa thấy: "hiệu ứng chân trời").
        "Đứng yên" (stand pat): bên đi có quyền không ăn nữa, nên điểm không thể thấp hơn eval hiện tại.
        """
        self._tick()
        stand_pat = evaluate(board)
        if ply >= MAX_PLY or stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        captures = list(board.generate_legal_captures())
        if USE_ORDERING:
            captures.sort(key=lambda m: -_mvv_lva(board, m))
        for move in captures:
            if USE_DELTA_PRUNING and not move.promotion:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                if stand_pat + PIECE_VALUE[victim] + DELTA_MARGIN <= alpha:
                    continue  # dù ăn được quân này vẫn chưa chạm alpha -> khỏi xét
            board.push(move)
            value = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if value >= beta:
                return value
            if value > alpha:
                alpha = value
        return alpha

    # ----------------------------------------------------------------------------------
    # Sắp xếp nước đi
    # ----------------------------------------------------------------------------------
    def _ordered_moves(self, board: chess.Board, tt_move, ply: int) -> list:
        """Nước có khả năng tốt xếp trước để alpha-beta cắt được nhiều (slide 04: "thứ tự duyệt thay đổi mức cắt").

        Thứ tự ưu tiên: nước từ bảng chuyển vị > ăn quân (MVV-LVA) > phong cấp > killer > còn lại.
        """
        moves = list(board.legal_moves)
        if not USE_ORDERING or len(moves) <= 1:
            return moves
        killers = self.killers[ply] if ply < len(self.killers) else []

        def priority(move: chess.Move) -> int:
            if move == tt_move:
                return 10_000_000
            if board.is_capture(move):
                return 1_000_000 + _mvv_lva(board, move)
            if move.promotion:
                return 900_000 + PIECE_VALUE[move.promotion]
            if move in killers:
                return 500_000
            if USE_HISTORY:
                return min(self.history[move.from_square][move.to_square], 100_000)
            return 0

        moves.sort(key=priority, reverse=True)
        return moves

    def _remember_killer(self, ply: int, move: chess.Move) -> None:
        """Killer move: nước yên lặng gây cắt nhánh ở độ sâu này; có thể cũng tốt ở nhánh anh em."""
        if ply >= len(self.killers):
            return
        slot = self.killers[ply]
        if move not in slot:
            slot.insert(0, move)
            del slot[2:]  # chỉ giữ 2 nước


# --------------------------------------------------------------------------------------
# Hàm phụ
# --------------------------------------------------------------------------------------
def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA: ăn quân giá trị CAO nhất (Most Valuable Victim) bằng quân giá trị THẤP nhất (Least Valuable Attacker)."""
    victim_type = board.piece_type_at(move.to_square)
    if victim_type is None:  # bắt tốt qua đường: ô đích trống nhưng vẫn ăn tốt
        victim_type = chess.PAWN
    attacker_type = board.piece_type_at(move.from_square)
    return 10 * PIECE_VALUE[victim_type] - PIECE_VALUE.get(attacker_type, 0) // 10 + (
        PIECE_VALUE[move.promotion] if move.promotion else 0
    )


def _to_tt(value: int, ply: int) -> int:
    """Điểm chiếu hết phụ thuộc ply của nút hiện tại; khi lưu vào bảng đổi thành "chiếu hết sau N nước kể từ nút này"."""
    if value >= MATE_BOUND:
        return value + ply
    if value <= -MATE_BOUND:
        return value - ply
    return value


def _from_tt(value: int, ply: int) -> int:
    """Phép ngược của `_to_tt` khi đọc từ bảng ở một ply khác."""
    if value >= MATE_BOUND:
        return value - ply
    if value <= -MATE_BOUND:
        return value + ply
    return value


def minimax_reference(board: chess.Board, depth: int) -> int:
    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0
    if depth == 0:
        return evaluate(board)
    best = -INF
    for move in moves:
        board.push(move)
        best = max(best, -minimax_reference(board, depth - 1))
        board.pop()
    return best
