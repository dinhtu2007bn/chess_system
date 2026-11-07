"""
Module search.py – Bộ máy tìm kiếm cờ vua chuyên sâu (Search Engine Core).

Kiến trúc thuật toán & Kỹ thuật tối ưu hóa:
1. Thuật toán tìm kiếm gốc:
   - Negamax kết hợp Alpha-Beta Pruning (biến thể đối xứng toán học của Minimax).
   - Iterative Deepening Search (IDS - Đào sâu lặp): Duyệt từ độ sâu 1 đến MAX_DEPTH,
     đảm bảo luôn có nước đi dự phòng tốt nhất khi hết thời gian.
   - Aspiration Windows (Cửa sổ kỳ vọng): Thu hẹp khoảng tìm kiếm [alpha, beta]
     quanh điểm số tầng trước để tăng tối đa tốc độ cắt tỉa ở gốc.
   - Principal Variation Search (PVS / NegaScout): Tìm kiếm nước đi đầu tiên với
     cửa sổ đầy đủ, các nước sau tìm với cửa sổ hẹp [alpha, alpha + 1] (Zero-Window).

2. Kỹ thuật cắt tỉa nâng cao (Advanced Pruning & Reductions):
   - Transposition Table (TT) với Zobrist Key: Bảng băm lưu trữ trạng thái O(1).
   - Reverse Futility Pruning (RFP / Static Null Move): Cắt tỉa sớm ở depth 1-2 nếu thế cờ tĩnh vượt trội beta.
   - Null Move Pruning (NMP): Nhường một lượt đi để kiểm tra xem đối thủ có tạo được đe dọa không.
   - Futility Pruning (FP): Cắt tỉa các nước đi yên lặng ở depth 1 nếu không có khả năng nâng alpha.
   - Late Move Reductions (LMR): Giảm độ sâu tìm kiếm đối với các nước đi yên lặng xếp sau.
   - Quiescence Search (QS): Tìm kiếm tĩnh lặng triệt tiêu hiệu ứng chân trời (Horizon Effect),
     kết hợp Stand-pat, Delta Pruning và sinh nước bắt quân bằng bitboard.

3. Sắp xếp nước đi (Move Ordering Heuristic):
   - Nước đi từ Transposition Table (TT Move).
   - Phong cấp Hậu (Queen Promotion).
   - Nước bắt quân có lợi (Good Captures với SEE >= 0) kết hợp MVV-LVA.
   - Killer Moves Heuristic (nước đi sát thủ gây cắt tỉa beta).
   - History Heuristic (bảng lịch sử các nước đi thành công).
   - Nước bắt quân bất lợi (Bad Captures với SEE < 0).
   - Nước đi yên lặng (Quiet Moves).
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple
import chess

from .time_manager import TimeManager
from .evaluation import evaluate

# =========================================================================
# 1. HẰNG SỐ VÀ CÁC CỜ CỦA TRANSPOSITION TABLE
# =========================================================================

# Kích thước bảng băm: 2^18 = 262,144 mục (vừa vặn bộ nhớ RAM, tốc độ tra cứu O(1))
# Tương ứng "Càn Khôn Giới Chỉ" - lưu trữ trạng thái và ký ức thế cờ
TT_SIZE = 1 << 19

# Điểm quy ước chiếu hết (Checkmate score): 30,000 centipawns
# Tương ứng "Tuyệt Sát Đạo" - cảnh giới đoạt mạng tối cao trong cờ vua
MATE_SCORE = 30_000

# Ngưỡng nhận diện điểm chiếu hết: Các giá trị >= 29,000 được xem là chuỗi chiếu hết
MATE_THRESHOLD = 29_000

# Điểm hòa cờ cân bằng (Draw score): 0 centipawn (Cục Diện Thái Hòa)
DRAW_SCORE = 0

# Độ sâu tìm kiếm tối đa trên lý thuyết
MAX_DEPTH = 64

# Bảng giá trị vật chất quy chuẩn phục vụ tính toán Static Exchange Evaluation (SEE)
PIECE_VALS = {
    None: 0,
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 1000,
    chess.KING: 20000,
}

# Điểm ưu tiên quân bị ăn và quân đi ăn (Most Valuable Victim - Least Valuable Attacker: MVV-LVA)
_MVV_LVA_SCORES = {
    None: 0,
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

# Giới hạn độ sâu tối đa cho Quiescence Search để tránh bùng nổ tổ hợp trong chuỗi đổi quân
MAX_QSEARCH_DEPTH = 8

# Ba loại cận (bound flags) lưu trong Transposition Table theo lý thuyết Alpha-Beta:
TT_EXACT = 0  # Điểm chính xác (PV-Node: alpha < score < beta)
TT_LOWER = 1  # Cận dưới (Cut-Node / Fail-High: score >= beta, nước đi quá mạnh)
TT_UPPER = 2  # Cận trên (All-Node / Fail-Low: score <= alpha, nước đi quá yếu)


# =========================================================================
# 2. HÀM CHUẨN HÓA ĐIỂM CHIẾU HẾT (MATE SCORE NORMALIZATION)
# =========================================================================

def score_to_tt(score: int, ply: int) -> int:
    """
    Chuẩn hóa điểm chiếu hết theo độ sâu (ply) trước khi lưu vào Transposition Table.

    Tại sao cần chuẩn hóa?
    Điểm chiếu hết phụ thuộc vào khoảng cách từ thế cờ hiện tại đến khi bị chiếu hết.
    Nếu ở ply = 3 ta thấy chiếu hết sau 2 nước (score = MATE - 5), khi lưu vào TT phải
    cộng thêm ply thành (MATE - 5 + 3 = MATE - 2) để thể hiện số nước chiếu hết tuyệt đối
    tính từ vị trí đó, không phụ thuộc vào vị trí nhánh cây đang đứng.

    Args:
        score: Điểm số tính toán từ nút hiện tại.
        ply: Khoảng cách từ gốc cây tìm kiếm đến nút hiện tại.

    Returns:
        int: Điểm số độc lập với ply để lưu vào TT.
    """
    if score >= MATE_THRESHOLD:
        return score + ply
    elif score <= -MATE_THRESHOLD:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    """
    Khôi phục lại điểm chiếu hết theo khoảng cách ply khi lấy từ Transposition Table ra.

    Args:
        score: Điểm số lưu trong TT (đã chuẩn hóa).
        ply: Khoảng cách từ gốc cây tìm kiếm đến nút hiện tại.

    Returns:
        int: Điểm số tương ứng với khoảng cách thực tế từ vị trí hiện tại.
    """
    if score >= MATE_THRESHOLD:
        return score - ply
    elif score <= -MATE_THRESHOLD:
        return score + ply
    return score


# =========================================================================
# 3. ĐÁNH GIÁ ĐỔI QUÂN TĨNH (STATIC EXCHANGE EVALUATION - SEE)
# =========================================================================

def see(board: chess.Board, move: chess.Move) -> int:
    """
    Static Exchange Evaluation (SEE) – "Thấu Thị Chi Nhãn" đánh giá chuỗi đổi quân tĩnh.

    Chức năng:
    Ước lượng sự chênh lệch vật chất thuần sau chuỗi đổi quân tại ô đích (move.to_square)
    mà không cần đệ quy sinh toàn bộ cây tìm kiếm.

    Nguyên lý hoạt động (1-ply fast SEE):
    1. Lấy giá trị quân bị ăn (victim) và quân đi ăn (attacker).
    2. Nếu quân đi ăn có giá trị nhỏ hơn hoặc bằng quân bị ăn (ví dụ Tốt ăn Hậu, Tượng ăn Xe):
       Chắc chắn có lợi hoặc ít nhất đổi ngang => trả về (v_val - a_val) >= 0 ngay lập tức.
    3. Nếu quân lớn ăn quân nhỏ (ví dụ Hậu ăn Tốt):
       Kiểm tra ô đích có được đối phương bảo vệ (attacked) hay không:
       - Nếu không được bảo vệ: Quân treo tự do => thu lời trọn vẹn v_val.
       - Nếu được bảo vệ: Quân tấn công sẽ bị ăn lại ngay => lỗ (v_val - a_val) < 0.

    Args:
        board: Bàn cờ trước khi thực hiện nước đi.
        move: Nước bắt quân cần thẩm định.

    Returns:
        int (centipawn):
            >= 0: Nước bắt quân có lợi hoặc đổi ngang giá trị.
            < 0: Nước bắt quân bị lỗ (ví dụ: Hậu ăn Tốt có bảo vệ).
    """
    to_sq = move.to_square
    from_sq = move.from_square

    victim = board.piece_type_at(to_sq)
    if victim is None:
        if board.is_en_passant(move):
            victim = chess.PAWN
        else:
            return 0

    attacker = board.piece_type_at(from_sq)
    if attacker is None:
        return 0

    v_val = PIECE_VALS[victim]
    a_val = PIECE_VALS[attacker]

    # Nếu quân đi ăn có giá trị <= quân bị ăn (ví dụ Tốt ăn Hậu, Xe ăn Hậu, Tượng ăn Mã...):
    # Luôn là nước bắt quân có lợi hoặc ít nhất là đổi quân ngang
    if a_val <= v_val:
        return v_val - a_val

    # Nếu quân lớn ăn quân nhỏ (ví dụ Hậu ăn Tốt):
    # Kiểm tra xem ô đích có được đối phương bảo vệ không
    opp = not board.turn
    opp_attackers = board.attackers(opp, to_sq)
    if not opp_attackers:
        return v_val  # Quân treo tự do!

    # Nếu ô đích được quân ta bảo vệ và quân duy nhất của địch ngắm tới là Vua,
    # Vua địch không thể ăn vì sẽ bị chiếu phạm luật!
    if board.is_attacked_by(board.turn, to_sq):
        king_sq = board.king(opp)
        if king_sq is not None and opp_attackers == chess.SquareSet([king_sq]):
            return v_val

    # Ô đích được bảo vệ: quân tấn công sẽ bị ăn lại và lỗ vật chất
    return v_val - a_val


# =========================================================================
# 4. BẢNG CHUYỂN VỊ (TRANSPOSITION TABLE - TT)
# =========================================================================

class TTEntry(NamedTuple):
    """
    Mục lưu trữ trong Transposition Table.

    Attributes:
        key: Khóa băm Zobrist đại diện duy nhất cho thế cờ (board._transposition_key()).
        depth: Độ sâu tìm kiếm của kết quả lưu trữ.
        score: Điểm số đánh giá (đã qua chuẩn hóa mate score).
        flag: Cận của điểm số (TT_EXACT, TT_LOWER, TT_UPPER).
        best_move: Nước đi tốt nhất tìm được tại thế cờ này.
        age: Thế hệ tìm kiếm (lượt đi của ván) phục vụ cơ chế thay thế mục cũ.
    """
    key: Any
    depth: int
    score: int
    flag: int
    best_move: chess.Move | None
    age: int


class TranspositionTable:
    """
    Bảng băm lưu trữ trạng thái tìm kiếm (Transposition Table) với kích thước cố định.

    Tương ứng "Càn Khôn Giới Chỉ":
    Ghi nhớ kết quả của các thế cờ đã duyệt qua để tái sử dụng khi thế cờ lặp lại
    theo các thứ tự nước đi khác nhau (transposition), giúp tiết kiệm hàng triệu phép tính.
    """

    def __init__(self, size: int = TT_SIZE) -> None:
        """Khởi tạo bảng băm với kích thước cố định là lũy thừa của 2."""
        self.size = size
        self._table: dict[int, TTEntry] = {}

    def probe(self, key: Any) -> TTEntry | None:
        """
        Tra cứu trạng thái bàn cờ trong bảng băm.

        Args:
            key: Khóa băm Zobrist của bàn cờ.

        Returns:
            TTEntry nếu tìm thấy và trùng khớp khóa, ngược lại trả về None.
        """
        idx = hash(key) & (self.size - 1)
        entry = self._table.get(idx)
        if entry is not None and entry.key == key:
            return entry
        return None

    def store(
        self,
        key: Any,
        depth: int,
        score: int,
        flag: int,
        best_move: chess.Move | None,
        age: int,
    ) -> None:
        """
        Lưu hoặc ghi đè trạng thái vào bảng băm theo chính sách ưu tiên:
        1. Ghi đè nếu ô băm còn trống.
        2. Ghi đè nếu trùng khóa (cập nhật thông tin mới nhất).
        3. Ghi đè nếu mục hiện tại thuộc thế hệ cũ (age != existing.age).
        4. Ghi đè nếu lượt tìm kiếm mới sâu hơn hoặc bằng (depth >= existing.depth).
        """
        idx = hash(key) & (self.size - 1)
        existing = self._table.get(idx)
        if (
            existing is None
            or existing.key == key
            or existing.age != age
            or depth >= existing.depth
        ):
            self._table[idx] = TTEntry(
                key, depth, score, flag, best_move, age
            )

    def clear(self) -> None:
        """Xóa sạch bảng băm."""
        self._table.clear()


# =========================================================================
# 5. BỘ MÁY TÌM KIẾM CHÍNH (SEARCHER)
# =========================================================================

class Searcher:
    """
    Bộ máy tìm kiếm cờ vua Negamax tích hợp đầy đủ các kỹ thuật hiện đại:
    Alpha-Beta Pruning, PVS, LMR, NMP, RFP, Futility Pruning, Quiescence Search,
    Move Ordering và Aspiration Windows.
    """

    def __init__(self) -> None:
        """Khởi tạo bộ máy tìm kiếm với bảng chuyển vị, killer moves và history table."""
        self.tt = TranspositionTable()
        # Bảng Killer Moves: 2 nước đi sát thủ cho mỗi ply (độ sâu tính từ gốc)
        self.killer_moves: list[list[chess.Move | None]] = [
            [None, None] for _ in range(MAX_DEPTH + 16)
        ]
        # Bảng History Heuristic: lưu tần suất gây cắt tỉa beta của các nước đi yên lặng
        self.history: dict[tuple[chess.Color, int, int], int] = {}
        self._timeout: bool = False
        self._nodes: int = 0
        self._age: int = 0

    def _decay_history(self) -> None:
        """Làm suy giảm điểm History giữa các lượt đi để tránh tích lũy dữ liệu lỗi thời."""
        for k in list(self.history.keys()):
            val = self.history[k] // 2
            if val > 0:
                self.history[k] = val
            else:
                del self.history[k]

    def _store_killer(self, move: chess.Move, ply: int) -> None:
        """
        Lưu nước đi sát thủ (Killer Move) gây ra cắt tỉa beta tại ply hiện tại.
        Mỗi ply lưu tối đa 2 killer moves (dịch chuyển killer cũ xuống slot 2).
        """
        if ply < len(self.killer_moves):
            if self.killer_moves[ply][0] != move:
                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                self.killer_moves[ply][0] = move

    def _update_history(self, move: chess.Move, color: chess.Color, depth: int) -> None:
        """
        Cập nhật điểm History Heuristic cho nước đi yên lặng gây cắt tỉa beta.
        Điểm cộng tỷ lệ với bình phương độ sâu (depth * depth).
        """
        key = (color, move.from_square, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth

    def _order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        ply: int,
        tt_move: chess.Move | None,
    ) -> list[chess.Move]:
        """
        Sắp xếp danh sách nước đi theo độ ưu tiên Heuristic (Move Ordering).

        Nguyên lý: Move Ordering chuẩn xác là yếu tố quan trọng nhất giúp Alpha-Beta
        đạt độ phức tạp tối ưu O(b^(d/2)). Nếu nước đi tốt nhất luôn được duyệt đầu tiên,
        tỷ lệ cắt tỉa beta ở các nước sau sẽ đạt trên 90%.

        Hệ thống phân tầng điểm số ưu tiên:
        1. Nước đi từ Transposition Table (TT Move): +2,000,000 (đã chứng minh là nước tốt nhất ở tầng trước)
        2. Phong cấp Hậu (Queen Promotion): +1,200,000
        3. Nước bắt quân có lợi (Good Captures với SEE >= 0) kết hợp MVV-LVA: +1,000,000 + bonus
        4. Killer Move 1: +700,000
        5. Killer Move 2: +650,000
        6. Nước phong cấp khác (Xe, Tượng, Mã): +40,000
        7. Nước bắt quân bất lợi (Bad Captures với SEE < 0): +50,000 + SEE
        8. Nước đi yên lặng có điểm History cao: +1 đến +500,000
        """
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        turn = board.turn

        def score_move(move: chess.Move) -> int:
            # 1. Ưu tiên cao nhất: Nước đi gợi ý từ Transposition Table (Hash Move)
            if move == tt_move:
                return 10_000_000

            # 2. Ưu tiên phong cấp Hậu
            if move.promotion == chess.QUEEN:
                return 1_200_000

            # 3. Đánh giá các nước bắt quân
            if board.is_capture(move):
                victim = (
                    chess.PAWN
                    if board.is_en_passant(move)
                    else board.piece_type_at(move.to_square)
                )
                attacker = board.piece_type_at(move.from_square)

                # MVV-LVA score: quân bị ăn càng to, quân đi ăn càng nhỏ thì điểm càng cao
                mvv_lva = (
                    _MVV_LVA_SCORES[victim] * 10
                    - _MVV_LVA_SCORES[attacker]
                )

                v_val = PIECE_VALS[victim]
                a_val = PIECE_VALS[attacker]

                if a_val <= v_val:
                    # Quân nhỏ/bằng ăn quân lớn: chắc chắn SEE >= 0, không cần quét tấn công đắt đỏ
                    return 1_000_000 + mvv_lva * 100 + (v_val - a_val)
                else:
                    # Quân to ăn quân bé: chỉ gọi see() khi thực sự cần phân loại
                    see_val = see(board, move)
                    if see_val >= 0:
                        return 1_000_000 + mvv_lva * 100 + see_val
                    else:
                        return 50_000 + see_val

            # 4. Killer Moves: các nước đi yên lặng từng gây cắt tỉa beta ở cùng ply
            if move == killers[0]:
                return 900_000
            if move == killers[1]:
                return 800_000

            # 5. Nước đi gây chiếu (Check Move): Nước yên lặng tạo đòn chiếu hiểm hóc
            if board.gives_check(move):
                return 650_000

            # 6. Các nước phong cấp khác (Mã, Tượng, Xe)
            if move.promotion is not None:
                return 40_000

            # 7. History Heuristic: sắp xếp các nước yên lặng theo tần suất thành công
            key = (turn, move.from_square, move.to_square)
            return min(self.history.get(key, 0), 500_000)

        moves.sort(key=score_move, reverse=True)
        return moves

    # ------------------------------------------------------------------
    # 6. TÌM KIẾM TĨNH LẶNG (QUIESCENCE SEARCH)
    # ------------------------------------------------------------------

    def _quiescence(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply: int,
        tm: TimeManager,
        qdepth: int = 0,
    ) -> int:
        """
        Quiescence Search – "Đao Quang Kiếm Ảnh" (Tìm kiếm tĩnh lặng tối ưu hóa).

        Chức năng:
        Giải quyết triệt để "Hiệu ứng chân trời" (Horizon Effect). Nếu dừng tìm kiếm
        ngay tại độ sâu depth = 0 khi bàn cờ đang trong chuỗi bắt quân qua lại, điểm số
        sẽ bị sai lệch nghiêm trọng. Quiescence Search tiếp tục duyệt riêng các nước
        bắt quân (captures) và giải chiếu cho đến khi bàn cờ đạt trạng thái "tĩnh lặng".

        Kỹ thuật tối ưu hóa hiệu năng cao:
        1. Stand-pat: Sử dụng điểm đánh giá tĩnh hiện tại làm ngưỡng sàn. Nếu bên đi
           đã đạt điểm >= beta mà không cần làm gì thêm, cắt tỉa ngay lập tức (fail-high).
        2. Delta Pruning: Nếu điểm stand-pat cộng với giá trị con Hậu (1150 cp) mà vẫn
           không đuổi kịp alpha, việc bắt quân bình thường không thể cứu vãn thế cờ => cắt tỉa.
        3. Bitboard Promo Check: Kiểm tra khả năng phong cấp siêu tốc bằng bitboard
           (board.pawns & promo_rank) thay vì duyệt danh sách legal moves.
        4. Fast Capture Generation: Sử dụng board.generate_legal_captures() kết hợp
           lọc nhanh see(board, move) >= 0 để loại bỏ các nước thí quân lỗ vốn.
        """
        self._nodes += 1

        # Kiểm tra deadline định kỳ mỗi 128 nodes để phản ứng nhanh nhạy với thời gian
        if (self._nodes & 127) == 0 and tm.should_stop():
            self._timeout = True
            return 0

        if self._timeout:
            return 0

        in_check = board.is_check()

        # Stand-pat và Delta Pruning (chỉ áp dụng khi không bị chiếu)
        if not in_check:
            # Kiểm tra luật hòa 50 nước và lặp cờ
            if board.halfmove_clock >= 100 or (ply > 0 and board.is_repetition(2)):
                return DRAW_SCORE

            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return beta

            # Kiểm tra phong cấp siêu nhanh bằng Bitboard: chỉ xét Tốt của bên đang đến lượt đi
            promo_rank = chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
            my_pawns = board.pawns & board.occupied_co[board.turn]
            has_promotion = bool(my_pawns & promo_rank)

            # Delta Pruning: không cắt tỉa nếu có nguy cơ đối phương hoặc mình phong cấp
            if qdepth > 0 and not has_promotion and stand_pat + 1150 < alpha:
                return alpha

            if stand_pat > alpha:
                alpha = stand_pat

        # Đạt giới hạn độ sâu của Quiescence Search: trả về alpha hiện tại
        if qdepth >= MAX_QSEARCH_DEPTH and not in_check:
            return alpha

        # Sinh nước đi cho Quiescence Search
        if in_check:
            # Nếu đang bị chiếu: phải xét mọi nước đi hợp lệ để tìm cách giải chiếu
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE_SCORE - ply)
        else:
            # Không bị chiếu: sinh thẳng các nước bắt quân qua generate_legal_captures()
            # Giữ lại các nước bắt quân có lợi (SEE >= 0) hoặc nước bắt quân gây chiếu (để không bỏ sót đòn sát thủ)
            promo_rank = chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
            my_pawns = board.pawns & board.occupied_co[board.turn]
            if my_pawns & promo_rank:
                caps = list(board.generate_legal_captures())
                moves = [m for m in caps if see(board, m) >= 0 or board.gives_check(m)]
                for m in board.legal_moves:
                    if m.promotion == chess.QUEEN and not board.is_capture(m):
                        moves.append(m)
            else:
                moves = [m for m in board.generate_legal_captures() if see(board, m) >= 0 or board.gives_check(m)]

        if not moves:
            return alpha

        # Sắp xếp các nước bắt quân theo MVV-LVA
        def q_order(m: chess.Move) -> int:
            if m.promotion == chess.QUEEN:
                return 1000
            v = (
                chess.PAWN
                if board.is_en_passant(m)
                else board.piece_type_at(m.to_square)
            )
            a = board.piece_type_at(m.from_square)
            return _MVV_LVA_SCORES[v] * 10 - _MVV_LVA_SCORES[a]

        moves.sort(key=q_order, reverse=True)

        best_score = -MATE_SCORE - 1 if in_check else alpha

        for move in moves:
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, tm, qdepth + 1)
            finally:
                board.pop()

            if self._timeout:
                return 0

            if score > best_score:
                best_score = score
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return best_score if in_check else alpha

    # ------------------------------------------------------------------
    # 7. THUẬT TOÁN TÌM KIẾM NEGAMAX VỚI ALPHA-BETA PRUNING & PVS
    # ------------------------------------------------------------------

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        tm: TimeManager,
        allow_nmp: bool = True,
    ) -> int:
        """
        Negamax với Alpha-Beta Pruning, PVS, NMP, RFP, Futility Pruning và LMR.

        Nguyên lý Negamax:
        Thay vì luân phiên Max cho Trắng và Min cho Đen như Minimax truyền thống,
        Negamax tận dụng tính chất đối xứng zero-sum: max(a, b) == -min(-a, -b).
        Điểm số trả về luôn theo góc nhìn của bên đang đến lượt đi (board.turn).

        Các tầng cắt tỉa toán học:
        1. Kiểm tra hòa lặp (Repetition) & Thiếu quân (Insufficient Material).
        2. Tra cứu Transposition Table: Cắt tỉa tức thì nếu đã có kết quả đủ độ sâu.
        3. Khi depth <= 0: Chuyển tiếp sang Quiescence Search để giải quyết hiệu ứng chân trời.
        4. Reverse Futility Pruning (RFP / Thần Uy Cắt Tỉa): Cắt tỉa ở depth 1-2 nếu điểm tĩnh >= beta.
        5. Null Move Pruning (NMP / Hư Vô Thần Bộ): Nhường nước đi ở depth >= 3 để kiểm định thế cờ.
        6. Futility Pruning (FP / Tịch Diệt Cắt Tỉa): Bỏ qua nước yên lặng ở depth 1 nếu không cứu vãn được alpha.
        7. Late Move Reductions (LMR / Tùy Phong Giảm Lực): Giảm độ sâu với các nước đi xếp sau.
        8. Principal Variation Search (PVS / Nhất Đạo Phá Vạn Pháp): Tìm kiếm nước đầu với dải đầy đủ,
           các nước sau kiểm định nhanh với cửa sổ hẹp Zero-Window [alpha, alpha + 1].
        """
        self._nodes += 1

        # Kiểm tra deadline định kỳ mỗi 128 nodes để phản ứng nhanh nhạy với thời gian
        if (self._nodes & 127) == 0 and tm.should_stop():
            self._timeout = True
            return 0

        if self._timeout:
            return 0

        # Kiểm tra hòa lặp và thiếu quân
        if ply > 0:
            if board.is_repetition(2) or board.halfmove_clock >= 100:
                if board.is_checkmate():
                    return -(MATE_SCORE - ply)
                # CONTEMPT FACTOR: Khinh địch khi đang hơn thế để ép bot tìm đường thắng
                eval_side = evaluate(board)
                if eval_side >= 80:
                    return -50  # Phạt hòa khi đang hơn thế
                elif eval_side <= -80:
                    return +50  # Thưởng hòa khi đang thua
                return DRAW_SCORE
            if board.is_insufficient_material():
                return DRAW_SCORE

        # --- Tra cứu Transposition Table ---
        tt_key = board._transposition_key()
        tt_entry = self.tt.probe(tt_key)
        tt_move: chess.Move | None = None

        if tt_entry is not None:
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth:
                tt_score = score_from_tt(tt_entry.score, ply)
                if tt_entry.flag == TT_EXACT:
                    return tt_score
                elif tt_entry.flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                elif tt_entry.flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        # Khi hết độ sâu: chuyển tiếp sang Quiescence Search
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, tm, 0)

        in_check = board.is_check()

        # Check Extension: Mở rộng 1 tầng khi bị chiếu để triệt tiêu Horizon Effect
        if in_check and ply < 24:
            depth += 1

        # Tính toán lượng giá tĩnh một lần duy nhất khi không bị chiếu
        # Dùng chung cho RFP, NMP và Futility Pruning để tránh gọi lại evaluate()
        static_eval = evaluate(board) if not in_check else None

        # --- 1. Reverse Futility Pruning (RFP / Static Null Move) - Thần Uy Cắt Tỉa ---
        # Cắt tỉa siêu hiệu quả ở độ sâu 1 và 2 nếu vị trí tĩnh đã quá áp đảo
        if not in_check and depth <= 2 and abs(beta) < MATE_THRESHOLD and ply > 0:
            has_pieces = bool(board.occupied_co[board.turn] & ~(board.pawns | board.kings))
            if has_pieces and static_eval is not None:
                rfp_margin = 120 * depth  # 120 CP ở depth 1, 240 CP ở depth 2
                if static_eval - rfp_margin >= beta:
                    return static_eval - rfp_margin

        # --- 2. Null Move Pruning (NMP) - Hư Vô Thần Bộ (Nhượng Nhất Chiêu) ---
        if allow_nmp and not in_check and depth >= 3 and ply > 0:
            # Kiểm tra nhanh bằng bitboard: có quân cờ lớn (không phải Tốt, Vua) để tránh bẫy Zugzwang
            # Chỉ thực hiện Null Move khi static_eval >= beta: nếu điểm tĩnh không vượt được beta,
            # nhường lượt đi cho đối thủ chắc chắn không thể vượt beta
            has_pieces = bool(board.occupied_co[board.turn] & ~(board.pawns | board.kings))
            if has_pieces and static_eval is not None and static_eval >= beta:
                r = 2 if depth <= 4 else 3
                board.push(chess.Move.null())
                try:
                    null_score = -self._negamax(
                        board, depth - 1 - r, -beta, -beta + 1, ply + 1, tm, False
                    )
                finally:
                    board.pop()

                if self._timeout:
                    return 0

                if null_score >= beta and abs(null_score) < MATE_THRESHOLD:
                    return beta

        # Sinh và sắp xếp nước đi hợp lệ
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            if in_check:
                return -(MATE_SCORE - ply)
            return DRAW_SCORE

        moves = self._order_moves(board, legal_moves, ply, tt_move)

        best_score = -MATE_SCORE - 1
        best_move: chess.Move | None = None
        original_alpha = alpha
        moves_searched = 0

        for move in moves:
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)

            # --- 3. Futility Pruning tại Depth 1 - Tịch Diệt Cắt Tỉa ---
            # Bỏ qua các nước yên lặng không có cơ hội nâng alpha
            if (
                depth == 1
                and not in_check
                and moves_searched > 0
                and not is_capture
                and not gives_check
                and move.promotion is None
                and move != tt_move
            ):
                if static_eval is not None and static_eval + 150 <= alpha:
                    continue

            board.push(move)
            try:
                moves_searched += 1

                # --- 4. Late Move Reductions (LMR) - Tùy Phong Giảm Lực ---
                do_lmr = (
                    depth >= 3
                    and moves_searched >= 4
                    and not in_check
                    and not is_capture
                    and move.promotion is None
                    and not gives_check
                    and move != tt_move
                )

                # --- 5. Principal Variation Search (PVS) - Nhất Đạo Phá Vạn Pháp ---
                if moves_searched == 1:
                    # Nước đi đầu tiên (nước hứa hẹn nhất): tìm kiếm với cửa sổ đầy đủ
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, tm, True)
                else:
                    # Các nước sau: tìm kiếm với cửa sổ hẹp (Zero-Window) để kiểm tra fail-low
                    r = 1 if do_lmr else 0
                    score = -self._negamax(board, depth - 1 - r, -alpha - 1, -alpha, ply + 1, tm, True)

                    # Nếu LMR bị fail-high (nước đi tốt hơn dự kiến): tìm kiếm lại ở độ sâu đầy đủ
                    if do_lmr and score > alpha:
                        score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, tm, True)

                    # Nếu PVS tìm thấy điểm vượt alpha nhưng chưa chạm beta: mở rộng tìm lại toàn dải
                    if score > alpha and score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, tm, True)

            finally:
                board.pop()

            if self._timeout:
                return 0

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Cắt tỉa Beta (Beta Cutoff): nước đi này quá mạnh, đối thủ sẽ không cho phép xảy ra
                if not is_capture:
                    self._store_killer(move, ply)
                    self._update_history(move, board.turn, depth)
                break

        # Lưu kết quả tìm kiếm vào Transposition Table
        if not self._timeout:
            if best_score <= original_alpha:
                flag = TT_UPPER
            elif best_score >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            tt_score = score_to_tt(best_score, ply)
            self.tt.store(tt_key, depth, tt_score, flag, best_move, self._age)

        return best_score

    # ------------------------------------------------------------------
    # 8. ĐÀO SÂU LẶP VỚI CỬA SỔ KỲ VỌNG (ITERATIVE DEEPENING & ASPIRATION)
    # ------------------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        tm: TimeManager,
    ) -> chess.Move:
        """
        Tìm kiếm lặp tăng dần độ sâu (Iterative Deepening Search - Bát Đại Cảnh Giới: Luyện Khí -> Độ Kiếp).

        Quy trình vận hành:
        1. Bắt đầu từ depth = 1 và tăng dần từng tầng (1, 2, 3, 4, 5, 6...).
        2. Tích hợp Aspiration Windows (Tụ Khí Ngưng Thần) từ depth >= 4:
           Thu hẹp cửa sổ tìm kiếm quanh điểm số của độ sâu trước [prev - delta, prev + delta]
           (delta = 40 cp). Nếu điểm rơi ra ngoài cửa sổ (fail-low hoặc fail-high),
           hệ thống tự động mở rộng về toàn dải [-MATE, +MATE] để tìm lại.
        3. Cơ chế kiểm soát thời gian:
           - Trước khi mở tầng mới: gọi tm.can_start_next_depth() (dừng nếu đã tiêu hơn 55% ngân sách).
           - Trong quá trình duyệt: kiểm tra tm.should_stop() sau mỗi 512 nodes.
           - Nếu timeout giữa chừng: hủy bỏ kết quả dở dang của tầng hiện tại và
             trả về best_move hoàn chỉnh của tầng độ sâu sâu nhất đã hoàn thành.

        Args:
            board: Bàn cờ hiện tại.
            tm: Đối tượng TimeManager quản lý ngân sách thời gian.

        Returns:
            chess.Move: Nước đi tốt nhất tìm được.
        """
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")

        # Nước đi duy nhất: trả về ngay
        if len(legal) == 1:
            return legal[0]

        best_move = legal[0]
        self._timeout = False
        self._nodes = 0
        self._age += 1
        self._decay_history()
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 16)]
        tm.start()

        prev_score = 0
        aspiration_delta = 40

        for depth in range(1, MAX_DEPTH + 1):
            if tm.should_stop():
                break

            # Kiểm tra ngân sách trước khi bước vào cảnh giới / tầng độ sâu mới
            if depth > 1 and not tm.can_start_next_depth():
                break

            depth_start_time = time.monotonic()
            self._timeout = False
            candidate: chess.Move | None = None
            best_score_this_depth = -MATE_SCORE - 1
            aspiration_active = (
                depth >= 4 and abs(prev_score) < MATE_THRESHOLD
            )

            # Thiết lập Aspiration Window ở depth >= 4
            if aspiration_active:
                alpha = max(-MATE_SCORE - 1, prev_score - aspiration_delta)
                beta = min(MATE_SCORE + 1, prev_score + aspiration_delta)
            else:
                alpha = -MATE_SCORE - 1
                beta = MATE_SCORE + 1

            research_move: chess.Move | None = None
            while True:
                candidate = None
                best_score_this_depth = -MATE_SCORE - 1

                # Lấy nước đi tốt nhất từ TT tại gốc (hoặc nước gây fail-high ở lần duyệt trước) để ưu tiên duyệt đầu tiên
                tt_entry = self.tt.probe(board._transposition_key())
                root_tt_move = research_move or (tt_entry.best_move if tt_entry is not None else None)

                moves = self._order_moves(board, list(board.legal_moves), ply=0, tt_move=root_tt_move)

                moves_searched = 0
                for move in moves:
                    board.push(move)
                    try:
                        moves_searched += 1

                        # PVS tại node gốc
                        if moves_searched == 1:
                            score = -self._negamax(board, depth - 1, -beta, -alpha, 1, tm, True)
                        else:
                            score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1, tm, True)
                            if score > alpha and score < beta:
                                score = -self._negamax(board, depth - 1, -beta, -alpha, 1, tm, True)

                    finally:
                        board.pop()

                    if self._timeout:
                        break

                    if score > best_score_this_depth:
                        best_score_this_depth = score
                        candidate = move

                    if score > alpha:
                        alpha = score

                if self._timeout:
                    break

                # Xử lý Aspiration Window fail-low hoặc fail-high: mở rộng toàn dải và tìm lại
                if aspiration_active and (
                    best_score_this_depth <= prev_score - aspiration_delta
                    or best_score_this_depth >= prev_score + aspiration_delta
                ):
                    research_move = candidate
                    alpha = -MATE_SCORE - 1
                    beta = MATE_SCORE + 1
                    aspiration_active = False
                    continue

                break

            # Chỉ cập nhật best_move khi tầng độ sâu này đã hoàn thành trọn vẹn (không bị timeout)
            if not self._timeout and candidate is not None:
                best_move = candidate
                prev_score = best_score_this_depth
                tm.last_iteration_ms = int((time.monotonic() - depth_start_time) * 1000)

                self.tt.store(
                    board._transposition_key(),
                    depth,
                    score_to_tt(best_score_this_depth, 0),
                    TT_EXACT,
                    best_move,
                    self._age,
                )

                # Nếu đã tìm thấy chiếu hết chắc chắn: không cần tìm sâu hơn
                if best_score_this_depth >= MATE_THRESHOLD:
                    break

        return best_move
