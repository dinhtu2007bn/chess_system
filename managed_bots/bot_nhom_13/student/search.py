
from time import perf_counter

import chess

from .evaluation import evaluate

# Các hằng số
INF = 10000000                  # Vô cực, lớn hơn mọi điểm khả thi, dùng cho alpha-beta
MATE = 1000000                  # Điểm cho nước đi chiếu hết cờ
MAX_PLY = 64                    # Giới hạn độ sâu tối đa 64 ply
EXACT, LOWER, UPPER = 0, 1, 2   # Loại giá trị trong bảng chuyển vị

# Nếu hết giờ thì ném ngoại lệ để thoát
class SearchTimeout(Exception):
    pass

class Searcher:
    def __init__(self):
        self.tt = {}                    # Bảng chuyển vị: từ điển thế cờ : kết quả
        self.nodes = 0                  # Số nút đã duyệt
        self.deadline = 0.0             # Thời điểm phải dừng (giây)
        self.root_color = chess.WHITE   # Màu của nút gốc MAX
        self.last_depth = 0             # Độ sâu hoàn thành gần nhất


    # ------------------------------------------- Đào sâu lặp -------------------------------------------
    def search(self, board, soft, hard, max_depth=40):

        # Thời gian bắt đầu, deadline, xóa bộ đếm, màu của nút gốc MAX
        start = perf_counter()
        self.deadline = start + hard
        self.nodes = 0                                  
        self.root_color = board.turn

        # Giới hạn kích thước bảng chuyển vị
        if len(self.tt) > 400000:
            self.tt.clear()

        # Các nước đi hợp lệ, nếu chỉ còn 1 nước đi thì trả về luôn
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]
        # Sắp xếp nước đi theo hàm ước lượng _order
        moves = self._order(board, moves, None, True)
        best = moves[0]

        for depth in range(1, max_depth + 1):

            # Tìm kiếm mỗi độ sâu, nếu hết giờ thì thoát vòng lặp và giữ best của độ sâu hoàn chỉnh cuối cùng
            try:
                best, score = self._root(board, depth, moves)
            except SearchTimeout:
                break
            self.last_depth = depth

            # best vừa tìm được chuyển lên đầu danh sách để ưu tiên cho các độ sâu tiếp theo
            moves.remove(best)
            moves.insert(0, best)

            # Nếu có nước chiếu hết hoặc đã dùng quá 45% soft limit thì dừng tìm kiếm
            if abs(score) > MATE - 100 or perf_counter() - start > soft * 0.45:
                break
        
        return best

    # Tìm kiếm từ nút gốc MAX, trả về nước đi tốt nhất và giá trị của nó
    def _root(self, board, depth, moves):

        alpha = -INF        # Giá trị alpha ban đầu
        v = -INF            # Giá trị tốt nhất hiện tại
        best = moves[0]     # Nước đi tốt nhất hiện tại

        # Với từng nước đi hợp lệ, đẩy vào board, gọi min_value để tìm giá trị của nút con MIN, sau đó pop nước đi ra khỏi board
        for m in moves:
            board.push(m)
            s = self.min_value(board, depth - 1, alpha, INF, 1)
            board.pop()
            # Nếu giá trị của nước đi này tốt hơn giá trị tốt nhất hiện tại thì cập nhật v và best
            if s > v:
                v, best = s, m
            alpha = max(alpha, v)
        return best, v



    # ------------------------------------------- Minimax, alpha-beta -------------------------------------------

    # Chọn giá trị tốt nhất cho nút MAX
    def max_value(self, board, depth, alpha, beta, ply):

        # Nếu là nút lá (hòa, chiếu hết cờ, đạt giới hạn độ sâu) và nếu có giá trị thì trả luôn
        v = self._leaf(board, depth, ply)
        if v is not None:
            return v

        # Tra bảng chuyển vị, nếu có giá trị thì trả luôn
        key, v, alpha, beta, tt_move = self._probe(board, depth, alpha, beta)
        if v is not None:
            return v

        alpha0, beta0 = alpha, beta

        # Lấy các nước đi hợp lệ và sắp xếp theo hàm _order. Nếu không còn nước đi thì gọi hàm _no_moves
        moves = self._order(board, list(board.legal_moves), tt_move, True, depth)
        if not moves:
            return self._no_moves(board, ply)

        # Khởi tạo giá trị tốt nhất là -INF, nước đi tốt nhất best = None
        v, best = -INF, None

        # Với từng nước đi hợp lệ, đẩy vào board, gọi min_value để tìm giá trị của nút con MIN rồi pop ra khỏi board
        for m in moves:
            board.push(m)
            s = self.min_value(board, depth - 1, alpha, beta, ply + 1)
            board.pop()

            # Nếu giá trị của nước đi này tốt hơn giá trị tốt nhất hiện tại thì cập nhật v và best
            if s > v:
                v, best = s, m

            # Nếu giá trị tốt nhất hiện tại >= beta thì cắt tỉa beta
            if v >= beta:
                break
            # Cập nhật alpha nếu giá trị tốt nhất hiện tại lớn hơn alpha
            alpha = max(alpha, v)

        # Ghi kết quả vào bảng chuyển vị
        self._store(key, depth, v, alpha0, beta0, best)
        return v


    # Chọn giá trị tốt nhất cho nút MIN
    def min_value(self, board, depth, alpha, beta, ply):

        # Nếu là nút lá (hòa, chiếu hết cờ, đạt giới hạn độ sâu) và nếu có giá trị thì trả luôn
        v = self._leaf(board, depth, ply)
        if v is not None:
            return v

        # Tra bảng chuyển vị, nếu có giá trị thì trả luôn
        key, v, alpha, beta, tt_move = self._probe(board, depth, alpha, beta)
        if v is not None:
            return v
        
        alpha0, beta0 = alpha, beta

        # Lấy các nước đi hợp lệ và sắp xếp theo hàm _order. Nếu không còn nước đi thì gọi hàm _no_moves
        moves = self._order(board, list(board.legal_moves), tt_move, False, depth)
        if not moves:
            return self._no_moves(board, ply)
        
        # Khởi tạo giá trị tốt nhất là +INF, nước đi tốt nhất best = None
        v, best = INF, None

        # Với từng nước đi hợp lệ, đẩy vào board, gọi max_value để tìm giá trị của nút con MAX rồi pop ra khỏi board
        for m in moves:
            board.push(m)
            s = self.max_value(board, depth - 1, alpha, beta, ply + 1)
            board.pop()

            # Nếu giá trị của nước đi này tốt hơn giá trị tốt nhất hiện tại thì cập nhật v và best
            if s < v:
                v, best = s, m

            # Nếu giá trị tốt nhất hiện tại <= alpha thì cắt tỉa alpha
            if v <= alpha:
                break

            # Cập nhật beta nếu giá trị tốt nhất hiện tại nhỏ hơn beta
            beta = min(beta, v)

        # Ghi kết quả vào bảng chuyển vị
        self._store(key, depth, v, alpha0, beta0, best)
        return v
 
    # ------------------------------------------- Các hàm hỗ trợ -------------------------------------------

    # Trả về giá trị nếu nút là nút lá (hòa, chiếu hết cờ, đạt giới hạn độ sâu), ngược lại trả về None
    def _leaf(self, board, depth, ply):

        # Cứ mỗi 512 nút được duyệt thì kiểm tra thời gian, nếu hết giờ thì ném ngoại lệ SearchTimeout
        self.nodes += 1
        if self.nodes % 512 == 0 and perf_counter() > self.deadline:
            raise SearchTimeout

        # Các luật hòa: 100 ply liên tiếp không ăn quân, không đủ quân để chiếu hết, lặp lại thế cờ
        if (board.halfmove_clock >= 100 or board.is_insufficient_material() or (board.halfmove_clock >= 4 and board.is_repetition(2))):
            return 0

        # Nếu đạt giới hạn độ sâu hoặc đạt giới hạn ply thì dùng hàm ước lượng để trả về giá trị
        if depth <= 0 or ply >= MAX_PLY:

            # Nếu chiếu hết thì gọi hàm _no_moves
            if board.is_checkmate():
                return self._no_moves(board, ply)
            return self._value(board)
        return None

    # Tra bảng chuyển vị, trả về (khóa, giá trị hoặc None, alpha, beta, nước gợi ý)
    def _probe(self, board, depth, alpha, beta):

        # key là khóa của thế cờ hiện tại, e là giá trị trong bảng chuyển vị nếu có
        key = board._transposition_key()
        e = self.tt.get(key)

        # Nếu không có giá trị trong bảng chuyển vị thì trả về giá trị và nước gợi ý là None, alpha và beta không đổi
        if e is None:
            return key, None, alpha, beta, None

        # ed: độ sâu đã tính, flag: loại giá trị (EXACT, LOWER, UPPER), ev: giá trị, tt_move: nước gợi ý
        ed, flag, ev, tt_move = e

        # Chỉ sử dụng giá trị trong bảng chuyển vị nếu độ sâu đã tính >= độ sâu hiện tại
        if ed >= depth:
            # Nếu flag = EXACT thì giá trị trong bảng chuyển vị là giá trị chính xác, trả về luôn
            if flag == EXACT:
                return key, ev, alpha, beta, tt_move
            # Nếu flag = LOWER thì giá trị thật >= ev, cập nhật alpha
            if flag == LOWER:
                alpha = max(alpha, ev)
            # Nếu flag = UPPER thì giá trị thật <= ev, cập nhật beta
            else:
                beta = min(beta, ev)
            # Nếu alpha >= beta thì trả về giá trị trong bảng chuyển vị, ngược lại trả về None
            if alpha >= beta:
                return key, ev, alpha, beta, tt_move
        return key, None, alpha, beta, tt_move

    # Ghi kết quả vào bảng chuyển vị
    def _store(self, key, depth, v, alpha0, beta0, best):
        # Nếu giá trị quá lớn thì không ghi vào bảng chuyển vị
        if abs(v) > MATE - 1000:
            return
        # Xác định loại giá trị (EXACT, LOWER, UPPER) dựa trên alpha0 và beta0
        flag = UPPER if v <= alpha0 else (LOWER if v >= beta0 else EXACT)
        # Ghi vào bảng chuyển vị
        self.tt[key] = (depth, flag, v, best)

    # Ước lượng giá trị của thế cờ hiện tại theo góc nhìn của MAX
    def _value(self, board):
        s = evaluate(board)
        # Nếu đang là lượt của đối thủ thì đổi dấu
        return s if board.turn == self.root_color else -s

    # Không còn nước đi hợp lệ, nếu không chiếu thì hòa (stalemate), nếu chiếu thì bên bị chiếu hết thua
    def _no_moves(self, board, ply):
        if not board.is_check():
            return 0
        return -(MATE - ply) if board.turn == self.root_color else MATE - ply

    # Sắp xếp nước đi theo hàm ước lượng (MAX: cao trước, MIN: thấp trước); nước từ bảng đi đầu.
    def _order(self, board, moves, tt_move, maximizing, depth=99):
        # Độ sâu 1 không cần sắp xếp
        if depth >= 2:
            scored = []
            # Với từng nước đi hợp lệ: đẩy vào board, tính giá trị ước lượng theo hàm _value, lưu vào scored, pop nước đi ra khỏi board
            for m in moves:
                board.push(m)
                scored.append((self._value(board), m))
                board.pop()
             # Sắp xếp scored theo giá trị ước lượng. Nếu maximizing = True thì sắp xếp giảm dần, ngược lại tăng dần. Lấy danh sách moves là các nước đi từ scored
            scored.sort(key=lambda x: x[0], reverse=maximizing)
            moves = [m for _, m in scored]
        # Nếu có nước đi từ bảng chuyển vị thì đưa lên đầu danh sách
        if tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
        return moves
 