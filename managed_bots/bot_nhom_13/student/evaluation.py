import chess
from chess import WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN

# Các hằng số
VALUES = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, chess.KING: 0}    # Giá trị quân cờ (đơn vị centipawn)

# Pha ván để xác định ván cờ đang trong trạng thái nào (trung cuộc, tàn cuộc) dựa trên trọng số các quân cờ còn lại
# Tổng lớn nhất là 24 (2 mã (2) + 2 tượng (2) + 2 xe (4) + 1 hậu (4) = 12 cho mỗi bên)
# Pha ván = 0 là tàn cuộc, pha ván = 24 là trung cuộc.
MAX_PHASE = 24
PHASE = {KNIGHT: 1, BISHOP: 1, ROOK: 2, QUEEN: 4}

PAWN_ADV = [0, 0, 4, 10, 20, 36, 60, 0]      # Điểm thưởng cho tốt khi tiến lên các hàng
PASSED = [0, 0, 10, 20, 35, 60, 100, 0]      # Điểm thưởng cho tốt thông (không bị chặn bởi tốt đối phương trong cột của nó và cột kề) khi tiến lên các hàng

# Bảng độ trung tâm của từng ô, dùng để đánh giá vị trí quân cờ càng trung tâm càng tốt
# 4 ô trung tâm có điểm cao nhất là 12, 4 ô góc có điểm thấp nhất là 0, các ô còn lại có điểm trung bình là 4-8
CENT = [14 - abs(2 * chess.square_file(s) - 7) - abs(2 * chess.square_rank(s) - 7) for s in chess.SQUARES]

# Tạo mask cho từng ô để xác định tốt thông
def _passed_mask(sq, color):
    f, r = chess.square_file(sq), chess.square_rank(sq)

    # Nếu là tốt trắng thì xét các hàng từ r+1 đến 7, nếu là tốt đen thì các hàng từ 0 đến r-1
    ranks = range(r + 1, 8) if color == WHITE else range(r)

    # Với mỗi hàng, lấy 3 cột f - 1, f, f + 1 với điều kiện các cột nằm trong khoảng [0, 7]. Trả về bitboard có 1 ở các ô này, 0 ở các ô còn lại
    return sum(chess.BB_SQUARES[chess.square(ff, rr)] for ff in (f - 1, f, f + 1) if 0 <= ff <= 7 for rr in ranks)

# Tạo mask cho từng ô để xác định tốt thông cho cả 2 màu
PASSED_MASK = {c: [_passed_mask(s, c) for s in chess.SQUARES] for c in (WHITE, BLACK)}

# Đánh giá thế cờ hiện tại theo góc nhìn của bên đang đi. Trả về điểm đánh giá (đơn vị centipawn) dương nếu MAX tốt hơn, âm nếu MAX thua hơn, 0 nếu hòa
def evaluate(board: chess.Board) -> int:
    mg = eg = phase = 0                                             # Điểm trung cuộc, tàn cuộc, tổng pha ván
    bishops = {WHITE: 0, BLACK: 0}                                  # Số lượng tượng của mỗi bên
    pawns = {c: board.pieces_mask(PAWN, c) for c in (WHITE, BLACK)} # Bitboard vị trí tốt của mỗi bên

    # Duyệt qua tất cả các quân cờ trên bàn cờ
    for sq, piece in board.piece_map().items():
        c, pt = piece.color, piece.piece_type
        # Nếu là quân trắng thì sign = 1, rr là rank. Nếu là quân đen thì sign = -1, rr là 7 - rank
        sign = 1 if c == WHITE else -1
        rr = chess.square_rank(sq) if c == WHITE else 7 - chess.square_rank(sq)
        cent = CENT[sq]
        # m là điểm trung cuộc, e là điểm tàn cuộc của quân cờ này. Khởi tạo bằng giá trị quân cờ
        m = e = VALUES[pt]
        # Cập nhật pha ván dựa trên loại quân cờ
        phase += PHASE.get(pt, 0)

        # Tốt
        if pt == PAWN:
            # Cộng điểm thưởng khi tốt tiến lên các hàng, gấp đôi điểm thưởng cho tàn cuộc
            m += PAWN_ADV[rr]
            e += PAWN_ADV[rr] * 2
            # Nếu tốt thông thì cộng điểm thưởng theo bảng PASSED, gấp đôi điểm thưởng cho tàn cuộc
            if not PASSED_MASK[c][sq] & pawns[not c]:
                m += PASSED[rr] // 2
                e += PASSED[rr]
            # Nếu có nhiều hơn 1 tốt trên cùng 1 cột (tốt chồng) thì trừ điểm. Trừ nhiều hơn cho tàn cuộc
            if chess.popcount(pawns[c] & chess.BB_FILES[chess.square_file(sq)]) > 1:
                m -= 8
                e -= 12

        # Mã: càng trung tâm càng tốt
        elif pt == KNIGHT:
            m += cent * 3
            e += cent * 2

        # Tượng: càng trung tâm càng tốt
        elif pt == BISHOP:
            m += cent * 2
            e += cent * 2
            bishops[c] += 1    # Đếm số tượng để cộng điểm thưởng cho cặp tượng

        # Xe: cộng điểm thưởng cho xe ở hàng 7, cộng điểm thưởng cho xe ở cột không có tốt của mình
        elif pt == ROOK:
            if rr == 6:
                m += 15
                e += 20
            if not pawns[c] & chess.BB_FILES[chess.square_file(sq)]:
                m += 10
                e += 8

        # Hậu: càng trung tâm càng tốt
        elif pt == QUEEN:
            m += cent
            e += cent

        # Vua: khi trung cuộc thì trừ điểm nếu ra giữa, cộng điểm khi ở hàng nhà. Khi tàn cuộc thì cộng điểm nếu ra giữa sân
        else:
            m += -cent * 3 + (10 if rr == 0 else 0)
            e += cent * 5

        # Cộng điểm trung cuộc và tàn cuộc của quân cờ này vào tổng điểm
        mg += sign * m
        eg += sign * e

    # Xét điểm thưởng cho cặp tượng
    for c in (WHITE, BLACK):
        if bishops[c] >= 2:
            mg += (1 if c == WHITE else -1) * 30
            eg += (1 if c == WHITE else -1) * 40

    # Nếu pha ván > MAX_PHASE thì gán bằng MAX_PHASE
    phase = min(phase, MAX_PHASE)
    # Tính điểm trung bình giữa trung cuộc và tàn cuộc
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    # Nếu là lượt của trắng thì trả về score, nếu là lượt của đen thì trả về -score. + 10 là do bên đi trước có lợi thế
    return (score if board.turn == WHITE else -score) + 10