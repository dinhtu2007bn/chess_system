"""Hàm ước lượng giá trị (evaluation function) cho cờ vua.

Tương ứng bài giảng: 04_adversarial search, slide "Hàm ước lượng giá trị":
    "Thực tế = tổng có trọng số của các đặc trưng"   f(s) = w1*f1(s) + w2*f2(s) + ...
Khi không tìm được tới trạng thái kết thúc (độ sâu bị giới hạn), ta thay hàm lợi ích
bằng hàm ước lượng này.

Quy ước: điểm tính bằng "centipawn" (100 = giá trị 1 quân tốt), và luôn tính theo
góc nhìn của BÊN ĐANG ĐẾN LƯỢT (điểm dương = bên đi đang lợi). Cách này cho phép
dùng negamax (xem search.py).

Các đặc trưng (mỗi đặc trưng là một hằng số trọng số W_* + một đoạn code ngắn):
    1. Vật chất               - quân nhiều/giá trị cao là tốt.
    2. Vị trí quân (PST)      - quân ở trung tâm tốt hơn ở rìa; tốt đi càng xa càng tốt.
    3. Cặp tượng              - hai tượng phối hợp rất mạnh trong thế mở.
    4. Cấu trúc tốt           - tốt chồng, tốt cô lập bị phạt; tốt thông được thưởng.
    5. Xe trên cột mở         - xe cần đường thông; xe ở hàng 7 gây áp lực.
    6. An toàn của vua        - trung cuộc: vua cần có tốt che chắn; tàn cuộc: vua ra giữa.

"Giai đoạn ván" (phase): trung cuộc và tàn cuộc có đánh giá khác nhau (vua nên trốn hay
nên tiến lên?), nên mỗi đặc trưng có 2 phiên bản và ta nội suy tuyến tính theo lượng quân
nặng còn trên bàn.

Lưu ý Chess960: không dùng ô nhập thành cố định nào cả, mọi thứ chỉ dựa vào vị trí hiện
tại của quân nên đúng cho cả 960 thế xuất phát.
Mọi con số dưới đây do nhóm tự đặt theo nguyên lý cờ vua, không lấy từ dữ liệu ngoài.
"""

from __future__ import annotations

import chess

# --------------------------------------------------------------------------------------
# Trọng số (đơn vị centipawn). Mỗi dòng kèm lý do chọn.
# --------------------------------------------------------------------------------------
PIECE_VALUE = {
    chess.PAWN: 100,    # đơn vị gốc
    chess.KNIGHT: 320,  # ~3 tốt
    chess.BISHOP: 330,  # hơn mã một chút (cặp tượng được thưởng riêng)
    chess.ROOK: 500,    # ~5 tốt
    chess.QUEEN: 900,   # ~9 tốt
    chess.KING: 0,      # vua không bao giờ bị ăn nên không tính vào vật chất
}

W_BISHOP_PAIR = 30     # có đủ 2 tượng
W_DOUBLED_PAWN = -15   # phạt mỗi tốt thừa trên cùng một cột
W_ISOLATED_PAWN = -12  # phạt tốt không có tốt bạn ở hai cột bên cạnh
PASSED_PAWN = (0, 10, 15, 25, 45, 75, 120, 0)  # thưởng tốt thông theo hàng đã tiến tới (0..7)
W_ROOK_OPEN_FILE = 20  # xe trên cột không còn tốt nào
W_ROOK_SEMI_OPEN = 10  # xe trên cột chỉ còn tốt đối phương
W_ROOK_ON_7TH = 20     # xe ở hàng 7 (hàng 2 của đối phương)
W_KING_SHIELD = 10     # mỗi cột cạnh vua có tốt che phía trước (chỉ trung cuộc)
W_TEMPO = 10           # bên đến lượt luôn có chút lợi thế
W_KNIGHT_OUTPOST = 20  # mã ở hàng 4-6 được tốt bạn bảo vệ và không bị tốt địch đuổi
W_ROOK_BEHIND_PASSER = 15  # xe cùng cột và đứng sau tốt thông của mình (tàn cuộc)
W_BLOCKED_PAWN = -8    # tốt bị quân khác chắn ngay phía trước
MOP_UP_MIN_LEAD = 300  # chỉ áp dụng "ép vua" khi bên mạnh hơn ít nhất ~3 tốt vật chất
W_MOP_EDGE = 10        # mỗi bậc vua thua bị đẩy gần mép/góc hơn
W_MOP_CLOSE = 5        # mỗi bậc vua mạnh tiến lại gần vua thua hơn

# Lượng quân nặng dùng để xác định giai đoạn ván: mã/tượng = 1, xe = 2, hậu = 4.
PHASE_WEIGHT = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
MAX_PHASE = 24  # đủ quân: 4 mã + 4 tượng + 4 xe + 2 hậu = 4*1 + 4*1 + 4*2 + 2*4 = 24

# Tốt: thưởng theo hàng đã tiến (0..7) - tốt càng gần phong hậu càng quý.
PAWN_ADVANCE_MG = (0, 0, 5, 10, 20, 35, 60, 0)
PAWN_ADVANCE_EG = (0, 0, 10, 20, 35, 60, 100, 0)  # tàn cuộc tốt tiến xa quan trọng hơn


# --------------------------------------------------------------------------------------
# Bảng giá trị vị trí (piece-square table) - SINH TỪ CÔNG THỨC thay vì gõ tay 64 số.
# --------------------------------------------------------------------------------------
def _centrality(square: int) -> int:
    """Độ gần trung tâm: 0 (góc bàn) .. 6 (4 ô giữa). = khoảng cách tới mép theo cột + theo hàng."""
    file_ = chess.square_file(square)
    rank = chess.square_rank(square)
    return min(file_, 7 - file_) + min(rank, 7 - rank)


def _relative_rank(square: int, color: chess.Color) -> int:
    """Hàng tính từ phía nhà của `color`: 0 = hàng nhà, 7 = hàng đối phương (để dùng chung cho cả hai bên)."""
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _build_tables() -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Tạo PST (từ góc nhìn quân TRẮNG; quân đen dùng ô phản chiếu `square ^ 56`)."""
    mg = {pt: [0] * 64 for pt in range(1, 7)}  # trung cuộc
    eg = {pt: [0] * 64 for pt in range(1, 7)}  # tàn cuộc
    for sq in range(64):
        c = _centrality(sq)
        file_ = chess.square_file(sq)
        rank = chess.square_rank(sq)
        # Mã: rất cần trung tâm (mã ở rìa "đứng ngoài cuộc").
        mg[chess.KNIGHT][sq] = eg[chess.KNIGHT][sq] = 10 * c - 30
        # Tượng: cần trung tâm nhưng ít hơn mã (đã đi xa được).
        mg[chess.BISHOP][sq] = eg[chess.BISHOP][sq] = 4 * c - 10
        # Hậu: thiên nhẹ về trung tâm, tránh ra sớm nên chỉ thưởng nhỏ.
        mg[chess.QUEEN][sq] = eg[chess.QUEEN][sq] = 2 * c - 6
        # Xe: không thưởng vị trí (đã có đặc trưng cột mở / hàng 7).
        # Tốt: thưởng theo hàng đã tiến + cột giữa (d,e) quý hơn cột rìa.
        center_file = 8 if file_ in (3, 4) else 4 if file_ in (2, 5) else 0
        mg[chess.PAWN][sq] = PAWN_ADVANCE_MG[rank] + (center_file if 1 <= rank <= 5 else 0)
        eg[chess.PAWN][sq] = PAWN_ADVANCE_EG[rank]
        # Vua: trung cuộc ở càng gần hàng nhà càng an toàn; tàn cuộc nên ra giữa để tham chiến.
        mg[chess.KING][sq] = -25 * rank
        eg[chess.KING][sq] = 10 * c - 20
    return mg, eg


PST_MG, PST_EG = _build_tables()


# --------------------------------------------------------------------------------------
# Các đặc trưng phụ (ngoài vật chất + PST). Mỗi hàm tính cho MỘT bên.
# --------------------------------------------------------------------------------------
def _side_bonus(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    """Trả về (điểm trung cuộc, điểm tàn cuộc) của các đặc trưng 3-6 cho bên `color`."""
    mg = 0
    eg = 0
    own_pawns = list(board.pieces(chess.PAWN, color))
    enemy_pawns = list(board.pieces(chess.PAWN, not color))

    own_per_file = [0] * 8
    enemy_per_file = [0] * 8
    for sq in own_pawns:
        own_per_file[chess.square_file(sq)] += 1
    for sq in enemy_pawns:
        enemy_per_file[chess.square_file(sq)] += 1

    # (3) Cặp tượng.
    if len(board.pieces(chess.BISHOP, color)) >= 2:
        mg += W_BISHOP_PAIR
        eg += W_BISHOP_PAIR

    # (4) Cấu trúc tốt.
    for file_ in range(8):
        if own_per_file[file_] > 1:  # tốt chồng
            penalty = W_DOUBLED_PAWN * (own_per_file[file_] - 1)
            mg += penalty
            eg += penalty
    for sq in own_pawns:
        file_ = chess.square_file(sq)
        rank_rel = _relative_rank(sq, color)
        left = own_per_file[file_ - 1] if file_ > 0 else 0
        right = own_per_file[file_ + 1] if file_ < 7 else 0
        if left == 0 and right == 0:  # tốt cô lập
            mg += W_ISOLATED_PAWN
            eg += W_ISOLATED_PAWN
        # Tốt thông: không có tốt địch ở cột này và hai cột cạnh, phía trước nó.
        passed = True
        for enemy_sq in enemy_pawns:
            if abs(chess.square_file(enemy_sq) - file_) <= 1:
                if _relative_rank(enemy_sq, color) > rank_rel:
                    passed = False
                    break
        if passed:
            mg += PASSED_PAWN[rank_rel] // 2  # trung cuộc: thưởng một nửa
            eg += PASSED_PAWN[rank_rel]       # tàn cuộc: thưởng đủ
            for rook_sq in board.pieces(chess.ROOK, color):
                if chess.square_file(rook_sq) == file_ and _relative_rank(rook_sq, color) < rank_rel:
                    eg += W_ROOK_BEHIND_PASSER  # xe đứng sau tốt thông: vừa hỗ trợ tốt vừa không bị chắn
        # Tốt bị chặn: ô ngay phía trước có quân (của bên nào cũng được) thì tốt không tiến được.
        if rank_rel < 7 and board.piece_type_at(sq + (8 if color == chess.WHITE else -8)) is not None:
            mg += W_BLOCKED_PAWN
            eg += W_BLOCKED_PAWN

    # (4b) Mã tiền đồn (outpost).
    for sq in board.pieces(chess.KNIGHT, color):
        rank_rel = _relative_rank(sq, color)
        if 3 <= rank_rel <= 5:
            file_ = chess.square_file(sq)
            supported = any(
                abs(chess.square_file(p) - file_) == 1 and _relative_rank(p, color) == rank_rel - 1
                for p in own_pawns
            )
            chased = any(
                abs(chess.square_file(p) - file_) == 1 and _relative_rank(p, color) > rank_rel
                for p in enemy_pawns
            )
            if supported and not chased:
                mg += W_KNIGHT_OUTPOST
                eg += W_KNIGHT_OUTPOST

    # (5) Xe.
    for sq in board.pieces(chess.ROOK, color):
        file_ = chess.square_file(sq)
        if own_per_file[file_] == 0:
            bonus = W_ROOK_OPEN_FILE if enemy_per_file[file_] == 0 else W_ROOK_SEMI_OPEN
            mg += bonus
            eg += bonus
        if _relative_rank(sq, color) == 6:
            mg += W_ROOK_ON_7TH
            eg += W_ROOK_ON_7TH

    # (6) Lá chắn tốt trước vua (chỉ có ý nghĩa ở trung cuộc).
    king_sq = board.king(color)
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        king_rank_rel = _relative_rank(king_sq, color)
        shielded_files = set()
        for sq in own_pawns:
            pawn_file = chess.square_file(sq)
            if abs(pawn_file - king_file) <= 1:
                if 1 <= _relative_rank(sq, color) - king_rank_rel <= 2:  # tốt nằm 1-2 hàng phía trước vua
                    shielded_files.add(pawn_file)
        mg += W_KING_SHIELD * len(shielded_files)

    return mg, eg


def _mop_up(board: chess.Board, lead_for_white: int) -> int:
    """Tàn cuộc: bên đang hơn nhiều quân được thưởng khi dồn vua đối phương ra mép và đưa vua mình lại gần.

    Không có đặc trưng này, bot có thể hơn hậu/xe mà không biết cách chiếu hết, ván sẽ bị hòa sau 300 nước.
    Điểm tính theo góc nhìn TRẮNG và chỉ cộng vào điểm tàn cuộc (nên tự yếu đi khi còn nhiều quân).
    """
    if abs(lead_for_white) < MOP_UP_MIN_LEAD:
        return 0
    strong = chess.WHITE if lead_for_white > 0 else chess.BLACK
    strong_king = board.king(strong)
    weak_king = board.king(not strong)
    if strong_king is None or weak_king is None:
        return 0
    bonus = W_MOP_EDGE * (6 - _centrality(weak_king))                      # vua thua càng gần mép/góc càng tốt
    bonus += W_MOP_CLOSE * (7 - chess.square_distance(strong_king, weak_king))  # hai vua càng gần càng tốt
    return bonus if strong == chess.WHITE else -bonus


# --------------------------------------------------------------------------------------
# Hàm ước lượng chính
# --------------------------------------------------------------------------------------
def evaluate(board: chess.Board) -> int:
    """Điểm của thế cờ theo góc nhìn bên đang đến lượt (centipawn)."""
    mg = 0  # điểm trung cuộc, góc nhìn TRẮNG
    eg = 0  # điểm tàn cuộc, góc nhìn TRẮNG
    phase = 0
    material = {chess.WHITE: 0, chess.BLACK: 0}  # vật chất (không tính vua) mỗi bên, để biết ai đang hơn

    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        if piece.color == chess.WHITE:
            mg += PIECE_VALUE[pt] + PST_MG[pt][sq]
            eg += PIECE_VALUE[pt] + PST_EG[pt][sq]
        else:
            mirrored = sq ^ 56  # lật bàn theo chiều dọc để dùng chung bảng của quân trắng
            mg -= PIECE_VALUE[pt] + PST_MG[pt][mirrored]
            eg -= PIECE_VALUE[pt] + PST_EG[pt][mirrored]
        phase += PHASE_WEIGHT.get(pt, 0)
        material[piece.color] += PIECE_VALUE[pt]

    white_mg, white_eg = _side_bonus(board, chess.WHITE)
    black_mg, black_eg = _side_bonus(board, chess.BLACK)
    mg += white_mg - black_mg
    eg += white_eg - black_eg

    eg += _mop_up(board, material[chess.WHITE] - material[chess.BLACK])

    phase = min(phase, MAX_PHASE)
    # Nội suy: nhiều quân nặng -> thiên về điểm trung cuộc; ít quân -> điểm tàn cuộc.
    score_white = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE

    score = score_white if board.turn == chess.WHITE else -score_white
    return score + W_TEMPO
