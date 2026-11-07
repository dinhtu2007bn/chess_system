"""Đánh giá thủ công: vật chất, hoạt động quân, tốt và an toàn vua."""

import chess

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
ATTACK_WEIGHTS = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 2,
                  chess.ROOK: 3, chess.QUEEN: 5, chess.KING: 0}


def is_passed_pawn(board, square, color):
    """Không có tốt địch phía trước trên cùng cột hoặc hai cột kề."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    enemies = board.pawns & board.occupied_co[not color]
    for sq in chess.scan_forward(enemies):
        if abs(chess.square_file(sq) - file) <= 1:
            if chess.square_rank(sq) > rank if color else chess.square_rank(sq) < rank:
                return False
    return True


def _passed_bonus(board, square, color, phase):
    rank = chess.square_rank(square)  # hàng của ô quân tốt (0 đến 7)
    advance = rank if color else 7 - rank  # số hàng tốt đã tiến: trắng tính từ hàng 0, đen tính ngược lại
    direction = 8 if color else -8  # hướng đi của tốt trên số ô: trắng +8 (lên), đen -8 (xuống)
    ahead = square + direction  # ô ngay phía trước quân tốt
    if not 0 <= ahead < 64:  # ô phía trước nằm ngoài bàn cờ
        return 0  # không có thưởng
    bonus = (0, 0, 8, 18, 40, 85, 170, 0)[advance]  # điểm thưởng cơ bản theo số hàng đã tiến, càng gần phong cấp càng cao
    bonus = bonus * (36 - phase) // 24  # điều chỉnh theo giai đoạn ván: tàn cuộc (phase thấp) thưởng nhiều hơn
    # Quân chặn làm giảm giá trị; không coi mọi tốt thông là sắp phong hậu.
    if board.occupied & chess.BB_SQUARES[ahead]:  # ô phía trước đang có quân đứng (bị chặn)
        return bonus // 3  # bị chặn thì chỉ giữ một phần ba điểm thưởng, kết thúc hàm
    if board.is_attacked_by(not color, ahead):  # ô phía trước bị đối phương kiểm soát
        bonus = bonus * 2 // 3  # giảm điểm thưởng còn hai phần ba
    else:  # ô phía trước không bị tấn công
        bonus += advance * 5  # cộng thêm, càng tiến xa cộng càng nhiều
    promotion = chess.square(chess.square_file(square), 7 if color else 0)  # ô phong cấp cùng cột (hàng 8 với trắng, hàng 1 với đen)
    king = board.king(not color)  # vị trí vua đối phương
    own_king = board.king(color)  # vị trí vua của bên mình
    if king is not None and own_king is not None:  # chỉ tính khi cả hai vua còn trên bàn
        bonus += (chess.square_distance(king, promotion)  # khoảng cách từ vua địch đến ô phong cấp
                  - chess.square_distance(own_king, promotion)) * advance * 2  # trừ khoảng cách vua mình; vua địch càng xa, vua mình càng gần thì càng được thưởng
    # Chỉ dùng quy tắc ô vuông khi địch chỉ còn vua và tốt, đường đi trống.
    enemy_nonpawns = board.occupied_co[not color] & ~(board.pawns | board.kings)  # các quân địch không phải tốt và vua
    path = range(ahead, promotion + direction, direction)  # các ô từ ô phía trước đến hết ô phong cấp
    clear_path = all(not board.piece_at(sq) for sq in path)  # True nếu không có quân nào nằm trên đường này
    steps = 7 - advance  # số bước tốt còn cần đi để phong cấp
    defender_tempo = int(board.turn != color)  # bằng 1 nếu đang đến lượt bên phòng thủ (được đi trước một nhịp), ngược lại 0
    if (king is not None and not enemy_nonpawns and clear_path  # vua địch tồn tại, địch hết quân nặng nhẹ, đường đi trống
            and chess.square_distance(king, promotion) > steps + defender_tempo):  # và vua địch không kịp tới ô phong cấp (quy tắc ô vuông)
        bonus += 180  # tốt chắc chắn phong cấp được, cộng thêm điểm lớn
    return max(0, bonus)  # không để điểm thưởng âm


def _king_safety(board, color, pawns, enemy_pawns, phase, attacks, attack_union):
    """Đánh giá quanh ô vua thực tế, không giả định vua Chess960 ở c/e/g."""
    king = board.king(color)
    if king is None or phase == 0:
        return 0
    file, rank = chess.square_file(king), chess.square_rank(king)
    direction = 1 if color else -1
    zone = chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]
    pressure = attackers = 0
    for square in chess.scan_forward(board.occupied_co[not color]):
        piece_type = board.piece_type_at(square)
        hits = (attacks[square] & zone).bit_count()
        if hits and piece_type != chess.KING:
            attackers += 1
            defended = bool(attack_union[not color] & chess.BB_SQUARES[square])
            pressure += hits * ATTACK_WEIGHTS[piece_type]
            if defended:
                pressure += 1

    shelter = 0
    enemy_heavy = board.rooks | board.queens
    for f in range(max(0, file - 1), min(8, file + 2)):
        nearest = min(((chess.square_rank(sq) - rank) * direction
                       for sq in pawns if chess.square_file(sq) == f
                       and (chess.square_rank(sq) - rank) * direction > 0),
                      default=8)
        shelter += {1: 20, 2: 8}.get(nearest, -20)
        if nearest == 8:
            if not (enemy_pawns & chess.BB_FILES[f]):
                shelter -= 12  # Cột mở gần vua.
            if enemy_heavy & board.occupied_co[not color] & chess.BB_FILES[f]:
                shelter -= 15
    # Nhiều quân phối hợp quanh vua nguy hiểm hơn một quân tấn công đơn lẻ.
    danger = min(350, pressure * pressure // 3 + max(0, attackers - 1) * 12)
    # Ước lượng đường thoát, không coi bản đồ tấn công là nước vua hợp lệ.
    # Chỉ thưởng siết vòng vây khi có nhiều quân phối hợp, tránh phạt vua
    # được che chắn bởi quân nhà ngay từ khai cuộc.
    ring = chess.BB_KING_ATTACKS[king]
    exits = ring & ~board.occupied_co[color]
    controlled = (exits & attack_union[not color]).bit_count()
    if attackers >= 2:
        danger = min(400, danger + controlled * 8)
    if not board.queens & board.occupied_co[not color]:
        danger = danger * 2 // 3
    return (shelter - danger) * phase // 24


def evaluate(board: chess.Board) -> int:
    """Điểm dương có lợi cho bên đến lượt; search xử lý chiếu hết và hòa."""
    score = 0
    pieces = board.piece_map()
    attacks = {sq: board.attacks_mask(sq) for sq in pieces}
    attack_union = {chess.WHITE: 0, chess.BLACK: 0}
    for sq, piece in pieces.items():
        attack_union[piece.color] |= attacks[sq]
    phase = min(24, sum(PHASE_WEIGHTS.get(p.piece_type, 0) for p in pieces.values()))
    for color in chess.COLORS:
        side = 0
        pawns = board.pawns & board.occupied_co[color]
        enemy_pawns = board.pawns & board.occupied_co[not color]
        pawn_squares = tuple(chess.scan_forward(pawns))
        enemy_pawn_squares = tuple(chess.scan_forward(enemy_pawns))
        files = [(pawns & chess.BB_FILES[f]).bit_count() for f in range(8)]
        enemy_pawn_attacks = 0
        for sq in enemy_pawn_squares:
            enemy_pawn_attacks |= chess.BB_PAWN_ATTACKS[not color][sq]
        own = board.occupied_co[color]
        undeveloped = 0
        for square, piece in pieces.items():
            if piece.color != color:
                continue
            file, rank = chess.square_file(square), chess.square_rank(square)
            advance = rank if color else 7 - rank
            center = 7 - abs(2 * file - 7) - abs(2 * rank - 7)
            side += PIECE_VALUES[piece.piece_type]
            # Phạt nhẹ quân bị đe dọa; search quyết định khả năng bắt thật.
            # Không dùng đếm attacker như bằng chứng chắc chắn mất quân.
            if piece.piece_type != chess.KING and attack_union[not color] & chess.BB_SQUARES[square]:
                defended = bool(attack_union[color] & chess.BB_SQUARES[square])
                if not defended:
                    side -= PIECE_VALUES[piece.piece_type] // 8
                elif piece.piece_type != chess.PAWN and enemy_pawn_attacks & chess.BB_SQUARES[square]:
                    side -= PIECE_VALUES[piece.piece_type] // 16
            if piece.piece_type == chess.PAWN:
                side += advance * 5 + center * 2
                if not any(files[f] for f in (file - 1, file + 1) if 0 <= f < 8):
                    side -= 12
                if is_passed_pawn(board, square, color):
                    side += _passed_bonus(board, square, color, phase)
            elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                side += center * (5 if piece.piece_type == chess.KNIGHT else 3)
                mobility = (attacks[square] & ~own & ~enemy_pawn_attacks).bit_count()
                side += (mobility - 4) * 4
                if advance == 0:
                    undeveloped += 1
                    side -= 16 * phase // 24
            elif piece.piece_type == chess.ROOK:
                side += advance * 2
                if not files[file]:
                    side += 12 if enemy_pawns & chess.BB_FILES[file] else 22
                side += (attacks[square] & ~own).bit_count() * 2
            elif piece.piece_type == chess.QUEEN:
                side += center
            elif piece.piece_type == chess.KING:
                # Trung cuộc cần được che chắn; tàn cuộc cần hoạt động.
                middle = -advance * 10 - center * 3
                side += (middle * phase + center * 8 * (24 - phase)) // 24
        if undeveloped >= 2 and phase >= 18:
            for queen in board.pieces(chess.QUEEN, color):
                advance = chess.square_rank(queen) if color else 7 - chess.square_rank(queen)
                if advance > 1:
                    side -= 8 * undeveloped
        side -= sum(max(0, count - 1) * 15 for count in files)
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            side += 25
        side += _king_safety(board, color, pawn_squares, enemy_pawns, phase, attacks, attack_union)
        score += side if color else -side
    return score if board.turn else -score
