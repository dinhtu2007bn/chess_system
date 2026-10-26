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
    rank = chess.square_rank(square)
    advance = rank if color else 7 - rank
    direction = 8 if color else -8
    ahead = square + direction
    if not 0 <= ahead < 64:
        return 0
    bonus = (0, 0, 8, 18, 40, 85, 170, 0)[advance]
    bonus = bonus * (36 - phase) // 24
    # Quân chặn làm giảm giá trị; không coi mọi tốt thông là sắp phong hậu.
    if board.occupied & chess.BB_SQUARES[ahead]:
        return bonus // 3
    if board.is_attacked_by(not color, ahead):
        bonus = bonus * 2 // 3
    else:
        bonus += advance * 5
    promotion = chess.square(chess.square_file(square), 7 if color else 0)
    king = board.king(not color)
    own_king = board.king(color)
    if king is not None and own_king is not None:
        bonus += (chess.square_distance(king, promotion)
                  - chess.square_distance(own_king, promotion)) * advance * 2
    # Chỉ dùng quy tắc ô vuông khi địch chỉ còn vua và tốt, đường đi trống.
    enemy_nonpawns = board.occupied_co[not color] & ~(board.pawns | board.kings)
    path = range(ahead, promotion + direction, direction)
    clear_path = all(not board.piece_at(sq) for sq in path)
    steps = 7 - advance
    defender_tempo = int(board.turn != color)
    if (king is not None and not enemy_nonpawns and clear_path
            and chess.square_distance(king, promotion) > steps + defender_tempo):
        bonus += 180
    return max(0, bonus)


def _king_safety(board, color, pawns, enemy_pawns, phase):
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
        hits = (board.attacks_mask(square) & zone).bit_count()
        if hits and piece_type != chess.KING:
            attackers += 1
            pressure += hits * ATTACK_WEIGHTS[piece_type]

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
    if not board.queens & board.occupied_co[not color]:
        danger = danger * 2 // 3
    return (shelter - danger) * phase // 24


def evaluate(board: chess.Board) -> int:
    """Điểm dương có lợi cho bên đến lượt; search xử lý chiếu hết và hòa."""
    score = 0
    pieces = board.piece_map()
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
            if piece.piece_type == chess.PAWN:
                side += advance * 5 + center * 2
                if not any(files[f] for f in (file - 1, file + 1) if 0 <= f < 8):
                    side -= 12
                if is_passed_pawn(board, square, color):
                    side += _passed_bonus(board, square, color, phase)
            elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                side += center * (5 if piece.piece_type == chess.KNIGHT else 3)
                mobility = (board.attacks_mask(square) & ~own & ~enemy_pawn_attacks).bit_count()
                side += (mobility - 4) * 4
                if advance == 0:
                    undeveloped += 1
                    side -= 16 * phase // 24
            elif piece.piece_type == chess.ROOK:
                side += advance * 2
                if not files[file]:
                    side += 12 if enemy_pawns & chess.BB_FILES[file] else 22
                side += (board.attacks_mask(square) & ~own).bit_count() * 2
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
        side += _king_safety(board, color, pawn_squares, enemy_pawns, phase)
        score += side if color else -side
    return score if board.turn else -score
