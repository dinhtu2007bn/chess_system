"""Ước lượng trao đổi bằng chuỗi quân bắt hợp lệ rẻ nhất; không phải search đầy đủ."""

import chess

from .evaluation import PIECE_VALUES


def exchange_score(board: chess.Board, move: chess.Move) -> int:
    """SEE gần đúng, xét ghim, vua, x-ray, en passant và phong cấp.

    Mỗi bên có thể dừng trao đổi. Không cắt nhánh search dựa trên điểm này.
    """
    if board.is_castling(move):
        return 0
    victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    gain = PIECE_VALUES.get(victim, 0)
    if move.promotion:
        gain += PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]
    gains = [gain]
    target = move.to_square
    pushed = 0
    try:
        board.push(move)
        pushed += 1
        for _ in range(16):
            attackers = board.attackers_mask(board.turn, target)
            if not attackers:
                break
            replies = list(board.generate_legal_captures(
                from_mask=attackers, to_mask=chess.BB_SQUARES[target]))
            if not replies:
                break
            reply = min(replies, key=lambda m: (
                20_000 if board.piece_type_at(m.from_square) == chess.KING
                else PIECE_VALUES[board.piece_type_at(m.from_square)],
                -PIECE_VALUES.get(m.promotion, 0)))
            captured = PIECE_VALUES.get(board.piece_type_at(target), 0)
            if reply.promotion:
                captured += PIECE_VALUES[reply.promotion] - PIECE_VALUES[chess.PAWN]
            gains.append(captured - gains[-1])
            board.push(reply)
            pushed += 1
        for index in range(len(gains) - 1, 0, -1):
            gains[index - 1] = -max(-gains[index - 1], gains[index])
        return gains[0]
    finally:
        for _ in range(pushed):
            board.pop()
