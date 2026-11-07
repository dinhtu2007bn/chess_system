"""Sắp thứ tự, không loại nước hợp lệ; heuristic chỉ dẫn việc thử trước/sau."""

import chess
from .evaluation import PIECE_VALUES


def score_move(board, move):
    score = 0
    if move.promotion:
        score += 20000 + PIECE_VALUES[move.promotion]
    if board.is_capture(move) and not board.is_castling(move):
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
        attacker = board.piece_type_at(move.from_square)
        score += 10000 + 10 * PIECE_VALUES.get(victim, 0) - PIECE_VALUES.get(attacker, 0)
    elif board.is_castling(move):
        score += 100
    return score


def order_moves(board, preferred_move=None, killers=(), history=None, captures_only=False):
    # Sinh legal captures riêng giúp quiescence không chấm mọi nước yên tĩnh.
    if captures_only:
        moves = list(board.generate_legal_captures())
        seventh = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
        if board.pawns & board.occupied_co[board.turn] & seventh:
            moves += [
                m for m in board.generate_legal_moves() if m.promotion and not board.is_capture(m)
            ]
    else:
        moves = list(board.legal_moves)

    def priority(move):
        if move == preferred_move:
            return 10000000
        score = score_move(board, move)
        if not board.is_capture(move) and not move.promotion:
            if move in killers:
                score += 8000 - killers.index(move) * 100
            if history is not None:
                score += min(7000, history.get((board.turn, move.from_square, move.to_square), 0))
        return score

    moves.sort(key=priority, reverse=True)
    return moves
