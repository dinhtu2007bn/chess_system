"""Heuristic tự thiết kế; điểm dương lợi Trắng, không dùng bảng/dữ liệu bên ngoài."""

import chess

PIECE_VALUES = {1: 100, 2: 320, 3: 335, 4: 500, 5: 930, 6: 0}
PHASE_WEIGHTS = {2: 1, 3: 1, 4: 2, 5: 4}
MAX_PHASE = 24
# Bảng nhỏ tính từ công thức, không phải bảng của engine khác.
MG, EG = {}, {}
PASSED, SHIELD = {}, {}
for color in (chess.WHITE, chess.BLACK):
    MG[color], EG[color] = {}, {}
    PASSED[color], SHIELD[color] = [0] * 64, [0] * 64
    for pt in range(1, 7):
        mg, eg = [], []
        for sq in range(64):
            f, r = sq % 8, sq // 8
            rr = r if color else 7 - r
            center = 6 - (min(abs(f - 3), abs(f - 4)) + min(abs(r - 3), abs(r - 4)))
            if pt == chess.PAWN:
                advance = max(0, rr - 1)
                m = 5 * advance + 4 * (3 - min(abs(f - 3), abs(f - 4)))
                e = 9 * advance
            elif pt == chess.KNIGHT:
                m, e = 9 * center, 7 * center
            elif pt == chess.BISHOP:
                m, e = 5 * center + (8 if rr > 0 else 0), 5 * center
            elif pt == chess.ROOK:
                m, e = (20 if rr == 6 else 0), (25 if rr == 6 else 0)
            elif pt == chess.QUEEN:
                m, e = 2 * center, 4 * center
            else:
                m, e = -14 * rr - 3 * center, 12 * center
            mg.append(m)
            eg.append(e)
        MG[color][pt], EG[color][pt] = mg, eg
    for sq in range(64):
        f, r = sq % 8, sq // 8
        for ff in range(max(0, f - 1), min(7, f + 1) + 1):
            for rr in range(r + 1, 8) if color else range(r):
                PASSED[color][sq] |= chess.BB_SQUARES[chess.square(ff, rr)]
            for step in (1, 2):
                rr = r + step if color else r - step
                if 0 <= rr < 8:
                    SHIELD[color][sq] |= chess.BB_SQUARES[chess.square(ff, rr)]
ADJ_FILES = [
    (chess.BB_FILES[f - 1] if f else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0) for f in range(8)
]


def evaluate_material(board):
    score = 0
    for pt, value in PIECE_VALUES.items():
        score += value * (
            board.pieces_mask(pt, True).bit_count() - board.pieces_mask(pt, False).bit_count()
        )
    score += 30 * (
        int(board.pieces_mask(3, True).bit_count() >= 2)
        - int(board.pieces_mask(3, False).bit_count() >= 2)
    )
    return score


def evaluate_position(board):
    mg = eg = phase = 0
    pawns = [board.pieces_mask(1, False), board.pieces_mask(1, True)]
    for color in (True, False):
        sign = 1 if color else -1
        own, enemy = pawns[color], pawns[not color]
        occ = board.occupied_co[color]
        sm = se = 0
        for f in range(8):
            count = (own & chess.BB_FILES[f]).bit_count()
            sm -= 12 * max(0, count - 1)
            se -= 18 * max(0, count - 1)
        for pt in range(1, 7):
            pieces = board.pieces_mask(pt, color)
            phase += PHASE_WEIGHTS.get(pt, 0) * pieces.bit_count()
            for sq in chess.scan_forward(pieces):
                sm += MG[color][pt][sq]
                se += EG[color][pt][sq]
                f, r = sq % 8, sq // 8
                rr = r if color else 7 - r
                if pt == 1:
                    if not own & ADJ_FILES[f]:
                        sm -= 12
                        se -= 16
                    if not enemy & PASSED[color][sq]:
                        advance = max(0, rr - 1)
                        sm += 5 + 3 * advance * advance
                        se += 10 + 7 * advance * advance
                elif pt in (2, 3, 4):
                    # Khả năng hoạt động xấp xỉ: ô tấn công không có quân mình.
                    mobility = (board.attacks_mask(sq) & ~occ).bit_count()
                    weight = 3 if pt in (2, 3) else 1
                    sm += weight * mobility
                    se += weight * mobility
                    if pt == 4 and not own & chess.BB_FILES[f]:
                        bonus = 12 if enemy & chess.BB_FILES[f] else 22
                        sm += bonus
                        se += bonus
                elif pt == 6:
                    sm += 9 * (own & SHIELD[color][sq]).bit_count()
        mg += sign * sm
        eg += sign * se
    phase = min(24, phase)
    return int((mg * phase + eg * (24 - phase)) / 24), phase


def mop_up_eval(board, material_score):
    """Thưởng điểm dồn vua đối phương vào biên/góc khi thắng thế vật chất ở cờ tàn."""
    if abs(material_score) < 250:
        return 0
    lead_white = material_score > 0
    winner_king = board.king(chess.WHITE if lead_white else chess.BLACK)
    loser_king = board.king(chess.BLACK if lead_white else chess.WHITE)
    if winner_king is None or loser_king is None:
        return 0
    f, r = loser_king % 8, loser_king // 8
    center_dist = max(3 - f, f - 4) + max(3 - r, r - 4)
    wf, wr = winner_king % 8, winner_king // 8
    king_dist = max(abs(f - wf), abs(r - wr))
    bonus = center_dist * 14 + (7 - king_dist) * 10
    return bonus if lead_white else -bonus


def evaluate(board):
    mat = evaluate_material(board)
    pos, phase = evaluate_position(board)
    eg_weight = (24 - phase) / 24.0
    mop = int(mop_up_eval(board, mat) * eg_weight)
    return mat + pos + mop
