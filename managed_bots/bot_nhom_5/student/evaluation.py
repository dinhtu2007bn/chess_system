"""TODO: implement board evaluation heuristics here."""
import chess
from chess import BLACK, WHITE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

_popcount = int.bit_count

PIECE_VALUE = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 0}
_EG_VALUE = {PAWN: 110, KNIGHT: 310, BISHOP: 325, ROOK: 510, QUEEN: 920, KING: 0}
_PHASE = {PAWN: 0, KNIGHT: 1, BISHOP: 1, ROOK: 2, QUEEN: 4, KING: 0}
_MAX_PHASE = 24
TEMPO = 8

_MOB_BASE = {KNIGHT: 4, BISHOP: 6, ROOK: 7, QUEEN: 13}
_MOB_MG = {KNIGHT: 4, BISHOP: 3, ROOK: 1, QUEEN: 1}
_MOB_EG = {KNIGHT: 4, BISHOP: 3, ROOK: 3, QUEEN: 2}
_PASSED_MG = (0, 0, 5, 12, 24, 45, 75, 0)
_PASSED_EG = (0, 0, 12, 25, 50, 85, 140, 0)


def _build_pst():
    mg = {pt: [0] * 64 for pt in range(1, 7)}
    eg = {pt: [0] * 64 for pt in range(1, 7)}
    pawn_mg = (0, 0, 4, 10, 20, 34, 50, 0)
    pawn_eg = (0, 0, 8, 18, 32, 52, 85, 0)
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        cm = abs(f - 3.5) + abs(r - 3.5)
        p = pawn_mg[r]
        if f in (3, 4) and r in (3, 4):
            p += 6
        if f in (0, 7):
            p -= 4
        mg[PAWN][sq] = p
        eg[PAWN][sq] = pawn_eg[r]
        mg[KNIGHT][sq] = int(28 - 9 * (cm - 1))
        eg[KNIGHT][sq] = int(24 - 8 * (cm - 1))
        mg[BISHOP][sq] = int(14 - 4 * (cm - 1))
        eg[BISHOP][sq] = int(12 - 3 * (cm - 1))
        mg[QUEEN][sq] = int(6 - 2 * (cm - 1))
        eg[QUEEN][sq] = int(12 - 3 * (cm - 1))
        mg[ROOK][sq] = (3 if f in (3, 4) else 0) + (12 if r == 6 else 0)
        eg[ROOK][sq] = 15 if r == 6 else 0
        mg[KING][sq] = int(-14 * r + 4 * abs(f - 3.5))
        eg[KING][sq] = int(22 - 8 * (cm - 1))
    return mg, eg


def _build_tables():
    mg, eg = _build_pst()
    out_mg = {WHITE: {}, BLACK: {}}
    out_eg = {WHITE: {}, BLACK: {}}
    for pt in range(1, 7):
        for color in (WHITE, BLACK):
            flip = 0 if color == WHITE else 56
            out_mg[color][pt] = [PIECE_VALUE[pt] + mg[pt][sq ^ flip] for sq in range(64)]
            out_eg[color][pt] = [_EG_VALUE[pt] + eg[pt][sq ^ flip] for sq in range(64)]
    return out_mg, out_eg


_MGV, _EGV = _build_tables()


def _passed_mask(color, sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    mask = 0
    ranks = range(r + 1, 8) if color == WHITE else range(0, r)
    for ff in (f - 1, f, f + 1):
        if 0 <= ff <= 7:
            for rr in ranks:
                mask |= 1 << chess.square(ff, rr)
    return mask


_PASSED_MASK = {c: [_passed_mask(c, s) for s in range(64)] for c in (WHITE, BLACK)}
_ADJ_FILES = []
for _f in range(8):
    _m = 0
    for _ff in (_f - 1, _f + 1):
        if 0 <= _ff <= 7:
            _m |= chess.BB_FILES[_ff]
    _ADJ_FILES.append(_m)


def _center_dist(sq):
    f, r = sq & 7, sq >> 3
    return max(3 - f, f - 4) + max(3 - r, r - 4)


_PAWN_CACHE: dict = {}
_PAWN_CACHE_MAX = 60_000


def _side_pawn_terms(color, own, enemy):
    mg = eg = 0
    passed_sqs = []
    for f in range(8):
        fm = chess.BB_FILES[f]
        n = _popcount(own & fm)
        if not n:
            continue
        if n > 1:
            mg -= 12 * (n - 1)
            eg -= 16 * (n - 1)
        if not own & _ADJ_FILES[f]:
            mg -= 10 * n
            eg -= 12 * n
    masks = _PASSED_MASK[color]
    bb = own
    while bb:
        lsb = bb & -bb
        sq = lsb.bit_length() - 1
        bb ^= lsb
        if not enemy & masks[sq]:
            r = sq >> 3
            rel = r if color == WHITE else 7 - r
            mg += _PASSED_MG[rel]
            eg += _PASSED_EG[rel]
            if rel >= 3:
                passed_sqs.append(sq)
    return mg, eg, tuple(passed_sqs)


def _pawn_entry(wp, bp):
    key = (wp, bp)
    e = _PAWN_CACHE.get(key)
    if e is not None:
        return e
    wm, we, w_pass = _side_pawn_terms(WHITE, wp, bp)
    bm, be, b_pass = _side_pawn_terms(BLACK, bp, wp)
    w_att = ((wp & ~chess.BB_FILE_A) << 7) | ((wp & ~chess.BB_FILE_H) << 9)
    b_att = ((bp & ~chess.BB_FILE_H) >> 7) | ((bp & ~chess.BB_FILE_A) >> 9)
    e = (wm - bm, we - be, w_att, b_att, w_pass, b_pass)
    if len(_PAWN_CACHE) > _PAWN_CACHE_MAX:
        _PAWN_CACHE.clear()
    _PAWN_CACHE[key] = e
    return e


def _king_shield(ksq, color, own_pawns, all_pawns, enemy_heavy, can_castle):
    """Kết hợp logic _king_shield chuẩn của Thiên Tuất với tốc độ bitboard của Final."""
    eh = min(enemy_heavy, 4)
    if eh == 0:
        return 0
    kf, kr = ksq & 7, ksq >> 3
    rel = kr if color == WHITE else 7 - kr
    if rel > 1:
        return -15 * eh // 4
    if can_castle:
        return 0
    step = 1 if color == WHITE else -1
    s = 0
    for ff in (kf - 1, kf, kf + 1):
        if not 0 <= ff <= 7:
            continue
        r1, r2 = kr + step, kr + 2 * step
        if 0 <= r1 <= 7 and (own_pawns & chess.BB_SQUARES[(r1 << 3) | ff]):
            s += 10
        elif 0 <= r2 <= 7 and (own_pawns & chess.BB_SQUARES[(r2 << 3) | ff]):
            s += 5
        else:
            s -= 12
            if not (all_pawns & chess.BB_FILES[ff]):
                s -= 8
    if rel == 0 and kf in (3, 4):
        s -= 15
    return s * eh // 4


def evaluate(board: chess.Board) -> int:
    pawns_all = board.pawns
    wp = pawns_all & board.occupied_co[WHITE]
    bp = pawns_all & board.occupied_co[BLACK]
    pmg, peg, w_att, b_att, w_pass, b_pass = _pawn_entry(wp, bp)

    mg_tot = [0, 0]
    eg_tot = [0, 0]
    phase = 0
    heavy = [0, 0]
    knights, bishops, rooks, queens = board.knights, board.bishops, board.rooks, board.queens
    occ_all = board.occupied
    kn_att = chess.BB_KNIGHT_ATTACKS
    DIAG, DMASK, RANK, RMASK, FILE, FMASK = (
        chess.BB_DIAG_ATTACKS, chess.BB_DIAG_MASKS,
        chess.BB_RANK_ATTACKS, chess.BB_RANK_MASKS,
        chess.BB_FILE_ATTACKS, chess.BB_FILE_MASKS,
    )
    king_sq = [board.king(BLACK), board.king(WHITE)]
    cr = board.castling_rights
    can_castle = (bool(cr & chess.BB_RANK_8), bool(cr & chess.BB_RANK_1))

    for ci, color in ((1, WHITE), (0, BLACK)):
        occ = board.occupied_co[color]
        enemy_att = b_att if color == WHITE else w_att
        safe = ~occ & ~enemy_att
        own_p = wp if color == WHITE else bp
        enemy_p = bp if color == WHITE else wp
        mgv, egv = _MGV[color], _EGV[color]
        ksq = king_sq[ci]
        home_rank = 0 if color == WHITE else 7
        mg = eg = 0

        bb = own_p
        tm, te = mgv[PAWN], egv[PAWN]
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            bb ^= lsb
            mg += tm[sq]
            eg += te[sq]

        for pt, pieces_bb in ((KNIGHT, knights), (BISHOP, bishops), (ROOK, rooks), (QUEEN, queens)):
            bb = pieces_bb & occ
            if not bb:
                continue
            tm, te = mgv[pt], egv[pt]
            base, wm, we = _MOB_BASE[pt], _MOB_MG[pt], _MOB_EG[pt]
            ph = _PHASE[pt]
            while bb:
                lsb = bb & -bb
                sq = lsb.bit_length() - 1
                bb ^= lsb
                phase += ph
                if pt == KNIGHT:
                    att = kn_att[sq]
                elif pt == BISHOP:
                    att = DIAG[sq][DMASK[sq] & occ_all]
                elif pt == ROOK:
                    att = RANK[sq][RMASK[sq] & occ_all] | FILE[sq][FMASK[sq] & occ_all]
                else:
                    att = DIAG[sq][DMASK[sq] & occ_all] | RANK[sq][RMASK[sq] & occ_all] | FILE[sq][FMASK[sq] & occ_all]
                mob = _popcount(att & safe) - base
                mg += tm[sq] + wm * mob
                eg += te[sq] + we * mob
                if enemy_att & lsb:
                    mg -= 25
                    eg -= 15
                if pt == ROOK:
                    rf = sq & 7
                    fm = chess.BB_FILES[rf]
                    if not own_p & fm:
                        if not enemy_p & fm:
                            mg += 14
                            eg += 8
                        else:
                            mg += 7
                            eg += 4
                    # Phạt Vua chặn Xe ở góc bàn cờ trong Chess960 (từ Thiên Tuất)
                    if ksq is not None and (sq >> 3) == home_rank and (ksq >> 3) == home_rank:
                        kf = ksq & 7
                        if (rf < kf <= 3 and rf in (0, 1)) or (rf > kf >= 4 and rf in (6, 7)):
                            mg -= 25
                if pt >= ROOK:
                    heavy[ci] += 2 if pt == QUEEN else 1

        if _popcount(bishops & occ) >= 2:
            mg += 30
            eg += 45
        mg_tot[ci] = mg
        eg_tot[ci] = eg

    for ci, color in ((1, WHITE), (0, BLACK)):
        ksq = king_sq[ci]
        if ksq is None:
            continue
        own_p = wp if color == WHITE else bp
        mg_tot[ci] += _MGV[color][KING][ksq] + _king_shield(
            ksq, color, own_p, pawns_all, heavy[1 - ci], can_castle[ci]
        )
        eg_tot[ci] += _EGV[color][KING][ksq]

    mg_score = mg_tot[1] - mg_tot[0] + pmg
    eg_score = eg_tot[1] - eg_tot[0] + peg

    wk, bk = king_sq[1], king_sq[0]
    if wk is not None and bk is not None:
        # Smooth mop-up
        lead = abs(eg_score)
        if lead > 100:
            w = min(lead - 100, 200)
            dist = chess.square_distance(wk, bk)
            if eg_score > 0:
                eg_score += (w * (15 * _center_dist(bk) + 12 * (7 - dist))) // 200
            else:
                eg_score -= (w * (15 * _center_dist(wk) + 12 * (7 - dist))) // 200

        # Tàn cuộc Vua hỗ trợ Tốt thông & Quy tắc hình vuông (từ Thiên Tuất, lấy sẵn từ cache)
        if phase <= 10:
            extra = 0
            for psq in w_pass:
                f, rel = psq & 7, psq >> 3
                target = (7 << 3) | f
                ek = chess.square_distance(bk, target)
                ok = chess.square_distance(wk, psq)
                bonus = (rel - 2) * 5 + max(-12, min(24, (ek - (7 - rel)) * 6)) + max(-12, 12 - 3 * ok)
                extra += bonus
            for psq in b_pass:
                f, rel = psq & 7, 7 - (psq >> 3)
                target = f
                ek = chess.square_distance(wk, target)
                ok = chess.square_distance(bk, psq)
                bonus = (rel - 2) * 5 + max(-12, min(24, (ek - (7 - rel)) * 6)) + max(-12, 12 - 3 * ok)
                extra -= bonus
            eg_score += extra

    if phase > _MAX_PHASE:
        phase = _MAX_PHASE
    score = (mg_score * phase + eg_score * (_MAX_PHASE - phase)) // _MAX_PHASE
    return (score + TEMPO) if board.turn == WHITE else (-score + TEMPO)

