
import math as _m
from collections import Counter
from dataclasses import dataclass
from operator import itemgetter
from typing import Optional

import chess
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

from .evaluation import evaluate as _evaluate_raw
from .time_manager import Deadline, SearchTimeout

_EVAL_CACHE: dict = {}
_EVAL_CACHE_MAX = 250_000
_BBS = chess.BB_SQUARES

INF = 10_000_000
MATE = 100_000
MAX_PLY = 64
MATE_BOUND = MATE - 1000
EXACT, LOWER, UPPER = 0, 1, 2
CONTEMPT = 15


def evaluate(board: chess.Board) -> int:
    """Pure function of the position: includes 2-bit castling availability used by _king_shield."""
    cr = board.castling_rights
    key = (
        (((((((board.pawns << 64 | board.knights) << 64 | board.bishops) << 64 | board.rooks) << 64
            | board.queens) << 64 | board.kings) << 64 | board.occupied_co[chess.WHITE]) << 3)
        | (4 if board.turn else 0)
        | (2 if (cr & chess.BB_RANK_8) else 0)
        | (1 if (cr & chess.BB_RANK_1) else 0)
    )
    v = _EVAL_CACHE.get(key)
    if v is None:
        v = _evaluate_raw(board)
        # Dành riêng giá trị ±CONTEMPT (±15) làm sentinel cho điểm hòa lặp đường đi
        if v == CONTEMPT:
            v = CONTEMPT + 1
        elif v == -CONTEMPT:
            v = -CONTEMPT - 1
        if len(_EVAL_CACHE) >= _EVAL_CACHE_MAX:
            _EVAL_CACHE.clear()
        _EVAL_CACHE[key] = v
    return v


_VAL = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 2000}
_FUTILITY = (0, 200, 350)
_TIME_CHECK_MASK = 255
_LMP = (0, 10, 16, 26)
_RFP_MARGIN = 110

_LMR = [[0] * 64 for _ in range(64)]
for _d in range(3, 64):
    for _i in range(4, 64):
        _LMR[_d][_i] = int(0.6 + _m.log(_d) * _m.log(_i) / 2.4)


@dataclass
class SearchConfig:
    use_tt: bool = True
    use_qsearch: bool = True
    use_delta: bool = True
    use_pvs: bool = True
    use_killers: bool = True
    use_history: bool = True
    use_check_order: bool = False
    use_null_move: bool = True
    use_lmr: bool = True
    use_futility: bool = True
    use_check_ext: bool = True
    use_rfp: bool = True
    use_lmp: bool = True
    use_aspiration: bool = True
    lmr_log: bool = True
    max_depth: int = 64
    tt_max: int = 250_000


@dataclass
class SearchResult:
    best_move: Optional[chess.Move]
    score: int
    depth: int
    nodes: int
    elapsed: float
    seldepth: int = 0


def position_key(board: chess.Board) -> int:
    ep = board.ep_square if board.has_legal_en_passant() else 64
    k = board.pawns
    k = k << 64 | board.knights
    k = k << 64 | board.bishops
    k = k << 64 | board.rooks
    k = k << 64 | board.queens
    k = k << 64 | board.kings
    k = k << 64 | board.occupied_co[chess.WHITE]
    k = k << 64 | board.clean_castling_rights()
    k = k << 7 | ep
    return k << 1 | (1 if board.turn else 0)


def transposition_key(board: chess.Board, rep_key: int):
    return (rep_key, board.halfmove_clock) if board.halfmove_clock >= 80 else rep_key


class Searcher:
    def __init__(self, cfg: Optional[SearchConfig] = None) -> None:
        self.cfg = cfg or SearchConfig()
        self.tt: dict = {}
        self.killers = [[None, None] for _ in range(MAX_PLY + 8)]
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]
        self.nodes = 0
        self.seldepth = 0
        self.deadline: Optional[Deadline] = None
        self.game_history = Counter()
        self.search_path: set = set()
        self.last: Optional[SearchResult] = None
        self.iter_best = None

    def search(self, board: chess.Board, deadline: Deadline) -> SearchResult:
        cfg = self.cfg
        self.deadline = deadline
        self.nodes = 0
        self.seldepth = 0
        self.killers = [[None, None] for _ in range(MAX_PLY + 8)]
        for c in (0, 1):
            for row in self.history[c]:
                for i in range(64):
                    row[i] >>= 1
        if len(self.tt) > cfg.tt_max:
            self.tt.clear()
        self._init_seen(board)

        legal = list(board.legal_moves)
        if not legal:
            res = SearchResult(None, 0, 0, 0, deadline.elapsed(), 0)
            self.last = res
            return res
        moves = self._order(board, legal, None, 0, False)
        best_move, best_score, done_depth = moves[0], 0, 0
        if len(moves) == 1:
            res = SearchResult(best_move, 0, 0, 0, deadline.elapsed(), 0)
            self.last = res
            return res

        for depth in range(1, cfg.max_depth + 1):
            if depth > 1 and not deadline.can_start_iteration():
                break
            self.iter_best = None
            try:
                if cfg.use_aspiration and depth >= 4 and abs(best_score) < MATE_BOUND - 1000:
                    for w in (45, 160, None):
                        lo, hi = (-INF, INF) if w is None else (best_score - w, best_score + w)
                        self.iter_best = None
                        score, move, ordered = self._search_root(board, moves, depth, lo, hi)
                        if lo < score < hi:
                            moves = ordered
                            break
                else:
                    score, move, moves = self._search_root(board, moves, depth)
            except SearchTimeout:
                if self.iter_best is not None:
                    best_move, best_score = self.iter_best
                break
            best_move, best_score, done_depth = move, score, depth
            if score >= MATE_BOUND:
                break

        res = SearchResult(best_move, best_score, done_depth, self.nodes, deadline.elapsed(), self.seldepth)
        self.last = res
        return res

    def _init_seen(self, board: chess.Board) -> None:
        hist = Counter()
        b = board.copy()
        n = min(len(b.move_stack), b.halfmove_clock)
        for _ in range(n):
            b.pop()
            hist[position_key(b)] += 1
        self.game_history = hist
        self.search_path = set()

    def _search_root(self, board, moves, depth, alpha0=-INF, beta0=INF):
        cfg = self.cfg
        alpha, beta = alpha0, beta0
        best_score, best_move = -INF, None
        scored = []
        root_key = position_key(board)
        self.search_path.add(root_key)
        try:
            for idx, move in enumerate(moves):
                board.push(move)
                if idx == 0 or not cfg.use_pvs:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True)
                else:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1, True)
                    if score > alpha:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True)
                board.pop()
                scored.append((score, move))
                if score > best_score:
                    best_score, best_move = score, move
                    if idx > 0 and alpha0 < score < beta0:
                        self.iter_best = (move, score)
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
        finally:
            self.search_path.discard(root_key)

        scored.sort(key=itemgetter(0), reverse=True)
        ordered = [m for _, m in scored]
        if len(ordered) < len(moves):
            explored = set(ordered)
            ordered.extend(m for m in moves if m not in explored)
        if self.cfg.use_tt and best_move is not None and abs(best_score) != CONTEMPT:
            flag = UPPER if best_score <= alpha0 else (LOWER if best_score >= beta0 else EXACT)
            self._tt_store(transposition_key(board, root_key), depth, flag, best_score, best_move, 0)
        return best_score, best_move, ordered

    def _order(self, board, moves, tt_move, ply, use_checks):
        cfg = self.cfg
        killers = self.killers[ply]
        k0, k1 = killers[0], killers[1]
        turn = board.turn
        hist = self.history[1 if turn else 0]
        opp = board.occupied_co[not turn]
        own = board.occupied_co[turn]
        ep = board.ep_square
        pawns, kings, rooks, occupied = board.pawns, board.kings, board.rooks, board.occupied
        use_k, use_h = cfg.use_killers, cfg.use_history
        bbs = _BBS
        scored = []
        for m in moves:
            if m == tt_move:
                s = 10_000_000
            else:
                frm = m.from_square
                to = m.to_square
                to_bb = bbs[to]
                cap = to_bb & opp
                if cap or (ep is not None and to == ep and pawns & bbs[frm]
                           and abs(to - frm) in (7, 9) and not occupied & to_bb):
                    victim = board.piece_type_at(to) if cap else PAWN
                    attacker = board.piece_type_at(frm)
                    s = 1_000_000 + 10 * _VAL[victim] - _VAL[attacker] // 10
                    if m.promotion:
                        s += _VAL[m.promotion]
                elif m.promotion:
                    s = 900_000 + _VAL[m.promotion]
                elif use_k and (m == k0 or m == k1):
                    s = 800_000 if m == k0 else 790_000
                elif use_checks and cfg.use_check_order and board.gives_check(m):
                    s = 700_000
                elif kings & bbs[frm] and (abs((frm & 7) - (to & 7)) > 1 or rooks & own & to_bb):
                    s = 600_000
                else:
                    s = min(hist[frm][to], 499_999) if use_h else 0
            scored.append((s, m))
        scored.sort(key=itemgetter(0), reverse=True)
        return [m for _, m in scored]

    def _tt_store(self, key, depth, flag, score, move, ply):
        if score >= MATE_BOUND:
            score += ply
        elif score <= -MATE_BOUND:
            score -= ply
        tt = self.tt
        if len(tt) < self.cfg.tt_max * 2 or key in tt:
            tt[key] = (depth, flag, score, move)

    def _negamax(self, board, depth, alpha, beta, ply, allow_null, in_check=None):
        cfg = self.cfg
        self.nodes += 1
        if ply > self.seldepth:
            self.seldepth = ply
        if (self.nodes & _TIME_CHECK_MASK) == 0 and self.deadline.expired():
            raise SearchTimeout
        if ply >= MAX_PLY:
            return evaluate(board)

        key = position_key(board)
        if key in self.search_path or self.game_history.get(key, 0) >= 2:
            return -CONTEMPT if (ply & 1) == 0 else CONTEMPT

        if in_check is None:
            in_check = board.is_check()
        if board.halfmove_clock >= 100 and (not in_check or any(board.legal_moves)):
            return 0
        if in_check and cfg.use_check_ext:
            depth += 1
        if depth <= 0:
            if cfg.use_qsearch:
                return self._qsearch(board, alpha, beta, ply, 0, in_check)
            return evaluate(board)

        orig_alpha = alpha
        tt_move = None
        t_key = transposition_key(board, key)
        if cfg.use_tt:
            e = self.tt.get(t_key)
            if e is not None:
                e_depth, e_flag, e_score, e_move = e
                tt_move = e_move
                if e_depth >= depth and beta - alpha == 1:
                    if e_score >= MATE_BOUND:
                        e_score -= ply
                    elif e_score <= -MATE_BOUND:
                        e_score += ply
                    if (e_flag == EXACT
                            or (e_flag == LOWER and e_score >= beta)
                            or (e_flag == UPPER and e_score <= alpha)):
                        if e_move is None or board.is_legal(e_move):
                            return e_score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0

        if tt_move is not None and tt_move not in moves:
            tt_move = None

        static_eval = None
        if not in_check and (cfg.use_null_move or cfg.use_futility or cfg.use_rfp):
            static_eval = evaluate(board)

        non_pawn = bool(board.occupied_co[board.turn] & ~(board.pawns | board.kings))

        if (cfg.use_rfp and static_eval is not None and depth <= 3 and beta - alpha == 1
                and abs(beta) < MATE_BOUND and non_pawn and static_eval - _RFP_MARGIN * depth >= beta):
            return static_eval - _RFP_MARGIN * depth

        if (cfg.use_null_move and allow_null and not in_check and depth >= 3
                and beta - alpha == 1 and beta < MATE_BOUND
                and static_eval >= beta and non_pawn):
            r = 2 + (1 if depth >= 6 else 0)
            board.push(chess.Move.null())
            self.search_path.add(key)
            try:
                score = -self._negamax(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
            finally:
                self.search_path.discard(key)
                board.pop()
            if score >= beta:
                return beta if score >= MATE_BOUND else score

        moves = self._order(board, moves, tt_move, ply, False)
        killers = self.killers[ply]
        self.search_path.add(key)
        best_score, best_move = -INF, None
        turn = board.turn
        turn_i = 1 if turn else 0
        opp_occ = board.occupied_co[not turn]
        ep_sq = board.ep_square
        pawns_bb, occ_bb = board.pawns, board.occupied
        bbs = _BBS

        try:
            for idx, move in enumerate(moves):
                frm = move.from_square
                to = move.to_square
                to_bb = bbs[to]
                is_cap = bool(to_bb & opp_occ) or (ep_sq is not None and to == ep_sq and bool(pawns_bb & bbs[frm])
                                                   and abs(to - frm) in (7, 9) and not occ_bb & to_bb)
                is_pawn_push = bool(pawns_bb & bbs[frm]) and ((to >> 3) >= 5 if turn else (to >> 3) <= 2)
                quiet = not is_cap and move.promotion is None and not is_pawn_push
                gives_check = False
                if quiet and idx > 0 and not in_check:
                    if cfg.use_futility and depth <= 2:
                        gives_check = board.gives_check(move)
                        if (not gives_check and static_eval is not None
                                and alpha > -MATE_BOUND
                                and static_eval + _FUTILITY[depth] <= alpha):
                            continue
                    if (cfg.use_lmp and depth <= 3 and beta - alpha == 1 and alpha > -MATE_BOUND
                            and idx >= _LMP[depth] and move != killers[0] and move != killers[1]):
                        if not (gives_check or board.gives_check(move)):
                            continue

                board.push(move)
                child_in_check = board.is_check()
                new_depth = depth - 1
                if idx == 0 or not cfg.use_pvs:
                    score = -self._negamax(board, new_depth, -beta, -alpha, ply + 1, True, child_in_check)
                else:
                    red = 0
                    if (cfg.use_lmr and depth >= 3 and idx >= 4 and quiet
                            and not in_check and not child_in_check
                            and move != killers[0] and move != killers[1]):
                        red = _LMR[min(depth, 63)][min(idx, 63)] if cfg.lmr_log else 1 + (1 if (idx >= 10 and depth >= 5) else 0)
                        red = max(red, 1)
                        red = min(red, new_depth - 1)
                    score = -self._negamax(board, new_depth - red, -alpha - 1, -alpha, ply + 1, True, child_in_check)
                    if score > alpha and red > 0:
                        score = -self._negamax(board, new_depth, -alpha - 1, -alpha, ply + 1, True, child_in_check)
                    if alpha < score < beta:
                        score = -self._negamax(board, new_depth, -beta, -alpha, ply + 1, True, child_in_check)
                board.pop()

                if score > best_score:
                    best_score, best_move = score, move
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            if quiet:
                                if cfg.use_killers and move != killers[0]:
                                    killers[1] = killers[0]
                                    killers[0] = move
                                if cfg.use_history:
                                    h = self.history[turn_i][move.from_square]
                                    h[move.to_square] += depth * depth
                                    if h[move.to_square] > 400_000:
                                        h[move.to_square] = 400_000
                            break
        finally:
            self.search_path.discard(key)

        # Chỉ chặn lưu TT nếu chính best_score là điểm hòa lặp đường đi (±CONTEMPT)
        if cfg.use_tt and best_move is not None and abs(best_score) != CONTEMPT:
            if best_score <= orig_alpha:
                flag = UPPER
            elif best_score >= beta:
                flag = LOWER
            else:
                flag = EXACT
            self._tt_store(t_key, depth, flag, best_score, best_move, ply)
        return best_score

    def _qsearch(self, board, alpha, beta, ply, qd, in_check=None):
        cfg = self.cfg
        self.nodes += 1
        if ply > self.seldepth:
            self.seldepth = ply
        if (self.nodes & _TIME_CHECK_MASK) == 0 and self.deadline.expired():
            raise SearchTimeout
        if ply >= MAX_PLY:
            return evaluate(board)

        if in_check is None:
            in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            if qd >= 6:
                return evaluate(board)
            moves = self._order(board, moves, None, ply, False)
            best = -INF
            stand = None
        else:
            if board.halfmove_clock >= 100 or board.is_insufficient_material():
                return 0
            if board.occupied.bit_count() <= 7 and board.is_stalemate():
                return 0
            stand = evaluate(board)
            if stand >= beta:
                return stand
            if cfg.use_delta and stand + 1100 < alpha:
                return alpha
            if stand > alpha:
                alpha = stand
            best = stand
            moves = list(board.generate_legal_captures())
            turn = board.turn
            promo_from = board.pawns & board.occupied_co[turn] & (
                chess.BB_RANK_7 if turn else chess.BB_RANK_2)
            if promo_from:
                for m in board.generate_legal_moves(promo_from, ~board.occupied & chess.BB_ALL):
                    if m.promotion == QUEEN:
                        moves.append(m)
            if not moves:
                return stand
            moves = self._order(board, moves, None, ply, False)

        if stand is not None:
            q_ep = board.ep_square
            q_pawns, q_occ = board.pawns, board.occupied
        for move in moves:
            if stand is not None and cfg.use_delta and move.promotion is None:
                frm = move.from_square
                to = move.to_square
                is_ep = (q_ep is not None and to == q_ep and bool(q_pawns & _BBS[frm])
                         and abs(to - frm) in (7, 9) and not q_occ & _BBS[to])
                victim = PAWN if is_ep else board.piece_type_at(to)
                if victim is not None and stand + _VAL[victim] + 200 < alpha:
                    continue
            board.push(move)
            score = -self._qsearch(board, -beta, -alpha, ply + 1, qd + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best
