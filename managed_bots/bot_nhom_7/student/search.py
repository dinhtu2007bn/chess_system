
from time import perf_counter

import chess

from .evaluation import VALUES, evaluate
from .time_manager import GAME_PLY_LIMIT

MATE = 32_000
MATE_BOUND = MATE - 128
INF = 100_000
MAX_PLY = 62
TT_LIMIT = 130_000
EXACT, LOWER, UPPER = 0, 1, 2
EXCHANGE_VALUES = (*VALUES[:chess.KING], 20_000)

EDGE_DISTANCE = (0, 1, 2, 3, 3, 2, 1, 0)
CENTRALITY = tuple(
    EDGE_DISTANCE[square // 8] + EDGE_DISTANCE[square % 8] for square in chess.SQUARES
)


class SearchTimeout(Exception):
    pass


def exchange_score(board: chess.Board, move: chess.Move) -> int:
    target = move.to_square
    target_bit = chess.BB_SQUARES[target]
    piece = board.piece_type_at(move.from_square)
    victim = board.piece_type_at(target) or chess.PAWN
    occupied = (board.occupied & ~chess.BB_SQUARES[move.from_square]) | target_bit
    if board.is_en_passant(move):
        occupied &= ~chess.BB_SQUARES[target - 8 if board.turn else target + 8]

    gains = [VALUES[victim]]
    if move.promotion:
        gains[0] += VALUES[move.promotion] - VALUES[chess.PAWN]
    on_target = EXCHANGE_VALUES[move.promotion or piece]
    kings = [board.king(chess.BLACK), board.king(chess.WHITE)]
    if piece == chess.KING:
        kings[board.turn] = target

    side = not board.turn
    while True:
        attackers = board.attackers_mask(side, target, occupied) & occupied
        chosen = 0
        for kind in chess.PIECE_TYPES:
            for square in chess.scan_forward(attackers & board.pieces_mask(kind, side)):
                bit = chess.BB_SQUARES[square]
                after = occupied & ~bit
                king = target if kind == chess.KING else kings[side]
                if not board.attackers_mask(not side, king, after) & after & ~target_bit:
                    chosen = bit
                    break
            if chosen:
                break
        if not chosen:
            break

        promotes = kind == chess.PAWN and chess.square_rank(target) in (0, 7)
        promotion_gain = VALUES[chess.QUEEN] - VALUES[chess.PAWN] if promotes else 0
        gains.append(on_target + promotion_gain - gains[-1])
        if kind == chess.KING:
            break
        on_target = EXCHANGE_VALUES[chess.QUEEN if promotes else kind]
        occupied &= ~chosen
        side = not side

    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = min(gains[i - 1], -gains[i])
    return gains[0]


def mate_score_for_table(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


class Searcher:
    def __init__(self) -> None:
        self.table = {}
        self.history = [[0] * 4096 for _ in chess.COLORS]

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes & 31 == 0 and perf_counter() >= self.deadline:
            raise SearchTimeout

    def _static(self, board, key) -> int:
        score = self.evaluations.get(key)
        if score is None:
            score = self.evaluations[key] = evaluate(board)
        return score

    def _draw(self, board, key, ply) -> bool:
        if not ply:
            return False
        if key in self.path or key in self.repeated:
            return True
        if self.root_ply + ply >= GAME_PLY_LIMIT or board.halfmove_clock >= 100:
            return True
        if not (board.pawns | board.rooks | board.queens) and board.is_insufficient_material():
            return True
        return ply == 1 and (self.repeated or board.halfmove_clock >= 99) and board.can_claim_draw()

    def _order(self, board, moves, hint, ply):
        killer1, killer2 = self.killers[ply]
        history = self.history[board.turn]

        def priority(move):
            if move == hint:
                return 100_000
            if move.promotion:
                return 32_000 + VALUES[move.promotion]
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                attacker = board.piece_type_at(move.from_square)
                score = 20_000 + 10 * VALUES[victim] - VALUES[attacker]
                if VALUES[victim] < VALUES[attacker] and exchange_score(board, move) < 0:
                    score -= 32_000
                return score
            if move == killer1:
                return 13_000
            if move == killer2:
                return 12_000
            return history[64 * move.from_square + move.to_square] + CENTRALITY[move.to_square]

        return sorted(moves, key=priority, reverse=True)

    def _record_quiet_cutoff(self, board, move, depth, ply) -> None:
        first, _ = self.killers[ply]
        if move != first:
            self.killers[ply] = (move, first)
        index = 64 * move.from_square + move.to_square
        history = self.history[board.turn]
        history[index] = min(10_000, history[index] + 8 * depth * depth)

    def _quiescence(self, board, alpha, beta, ply, qply) -> int:
        self._tick()
        key = board._transposition_key()
        check = board.is_check()
        if self._draw(board, key, ply):
            return -MATE + ply if check and not any(board.generate_legal_moves()) else 0

        if check:
            moves = list(board.generate_legal_moves())
            if not moves:
                return -MATE + ply
            if ply >= MAX_PLY or qply >= 10:
                return self._static(board, key)
            best = -INF
        else:
            if not any(board.generate_legal_moves()):
                return 0
            best = self._static(board, key)
            if best >= beta or ply >= MAX_PLY or qply >= 8:
                return best
            alpha = max(alpha, best)
            stand_pat = best
            moves = list(board.generate_legal_captures())
            promotion_rank = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
            ready = board.pawns & board.occupied_co[board.turn] & promotion_rank
            if ready:
                moves.extend(move for move in board.generate_legal_moves(from_mask=ready)
                             if not board.is_capture(move))

        self.path.add(key)
        try:
            for move in self._order(board, moves, None, ply):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                board.push(move)
                try:
                    if (not check and not move.promotion
                            and stand_pat + VALUES[victim] + 210 < alpha and not board.is_check()):
                        continue
                    score = -self._quiescence(board, -beta, -alpha, ply + 1, qply + 1)
                finally:
                    board.pop()
                best = max(best, score)
                if score >= beta:
                    break
                alpha = max(alpha, score)
        finally:
            self.path.remove(key)
        return best

    def _child(self, board, depth, alpha, beta, ply, index, reduction):
        if index == 0:
            return -self._negamax(board, depth, -beta, -alpha, ply)
        score = -self._negamax(board, depth - reduction, -alpha - 1, -alpha, ply)
        if reduction and score > alpha:
            score = -self._negamax(board, depth, -alpha - 1, -alpha, ply)
        if alpha < score < beta:
            score = -self._negamax(board, depth, -beta, -alpha, ply)
        return score

    def _negamax(self, board, depth, alpha, beta, ply) -> int:
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, 0)
        self._tick()
        key = board._transposition_key()
        check = board.is_check()
        if self._draw(board, key, ply):
            return -MATE + ply if check and not any(board.generate_legal_moves()) else 0

        alpha_before = alpha
        narrow = beta == alpha + 1
        halfmove = board.halfmove_clock
        near_limit = max(0, self.root_ply + ply - (GAME_PLY_LIMIT - MAX_PLY))
        tt_key = (key, halfmove if halfmove >= 70 else 0, near_limit)
        old = self.table.get(tt_key)
        hint = self.root_hint if ply == 0 else None
        if old and ply:
            old_depth, old_flag, old_score, hint = old
            if old_depth >= depth and narrow:
                score = mate_score_for_table(old_score, -ply)
                if (old_flag == EXACT or (old_flag == LOWER and score >= beta)
                        or (old_flag == UPPER and score <= alpha)):
                    return score

        moves = self.root_moves if ply == 0 else list(board.generate_legal_moves())
        if not moves:
            return -MATE + ply if check else 0
        if ply >= MAX_PLY:
            return self._static(board, key)
        if check and len(moves) == 1 and ply < 14:
            depth += 1

        self.path.add(key)
        try:
            non_pawns = board.occupied_co[board.turn] & ~(board.pawns | board.kings)
            if (narrow and not check and ply and abs(beta) < MATE_BOUND
                    and non_pawns.bit_count() >= 2 and board.pawns
                    and halfmove < 76 and near_limit == 0):
                static = self._static(board, key)
                margin = 115 * depth + 30
                if depth <= 3 and static - margin >= beta:
                    return static - margin
                if depth >= 3 and static >= beta and board.peek() and not self.repeated:
                    board.push(chess.Move.null())
                    try:
                        reduced = max(0, depth - 3 - depth // 6)
                        score = -self._negamax(board, reduced, -beta, 1 - beta, ply + 1)
                    finally:
                        board.pop()
                    if score >= beta:
                        return beta

            best, best_move = -INF, moves[0]
            for index, move in enumerate(self._order(board, moves, hint, ply)):
                if ply == 0 and perf_counter() >= self.deadline:
                    raise SearchTimeout
                quiet = not move.promotion and not board.is_capture(move)
                soon_promotes = (chess.square_rank(move.to_square) in (1, 6)
                                 and board.piece_type_at(move.from_square) == chess.PAWN)
                board.push(move)
                try:
                    reduction = 0
                    if (depth >= 3 and index >= (4 if narrow else 6) and quiet
                            and not check and not soon_promotes and not board.is_check()
                            and move not in self.killers[ply]):
                        reduction = 1
                        if narrow and depth >= 6 and index >= 12:
                            reduction = 2
                    score = self._child(board, depth - 1, alpha, beta, ply + 1, index, reduction)
                finally:
                    board.pop()

                if score > best:
                    best, best_move = score, move
                    if ply == 0:
                        self.attempt_move = move
                alpha = max(alpha, score)
                if alpha >= beta:
                    if quiet:
                        self._record_quiet_cutoff(board, move, depth, ply)
                    break
        finally:
            self.path.remove(key)

        flag = UPPER if best <= alpha_before else LOWER if best >= beta else EXACT
        if old is None or depth >= old[0] or (flag == EXACT and depth + 1 >= old[0]):
            self.table[tt_key] = depth, flag, mate_score_for_table(best, ply), best_move
        return best

    def _prepare(self, board, moves, root_ply, deadline) -> None:
        self.root_ply, self.deadline = root_ply, deadline
        self.root_moves = moves
        self.nodes = 0
        self.evaluations = {}
        self.path = set()
        self.repeated = set()
        self.killers = [(None, None) for _ in range(MAX_PLY + 1)]
        self.history = [[score // 2 for score in history] for history in self.history]
        if len(self.table) > TT_LIMIT:
            self.table.clear()

        replay = board.copy(stack=min(board.halfmove_clock, len(board.move_stack)))
        visited = {board._transposition_key()}
        while replay.move_stack:
            replay.pop()
            key = replay._transposition_key()
            if key in visited:
                self.repeated.add(key)
            visited.add(key)

    def choose(self, board, moves, root_ply, soft_deadline, hard_deadline) -> chess.Move:
        self._prepare(board, moves, root_ply, hard_deadline)
        board = board.copy()
        best = self._order(board, moves, None, 0)[0]
        score = 0
        for depth in range(1, 26):
            started = perf_counter()
            if started >= hard_deadline:
                break
            last_move, last_score = best, score
            radius = 48 if depth >= 5 and abs(score) < MATE_BOUND else INF
            while True:
                alpha, beta = (-INF, INF) if radius >= INF else (score - radius, score + radius)
                self.root_hint = self.attempt_move = best
                try:
                    found = self._negamax(board, depth, alpha, beta, 0)
                except SearchTimeout:
                    return self.attempt_move if radius >= INF else best
                if alpha < found < beta or radius >= INF:
                    best, score = self.attempt_move, found
                    break
                radius = radius * 3 if radius < 350 else INF

            now = perf_counter()
            stable = best == last_move and abs(score - last_score) < 45
            if abs(score) >= MATE_BOUND:
                break
            if now >= soft_deadline and (stable or now + 0.5 * (now - started) >= hard_deadline):
                break
        return best
