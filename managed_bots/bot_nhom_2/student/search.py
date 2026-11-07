"""Single-threaded iterative deepening with PVS, alpha-beta, TT and quiescence.

All speculative pushes are paired with pops in finally blocks, including timeout.
Repetition history and the halfmove clock are included in transposition keys.
"""

import time

import chess

from .evaluation import VALUES, evaluate


INF = 32000
MATE = 30000
MATE_BOUND = 29000
MAX_PLY = 96
TT_LIMIT = 40000
EXACT, LOWER, UPPER = 0, 1, 2


class SearchTimeout(Exception):
    """Internal control flow; never escapes choose()."""


def position_key(board):
    # Public bitboards avoid relying on python-chess's private hash internals.
    return (
        board.pawns, board.knights, board.bishops, board.rooks, board.queens,
        board.kings, board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
        board.turn, board.castling_rights, board.ep_square, board.chess960,
    )


def repetition_key(board):
    key = position_key(board)
    # An en passant square only changes a repetition position if capture is legal.
    if board.ep_square is not None and not board.has_legal_en_passant():
        key = key[:10] + (None,) + key[11:]
    return key


class Searcher:
    def __init__(self):
        self.tt = {}
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}
        self.nodes = 0
        self.completed_depth = 0
        self.last_score = 0
        self.hard_deadline = 0.0
        self.soft_deadline = 0.0
        self.counts = {}
        self.history_hash = 0
        self.game_ply = 0

    def _check_time(self):
        if time.perf_counter() >= self.hard_deadline:
            raise SearchTimeout

    def _visit(self):
        self.nodes += 1
        if self.nodes & 15 == 0:
            self._check_time()

    def _init_history(self, board):
        cursor = board.copy(stack=True)
        self.counts = {}
        while True:
            key = repetition_key(cursor)
            self.counts[key] = self.counts.get(key, 0) + 1
            if not cursor.move_stack or cursor.halfmove_clock == 0:
                break
            cursor.pop()
        self.history_hash = 0
        for key, count in self.counts.items():
            self.history_hash ^= hash((key, count))

    def _push(self, board, move):
        old_hash = self.history_hash
        old_counts = self.counts
        board.push(move)
        key = repetition_key(board)
        if board.halfmove_clock == 0:
            self.counts = {key: 1}
            self.history_hash = hash((key, 1))
        else:
            count = self.counts.get(key, 0)
            if count:
                self.history_hash ^= hash((key, count))
            self.counts[key] = count + 1
            self.history_hash ^= hash((key, count + 1))
        return old_counts, old_hash, key

    def _pop(self, board, token):
        old_counts, old_hash, key = token
        if self.counts is old_counts:
            count = self.counts[key] - 1
            if count:
                self.counts[key] = count
            else:
                del self.counts[key]
        self.counts = old_counts
        self.history_hash = old_hash
        board.pop()

    def _draw(self, board, ply):
        return (
            self.game_ply + ply >= 300
            or board.halfmove_clock >= 100
            or self.counts.get(repetition_key(board), 0) >= 3
            or board.is_insufficient_material()
        )

    def _tt_key(self, board, ply):
        return (position_key(board), board.halfmove_clock, self.history_hash, self.game_ply + ply)

    def _pack_score(self, score, ply):
        return score + ply if score >= MATE_BOUND else score - ply if score <= -MATE_BOUND else score

    def _unpack_score(self, score, ply):
        return score - ply if score >= MATE_BOUND else score + ply if score <= -MATE_BOUND else score

    def _ordered(self, board, moves, preferred, ply):
        killers = self.killers[min(ply, MAX_PLY - 1)]
        color = board.turn

        def priority(move):
            if move == preferred:
                return 1000000
            score = 0
            if board.is_capture(move):
                victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                score = 100000 + 16 * VALUES[victim or 0] - VALUES[attacker or 0]
            if move.promotion:
                score += 90000 + VALUES[move.promotion]
            if score:
                return score
            if move == killers[0]:
                return 80000
            if move == killers[1]:
                return 79000
            return self.history.get((color, move.from_square, move.to_square), 0)

        return sorted(moves, key=priority, reverse=True)

    def _record_cutoff(self, board, move, depth, ply):
        if not board.is_capture(move) and not move.promotion:
            killers = self.killers[min(ply, MAX_PLY - 1)]
            if move != killers[0]:
                killers[1], killers[0] = killers[0], move
            key = (board.turn, move.from_square, move.to_square)
            self.history[key] = min(70000, self.history.get(key, 0) + depth * depth)

    def _quiescence(self, board, alpha, beta, ply, qdepth=0):
        self._visit()
        checked = board.is_check()
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if checked else 0
        if self._draw(board, ply):
            return 0
        if ply >= MAX_PLY - 1:
            return evaluate(board)
        if not checked:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            if qdepth >= 12:
                return alpha
            moves = [move for move in moves if board.is_capture(move) or move.promotion]
        # No stand pat or capture-only filtering while in check: search all evasions.
        for move in self._ordered(board, moves, None, ply):
            token = self._push(board, move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
            finally:
                self._pop(board, token)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _negamax(self, board, depth, alpha, beta, ply):
        self._visit()
        checked = board.is_check()
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if checked else 0
        if self._draw(board, ply):
            return 0
        if ply >= MAX_PLY - 1:
            return evaluate(board)
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        key = self._tt_key(board, ply)
        entry = self.tt.get(key)
        preferred = entry[3] if entry else None
        original_alpha = alpha
        if entry and entry[0] >= depth:
            score = self._unpack_score(entry[1], ply)
            if entry[2] == EXACT:
                return score
            if entry[2] == LOWER and score >= beta:
                return score
            if entry[2] == UPPER and score <= alpha:
                return score

        best_score, best_move = -INF, None
        for index, move in enumerate(self._ordered(board, moves, preferred, ply)):
            self._check_time()
            quiet = not board.is_capture(move) and not move.promotion
            token = self._push(board, move)
            try:
                # Late quiet moves get a reduced probe and a full re-search if needed.
                reduction = int(depth >= 3 and index >= 4 and quiet and not checked and not board.is_check())
                if index == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self._negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                    if reduction and score > alpha:
                        score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                self._pop(board, token)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                self._record_cutoff(board, move, depth, ply)
                break

        flag = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
        if len(self.tt) < TT_LIMIT or key in self.tt:
            self.tt[key] = (depth, self._pack_score(best_score, ply), flag, best_move)
        return best_score

    def _root(self, board, moves, preferred, depth, alpha, beta):
        best_score, best_move = -INF, preferred
        for index, move in enumerate(self._ordered(board, moves, preferred, 0)):
            self._check_time()
            token = self._push(board, move)
            try:
                if index == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                self._pop(board, token)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_move

    def choose(self, board, fallback, started, budget, game_ply):
        self.hard_deadline = started + budget.hard_ms / 1000.0
        self.soft_deadline = started + budget.soft_ms / 1000.0
        self.nodes = self.completed_depth = 0
        self.last_score = 0
        self.game_ply = max(0, game_ply)
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]
        best_move = fallback
        try:
            self._check_time()
            # A full table is discarded between turns, never inside a search.
            if len(self.tt) >= TT_LIMIT:
                self.tt.clear()
            self.history = {key: value // 2 for key, value in self.history.items() if value > 1}
            self._init_history(board)
            score = 0
            for depth in range(1, MAX_PLY):
                self._check_time()
                alpha, beta = (score - 45, score + 45) if depth >= 4 else (-INF, INF)
                candidate_score, candidate = self._root(board, moves, best_move, depth, alpha, beta)
                if candidate_score <= alpha or candidate_score >= beta:
                    candidate_score, candidate = self._root(board, moves, best_move, depth, -INF, INF)
                # Only publish a fully completed iteration.
                score, best_move = candidate_score, candidate
                self.completed_depth, self.last_score = depth, score
                if abs(score) >= MATE_BOUND or time.perf_counter() >= self.soft_deadline:
                    break
        except SearchTimeout:
            pass
        return best_move
