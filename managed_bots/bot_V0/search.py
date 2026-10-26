from __future__ import annotations

import time
from typing import ClassVar

import chess
import chess.polyglot

from .evaluation import PIECE_VALUES_MG, evaluate_board

# Transposition Table Entry Flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

# Piece values for MVV-LVA
MVV_LVA_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000,
}


class SearchTimeout(Exception):
    """Raised when search hard time limit is reached."""
    pass


class TranspositionTable:
    """Fixed-size transposition table for Chess960 alpha-beta search."""

    def __init__(self, max_entries: int = 150_000) -> None:
        self.max_entries = max_entries
        self.table: dict[int, tuple[int, int, int, chess.Move | None]] = {}

    def clear(self) -> None:
        self.table.clear()

    def lookup(self, key: int) -> tuple[int, int, int, chess.Move | None] | None:
        return self.table.get(key)

    def store(
        self,
        key: int,
        depth: int,
        score: int,
        flag: int,
        best_move: chess.Move | None,
    ) -> None:
        existing = self.table.get(key)
        if existing is None or depth >= existing[0]:
            if len(self.table) >= self.max_entries:
                # Evict 15% random entries when full to keep memory low
                items = list(self.table.keys())[:20_000]
                for k in items:
                    del self.table[k]
            self.table[key] = (depth, score, flag, best_move)


class Searcher:
    """Advanced Classical Search Engine with Alpha-Beta, Iterative Deepening,
    Transposition Table, Move Ordering, Killer/History Heuristics, and Quiescence Search.
    """

    MAX_PLY: ClassVar[int] = 64

    def __init__(self) -> None:
        self.tt = TranspositionTable()
        self.killer_moves: list[list[chess.Move | None]] = [
            [None, None] for _ in range(self.MAX_PLY)
        ]
        self.history_heuristic: dict[tuple[int, int], int] = {}
        self.nodes = 0
        self.hard_deadline = 0.0

    def reset_heuristics(self) -> None:
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY)]
        self.history_heuristic.clear()

    def check_time(self) -> None:
        if self.nodes & 1023 == 0:
            if time.monotonic() >= self.hard_deadline:
                raise SearchTimeout()

    def score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        hash_move: chess.Move | None,
        ply: int,
    ) -> int:
        # 1. Transposition table move gets highest priority
        if hash_move is not None and move == hash_move:
            return 2_000_000

        # 2. Captures sorted by MVV-LVA
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_val = MVV_LVA_VALUES.get(victim.piece_type if victim else chess.PAWN, 100)
            attacker_val = MVV_LVA_VALUES.get(attacker.piece_type if attacker else chess.PAWN, 100)
            return 1_000_000 + (victim_val * 10) - attacker_val

        # 3. Promotions
        if move.promotion:
            return 900_000 + MVV_LVA_VALUES.get(move.promotion, 0)

        # 4. Killer moves
        if ply < self.MAX_PLY:
            if move == self.killer_moves[ply][0]:
                return 800_000
            if move == self.killer_moves[ply][1]:
                return 700_000

        # 5. History heuristic
        move_key = (move.from_square, move.to_square)
        return self.history_heuristic.get(move_key, 0)

    def order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        hash_move: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        return sorted(
            moves,
            key=lambda m: self.score_move(board, m, hash_move, ply),
            reverse=True,
        )

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self.check_time()

        raw_eval = evaluate_board(board)
        stand_pat = raw_eval if board.turn == chess.WHITE else -raw_eval

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        if ply >= self.MAX_PLY:
            return stand_pat

        # Generate legal captures only
        legal_captures = [m for m in board.generate_legal_captures()]
        ordered_captures = sorted(
            legal_captures,
            key=lambda m: self.score_move(board, m, None, ply),
            reverse=True,
        )

        for move in ordered_captures:
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
    ) -> int:
        self.nodes += 1
        self.check_time()

        if board.is_checkmate():
            return -29000 + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        # Transposition table probe
        pos_key = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.lookup(pos_key)
        hash_move = None

        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_best_move = tt_entry
            hash_move = tt_best_move
            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWERBOUND and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPERBOUND and tt_score <= alpha:
                    return tt_score

        if depth <= 0 or ply >= self.MAX_PLY:
            return self.quiescence(board, alpha, beta, ply)

        moves = list(board.legal_moves)
        if not moves:
            return 0

        ordered_moves = self.order_moves(board, moves, hash_move, ply)
        original_alpha = alpha
        best_score = -40000
        best_move_found = None

        for move in ordered_moves:
            board.push(move)
            try:
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move_found = move

            if score > alpha:
                alpha = score
                if not board.is_capture(move):
                    move_key = (move.from_square, move.to_square)
                    self.history_heuristic[move_key] = (
                        self.history_heuristic.get(move_key, 0) + depth * depth
                    )

            if alpha >= beta:
                # Beta cutoff
                if not board.is_capture(move) and ply < self.MAX_PLY:
                    if self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                break

        # Transposition table store
        if best_score <= original_alpha:
            flag = TT_UPPERBOUND
        elif best_score >= beta:
            flag = TT_LOWERBOUND
        else:
            flag = TT_EXACT

        self.tt.store(pos_key, depth, best_score, flag, best_move_found)
        return best_score

    def search(
        self,
        board: chess.Board,
        target_time: float,
        hard_limit: float,
        max_depth: int = 30,
    ) -> tuple[chess.Move, int, int]:
        """Runs iterative deepening alpha-beta search up to target_time / hard_limit."""
        start_time = time.monotonic()
        self.hard_deadline = start_time + hard_limit
        self.nodes = 0
        self.reset_heuristics()

        board = board.copy()
        moves = list(board.legal_moves)
        if not moves:
            raise RuntimeError("No legal moves available")
        if len(moves) == 1:
            return moves[0], 1, 1

        overall_best_move = moves[0]
        completed_depth = 0

        for depth in range(1, max_depth + 1):
            iteration_best_move = None
            alpha = -35000
            beta = 35000

            try:
                pos_key = chess.polyglot.zobrist_hash(board)
                tt_entry = self.tt.lookup(pos_key)
                hash_move = tt_entry[3] if tt_entry else overall_best_move
                ordered_moves = self.order_moves(board, moves, hash_move, 0)

                best_iteration_score = -40000
                for move in ordered_moves:
                    self.check_time()
                    board.push(move)
                    try:
                        score = -self.alpha_beta(board, depth - 1, -beta, -alpha, 1)
                    finally:
                        board.pop()

                    if score > best_iteration_score:
                        best_iteration_score = score
                        iteration_best_move = move

                    if score > alpha:
                        alpha = score

                if iteration_best_move is not None:
                    overall_best_move = iteration_best_move
                    completed_depth = depth

                # Stop if mating score found
                if abs(best_iteration_score) > 28000:
                    break

                # Stop iteration if soft target time exceeded
                elapsed = time.monotonic() - start_time
                if elapsed >= target_time:
                    break

            except SearchTimeout:
                # Hard limit reached: safely return best move from completed depth
                break

        return overall_best_move, completed_depth, self.nodes
