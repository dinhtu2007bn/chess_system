"""Các thuật toán tìm kiếm của bot cờ vua.

Bao gồm:
- Negamax và Alpha-Beta Pruning.
- Quiescence Search.
- Transposition Table có xét hoàn cảnh hòa.
- Killer Move và History Heuristic.
- Check Extension có giới hạn.
"""

import time

import chess

from .evaluation import evaluate, PIECE_VALUES
from .move_ordering import order_moves


MATE_SCORE = 100000
MATE_THRESHOLD = 99000
INF = 1000000

EXACT = 0
LOWER = 1
UPPER = 2


class SearchTimeout(Exception):
    """Được phát sinh khi bot đã dùng hết thời gian tìm kiếm."""

    pass


def position_key(board):
    """Tạo khóa đại diện cho trạng thái chính của bàn cờ."""

    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.turn,
        board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def terminal_score(board):
    """Trả điểm kết thúc ván theo góc nhìn của quân Trắng."""

    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        return None

    if outcome.winner is None:
        return 0

    return MATE_SCORE if outcome.winner == chess.WHITE else -MATE_SCORE


class SearchEngine:
    def __init__(
        self,
        board,
        deadline,
        table=None,
        root_ply=None,
        history=None,
    ):
        self.deadline = deadline

        # Bảng chuyển vị có thể được dùng chung giữa nhiều lượt.
        self.table = {} if table is None else table

        # Killer Move được lưu riêng trong một lần tìm kiếm.
        self.killers = {}

        # History có thể được truyền từ StudentBot để dùng qua nhiều lượt.
        self.history = {} if history is None else history

        self.nodes = 0
        self.tt_hits = 0

        self.root_moves = None
        self.root_ply = (
            len(board.move_stack)
            if root_ply is None
            else root_ply
        )

        # Lưu số lần xuất hiện của các trạng thái để kiểm tra luật hòa.
        replay = board.copy(stack=True)
        self.counts = {}

        while True:
            key = position_key(replay)
            self.counts[key] = self.counts.get(key, 0) + 1

            if not replay.move_stack or replay.halfmove_clock == 0:
                break

            replay.pop()

        # Chữ ký lịch sử giúp TT phân biệt các thế bàn giống nhau
        # nhưng có lịch sử lặp lại khác nhau.
        self.signature = 0

        for key, count in self.counts.items():
            self.signature ^= hash((key, count))

        self.doubles = sum(
            count >= 2
            for count in self.counts.values()
        )

    def check_time(self):
        """Dừng tìm kiếm khi chạm thời hạn."""
        if (self.nodes & 127) == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout()

    def enter(self, board):
        """Cập nhật lịch sử trạng thái sau khi đi một nước."""

        key = position_key(board)

        # Nước đi tốt hoặc nước bắt quân đặt lại đồng hồ 50 nước.
        # Khi đó lịch sử lặp trước vị trí này không còn cần thiết.
        if board.halfmove_clock == 0:
            saved = (
                self.counts,
                self.signature,
                self.doubles,
            )

            self.counts = {key: 1}
            self.signature = hash((key, 1))
            self.doubles = 0

            return key, saved

        old = self.counts.get(key, 0)

        if old:
            self.signature ^= hash((key, old))

        self.counts[key] = old + 1
        self.signature ^= hash((key, old + 1))

        if old == 1:
            self.doubles += 1

        return key, None

    def leave(self, token):
        """Khôi phục lịch sử khi quay lại nút cha."""

        key, saved = token

        if saved is not None:
            (
                self.counts,
                self.signature,
                self.doubles,
            ) = saved

            return

        count = self.counts[key]
        self.signature ^= hash((key, count))

        if count == 2:
            self.doubles -= 1

        if count == 1:
            del self.counts[key]
        else:
            self.counts[key] = count - 1
            self.signature ^= hash((key, count - 1))

    def drawn(self, board, ply):
        """Kiểm tra các trường hợp hòa cần xử lý trong search."""

        if board.is_insufficient_material():
            return True

        if (
            board.halfmove_clock >= 99
            and board.can_claim_fifty_moves()
        ):
            return True

        if (
            self.doubles
            and board.can_claim_threefold_repetition()
        ):
            return True

        # Ván đấu được xử hòa nếu đạt giới hạn 300 ply.
        return self.root_ply + ply >= 300

    @staticmethod
    def pack_score(score, ply):
        """Chuẩn hóa điểm chiếu hết trước khi lưu vào TT."""

        if score >= MATE_THRESHOLD:
            return score + ply

        if score <= -MATE_THRESHOLD:
            return score - ply

        return score

    @staticmethod
    def unpack_score(score, ply):
        """Khôi phục điểm chiếu hết khi đọc từ TT."""

        if score >= MATE_THRESHOLD:
            return score - ply

        if score <= -MATE_THRESHOLD:
            return score + ply

        return score

    def quiet_cutoff(self, board, move, depth, ply):
        """Ghi nhận nước yên tĩnh gây ra Alpha-Beta Cutoff."""

        if move.promotion or board.is_capture(move):
            return

        # Lưu tối đa hai Killer Move tại mỗi độ sâu.
        killers = self.killers.setdefault(ply, [])

        if move not in killers:
            killers.insert(0, move)
            del killers[2:]

        # History lưu hiệu quả của nước đi theo màu và hai ô.
        key = (
            board.turn,
            move.from_square,
            move.to_square,
        )

        bonus = depth * depth

        self.history[key] = min(
            7000,
            self.history.get(key, 0) + bonus,
        )

    def quiescence(
        self,
        board,
        alpha,
        beta,
        ply,
        qdepth=0,
    ):
        """Tìm tiếp các nước chiến thuật tại điểm dừng của Negamax."""

        self.check_time()
        self.nodes += 1

        checked = board.is_check()

        # Chiếu hết phải được kiểm tra trước các trường hợp hòa.
        if checked and board.is_checkmate():
            return -MATE_SCORE + ply

        if self.drawn(board, ply):
            return 0

        if not checked:
            # Không còn nước hợp lệ nhưng không bị chiếu là bí nước.
            if not any(board.legal_moves):
                return 0

            # Điểm đứng yên được tính theo bên đang tới lượt.
            stand = evaluate(board) * (
                1 if board.turn == chess.WHITE else -1
            )

            if stand >= beta:
                return stand

            alpha = max(alpha, stand)

            # Giới hạn độ sâu cho các chuỗi bắt quân.
            if qdepth >= 8:
                return stand

        # Khi bị chiếu phải xét mọi nước đỡ chiếu.
        # Khi không bị chiếu chỉ xét nước bắt quân và phong cấp.
        moves = order_moves(
            board,
            captures_only=not checked,
        )

        if checked and not moves:
            return -MATE_SCORE + ply

        for move in moves:
            if not checked and not move.promotion:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                if stand + PIECE_VALUES.get(victim, 100) + 200 <= alpha:
                    continue

            board.push(move)
            token = self.enter(board)

            try:
                score = -self.quiescence(
                    board,
                    -beta,
                    -alpha,
                    ply + 1,
                    qdepth + 1,
                )
            finally:
                self.leave(token)
                board.pop()

            if score >= beta:
                return score

            alpha = max(alpha, score)

        return alpha

    def negamax(
        self,
        board,
        depth,
        alpha,
        beta,
        ply,
        check_extensions_left=1,
    ):
        """Tìm điểm tốt nhất bằng Negamax và Alpha-Beta Pruning."""

        self.check_time()
        self.nodes += 1

        checked = board.is_check()

        if checked and board.is_checkmate():
            return -MATE_SCORE + ply

        if self.drawn(board, ply):
            return 0

        # Khi bên tới lượt đang bị chiếu, tìm thêm một tầng.
        # Mỗi nhánh chỉ được mở rộng một lần để kiểm soát kích thước cây.
        if checked and check_extensions_left > 0:
            depth += 1
            check_extensions_left -= 1

        if depth <= 0:
            return self.quiescence(
                board,
                alpha,
                beta,
                ply,
            )

        # Null Move Pruning
        if (
            depth >= 3
            and not checked
            and ply > 0
            and (board.occupied_co[board.turn] & ~board.pawns & ~board.kings)
        ):
            board.push(chess.Move.null())
            try:
                null_score = -self.negamax(
                    board,
                    depth - 1 - 2,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    0,
                )
            finally:
                board.pop()
            if null_score >= beta:
                return beta

        # Bảng chuyển vị dùng Zobrist hash nhanh của python-chess
        key = (board._transposition_key(), check_extensions_left)

        entry = self.table.get(key)
        preferred = None

        original_alpha = alpha
        original_beta = beta

        if entry:
            (
                stored_depth,
                value,
                flag,
                preferred,
            ) = entry

            if stored_depth >= depth:
                self.tt_hits += 1
                value = self.unpack_score(value, ply)

                if flag == EXACT:
                    return value

                if flag == LOWER:
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)

                if alpha >= beta:
                    return value

        moves = order_moves(
            board,
            preferred,
            self.killers.get(ply, ()),
            self.history,
        )

        if not moves:
            if checked:
                return -MATE_SCORE + ply

            return 0

        best = -INF
        best_move = moves[0]

        for move in moves:
            board.push(move)
            token = self.enter(board)

            try:
                score = -self.negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    check_extensions_left,
                )
            finally:
                self.leave(token)
                board.pop()

            if score > best:
                best = score
                best_move = move

            alpha = max(alpha, score)

            if alpha >= beta:
                self.quiet_cutoff(
                    board,
                    move,
                    depth,
                    ply,
                )
                break

        if best <= original_alpha:
            flag = UPPER
        elif best >= original_beta:
            flag = LOWER
        else:
            flag = EXACT

        if len(self.table) >= 60000:
            self.table.clear()

        if not entry or depth >= entry[0]:
            self.table[key] = (
                depth,
                self.pack_score(best, ply),
                flag,
                best_move,
            )

        return best

    def search(
        self,
        board,
        depth,
        preferred_move=None,
    ):
        """Tìm nước tốt nhất tại một độ sâu xác định."""

        self.check_time()

        if self.root_moves:
            moves = list(self.root_moves)
            if preferred_move in moves:
                moves.remove(preferred_move)
                moves.insert(0, preferred_move)
        else:
            moves = order_moves(
                board,
                preferred_move,
                self.killers.get(0, ()),
                self.history,
            )

        if not moves:
            raise ValueError("Board has no legal moves")

        best_move = moves[0]
        best = -INF
        alpha = -INF
        scored_moves = []

        for move in moves:
            board.push(move)
            token = self.enter(board)

            try:
                score = -self.negamax(
                    board,
                    depth - 1,
                    -INF,
                    -alpha,
                    1,
                )
            finally:
                self.leave(token)
                board.pop()

            scored_moves.append((score, move))

            if score > best:
                best = score
                best_move = move

            alpha = max(alpha, score)

        scored_moves.sort(key=lambda item: -item[0])
        self.root_moves = [m for _, m in scored_moves]

        self.check_time()

        # Giao diện của hàm trả điểm theo góc nhìn của quân Trắng.
        white_score = best * (
            1 if board.turn == chess.WHITE else -1
        )

        return best_move, white_score


def search_at_depth(
    board,
    depth,
    deadline,
    preferred_move=None,
):
    """Tìm nước tốt nhất tại độ sâu được yêu cầu."""

    if depth < 1:
        raise ValueError("depth must be at least 1")

    engine = SearchEngine(
        board,
        deadline,
    )

    return engine.search(
        board,
        depth,
        preferred_move,
    )


def max_value(
    board,
    depth,
    alpha,
    beta,
    deadline,
):
    """Tìm giá trị khi bên tối đa hóa đang tới lượt."""

    engine = SearchEngine(
        board,
        deadline,
    )

    return engine.negamax(
        board,
        depth,
        alpha,
        beta,
        0,
    )


def min_value(
    board,
    depth,
    alpha,
    beta,
    deadline,
):
    """Tìm giá trị khi bên tối thiểu hóa đang tới lượt."""

    engine = SearchEngine(
        board,
        deadline,
    )

    return -engine.negamax(
        board,
        depth,
        -beta,
        -alpha,
        0,
    )