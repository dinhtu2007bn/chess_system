"""Negamax + alpha-beta, sâu dần theo ngân sách thời gian.

Tầng giữa duy nhất của cây phụ thuộc: chỉ import ``EvalFn`` (evaluation) và
``TimeBudget`` (time_manager), không import ``chesslab_sdk`` để test độc lập.

Quy ước:
- ``evaluate(board)`` dương = tốt cho bên đang đi, nên lá negamax trả thẳng,
  không đổi dấu (đổi dấu chỉ xảy ra một lần khi quay lên cha).
- ``deadline_s`` là hạn cứng của cả lượt; ``soft_deadline_s`` chỉ chặn việc mở
  độ sâu mới. Khi một độ sâu bị ngắt, nước tốt nhất của **tiền tố đã duyệt xong**
  vẫn được dùng (``SearchResult.partial``), vì nước gốc đầu tiên chạy cửa sổ đầy
  đủ nên điểm của nó chính xác và không thể tệ hơn độ sâu trước. Khi đó ``score``
  là cận dưới fail-soft, không dùng để công bố chiếu hết.
- Mọi ``push()`` đều ``pop()`` trong ``try/finally``; bàn cờ không bị sửa.
- Hòa được nhận ở mọi node, sau kiểm mate/bí và trước khi chấm lá, **đúng như
  trọng tài** ``outcome(claim_draw=True)``: bên đi có thể claim lặp 3 lần / 50 nước,
  thiếu quân, hoặc chạm 300 nửa nước. Trả 0 điểm.
- Thứ tự nước đi (``_ordered``): ăn quân theo MVV-LVA, phong cấp, killer.
  Chỉ đổi thứ tự duyệt, không đổi điểm: alpha-beta vẫn trả đúng giá trị minimax.
- Lá (``depth <= 0``) đi vào ``_quiesce``: chỉ duyệt ăn quân/phong hậu (hoặc mọi
  nước thoát chiếu) tới khi thế yên, nên không chấm điểm giữa một chuỗi đổi quân.
"""

import time
from dataclasses import dataclass, field
from typing import Callable

import chess

from .evaluation import PIECE_VALUES, EvalFn
from .time_manager import TimeBudget

# Điểm chiếu hết: lớn hơn mọi điểm lượng giá, cộng/trừ khoảng cách tới mate để
# bên thắng chọn mate ngắn nhất và bên thua chọn đường chống dai nhất.
MATE = 100_000
INF = 1_000_000


@dataclass(slots=True)
class SearchState:
    """Trạng thái của một lượt tìm; tạo mới mỗi nước (không tái dùng giữa các lượt).

    """

    evaluate: EvalFn
    now: Callable[[], float]
    deadline_s: float
    soft_deadline_s: float
    max_depth: int = 64
    ply: int = 0
    nodes: int = 0
    stopped: bool = False
    # Khóa thế cờ trong cửa sổ thuận nghịch trước gốc + dọc đường đi hiện tại.
    history: list[tuple] = field(default_factory=list)
    # Tối đa 2 nước yên tĩnh gây cắt nhánh, theo khoảng cách tới gốc.
    killers: dict[int, list[chess.Move]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Kết quả một lượt. ``move`` là field bắt buộc duy nhất; phần còn lại để chẩn đoán."""

    move: chess.Move
    score: int = 0
    depth: int = 0
    nodes: int = 0
    pv: tuple[chess.Move, ...] = ()
    # True khi `move` đến từ một độ sâu bị ngắt giữa chừng (ít nhất một nước gốc
    # đã duyệt xong). Khi đó `score` là cận dưới fail-soft, không phải điểm chính xác.
    partial: bool = False


def _key(board: chess.Board) -> tuple:
    """Khóa thế cờ theo nghĩa luật lặp (giống python-chess), chỉ dùng API công khai."""
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


def _history_keys(board: chess.Board) -> list[tuple]:
    """Khóa các thế đã qua, từ nước không ăn quân/đẩy tốt gần nhất tới gốc. Gọi 1 lần mỗi lượt."""
    popped = [board.pop() for _ in range(min(board.halfmove_clock, len(board.move_stack)))]
    keys = []
    for move in reversed(popped):
        keys.append(_key(board))
        board.push(move)
    keys.append(_key(board))  # chính gốc
    return keys


def _claimable_draw(board: chess.Board, history: list[tuple]) -> bool:
    """Bằng ``board.can_claim_draw()`` nhưng rẻ: dùng khóa đã lưu, không dựng lại lịch sử.

    ``history``: khóa các thế trước ``board``. Chỉ ``halfmove_clock`` phần tử cuối
    còn với tới được (trước đó có nước không đảo ngược).
    """
    halfmoves = board.halfmove_clock
    if halfmoves >= 99 and board.can_claim_fifty_moves():
        return True
    # X ... X ... (nay) -> X cần >= 7 nửa nước thuận nghịch; dưới đó không thể lặp 3 lần.
    if halfmoves < 7:
        return False
    seen: dict[tuple, int] = {}
    for key in history[-halfmoves:]:
        seen[key] = seen.get(key, 0) + 1
    if seen.get(_key(board), 0) >= 2:  # thế hiện tại là lần thứ 3
        return True
    twice = {key for key, count in seen.items() if count >= 2}
    if not twice:
        return False
    # Nước đi tới thế đã gặp 2 lần không thể là nước ăn quân, nên quân đối phương
    # phải y hệt bây giờ, và chỉ cần thử nước đi từ các ô quân mình đã rời.
    us = board.turn
    mine, theirs = board.occupied_co[us], board.occupied_co[not us]
    from_mask = 0
    for key in twice:
        key_mine, key_theirs = (key[6], key[7]) if us == chess.WHITE else (key[7], key[6])
        if key[8] != us and key_theirs == theirs:
            from_mask |= (mine & ~key_mine) or (board.kings & mine)  # nhập thành đổi chỗ vua-xe
    for move in board.generate_legal_moves(from_mask):
        board.push(move)
        try:
            if _key(board) in twice:
                return True
        finally:
            board.pop()
    return False


def _ordered(board: chess.Board, moves: list[chess.Move], state: SearchState) -> list[chess.Move]:
    """Ăn quân (MVV-LVA) > phong cấp > killer > còn lại (giữ thứ tự sinh nước)."""
    killers = state.killers.get(board.ply() - state.ply, ())

    def score(move: chess.Move) -> int:
        # is_capture(): nhập thành 960 mã hóa "vua đi vào ô xe mình" nên không được
        # suy ăn quân từ việc ô đích có quân.
        if board.is_capture(move):
            # Ô đích trống chỉ khi bắt tốt qua đường; quân đi luôn có (``or`` chỉ cho mypy).
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or chess.KING
            # Nước hợp lệ của vua không bao giờ bị ăn lại, nên vua coi như kẻ ăn rẻ nhất.
            return 30_000 + 10 * PIECE_VALUES[victim] - PIECE_VALUES.get(attacker, 0)
        if move.promotion:
            return 20_000 + PIECE_VALUES[move.promotion]
        return 10_000 if move in killers else 0

    return sorted(moves, key=score, reverse=True)


def _remember_killer(board: chess.Board, move: chess.Move, state: SearchState) -> None:
    """Ghi nước yên tĩnh gây cắt nhánh (ăn quân/phong cấp đã đứng đầu sẵn)."""
    if board.is_capture(move) or move.promotion:
        return
    slot = state.killers.setdefault(board.ply() - state.ply, [])
    if move not in slot:
        slot.insert(0, move)
        del slot[2:]


def _quiesce(state: SearchState, board: chess.Board, alpha: int, beta: int) -> int:
    """Điểm của một lá khi đã hết ăn quân: bên đi được "đứng yên" (stand pat) với
    điểm lượng giá, trừ khi bị chiếu (khi đó phải xét mọi nước thoát chiếu)."""
    if state.stopped:
        return 0
    state.nodes += 1
    if state.now() >= state.deadline_s:
        state.stopped = True
        return 0

    in_check = board.is_check()
    if in_check:
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + (board.ply() - state.ply)
    elif not any(board.generate_legal_moves()):  # dừng ở nước hợp lệ đầu tiên
        return 0
    if (
        _claimable_draw(board, state.history)
        or board.is_insufficient_material()
        or board.ply() >= 300
    ):
        return 0

    if in_check:
        best = -INF
        # Nước thoát chiếu có thể không ăn quân -> con của node này vẫn có thể lặp.
        state.history.append(_key(board))
    else:
        best = state.evaluate(board)
        if best >= beta:
            return best
        alpha = max(alpha, best)
        moves = list(board.generate_legal_captures())
        moves += [
            move
            for move in board.generate_legal_moves(
                board.pawns, chess.BB_BACKRANKS & ~board.occupied
            )
            if move.promotion == chess.QUEEN
        ]

    for move in _ordered(board, moves, state):
        board.push(move)
        try:
            score = -_quiesce(state, board, -beta, -alpha)
        finally:
            board.pop()
        if state.stopped:
            break
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    if in_check:
        state.history.pop()
    return 0 if state.stopped else best


def find_best_move(
    board: chess.Board,
    *,
    evaluate: EvalFn,
    budget: TimeBudget,
    now: Callable[[], float] = time.monotonic,
    max_depth: int = 64,
) -> SearchResult:
    """Trả nước tốt nhất tìm được trong ngân sách ``budget``.

    ``board`` được coi là hộp đen Chess960: chỉ dùng các API có sẵn
    (``legal_moves``, ``push``/``pop``, ``is_check``, ``ply`` và các hàm kiểm
    hòa) — không tự dựng nhập thành.

    Raise ``ValueError`` nếu bàn không có nước đi hợp lệ (ván đã kết thúc).

    Độ sâu 1 luôn được bắt đầu (kể cả khi ``soft_ms == 0``); nếu nó chạy xong,
    ta có một nước đã lượng giá thay vì nước đầu danh sách. Nếu bị ngắt khi chưa
    duyệt xong nước gốc nào, kết quả bị bỏ và hàm trả ``moves[0]``. Nếu bị ngắt
    sau khi đã duyệt xong ít nhất một nước gốc, kết quả dở được dùng và
    ``partial`` bằng True. Từ độ sâu 2 mới áp ``soft_deadline_s``. Nếu
    ``hard_ms <= 0`` (đã cạn giờ) trả ngay nước hợp lệ đầu tiên mà không gọi
    ``evaluate``.
    """
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("find_best_move: board has no legal moves")
    if budget.hard_ms <= 0:
        return SearchResult(move=moves[0], nodes=0, depth=0)

    start = now()
    state = SearchState(
        evaluate=evaluate,
        now=now,
        deadline_s=start + budget.hard_ms / 1000.0,
        soft_deadline_s=start + budget.soft_ms / 1000.0,
        max_depth=max(1, max_depth),  # max_depth <= 0 vẫn chạy depth 1 (lưới an toàn)
        ply=board.ply(),
        history=_history_keys(board),
    )

    best = moves[0]
    best_score = 0
    completed = 0
    partial = False
    for depth in range(1, state.max_depth + 1):
        if depth > 1 and now() >= state.soft_deadline_s:
            break
        # Nước tốt nhất độ sâu trước lên đầu, phần còn lại theo ``_ordered``.
        score, move = _search_root(board, state, depth, moves, best if completed else None)
        if state.stopped:
            # `_search_root` chỉ trả về điểm hữu hạn khi đã có ít nhất một nước gốc
            # duyệt xong (nếu không, `best_score` vẫn là -INF). Nước gốc đầu tiên
            # chạy cửa sổ đầy đủ nên điểm của nó chính xác; các nước sau chỉ có thể
            # thay thế khi thực sự cao hơn. Vì vậy dùng kết quả dở là an toàn và
            # không bao giờ tệ hơn kết quả của độ sâu trước.
            if score > -INF:
                best, best_score = move, score
                partial = True
            break
        best, best_score, completed = move, score, depth

    return SearchResult(
        move=best,
        score=best_score,
        depth=completed,
        nodes=state.nodes,
        partial=partial,
    )


def _search_root(
    board: chess.Board,
    state: SearchState,
    depth: int,
    moves: list[chess.Move],
    first: chess.Move | None,
) -> tuple[int, chess.Move]:
    """Duyệt gốc với cửa sổ mở; trả (điểm, nước). Kết quả dở nếu ``state.stopped``."""
    ordered = _ordered(board, moves, state)
    if first is not None:
        ordered = [first, *(move for move in ordered if move != first)]

    alpha = -INF
    best_score = -INF
    best_move = ordered[0]
    for move in ordered:
        board.push(move)
        try:
            score = -_negamax(state, board, depth - 1, -INF, -alpha)
        finally:
            board.pop()
        if state.stopped:
            return best_score, best_move
        if score > best_score:
            best_score, best_move = score, move
            alpha = score
    return best_score, best_move


def _negamax(
    state: SearchState,
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
) -> int:
    """Điểm theo bên đang đi tại ``board``, cửa sổ (alpha, beta)."""
    if state.stopped:
        return 0
    if depth <= 0:
        return _quiesce(state, board, alpha, beta)

    state.nodes += 1
    # Kiểm hạn ở mỗi node: gọi đồng hồ rẻ hơn nhiều so với một node (vốn đã tốn
    # hàng chục µs trở lên). Trên Windows + Python 3.12, ``time.monotonic()`` chỉ
    # phân giải 15,6 ms (GetTickCount64), nên bot truyền ``now=time.perf_counter``.
    if state.now() >= state.deadline_s:
        state.stopped = True
        return 0

    # Sinh nước một lần: dùng cho cả kiểm chiếu hết/bí nước lẫn vòng duyệt.
    moves = list(board.legal_moves)
    if not moves:
        if board.is_check():
            # Khoảng cách ply từ gốc (không phải ply tuyệt đối): mate ngắn hơn
            # được điểm cao hơn, bên thua chọn đường chống dai hơn.
            return -MATE + (board.ply() - state.ply)
        return 0
    # Hòa theo luật là hòa thật: phía chạy ván dùng outcome(claim_draw=True) và
    # dừng ở 300 nửa nước. Kiểm ở mọi node, sau mate/bí và trước khi chấm lá.
    if (
        _claimable_draw(board, state.history)
        or board.is_insufficient_material()
        or board.ply() >= 300
    ):
        return 0
    best = -INF
    state.history.append(_key(board))
    for move in _ordered(board, moves, state):
        board.push(move)
        try:
            score = -_negamax(state, board, depth - 1, -beta, -alpha)
        finally:
            board.pop()
        if state.stopped:
            break
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    _remember_killer(board, move, state)
                    break
    state.history.pop()
    return 0 if state.stopped else best
