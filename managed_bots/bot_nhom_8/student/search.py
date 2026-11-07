import chess

from .strategies.search import search_MVP, search_v2


def find_best_move(
    board: chess.Board,
    time_limit_ms: int,
    brain: dict[str, str],
) -> chess.Move:
    search_brain = brain["search"]
    eval_brain = brain["eval"]
    if search_brain == "MVP":
        return search_MVP.find_best_move(board, time_limit_ms, eval_brain)
    if search_brain == "v2":
        return search_v2.find_best_move(board, time_limit_ms, eval_brain)
    raise ValueError(f"Unknown search brain: {search_brain}")
