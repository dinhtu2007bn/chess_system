import chess

from .strategies.evaluation import eval_MVP, eval_v2, eval_v3


def evaluate_board(board: chess.Board, brain: str) -> float:
    if brain == "MVP":
        return eval_MVP.evaluate_board(board)
    if brain == "v2":
        return eval_v2.evaluate_board(board)
    if brain == "v3":
        return eval_v3.evaluate_board(board)
    raise ValueError(f"Unknown evaluation brain: {brain}")
