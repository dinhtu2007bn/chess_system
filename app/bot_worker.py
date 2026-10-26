from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import chess
import numpy as np

from chesslab_sdk import BotConfig, SearchContext

PROTOCOL_VERSION = "1.0"
PROTOCOL_OUT = sys.stdout


def emit(payload: dict[str, Any]) -> None:
    PROTOCOL_OUT.write(json.dumps(payload, separators=(",", ":")) + "\n")
    PROTOCOL_OUT.flush()


def build_board(position_id: int, moves: list[str], fen: str | None = None) -> chess.Board:
    if fen:
        board = chess.Board(fen, chess960=True)
    else:
        board = chess.Board.from_chess960_pos(position_id)
    for token in moves:
        board.push_uci(token)
    return board


def load_bot_class(bot_dir: str | None) -> type:
    if not bot_dir:
        module = importlib.import_module("student.bot")
        return module.StudentBot

    bot_path = Path(bot_dir).resolve()
    student_dir = bot_path / "student" if (bot_path / "student").is_dir() else bot_path

    # Invalidate cache for student and helper modules
    for mod_name in list(sys.modules.keys()):
        if mod_name == "student" or mod_name.startswith("student.") or mod_name in ["bot", "search", "evaluation", "time_manager"]:
            del sys.modules[mod_name]

    for p in [str(student_dir), str(bot_path)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        module = importlib.import_module("student.bot")
        return module.StudentBot
    except Exception:
        module = importlib.import_module("bot")
        return module.StudentBot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot-dir", type=str, default=None)
    cli_args, _ = parser.parse_known_args()
    active_bot_dir = cli_args.bot_dir

    bot: Any = None
    emit({"type": "ready", "protocol_version": PROTOCOL_VERSION})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_type = request.get("type")

            if request_type == "init":
                seed = int(request["seed"])
                random.seed(seed)
                np.random.seed(seed & 0xFFFFFFFF)
                bot_dir_param = request.get("bot_dir") or active_bot_dir
                if bot_dir_param:
                    active_bot_dir = bot_dir_param

                config = BotConfig(
                    color=bool(request["color"]),
                    chess960_id=int(request.get("chess960_id", 518)),
                    initial_time_ms=int(request["initial_time_ms"]),
                    seed=seed,
                )
                with contextlib.redirect_stdout(sys.stderr):
                    bot_cls = load_bot_class(active_bot_dir)
                    bot = bot_cls(config)
                emit({"type": "initialized"})

            elif request_type == "move":
                if bot is None:
                    raise RuntimeError("Bot has not been initialized")
                fen = request.get("fen")
                chess960_id = int(request.get("chess960_id", 518))
                board = build_board(chess960_id, list(request["moves"]), fen=fen)

                context = SearchContext(
                    time_remaining_ms=int(request["time_remaining_ms"]),
                    opponent_time_remaining_ms=int(request["opponent_time_remaining_ms"]),
                    ply=len(request["moves"]),
                )
                with contextlib.redirect_stdout(sys.stderr):
                    move = bot.choose_move(board.copy(stack=True), context)
                if not isinstance(move, chess.Move):
                    raise TypeError("choose_move() must return chess.Move")

                emit({
                    "type": "move",
                    "move": board.uci(move, chess960=True),
                })

            elif request_type == "shutdown":
                emit({"type": "bye"})
                return

            else:
                raise ValueError(f"Unknown request type: {request_type!r}")

        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"type": "error", "error": type(exc).__name__, "message": str(exc)})


if __name__ == "__main__":
    main()
