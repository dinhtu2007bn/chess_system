from __future__ import annotations

import asyncio
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import chess
import chess.engine

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE_PATH = ROOT_DIR / "engines" / "stockfish.exe"


class ExternalEngineSparring:
    """Manages an external UCI chess engine (Stockfish) with calibrated Elo (1000 - 2500)
    and native Chess960 support for benchmarking and sparring matches.
    """

    def __init__(
        self,
        elo: int = 1700,
        engine_path: str | None = None,
    ) -> None:
        self.elo = max(1320, min(2500, elo))
        self.engine_path = engine_path or str(DEFAULT_ENGINE_PATH)
        self.engine: chess.engine.UciProtocol | None = None
        self.transport: asyncio.SubprocessTransport | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if not os.path.isfile(self.engine_path):
                raise FileNotFoundError(
                    f"Không tìm thấy Stockfish binary tại: {self.engine_path}"
                )
            self.transport, self.engine = await chess.engine.popen_uci(self.engine_path)
            try:
                await self.engine.configure({
                    "UCI_LimitStrength": True,
                    "UCI_Elo": self.elo,
                })
            except Exception as exc:
                print(f"[ExternalEngine] Warning setting Elo config: {exc}", file=sys.stderr)

    async def choose_move(
        self,
        board: chess.Board,
        time_limit_seconds: float = 1.0,
    ) -> chess.Move:
        async with self._lock:
            if self.engine is None:
                raise RuntimeError("External engine is not running")

            # Ensure board has chess960 flag set
            if not board.chess960:
                board = board.copy(stack=True)
                board.chess960 = True

            time_limit = max(0.05, time_limit_seconds)
            limit = chess.engine.Limit(time=time_limit)
            result = await self.engine.play(board, limit)
            if result.move is None:
                # Fallback to first legal move
                return next(iter(board.legal_moves))
            return result.move

    async def stop(self) -> None:
        async with self._lock:
            if self.engine is not None:
                try:
                    await self.engine.quit()
                except Exception:
                    pass
                self.engine = None
                self.transport = None


class FallbackSparringBot:
    """Heuristic sparring bot in case UCI engine is unavailable."""

    def __init__(self, elo: int = 1500) -> None:
        self.elo = elo

    async def choose_move(
        self,
        board: chess.Board,
        time_limit_seconds: float = 0.5,
    ) -> chess.Move:
        await asyncio.sleep(min(0.2, time_limit_seconds * 0.3))
        moves = list(board.legal_moves)
        if not moves:
            raise RuntimeError("No legal moves")

        # Pick captures or checks with higher probability
        tactical = [m for m in moves if board.is_capture(m) or board.gives_check(m)]
        if tactical and random.random() < 0.7:
            return random.choice(tactical)
        return random.choice(moves)

    async def stop(self) -> None:
        pass

