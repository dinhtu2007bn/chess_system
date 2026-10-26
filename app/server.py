from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import random
import secrets
import sys
from typing import Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .benchmark_manager import ArenaRunner, BenchmarkRunner, load_positions
from .bot_manager import BotManager
from .chess960_rules import new_board
from .game_controller import GameController
from .schemas import (
    BotRenameRequest,
    MoveRequest,
    StartArenaRequest,
    StartBenchmarkRequest,
    StartGameRequest,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static" / "built-ui"
ENGINE_PATH = ROOT / "engines" / "stockfish.exe"

connections: set[WebSocket] = set()


async def broadcast(payload: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    for socket in list(connections):
        try:
            await socket.send_json(payload)
        except Exception:
            dead.append(socket)
    for socket in dead:
        connections.discard(socket)


bot_manager = BotManager()
game = GameController(broadcast)
benchmark_runner = BenchmarkRunner(bot_manager, broadcast)
arena_runner = ArenaRunner(bot_manager, broadcast)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await game.close()
    await benchmark_runner.stop()
    await arena_runner.stop()


app = FastAPI(title="ChessLab 960 Arena & Benchmark", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== GAME CONTROLLER API ====================

@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return game.state()


@app.get("/api/preview/{position_id}")
async def preview_position(position_id: int) -> dict[str, Any]:
    if not 0 <= position_id <= 959:
        raise HTTPException(status_code=422, detail="Chess960 ID must be from 0 to 959")
    board = new_board(position_id)
    return {"chess960_id": position_id, "fen": board.fen(shredder=True)}


@app.post("/api/start")
async def start_game(request: StartGameRequest) -> dict[str, Any]:
    bot_dir = None
    if request.bot_id:
        b_info = bot_manager.get_bot(request.bot_id)
        if b_info and b_info.get("dir_path"):
            bot_dir = b_info["dir_path"]

    await game.start(
        request.bot_color,
        request.time_seconds,
        request.chess960_id,
        bot_dir=bot_dir,
    )
    return game.state()


@app.post("/api/stop")
async def stop_game() -> dict[str, Any]:
    await game.stop()
    return game.state()


@app.post("/api/move")
async def submit_move(request: MoveRequest) -> dict[str, Any]:
    try:
        await game.human_move(request.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return game.state()


# ==================== BOT MANAGER API ====================

@app.get("/api/bots")
async def list_bots() -> list[dict[str, Any]]:
    return bot_manager.list_bots()


@app.post("/api/bots/upload")
async def upload_bot(
    name: str = Form(...),
    description: str = Form(""),
    relative_paths: str | None = Form(None),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    rel_path_list = []
    if relative_paths:
        try:
            rel_path_list = json.loads(relative_paths)
        except Exception:
            rel_path_list = []

    file_tuples: list[tuple[str, bytes]] = []
    for idx, upload_file in enumerate(files):
        content = await upload_file.read()
        rel_path = (
            rel_path_list[idx]
            if idx < len(rel_path_list) and rel_path_list[idx]
            else (upload_file.filename or "unknown")
        )
        file_tuples.append((rel_path, content))

    try:
        new_bot = bot_manager.save_uploaded_files(name, file_tuples, description)
        return new_bot
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/bots/{bot_id}")
async def rename_bot(bot_id: str, request: BotRenameRequest) -> dict[str, Any]:
    try:
        return bot_manager.rename_bot(bot_id, request.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/bots/{bot_id}")
async def delete_bot(bot_id: str) -> dict[str, Any]:
    try:
        bot_manager.delete_bot(bot_id)
        return {"success": True, "deleted_id": bot_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ==================== POSITION SUITE API ====================

@app.get("/api/positions")
async def get_positions() -> list[dict[str, Any]]:
    return load_positions()


@app.get("/api/positions/random")
async def get_random_positions(count: int = Query(default=5, ge=1, le=40)) -> list[dict[str, Any]]:
    all_pos = load_positions()
    if not all_pos:
        return []
    sample_size = min(count, len(all_pos))
    return random.sample(all_pos, sample_size)


@app.post("/api/positions/generate-opening")
async def generate_opening_position() -> dict[str, Any]:
    sp_id = secrets.randbelow(960)
    board = new_board(sp_id)
    return {
        "id": f"gen-open-{sp_id}",
        "name": f"Khai cuộc ngẫu nhiên Chess960 #{sp_id}",
        "category": "opening",
        "fen": board.fen(shredder=True),
        "chess960_id": sp_id,
        "description": f"Thế cờ xuất phát ngẫu nhiên từ biến thể Chess960 mã số #{sp_id}.",
        "recommended_time_ms": 2000,
    }


# ==================== BENCHMARK API ====================

@app.get("/api/benchmark/status")
async def get_benchmark_status() -> dict[str, Any]:
    return benchmark_runner.get_status()


@app.post("/api/benchmark/start")
async def start_benchmark(request: StartBenchmarkRequest) -> dict[str, Any]:
    try:
        game_time = request.game_time_seconds or request.global_time_seconds or 60
        session = await benchmark_runner.start_benchmark(
            bot_id=request.bot_id,
            position_ids=request.position_ids,
            opponent_type=request.opponent_type,
            opponent_elo=request.opponent_elo,
            opponent_bot_id=request.opponent_bot_id,
            time_mode=request.time_mode,
            game_time_seconds=game_time,
            increment_seconds=request.increment_seconds,
            per_position_times=request.per_position_times,
            runs_per_position=request.runs_per_position,
            concurrency=request.concurrency,
        )
        return session
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/benchmark/stop")
async def stop_benchmark() -> dict[str, Any]:
    await benchmark_runner.stop()
    return {"success": True, "status": "stopped"}


# ==================== ARENA API ====================

@app.get("/api/arena/status")
async def get_arena_status() -> dict[str, Any]:
    return arena_runner.get_status()


@app.post("/api/arena/start")
async def start_arena(request: StartArenaRequest) -> dict[str, Any]:
    try:
        game_time = request.game_time_seconds or request.time_seconds or 60
        session = await arena_runner.start_arena(
            bot_ids=request.bot_ids,
            position_ids=request.position_ids,
            game_time_seconds=game_time,
            increment_seconds=request.increment_seconds,
            format_type=request.format_type,
        )
        return session
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/arena/stop")
async def stop_arena() -> dict[str, Any]:
    await arena_runner.stop()
    return {"success": True, "status": "stopped"}


# ==================== EXTERNAL ENGINE STATUS ====================

@app.get("/api/engine/status")
async def get_engine_status() -> dict[str, Any]:
    is_installed = ENGINE_PATH.is_file()
    return {
        "engine": "Stockfish 19 Universal",
        "path": str(ENGINE_PATH),
        "is_installed": is_installed,
        "chess960_supported": True,
        "elo_range": [1000, 2500],
        "default_elo": 1700,
    }


# ==================== WEBSOCKET ====================

@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket) -> None:
    await socket.accept()
    connections.add(socket)
    await socket.send_json(game.state())
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        connections.discard(socket)


# ==================== FRONTEND STATIC FILES ====================

if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")


@app.get("/chesslab-icon.svg", include_in_schema=False)
async def app_icon() -> FileResponse:
    return FileResponse(STATIC / "chesslab-icon.svg", media_type="image/svg+xml")


@app.get("/manifest.webmanifest", include_in_schema=False)
async def web_manifest() -> FileResponse:
    return FileResponse(
        STATIC / "manifest.webmanifest",
        media_type="application/manifest+json",
    )


@app.get("/{path:path}")
async def frontend(path: str) -> FileResponse:
    index = STATIC / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="Frontend has not been built")
    return FileResponse(
        index,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
