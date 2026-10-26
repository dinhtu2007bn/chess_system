from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import secrets
import time
from typing import Any, Callable, Coroutine

import chess

from .bot_manager import BotManager
from .bot_process import BotProcess, BotProcessError
from .chess960_rules import display_legal_moves, move_for_token, new_board
from .clock import ChessClock
from .external_engine import ExternalEngineSparring, FallbackSparringBot

ROOT_DIR = Path(__file__).resolve().parents[1]
POSITIONS_FILE = ROOT_DIR / "app" / "chess960_positions.json"


def load_positions() -> list[dict[str, Any]]:
    if not POSITIONS_FILE.exists():
        return []
    with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_optimization_metrics(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculates quantitative benchmark score and diagnostic time management comment."""
    total = len(games)
    if total == 0:
        return {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "timeout_losses": 0,
            "checkmate_losses": 0,
            "win_rate": 0.0,
            "timeout_penalty": 0.0,
            "time_efficiency_factor": 1.0,
            "optimization_score": 0.0,
            "avg_move_time_ms": 0.0,
            "peak_move_time_ms": 0.0,
            "verdict": "Chưa có ván đấu nào được hoàn thành.",
        }

    wins = sum(1 for g in games if g["bot_outcome"] == "win")
    draws = sum(1 for g in games if g["bot_outcome"] == "draw")
    losses = sum(1 for g in games if g["bot_outcome"] == "loss")
    timeout_losses = sum(1 for g in games if g["is_timeout"] and g["bot_outcome"] == "loss")
    checkmate_losses = losses - timeout_losses

    win_rate = ((wins + 0.5 * draws) / total) * 100.0
    timeout_penalty = timeout_losses / total

    move_times = [g["avg_move_time_ms"] for g in games if g.get("avg_move_time_ms", 0) > 0]
    peak_times = [g["peak_move_time_ms"] for g in games if g.get("peak_move_time_ms", 0) > 0]

    avg_move_time = sum(move_times) / len(move_times) if move_times else 0.0
    peak_move_time = max(peak_times) if peak_times else 0.0

    # Time Efficiency Factor
    if timeout_penalty == 0:
        time_efficiency = 1.0
    elif timeout_penalty <= 0.1:
        time_efficiency = 0.85
    elif timeout_penalty <= 0.25:
        time_efficiency = 0.65
    elif timeout_penalty <= 0.5:
        time_efficiency = 0.4
    else:
        time_efficiency = 0.15

    opt_score = win_rate * (1.0 - timeout_penalty) * time_efficiency

    # Quantitative verdict
    if timeout_losses == 0:
        if win_rate >= 70.0:
            verdict = (
                f"XUẤT SẮC: Thuật toán quản lý thời gian (time_manager.py) tối ưu tuyệt đối (0% Timeout, "
                f"WinRate {win_rate:.1f}%). Khả năng phân bổ thời gian giữa các giai đoạn khai cuộc - trung cuộc "
                f"và tàn cuộc rất nhịp nhàng, khai thác triệt để độ sâu mà không gặp bất kỳ nguy cơ cạn giờ nào."
            )
        elif win_rate >= 45.0:
            verdict = (
                f"ỔN ĐỊNH: Quản lý thời gian an toàn và hiệu quả (0% Timeout, WinRate {win_rate:.1f}%). "
                f"Thời gian trung bình {avg_move_time:.0f}ms/nước, đỉnh điểm {peak_move_time:.0f}ms. "
                f"Có thể cân nhắc tăng nhẹ độ sâu tìm kiếm trong các thế cờ trung cuộc ít quân để gia tăng sức tấn công."
            )
        else:
            verdict = (
                f"AN TOÀN NHƯNG THỤ ĐỘNG: Không bị lỗi tràn giờ (0% Timeout), nhưng tỷ lệ thắng chưa cao "
                f"({win_rate:.1f}%). Bot có xu hướng đánh quá nhanh hoặc nhường thời gian dư thừa. "
                f"Nên tăng hệ số thời gian cho các thế cờ chiến thuật phức tạp."
            )
    elif timeout_penalty <= 0.15:
        verdict = (
            f"CẢNH BÁO NHẸ: Xuất hiện {timeout_losses}/{total} ván thua do tràn giờ ({timeout_penalty*100:.1f}%). "
            f"Thuật toán time_manager.py còn hơi mạo hiểm ở các thế cờ trung cuộc có nhiều nhánh ăn quân. "
            f"Khuyến nghị: Tăng ngưỡng safety margin thêm 100-200ms để triệt tiêu hoàn toàn rủi ro bị xử thua theo giờ."
        )
    else:
        verdict = (
            f"NGUY HIỂM CAO: Tỷ lệ tràn giờ nghiêm trọng ({timeout_losses}/{total} ván, {timeout_penalty*100:.1f}%). "
            f"Thuật toán Iterative Deepening chưa dừng kịp khi chạm ngưỡng hard limit hoặc hàm time_manager.py "
            f"dự đoán số nước còn lại quá thấp. Bắt buộc hạ bớt độ sâu duyệt hoặc siết chặt kiểm tra thời gian giữa các node."
        )

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "timeout_losses": timeout_losses,
        "checkmate_losses": checkmate_losses,
        "win_rate": round(win_rate, 2),
        "timeout_penalty": round(timeout_penalty, 4),
        "time_efficiency_factor": round(time_efficiency, 2),
        "optimization_score": round(opt_score, 2),
        "avg_move_time_ms": round(avg_move_time, 1),
        "peak_move_time_ms": round(peak_move_time, 1),
        "verdict": verdict,
    }


class BenchmarkRunner:
    """Coordinates a 3-run benchmark session across selected Chess960 positions
    with live real-time step-by-step chessboard updates.
    """

    def __init__(
        self,
        bot_manager: BotManager,
        broadcast_fn: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        self.bot_manager = bot_manager
        self.broadcast_fn = broadcast_fn
        self.active_session: dict[str, Any] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = False

    def get_status(self) -> dict[str, Any]:
        if self.active_session is None:
            return {"status": "idle", "session": None}
        return {
            "status": self.active_session.get("status", "idle"),
            "session": self.active_session,
        }

    async def stop(self) -> None:
        self._stop_requested = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self.active_session:
            self.active_session["status"] = "stopped"
            await self.broadcast_fn({"type": "benchmark_update", "session": self.active_session})

    async def start_benchmark(
        self,
        bot_id: str,
        position_ids: list[str],
        opponent_type: str = "stockfish",
        opponent_elo: int = 1700,
        opponent_bot_id: str | None = None,
        time_mode: str = "global",
        game_time_seconds: int = 60,
        increment_seconds: int = 1,
        global_time_seconds: int | None = None,
        per_position_times: dict[str, int] | None = None,
        runs_per_position: int = 3,
        concurrency: int = 3,
    ) -> dict[str, Any]:
        await self.stop()
        self._stop_requested = False
        effective_time = global_time_seconds if global_time_seconds is not None else game_time_seconds

        bot_info = self.bot_manager.get_bot(bot_id)
        if not bot_info:
            raise KeyError(f"Bot '{bot_id}' không tồn tại")

        all_positions = load_positions()
        pos_map = {p["id"]: p for p in all_positions}
        selected_positions = [pos_map[pid] for pid in position_ids if pid in pos_map]
        if not selected_positions:
            raise ValueError("Không có thế cờ hợp lệ nào được chọn")

        total_games = len(selected_positions) * runs_per_position
        effective_concurrency = max(1, min(concurrency, total_games, 8))
        session_id = f"bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        live_boards = {}
        for s_id in range(effective_concurrency):
            pos_sample = selected_positions[min(s_id, len(selected_positions) - 1)]
            live_boards[str(s_id)] = {
                "slot_id": s_id,
                "position_id": pos_sample["id"],
                "position_name": pos_sample["name"],
                "run_number": 1,
                "fen": pos_sample["fen"],
                "turn": "white",
                "bot_color": "white",
                "white_ms": effective_time * 1000,
                "black_ms": effective_time * 1000,
                "history": [],
                "status": "waiting",
            }

        self.active_session = {
            "id": session_id,
            "bot_id": bot_id,
            "bot_name": bot_info["name"],
            "bot_dir": bot_info["dir_path"],
            "opponent_type": opponent_type,
            "opponent_elo": opponent_elo,
            "opponent_bot_id": opponent_bot_id,
            "opponent_name": f"Stockfish ({opponent_elo} Elo)" if opponent_type == "stockfish" else (opponent_bot_id or "Bot đối thủ"),
            "time_mode": time_mode,
            "game_time_seconds": effective_time,
            "increment_seconds": increment_seconds,
            "per_position_times": per_position_times or {},
            "runs_per_position": runs_per_position,
            "concurrency": effective_concurrency,
            "selected_positions": selected_positions,
            "total_games": total_games,
            "games_completed": 0,
            "current_game_index": 1,
            "current_position_name": selected_positions[0]["name"] if selected_positions else "",
            "current_position_fen": selected_positions[0]["fen"] if selected_positions else "",
            "current_run": 1,
            "status": "running",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "games": [],
            "metrics": calculate_optimization_metrics([]),
            "live_boards": live_boards,
            "live_board": live_boards.get("0", {
                "fen": selected_positions[0]["fen"] if selected_positions else "",
                "turn": "white",
                "white_ms": effective_time * 1000,
                "black_ms": effective_time * 1000,
                "history": [],
            }),
        }

        self._task = asyncio.create_task(self._run_benchmark_loop())
        return self.active_session

    async def _run_benchmark_loop(self) -> None:
        session = self.active_session
        if not session:
            return

        selected_positions = session["selected_positions"]
        runs_per_pos = session["runs_per_position"]
        bot_dir = session["bot_dir"]
        time_mode = session["time_mode"]
        game_time_sec = session["game_time_seconds"]
        inc_sec = session["increment_seconds"]
        per_pos_times = session["per_position_times"]
        opponent_type = session["opponent_type"]
        opponent_elo = session["opponent_elo"]
        opponent_bot_id = session["opponent_bot_id"]
        concurrency = session.get("concurrency", 3)

        opponent_bot_dir = None
        if opponent_type == "managed_bot" and opponent_bot_id:
            opp_info = self.bot_manager.get_bot(opponent_bot_id)
            if opp_info:
                opponent_bot_dir = opp_info.get("dir_path")

        # Build game tasks queue
        game_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        game_idx = 0
        for pos in selected_positions:
            pos_id = pos["id"]
            pos_name = pos["name"]
            fen = pos["fen"]
            chess960_id = pos.get("chess960_id", 518)

            if time_mode == "per_position":
                pos_sec = per_pos_times.get(pos_id, pos.get("recommended_time_ms", 3000) // 1000)
                total_clock_ms = max(500, pos_sec * 1000)
            else:
                total_clock_ms = max(500, game_time_sec * 1000)

            increment_ms = max(0, inc_sec * 1000)

            for run_num in range(1, runs_per_pos + 1):
                game_idx += 1
                bot_is_white = (run_num % 2 != 0)
                game_queue.put_nowait({
                    "game_index": game_idx,
                    "pos_id": pos_id,
                    "pos_name": pos_name,
                    "fen": fen,
                    "chess960_id": chess960_id,
                    "run_num": run_num,
                    "bot_is_white": bot_is_white,
                    "total_clock_ms": total_clock_ms,
                    "increment_ms": increment_ms,
                })

        num_workers = min(concurrency, game_queue.qsize())
        if num_workers < 1:
            num_workers = 1

        lock = asyncio.Lock()

        async def worker(slot_id: int) -> None:
            while not game_queue.empty() and not self._stop_requested:
                try:
                    task = game_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                pos_id = task["pos_id"]
                pos_name = task["pos_name"]
                fen = task["fen"]
                chess960_id = task["chess960_id"]
                run_num = task["run_num"]
                bot_is_white = task["bot_is_white"]
                total_clock_ms = task["total_clock_ms"]
                increment_ms = task["increment_ms"]

                async with lock:
                    session["current_game_index"] = task["game_index"]
                    session["current_position_name"] = pos_name
                    session["current_position_fen"] = fen
                    session["current_run"] = run_num
                    await self.broadcast_fn({"type": "benchmark_update", "session": session})

                try:
                    game_result = await self._play_single_game(
                        slot_id=slot_id,
                        pos_id=pos_id,
                        pos_name=pos_name,
                        fen=fen,
                        chess960_id=chess960_id,
                        run_num=run_num,
                        bot_dir=bot_dir,
                        bot_is_white=bot_is_white,
                        game_time_ms=total_clock_ms,
                        increment_ms=increment_ms,
                        opponent_type=opponent_type,
                        opponent_elo=opponent_elo,
                        opponent_bot_dir=opponent_bot_dir,
                    )
                except Exception as g_exc:
                    import traceback
                    traceback.print_exc()
                    game_result = {
                        "game_id": f"{pos_id}-r{run_num}",
                        "position_id": pos_id,
                        "position_name": pos_name,
                        "run_number": run_num,
                        "bot_color": "white" if bot_is_white else "black",
                        "result": "0-1" if bot_is_white else "1-0",
                        "bot_outcome": "loss",
                        "termination": f"error: {type(g_exc).__name__}: {g_exc}",
                        "is_timeout": False,
                        "moves_count": 0,
                        "bot_moves_count": 0,
                        "avg_move_time_ms": 0.0,
                        "peak_move_time_ms": 0.0,
                        "total_time_ms": 0,
                        "final_fen": fen,
                    }

                async with lock:
                    session["games"].append(game_result)
                    session["games_completed"] = len(session["games"])
                    session["metrics"] = calculate_optimization_metrics(session["games"])
                    if str(slot_id) in session.get("live_boards", {}):
                        session["live_boards"][str(slot_id)]["status"] = "finished"
                        session["live_boards"][str(slot_id)]["result"] = game_result.get("result")
                        session["live_boards"][str(slot_id)]["termination"] = game_result.get("termination")
                    await self.broadcast_fn({"type": "benchmark_update", "session": session})

                game_queue.task_done()

        try:
            workers = [asyncio.create_task(worker(i)) for i in range(num_workers)]
            await asyncio.gather(*workers)

            if not self._stop_requested:
                session["status"] = "finished"
                session["end_time"] = datetime.now(timezone.utc).isoformat()
                await self.broadcast_fn({"type": "benchmark_update", "session": session})

        except asyncio.CancelledError:
            session["status"] = "stopped"
        except Exception as exc:
            import traceback
            traceback.print_exc()
            session["status"] = "error"
            session["error"] = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            await self.broadcast_fn({"type": "benchmark_update", "session": session})

    async def _play_single_game(
        self,
        slot_id: int,
        pos_id: str,
        pos_name: str,
        fen: str,
        chess960_id: int,
        run_num: int,
        bot_dir: str,
        bot_is_white: bool,
        game_time_ms: int,
        increment_ms: int,
        opponent_type: str,
        opponent_elo: int,
        opponent_bot_dir: str | None,
    ) -> dict[str, Any]:
        """Plays one complete match with accurate dual clock and live move broadcasting."""
        board = chess.Board(fen, chess960=True)
        clock = ChessClock(game_time_ms, increment_ms=increment_ms)
        bot_color = chess.WHITE if bot_is_white else chess.BLACK

        bot_proc = BotProcess(bot_dir=bot_dir)
        opp_proc: BotProcess | None = None
        engine_opponent: ExternalEngineSparring | None = None
        fallback_opponent: FallbackSparringBot | None = None

        bot_move_times: list[int] = []
        moves_played: list[str] = []
        move_history_records: list[dict[str, Any]] = []
        result = "1/2-1/2"
        termination = "normal"
        bot_outcome = "draw"
        is_timeout = False

        # Broadcast initial board state for this game
        await self._broadcast_live_step(
            slot_id, pos_id, pos_name, run_num, board, clock, bot_color, None, None, []
        )

        try:
            await bot_proc.start()
            # Initialize student bot with full game clock
            await bot_proc.request(
                {
                    "type": "init",
                    "color": bot_color,
                    "chess960_id": chess960_id,
                    "initial_time_ms": game_time_ms,
                    "seed": secrets.randbits(63),
                    "bot_dir": bot_dir,
                },
                timeout=max(10.0, game_time_ms / 1000.0),
            )

            # Initialize opponent
            if opponent_type == "stockfish":
                try:
                    engine_opponent = ExternalEngineSparring(elo=opponent_elo)
                    await engine_opponent.start()
                except Exception:
                    fallback_opponent = FallbackSparringBot(elo=opponent_elo)
            elif opponent_type == "managed_bot" and opponent_bot_dir:
                opp_proc = BotProcess(bot_dir=opponent_bot_dir)
                await opp_proc.start()
                await opp_proc.request(
                    {
                        "type": "init",
                        "color": not bot_color,
                        "chess960_id": chess960_id,
                        "initial_time_ms": game_time_ms,
                        "seed": secrets.randbits(63),
                        "bot_dir": opponent_bot_dir,
                    },
                    timeout=max(10.0, game_time_ms / 1000.0),
                )
            else:
                fallback_opponent = FallbackSparringBot(elo=opponent_elo)

            max_plies = 180

            while len(moves_played) < max_plies:
                if self._stop_requested:
                    termination = "stopped_by_user"
                    break

                turn = board.turn
                clock.start(turn)
                curr_remaining = clock.value(turn)
                opp_remaining = clock.value(not turn)

                if curr_remaining <= 0:
                    clock.stop()
                    termination = "timeout"
                    result = "0-1" if turn == chess.WHITE else "1-0"
                    is_timeout = True
                    bot_outcome = "loss" if turn == bot_color else "win"
                    break

                if turn == bot_color:
                    # ================= BOT TURN =================
                    t0 = time.monotonic()
                    try:
                        resp = await bot_proc.request(
                            {
                                "type": "move",
                                "chess960_id": chess960_id,
                                "moves": moves_played,
                                "fen": fen,
                                "time_remaining_ms": curr_remaining,
                                "opponent_time_remaining_ms": opp_remaining,
                            },
                            timeout=max(0.5, curr_remaining / 1000.0 + 1.0),
                        )
                        # README Rule: Minimum 500ms per move
                        calc_ms = max(0, int((time.monotonic() - t0) * 1000))
                        delay_ms = max(0, 500 - calc_ms)
                        if delay_ms > 0:
                            await asyncio.sleep(min(delay_ms, clock.value(turn)) / 1000.0)

                        elapsed_ms = clock.stop()
                        bot_move_times.append(elapsed_ms)

                        # Check if bot exhausted clock
                        if clock.remaining[bot_color] <= 0:
                            termination = "timeout"
                            result = "0-1" if bot_color == chess.WHITE else "1-0"
                            bot_outcome = "loss"
                            is_timeout = True
                            break

                        move_str = str(resp.get("move", ""))
                        move = move_for_token(board, move_str)
                    except (BotProcessError, Exception) as exc:
                        clock.stop()
                        err_str = str(exc).lower()
                        if "timed out" in err_str:
                            termination = "timeout"
                            is_timeout = True
                        else:
                            termination = f"bot_failure: {exc}"
                        result = "0-1" if bot_color == chess.WHITE else "1-0"
                        bot_outcome = "loss"
                        break
                else:
                    # ================= OPPONENT TURN =================
                    try:
                        # Stockfish thinks for reasonable slice of its remaining clock
                        opp_time_sec = max(0.1, min(2.5, curr_remaining / 1000.0 * 0.06))
                        if engine_opponent:
                            move = await engine_opponent.choose_move(board, time_limit_seconds=opp_time_sec)
                        elif opp_proc:
                            resp = await opp_proc.request(
                                {
                                    "type": "move",
                                    "chess960_id": chess960_id,
                                    "moves": moves_played,
                                    "fen": fen,
                                    "time_remaining_ms": curr_remaining,
                                    "opponent_time_remaining_ms": opp_remaining,
                                },
                                timeout=max(0.5, curr_remaining / 1000.0 + 1.0),
                            )
                            move = move_for_token(board, str(resp.get("move", "")))
                        else:
                            move = await fallback_opponent.choose_move(board, time_limit_seconds=opp_time_sec)

                        elapsed_ms = clock.stop()

                        if clock.remaining[not bot_color] <= 0:
                            termination = "timeout"
                            result = "1-0" if bot_color == chess.WHITE else "0-1"
                            bot_outcome = "win"
                            break

                    except Exception as exc:
                        clock.stop()
                        termination = f"opponent_failure: {exc}"
                        result = "1-0" if bot_color == chess.WHITE else "0-1"
                        bot_outcome = "win"
                        break

                # Execute Move on Board
                san = board.san(move)
                token = board.uci(move, chess960=True)
                board.push(move)
                moves_played.append(token)
                move_history_records.append({
                    "ply": len(moves_played),
                    "san": san,
                    "uci": token,
                    "turn": "white" if turn == chess.WHITE else "black",
                    "elapsed_ms": elapsed_ms,
                })

                # Broadcast live step so the frontend updates move-by-move!
                await self._broadcast_live_step(
                    slot_id, pos_id, pos_name, run_num, board, clock, bot_color, token, san, move_history_records
                )

                # Tiny pacing delay for smooth visual chess movement
                await asyncio.sleep(0.04)

                # Check game termination conditions
                if board.is_checkmate():
                    winner = not board.turn
                    result = "1-0" if winner == chess.WHITE else "0-1"
                    termination = "checkmate"
                    bot_outcome = "win" if winner == bot_color else "loss"
                    break
                elif board.is_stalemate():
                    result = "1/2-1/2"
                    termination = "stalemate"
                    bot_outcome = "draw"
                    break
                elif board.is_insufficient_material():
                    result = "1/2-1/2"
                    termination = "insufficient_material"
                    bot_outcome = "draw"
                    break
                elif board.can_claim_threefold_repetition():
                    result = "1/2-1/2"
                    termination = "threefold_repetition"
                    bot_outcome = "draw"
                    break
                elif board.can_claim_fifty_moves():
                    result = "1/2-1/2"
                    termination = "fifty_moves"
                    bot_outcome = "draw"
                    break

        finally:
            await bot_proc.stop()
            if opp_proc:
                await opp_proc.stop()
            if engine_opponent:
                await engine_opponent.stop()

        avg_time = sum(bot_move_times) / len(bot_move_times) if bot_move_times else 0.0
        peak_time = max(bot_move_times) if bot_move_times else 0.0

        return {
            "game_id": f"{pos_id}-r{run_num}",
            "position_id": pos_id,
            "position_name": pos_name,
            "run_number": run_num,
            "bot_color": "white" if bot_is_white else "black",
            "result": result,
            "bot_outcome": bot_outcome,
            "termination": termination,
            "is_timeout": is_timeout,
            "moves_count": len(moves_played),
            "bot_moves_count": len(bot_move_times),
            "avg_move_time_ms": round(avg_time, 1),
            "peak_move_time_ms": peak_time,
            "total_time_ms": sum(bot_move_times),
            "final_fen": board.fen(shredder=True),
        }

    async def _broadcast_live_step(
        self,
        slot_id: int,
        pos_id: str,
        pos_name: str,
        run_num: int,
        board: chess.Board,
        clock: ChessClock,
        bot_color: bool,
        last_move: str | None,
        last_move_san: str | None,
        history: list[dict[str, Any]],
    ) -> None:
        fen_str = board.fen(shredder=True)
        turn_str = "white" if board.turn == chess.WHITE else "black"
        bot_color_str = "white" if bot_color == chess.WHITE else "black"
        white_ms = clock.value(chess.WHITE)
        black_ms = clock.value(chess.BLACK)

        board_state = {
            "slot_id": slot_id,
            "position_id": pos_id,
            "position_name": pos_name,
            "run_number": run_num,
            "fen": fen_str,
            "turn": turn_str,
            "bot_color": bot_color_str,
            "white_ms": white_ms,
            "black_ms": black_ms,
            "last_move": last_move,
            "last_move_san": last_move_san,
            "moves_count": len(history),
            "history": history[-10:],
            "status": "playing",
        }

        if self.active_session:
            if "live_boards" not in self.active_session:
                self.active_session["live_boards"] = {}
            self.active_session["live_boards"][str(slot_id)] = board_state
            if slot_id == 0 or "live_board" not in self.active_session:
                self.active_session["live_board"] = board_state

        await self.broadcast_fn({
            "type": "benchmark_live_move",
            "slot_id": slot_id,
            "position_id": pos_id,
            "position_name": pos_name,
            "run_number": run_num,
            "fen": fen_str,
            "white_ms": white_ms,
            "black_ms": black_ms,
            "turn": turn_str,
            "bot_color": bot_color_str,
            "last_move": last_move,
            "last_move_san": last_move_san,
            "moves_count": len(history),
            "history": history[-10:],
        })


class ArenaRunner:
    """Coordinates Bot-vs-Bot tournaments and head-to-head matches with internal Elo calculation."""

    def __init__(
        self,
        bot_manager: BotManager,
        broadcast_fn: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        self.bot_manager = bot_manager
        self.broadcast_fn = broadcast_fn
        self.active_session: dict[str, Any] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = False

    def get_status(self) -> dict[str, Any]:
        if self.active_session is None:
            return {"status": "idle", "session": None}
        return {
            "status": self.active_session.get("status", "idle"),
            "session": self.active_session,
        }

    async def stop(self) -> None:
        self._stop_requested = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self.active_session:
            self.active_session["status"] = "stopped"
            if "live_board" in self.active_session:
                self.active_session["live_board"]["status"] = "stopped"
            await self.broadcast_fn({"type": "arena_update", "session": self.active_session})

    async def start_arena(
        self,
        bot_ids: list[str],
        position_ids: list[str],
        game_time_seconds: int = 60,
        increment_seconds: int = 0,
        time_seconds: int | None = None,
        format_type: str = "round_robin",
    ) -> dict[str, Any]:
        await self.stop()
        self._stop_requested = False
        effective_time = time_seconds if time_seconds is not None else game_time_seconds

        if len(bot_ids) < 2:
            raise ValueError("Đấu trường yêu cầu tối thiểu 2 bot để thi đấu")

        all_bots = {b["id"]: b for b in self.bot_manager.list_bots()}
        participating_bots = []
        for bid in bot_ids:
            if bid not in all_bots:
                raise KeyError(f"Bot '{bid}' không tồn tại")
            participating_bots.append(all_bots[bid])

        all_positions = load_positions()
        pos_map = {p["id"]: p for p in all_positions}
        selected_positions = [pos_map[pid] for pid in position_ids if pid in pos_map]
        if not selected_positions:
            selected_positions = all_positions[:5]

        # Generate pairings (both colors for fairness on each position)
        matches_to_play: list[dict[str, Any]] = []
        match_id_counter = 1

        for pos in selected_positions:
            for i in range(len(participating_bots)):
                for j in range(i + 1, len(participating_bots)):
                    b1 = participating_bots[i]
                    b2 = participating_bots[j]

                    matches_to_play.append({
                        "match_id": match_id_counter,
                        "position_id": pos["id"],
                        "position_name": pos["name"],
                        "chess960_id": pos.get("chess960_id", 518),
                        "fen": pos["fen"],
                        "white_bot_id": b1["id"],
                        "white_bot_name": b1["name"],
                        "white_bot_dir": b1["dir_path"],
                        "black_bot_id": b2["id"],
                        "black_bot_name": b2["name"],
                        "black_bot_dir": b2["dir_path"],
                        "status": "pending",
                    })
                    match_id_counter += 1

                    matches_to_play.append({
                        "match_id": match_id_counter,
                        "position_id": pos["id"],
                        "position_name": pos["name"],
                        "chess960_id": pos.get("chess960_id", 518),
                        "fen": pos["fen"],
                        "white_bot_id": b2["id"],
                        "white_bot_name": b2["name"],
                        "white_bot_dir": b2["dir_path"],
                        "black_bot_id": b1["id"],
                        "black_bot_name": b1["name"],
                        "black_bot_dir": b1["dir_path"],
                        "status": "pending",
                    })
                    match_id_counter += 1

        session_id = f"arena-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        elo_table = {b["id"]: 1700.0 for b in participating_bots}
        score_table = {
            b["id"]: {"wins": 0, "draws": 0, "losses": 0, "points": 0.0, "timeouts": 0}
            for b in participating_bots
        }

        self.active_session = {
            "id": session_id,
            "format": format_type,
            "bots": participating_bots,
            "positions": selected_positions,
            "game_time_seconds": effective_time,
            "increment_seconds": increment_seconds,
            "total_matches": len(matches_to_play),
            "completed_matches": 0,
            "current_match": None,
            "status": "running",
            "matches": matches_to_play,
            "elo_ratings": elo_table,
            "score_table": score_table,
            "leaderboard": [],
            "live_board": {
                "match_id": 1,
                "fen": selected_positions[0]["fen"] if selected_positions else "",
                "turn": "white",
                "white_ms": effective_time * 1000,
                "black_ms": effective_time * 1000,
                "history": [],
                "white_bot_name": matches_to_play[0]["white_bot_name"] if matches_to_play else "",
                "black_bot_name": matches_to_play[0]["black_bot_name"] if matches_to_play else "",
                "position_name": matches_to_play[0]["position_name"] if matches_to_play else "",
                "status": "waiting",
            },
        }
        self._update_leaderboard()

        self._task = asyncio.create_task(self._run_arena_loop())
        return self.active_session

    def _update_leaderboard(self) -> None:
        session = self.active_session
        if not session:
            return
        bots = session["bots"]
        scores = session["score_table"]
        elos = session["elo_ratings"]

        leaderboard = []
        for b in bots:
            bid = b["id"]
            sc = scores.get(bid, {"wins": 0, "draws": 0, "losses": 0, "points": 0.0, "timeouts": 0})
            played = sc["wins"] + sc["draws"] + sc["losses"]
            winrate = (sc["points"] / played * 100.0) if played > 0 else 0.0
            leaderboard.append({
                "bot_id": bid,
                "bot_name": b["name"],
                "elo": round(elos.get(bid, 1700.0), 1),
                "points": sc["points"],
                "wins": sc["wins"],
                "draws": sc["draws"],
                "losses": sc["losses"],
                "timeouts": sc["timeouts"],
                "games_played": played,
                "winrate": round(winrate, 1),
            })

        leaderboard.sort(key=lambda x: (x["points"], x["elo"]), reverse=True)
        session["leaderboard"] = leaderboard

    async def _run_arena_loop(self) -> None:
        session = self.active_session
        if not session:
            return

        matches = session["matches"]
        time_ms = max(5000, session["game_time_seconds"] * 1000)
        inc_ms = max(0, session["increment_seconds"] * 1000)

        try:
            for idx, match in enumerate(matches):
                if self._stop_requested:
                    break

                session["current_match"] = match
                match["status"] = "playing"
                await self.broadcast_fn({"type": "arena_update", "session": session})

                try:
                    res = await self._play_arena_game(match, time_ms, inc_ms)
                except Exception as m_exc:
                    import traceback
                    traceback.print_exc()
                    res = {
                        "result": "1/2-1/2",
                        "termination": f"error: {m_exc}",
                        "is_timeout": False,
                        "loser_id": None,
                        "moves_count": 0,
                    }
                match.update(res)
                match["status"] = "finished"

                w_id = match["white_bot_id"]
                b_id = match["black_bot_id"]
                result_str = res["result"]

                if result_str == "1-0":
                    session["score_table"][w_id]["wins"] += 1
                    session["score_table"][w_id]["points"] += 1.0
                    session["score_table"][b_id]["losses"] += 1
                    s_w, s_b = 1.0, 0.0
                elif result_str == "0-1":
                    session["score_table"][b_id]["wins"] += 1
                    session["score_table"][b_id]["points"] += 1.0
                    session["score_table"][w_id]["losses"] += 1
                    s_w, s_b = 0.0, 1.0
                else:
                    session["score_table"][w_id]["draws"] += 1
                    session["score_table"][w_id]["points"] += 0.5
                    session["score_table"][b_id]["draws"] += 1
                    session["score_table"][b_id]["points"] += 0.5
                    s_w, s_b = 0.5, 0.5

                if res.get("is_timeout"):
                    timeout_loser = res.get("loser_id")
                    if timeout_loser and timeout_loser in session["score_table"]:
                        session["score_table"][timeout_loser]["timeouts"] += 1

                # Standard FIDE Elo Update with K=32
                r_w = session["elo_ratings"][w_id]
                r_b = session["elo_ratings"][b_id]
                e_w = 1.0 / (1.0 + math.pow(10, (r_b - r_w) / 400.0))
                e_b = 1.0 - e_w
                k = 32.0
                session["elo_ratings"][w_id] = max(100.0, r_w + k * (s_w - e_w))
                session["elo_ratings"][b_id] = max(100.0, r_b + k * (s_b - e_b))

                session["completed_matches"] += 1
                self._update_leaderboard()
                await self.broadcast_fn({"type": "arena_update", "session": session})

            if not self._stop_requested:
                session["status"] = "finished"
                session["current_match"] = None
                await self.broadcast_fn({"type": "arena_update", "session": session})

        except asyncio.CancelledError:
            session["status"] = "stopped"
        except Exception as exc:
            session["status"] = "error"
            session["error"] = str(exc)
            await self.broadcast_fn({"type": "arena_update", "session": session})

    async def _play_arena_game(
        self,
        match: dict[str, Any],
        time_ms: int,
        inc_ms: int,
    ) -> dict[str, Any]:
        fen = match["fen"]
        chess960_id = match.get("chess960_id", 518)
        board = chess.Board(fen, chess960=True)
        # README Section 2.3: No increment allowed in competition
        clock = ChessClock(time_ms, increment_ms=inc_ms)

        w_proc = BotProcess(bot_dir=match["white_bot_dir"])
        b_proc = BotProcess(bot_dir=match["black_bot_dir"])

        moves_played: list[str] = []
        history_records: list[dict[str, Any]] = []
        result = "1/2-1/2"
        termination = "draw"
        is_timeout = False
        loser_id = None

        # Set initial live board state and broadcast
        initial_live_board = {
            "match_id": match["match_id"],
            "position_id": match.get("position_id"),
            "position_name": match.get("position_name", ""),
            "white_bot_id": match.get("white_bot_id"),
            "white_bot_name": match.get("white_bot_name", ""),
            "black_bot_id": match.get("black_bot_id"),
            "black_bot_name": match.get("black_bot_name", ""),
            "fen": board.fen(shredder=True),
            "white_ms": clock.value(chess.WHITE),
            "black_ms": clock.value(chess.BLACK),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "last_move": None,
            "last_move_san": None,
            "moves_count": 0,
            "history": [],
            "status": "playing",
        }
        if self.active_session:
            self.active_session["live_board"] = initial_live_board

        await self.broadcast_fn({
            "type": "arena_live_move",
            **initial_live_board,
        })

        ticker_stop = asyncio.Event()
        ticker_task: asyncio.Task[None] | None = None

        async def match_ticker():
            while not ticker_stop.is_set():
                await asyncio.sleep(0.08)
                if self._stop_requested:
                    break
                w_val = clock.value(chess.WHITE)
                b_val = clock.value(chess.BLACK)
                if self.active_session and "live_board" in self.active_session:
                    self.active_session["live_board"]["white_ms"] = w_val
                    self.active_session["live_board"]["black_ms"] = b_val
                await self.broadcast_fn({
                    "type": "arena_live_clock",
                    "match_id": match["match_id"],
                    "white_ms": w_val,
                    "black_ms": b_val,
                    "turn": "white" if board.turn == chess.WHITE else "black",
                })

        try:
            await asyncio.gather(w_proc.start(), b_proc.start())
            await asyncio.gather(
                w_proc.request(
                    {
                        "type": "init",
                        "color": chess.WHITE,
                        "chess960_id": chess960_id,
                        "initial_time_ms": time_ms,
                        "seed": secrets.randbits(63),
                        "bot_dir": match["white_bot_dir"],
                    },
                    timeout=max(10.0, time_ms / 1000.0),
                ),
                b_proc.request(
                    {
                        "type": "init",
                        "color": chess.BLACK,
                        "chess960_id": chess960_id,
                        "initial_time_ms": time_ms,
                        "seed": secrets.randbits(63),
                        "bot_dir": match["black_bot_dir"],
                    },
                    timeout=max(10.0, time_ms / 1000.0),
                ),
            )

            ticker_task = asyncio.create_task(match_ticker())
            max_plies = 300

            while len(moves_played) < max_plies:
                if self._stop_requested:
                    termination = "stopped"
                    break

                turn = board.turn
                clock.start(turn)
                remaining = clock.value(turn)
                opponent_remaining = clock.value(not turn)
                curr_proc = w_proc if turn == chess.WHITE else b_proc
                curr_bot_id = match["white_bot_id"] if turn == chess.WHITE else match["black_bot_id"]

                if remaining <= 0:
                    clock.stop()
                    termination = "timeout"
                    result = "0-1" if turn == chess.WHITE else "1-0"
                    is_timeout = True
                    loser_id = curr_bot_id
                    break

                t_req_start = time.monotonic_ns()
                try:
                    resp = await curr_proc.request(
                        {
                            "type": "move",
                            "chess960_id": chess960_id,
                            "moves": moves_played,
                            "fen": fen,
                            "time_remaining_ms": remaining,
                            "opponent_time_remaining_ms": opponent_remaining,
                        },
                        timeout=max(0.5, remaining / 1000.0 + 1.0),
                    )

                    # README Section 2.3 Rule:
                    # Every move MUST deduct at least 500ms (0.5s) from the clock
                    calc_ms = max(0, (time.monotonic_ns() - t_req_start) // 1_000_000)
                    delay_ms = max(0, 500 - calc_ms)
                    if delay_ms > 0:
                        rem_curr = clock.value(turn)
                        await asyncio.sleep(min(delay_ms, rem_curr) / 1000.0)

                    elapsed_ms = clock.stop()

                    if clock.remaining[turn] <= 0:
                        termination = "timeout"
                        result = "0-1" if turn == chess.WHITE else "1-0"
                        is_timeout = True
                        loser_id = curr_bot_id
                        break

                    move = move_for_token(board, str(resp.get("move", "")))
                except Exception as exc:
                    clock.stop()
                    termination = f"failure: {exc}"
                    result = "0-1" if turn == chess.WHITE else "1-0"
                    loser_id = curr_bot_id
                    if "timed out" in str(exc).lower():
                        is_timeout = True
                        termination = "timeout"
                    break

                san = board.san(move)
                token = board.uci(move, chess960=True)
                board.push(move)
                moves_played.append(token)
                history_records.append({
                    "ply": len(moves_played),
                    "san": san,
                    "uci": token,
                    "turn": "white" if turn == chess.WHITE else "black",
                    "elapsed_ms": elapsed_ms,
                })

                live_board_data = {
                    "match_id": match["match_id"],
                    "position_id": match.get("position_id"),
                    "position_name": match.get("position_name", ""),
                    "white_bot_id": match.get("white_bot_id"),
                    "white_bot_name": match.get("white_bot_name", ""),
                    "black_bot_id": match.get("black_bot_id"),
                    "black_bot_name": match.get("black_bot_name", ""),
                    "fen": board.fen(shredder=True),
                    "white_ms": clock.value(chess.WHITE),
                    "black_ms": clock.value(chess.BLACK),
                    "turn": "white" if board.turn == chess.WHITE else "black",
                    "last_move": token,
                    "last_move_san": san,
                    "moves_count": len(moves_played),
                    "history": history_records[-10:],
                    "status": "playing",
                }
                if self.active_session:
                    self.active_session["live_board"] = live_board_data

                # Broadcast live arena step
                await self.broadcast_fn({
                    "type": "arena_live_move",
                    **live_board_data,
                })

                if board.is_checkmate():
                    winner = not board.turn
                    result = "1-0" if winner == chess.WHITE else "0-1"
                    termination = "checkmate"
                    break
                elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                    result = "1/2-1/2"
                    termination = "draw"
                    break

        finally:
            ticker_stop.set()
            if ticker_task and not ticker_task.done():
                ticker_task.cancel()
                try:
                    await ticker_task
                except (asyncio.CancelledError, Exception):
                    pass
            await asyncio.gather(w_proc.stop(), b_proc.stop(), return_exceptions=True)

        if self.active_session and "live_board" in self.active_session:
            self.active_session["live_board"]["status"] = "finished"
            self.active_session["live_board"]["result"] = result
            self.active_session["live_board"]["termination"] = termination

        return {
            "result": result,
            "termination": termination,
            "moves_count": len(moves_played),
            "is_timeout": is_timeout,
            "loser_id": loser_id,
            "final_fen": board.fen(shredder=True),
        }
