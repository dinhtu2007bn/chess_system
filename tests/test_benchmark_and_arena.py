import asyncio
import json
import pytest
import chess

from app.benchmark_manager import (
    calculate_optimization_metrics,
    load_positions,
    BenchmarkRunner,
    ArenaRunner,
)
from app.bot_manager import BotManager
from app.external_engine import ExternalEngineSparring, FallbackSparringBot


def test_positions_suite_structure() -> None:
    positions = load_positions()
    assert len(positions) == 40, f"Expected 40 positions, got {len(positions)}"

    openings = [p for p in positions if p["category"] == "opening"]
    midgames = [p for p in positions if p["category"] == "midgame"]
    endgames = [p for p in positions if p["category"] == "endgame"]

    assert len(openings) == 10, f"Expected 10 openings, got {len(openings)}"
    assert len(midgames) == 20, f"Expected 20 midgames, got {len(midgames)}"
    assert len(endgames) == 10, f"Expected 10 endgames, got {len(endgames)}"

    for pos in positions:
        assert "id" in pos
        assert "name" in pos
        assert "fen" in pos
        assert "recommended_time_ms" in pos
        board = chess.Board(pos["fen"], chess960=True)
        assert board.is_valid(), f"Position {pos['id']} has invalid FEN: {pos['fen']}"


def test_optimization_metrics_formula() -> None:
    # Scenario: 10 games, 6 wins, 2 draws, 2 losses (1 by timeout, 1 by checkmate)
    mock_games = [
        {"bot_outcome": "win", "is_timeout": False, "avg_move_time_ms": 500, "peak_move_time_ms": 900},
        {"bot_outcome": "win", "is_timeout": False, "avg_move_time_ms": 600, "peak_move_time_ms": 950},
        {"bot_outcome": "win", "is_timeout": False, "avg_move_time_ms": 550, "peak_move_time_ms": 800},
        {"bot_outcome": "win", "is_timeout": False, "avg_move_time_ms": 500, "peak_move_time_ms": 850},
        {"bot_outcome": "win", "is_timeout": False, "avg_move_time_ms": 700, "peak_move_time_ms": 1100},
        {"bot_outcome": "win", "is_timeout": False, "avg_move_time_ms": 650, "peak_move_time_ms": 1050},
        {"bot_outcome": "draw", "is_timeout": False, "avg_move_time_ms": 600, "peak_move_time_ms": 1000},
        {"bot_outcome": "draw", "is_timeout": False, "avg_move_time_ms": 550, "peak_move_time_ms": 900},
        {"bot_outcome": "loss", "is_timeout": True, "avg_move_time_ms": 1200, "peak_move_time_ms": 3500},
        {"bot_outcome": "loss", "is_timeout": False, "avg_move_time_ms": 600, "peak_move_time_ms": 950},
    ]

    metrics = calculate_optimization_metrics(mock_games)

    assert metrics["wins"] == 6
    assert metrics["draws"] == 2
    assert metrics["losses"] == 2
    assert metrics["timeout_losses"] == 1
    assert metrics["checkmate_losses"] == 1

    expected_winrate = ((6 + 0.5 * 2) / 10) * 100.0  # 70.0%
    assert abs(metrics["win_rate"] - expected_winrate) < 0.01

    expected_timeout_penalty = 1 / 10  # 0.1
    assert abs(metrics["timeout_penalty"] - expected_timeout_penalty) < 0.01

    expected_score = expected_winrate * (1.0 - expected_timeout_penalty) * metrics["time_efficiency_factor"]
    assert abs(metrics["optimization_score"] - expected_score) < 0.05
    assert len(metrics["verdict"]) > 20


def test_bot_manager_crud() -> None:
    bm = BotManager()
    bots = bm.list_bots()
    assert len(bots) >= 2

    default_bot = next(b for b in bots if b.get("is_default") or b["id"] in ("bot_V1", "student-default"))
    assert default_bot["is_valid"] is True

    # In-place rename
    orig_name = default_bot["name"]
    bot_id = default_bot["id"]
    renamed = bm.rename_bot(bot_id, "Renamed Test Bot")
    assert renamed["name"] == "Renamed Test Bot"

    # Restore name
    bm.rename_bot(bot_id, orig_name)


@pytest.mark.anyio
async def test_external_engine_sparring() -> None:
    engine = ExternalEngineSparring(elo=1600)
    await engine.start()
    try:
        board = chess.Board.from_chess960_pos(518)
        move = await engine.choose_move(board, time_limit_seconds=0.2)
        assert isinstance(move, chess.Move)
        assert move in board.legal_moves
    finally:
        await engine.stop()


@pytest.mark.anyio
async def test_benchmark_runner_single_game() -> None:
    bm = BotManager()
    events = []

    async def broadcast(payload):
        events.append(payload)

    runner = BenchmarkRunner(bm, broadcast)
    bots = bm.list_bots()
    target_bot = bots[0]

    # Benchmark on 1 position with 1 run for speed
    session = await runner.start_benchmark(
        bot_id=target_bot["id"],
        position_ids=["open-01"],
        opponent_type="stockfish",
        opponent_elo=1350,
        time_mode="global",
        game_time_seconds=2,
        increment_seconds=0,
        runs_per_position=1,
    )
    assert session["status"] == "running"

    # Wait for completion (fast 2-sec game)
    for _ in range(40):
        await asyncio.sleep(0.5)
        if runner.active_session and runner.active_session.get("status") in ("finished", "stopped"):
            break

    assert runner.active_session["games_completed"] >= 1
    metrics = runner.active_session["metrics"]
    assert "optimization_score" in metrics
    assert "verdict" in metrics


@pytest.mark.anyio
async def test_benchmark_runner_concurrent_multi_board() -> None:
    bm = BotManager()
    events = []

    async def broadcast(payload):
        events.append(payload)

    runner = BenchmarkRunner(bm, broadcast)
    bots = bm.list_bots()
    target_bot = bots[0]

    # Benchmark on 3 positions with concurrency=3
    session = await runner.start_benchmark(
        bot_id=target_bot["id"],
        position_ids=["open-01", "open-02", "open-03"],
        opponent_type="stockfish",
        opponent_elo=1350,
        time_mode="global",
        game_time_seconds=2,
        increment_seconds=0,
        runs_per_position=1,
        concurrency=3,
    )
    assert session["status"] == "running"
    assert session["concurrency"] == 3
    assert len(session["live_boards"]) == 3
    assert "0" in session["live_boards"]
    assert "1" in session["live_boards"]
    assert "2" in session["live_boards"]

    # Wait for completion (fast 2-sec game)
    for _ in range(60):
        await asyncio.sleep(0.5)
        if runner.active_session and runner.active_session.get("status") in ("finished", "stopped"):
            break

    assert runner.active_session["games_completed"] == 3
    assert len(runner.active_session["games"]) == 3
    assert any("slot_id" in ev for ev in events)

