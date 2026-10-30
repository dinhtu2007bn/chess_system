import asyncio
import pytest
import chess

from app.history_manager import (
    save_match_pgn,
    list_saved_matches,
    get_match_detail,
    delete_history_game,
    clear_all_history,
)
from app.benchmark_manager import ArenaRunner
from app.bot_manager import BotManager


def test_history_pgn_save_and_detail(tmp_path) -> None:
    board = chess.Board("bbqrnnkr/pppppppp/8/8/8/8/PPPPPPPP/BBQRNNKR w KQkq - 0 1", chess960=True)
    initial_fen = board.fen(shredder=True)

    moves = ["f1g3", "f8g6", "g3h5", "g8f8"]
    move_details = [
        {"elapsed_ms": 500},
        {"elapsed_ms": 510},
        {"elapsed_ms": 520},
        {"elapsed_ms": 505},
    ]

    saved = save_match_pgn(
        session_type="arena",
        session_id="arena-test-01",
        match_id=1,
        white_name="Bot V6",
        black_name="Bot V0",
        chess960_id=518,
        initial_fen=initial_fen,
        moves=moves,
        result="1-0",
        termination="checkmate",
        time_control="60+0",
        move_details=move_details,
    )

    assert "filename" in saved
    filename = saved["filename"]

    # List matches
    matches = list_saved_matches()
    match_entry = next((m for m in matches if m["filename"] == filename), None)
    assert match_entry is not None
    assert match_entry["white"] == "Bot V6"
    assert match_entry["black"] == "Bot V0"
    assert match_entry["result"] == "1-0"
    assert match_entry["chess960_id"] == 518

    # Detail
    detail = get_match_detail(filename)
    assert detail["white"] == "Bot V6"
    assert detail["black"] == "Bot V0"
    assert detail["result"] == "1-0"
    assert detail["total_plies"] == 4
    assert len(detail["moves"]) == 4

    # Check move 1
    m1 = detail["moves"][0]
    assert m1["ply"] == 1
    assert m1["turn"] == "white"
    assert m1["san"] == "Ng3"
    assert m1["uci"] == "f1g3"
    assert m1["elapsed_ms"] == 500
    assert m1["fen"] != ""

    # Clean up test match
    deleted = delete_history_game(filename)
    assert deleted is True


@pytest.mark.anyio
async def test_arena_runner_concurrency_setup() -> None:
    events = []

    async def mock_broadcast(payload):
        events.append(payload)

    bot_mgr = BotManager()
    bots = bot_mgr.list_bots()
    if len(bots) < 2:
        pytest.skip("Not enough bots for arena test")

    runner = ArenaRunner(bot_mgr, mock_broadcast)
    session = await runner.start_arena(
        bot_ids=[bots[0]["id"], bots[1]["id"]],
        position_ids=["pos-01"],
        game_time_seconds=60,
        increment_seconds=0,
        concurrency=2,
    )

    try:
        assert session["status"] == "running"
        assert session["concurrency"] == 2
        assert "live_boards" in session
        assert "0" in session["live_boards"]
        assert "1" in session["live_boards"]
    finally:
        await runner.stop()
