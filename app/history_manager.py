from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path
import re
from typing import Any

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "history_matches"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _clean_slug(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip().replace(" ", "_"))
    return cleaned[:30] if cleaned else "bot"


def save_match_pgn(
    session_type: str,
    session_id: str,
    match_id: int | str,
    white_name: str,
    black_name: str,
    chess960_id: int,
    initial_fen: str,
    moves: list[str],
    result: str,
    termination: str,
    time_control: str = "60+1",
    move_details: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Saves a completed match to standard PGN format in the history_matches directory.
    Fully compatible with Chess960 specifications, standard chess viewers (ChessBase, Lichess, Scid),
    and the built-in Interactive Game Replayer.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        board = chess.Board(initial_fen, chess960=True)
    except Exception:
        board = chess.Board(chess960=True)

    game = chess.pgn.Game()
    game.setup(board)

    now = datetime.now()
    event_label = "Đấu Trường Nội Bộ" if session_type == "arena" else "Benchmark Chess960"

    game.headers["Event"] = f"{event_label}: {white_name} vs {black_name}"
    game.headers["Site"] = "ChessLab 960 System"
    game.headers["Date"] = now.strftime("%Y.%m.%d")
    game.headers["Time"] = now.strftime("%H:%M:%S")
    game.headers["Round"] = str(match_id)
    game.headers["White"] = str(white_name)
    game.headers["Black"] = str(black_name)
    game.headers["Result"] = str(result)
    game.headers["Variant"] = "Chess960"
    game.headers["SetUp"] = "1"
    game.headers["FEN"] = initial_fen
    game.headers["Chess960Id"] = str(chess960_id)
    game.headers["Termination"] = str(termination)
    game.headers["TimeControl"] = str(time_control)
    game.headers["SessionId"] = str(session_id)
    game.headers["Mode"] = str(session_type)

    if metadata:
        for k, v in metadata.items():
            if k not in game.headers and isinstance(v, (str, int, float)):
                game.headers[str(k)] = str(v)

    # Populate moves on mainline
    curr_node = game
    temp_board = chess.Board(initial_fen, chess960=True)

    for idx, mv_str in enumerate(moves):
        try:
            mv = None
            try:
                mv = temp_board.parse_uci(mv_str)
            except Exception:
                for legal_mv in temp_board.legal_moves:
                    if temp_board.uci(legal_mv, chess960=True) == mv_str:
                        mv = legal_mv
                        break

            if mv and mv in temp_board.legal_moves:
                curr_node = curr_node.add_variation(mv)
                if move_details and idx < len(move_details):
                    d = move_details[idx]
                    elapsed = d.get("elapsed_ms")
                    if elapsed is not None:
                        curr_node.comment = f"[{elapsed}ms]"
                temp_board.push(mv)
            else:
                break
        except Exception:
            break

    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_text = game.accept(exporter)

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    w_slug = _clean_slug(white_name)
    b_slug = _clean_slug(black_name)
    filename = f"{session_type}_{timestamp}_pos{chess960_id}_m{match_id}_{w_slug}_vs_{b_slug}.pgn"
    file_path = HISTORY_DIR / filename
    file_path.write_text(pgn_text, encoding="utf-8")

    return {
        "filename": filename,
        "path": str(file_path),
        "result": result,
        "termination": termination,
        "moves_count": len(moves),
        "created_at": now.isoformat(),
    }


def list_saved_matches(
    session_type: str | None = None,
    bot_filter: str | None = None,
    limit: int = 150,
) -> list[dict[str, Any]]:
    """
    Returns a list of saved matches from history_matches, sorted newest first.
    Reads PGN headers efficiently without parsing full game trees.
    """
    if not HISTORY_DIR.exists():
        return []

    files = list(HISTORY_DIR.glob("*.pgn"))
    # Sort files by modification time descending
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    matches: list[dict[str, Any]] = []

    for f in files:
        if len(matches) >= limit:
            break

        try:
            with open(f, encoding="utf-8", errors="replace") as pgn_file:
                headers = chess.pgn.read_headers(pgn_file)
                if not headers:
                    continue

                mode = headers.get("Mode", "arena" if f.name.startswith("arena") else "benchmark")
                if session_type and session_type != "all" and mode != session_type:
                    continue

                white = headers.get("White", "White")
                black = headers.get("Black", "Black")

                if bot_filter:
                    bf_lower = bot_filter.lower()
                    if bf_lower not in white.lower() and bf_lower not in black.lower():
                        continue

                stat = f.stat()
                mtime = stat.st_mtime
                created_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

                try:
                    chess960_id = int(headers.get("Chess960Id", "518"))
                except ValueError:
                    chess960_id = 518

                matches.append({
                    "filename": f.name,
                    "session_type": mode,
                    "event": headers.get("Event", f"{white} vs {black}"),
                    "date": headers.get("Date", ""),
                    "time": headers.get("Time", ""),
                    "created_at": created_str,
                    "mtime": mtime,
                    "round": headers.get("Round", "1"),
                    "white": white,
                    "black": black,
                    "result": headers.get("Result", "*"),
                    "termination": headers.get("Termination", "unknown"),
                    "chess960_id": chess960_id,
                    "initial_fen": headers.get("FEN", ""),
                    "time_control": headers.get("TimeControl", ""),
                    "session_id": headers.get("SessionId", ""),
                    "file_size": stat.st_size,
                })
        except Exception:
            continue

    return matches


def get_match_detail(filename: str) -> dict[str, Any]:
    """
    Reads a PGN file and returns full move-by-move replay information,
    including each ply's SAN notation, UCI move, board FEN, and elapsed time.
    """
    # Prevent path traversal
    safe_name = Path(filename).name
    file_path = HISTORY_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Tập tin ván đấu '{safe_name}' không tồn tại")

    content = file_path.read_text(encoding="utf-8", errors="replace")
    game = chess.pgn.read_game(io.StringIO(content))
    if not game:
        raise ValueError("Không thể phân tích dữ liệu PGN của ván đấu")

    headers = dict(game.headers)
    board = game.board()
    initial_fen = headers.get("FEN", board.fen(shredder=True))

    try:
        chess960_id = int(headers.get("Chess960Id", "518"))
    except ValueError:
        chess960_id = 518

    moves_data: list[dict[str, Any]] = []
    ply = 0

    for node in game.mainline():
        ply += 1
        move = node.move
        san = board.san(move)
        uci = board.uci(move, chess960=True)
        comment = node.comment or ""

        elapsed_ms: int | None = None
        if comment:
            m = re.search(r"\[(\d+)ms\]", comment)
            if m:
                elapsed_ms = int(m.group(1))

        board.push(move)

        moves_data.append({
            "ply": ply,
            "move_number": (ply + 1) // 2,
            "turn": "white" if ply % 2 != 0 else "black",
            "san": san,
            "uci": uci,
            "fen": board.fen(shredder=True),
            "elapsed_ms": elapsed_ms,
            "comment": comment,
        })

    return {
        "filename": safe_name,
        "headers": headers,
        "session_type": headers.get("Mode", "arena" if safe_name.startswith("arena") else "benchmark"),
        "event": headers.get("Event", ""),
        "date": headers.get("Date", ""),
        "time": headers.get("Time", ""),
        "white": headers.get("White", ""),
        "black": headers.get("Black", ""),
        "result": headers.get("Result", "*"),
        "termination": headers.get("Termination", ""),
        "time_control": headers.get("TimeControl", ""),
        "chess960_id": chess960_id,
        "initial_fen": initial_fen,
        "final_fen": board.fen(shredder=True),
        "total_plies": ply,
        "total_moves": (ply + 1) // 2,
        "moves": moves_data,
        "pgn_raw": content,
    }


def get_pgn_file_path(filename: str) -> Path:
    safe_name = Path(filename).name
    file_path = HISTORY_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Tập tin '{safe_name}' không tồn tại")
    return file_path


def delete_history_game(filename: str) -> bool:
    safe_name = Path(filename).name
    file_path = HISTORY_DIR / safe_name
    if file_path.exists() and file_path.is_file():
        file_path.unlink()
        return True
    return False


def clear_all_history() -> int:
    if not HISTORY_DIR.exists():
        return 0
    count = 0
    for f in HISTORY_DIR.glob("*.pgn"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count

