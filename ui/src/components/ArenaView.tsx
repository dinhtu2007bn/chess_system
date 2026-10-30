import React, { useEffect, useState } from "react";
import { Chessboard } from "react-chessboard";
import { ArenaSession, ChessPosition, LiveBoardState, ManagedBot } from "../types";

type Props = {
  bots: ManagedBot[];
  positions: ChessPosition[];
  arenaSessionEvent?: any;
  arenaLiveEvent?: any;
  onRefreshPositions?: () => void;
  onViewReplay?: (filename: string) => void;
};

function formatClock(milliseconds: number) {
  const total = Math.max(0, Math.floor(milliseconds));
  const minutes = Math.floor(total / 60000);
  const seconds = Math.floor((total % 60000) / 1000);
  const millis = total % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(Math.floor(millis / 100))}`;
}

export const ArenaView: React.FC<Props> = ({
  bots,
  positions,
  arenaSessionEvent,
  arenaLiveEvent,
  onRefreshPositions,
  onViewReplay,
}) => {
  const [selectedBotIds, setSelectedBotIds] = useState<Set<string>>(new Set());
  const [selectedPosIds, setSelectedPosIds] = useState<Set<string>>(new Set());
  const [posFilter, setPosFilter] = useState<"all" | "opening" | "midgame" | "endgame">("all");
  const [randomCount, setRandomCount] = useState<number>(5);

  // Preview position board (when idle or inspecting bank)
  const [previewPos, setPreviewPos] = useState<ChessPosition | null>(
    positions && positions.length > 0 ? positions[0] : null
  );

  // Time controls: Full Game Clock per player + Increment per move
  const [gameTimeSec, setGameTimeSec] = useState<number>(60); // Default: 60s per player
  const [incrementSec, setIncrementSec] = useState<number>(0);  // Default: +0s (README Rule: no increment)
  const [formatType, setFormatType] = useState<"round_robin" | "1v1">("round_robin");
  const [concurrency, setConcurrency] = useState<number>(1); // Multi-board concurrent matches

  const [session, setSession] = useState<ArenaSession | null>(null);
  const [liveBoard, setLiveBoard] = useState<LiveBoardState | null>(null);
  const [liveBoards, setLiveBoards] = useState<Record<string, LiveBoardState>>({});
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Initialize selected bots (select first 2 or all bots by default)
  useEffect(() => {
    if (bots.length >= 2 && selectedBotIds.size === 0) {
      setSelectedBotIds(new Set(bots.slice(0, Math.min(4, bots.length)).map((b) => b.id)));
    }
  }, [bots]);

  // Initialize selected positions (select 5 positions by default)
  useEffect(() => {
    if (positions.length > 0 && selectedPosIds.size === 0) {
      const initPos = new Set<string>();
      positions.slice(0, 5).forEach((p) => initPos.add(p.id));
      setSelectedPosIds(initPos);
      if (!previewPos) setPreviewPos(positions[0]);
    }
  }, [positions]);

  // Real-time Arena live move and live clock via WebSocket
  useEffect(() => {
    if (arenaLiveEvent) {
      const slotKey = String(arenaLiveEvent.slot_id ?? 0);

      // Dedicated Monotonic Clock Update from backend (every 80ms)
      if (arenaLiveEvent.type === "arena_live_clock") {
        setLiveBoards((prev) => {
          const b = prev[slotKey];
          if (!b) return prev;
          return {
            ...prev,
            [slotKey]: {
              ...b,
              white_ms: arenaLiveEvent.white_ms,
              black_ms: arenaLiveEvent.black_ms,
              turn: arenaLiveEvent.turn,
            },
          };
        });

        if (arenaLiveEvent.slot_id === 0 || !liveBoard) {
          setLiveBoard((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              white_ms: arenaLiveEvent.white_ms,
              black_ms: arenaLiveEvent.black_ms,
              turn: arenaLiveEvent.turn,
            };
          });
        }
        return;
      }

      // Live Move Update
      setLiveBoards((prev) => ({
        ...prev,
        [slotKey]: {
          ...(prev[slotKey] || {}),
          ...arenaLiveEvent,
          slot_id: arenaLiveEvent.slot_id ?? 0,
          fen: arenaLiveEvent.fen || prev[slotKey]?.fen || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
          history: arenaLiveEvent.history || prev[slotKey]?.history || [],
          white_ms: arenaLiveEvent.white_ms !== undefined ? arenaLiveEvent.white_ms : prev[slotKey]?.white_ms ?? gameTimeSec * 1000,
          black_ms: arenaLiveEvent.black_ms !== undefined ? arenaLiveEvent.black_ms : prev[slotKey]?.black_ms ?? gameTimeSec * 1000,
          turn: arenaLiveEvent.turn || prev[slotKey]?.turn || "white",
          status: arenaLiveEvent.status || prev[slotKey]?.status || "playing",
        },
      }));

      if (arenaLiveEvent.slot_id === 0 || !liveBoard) {
        setLiveBoard((prev) => ({
          ...(prev || {}),
          ...arenaLiveEvent,
          fen: arenaLiveEvent.fen || prev?.fen || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
          history: arenaLiveEvent.history || prev?.history || [],
          white_ms: arenaLiveEvent.white_ms !== undefined ? arenaLiveEvent.white_ms : prev?.white_ms ?? gameTimeSec * 1000,
          black_ms: arenaLiveEvent.black_ms !== undefined ? arenaLiveEvent.black_ms : prev?.black_ms ?? gameTimeSec * 1000,
          turn: arenaLiveEvent.turn || prev?.turn || "white",
          status: arenaLiveEvent.status || prev?.status || "playing",
        }));
      }
    }
  }, [arenaLiveEvent]);

  // Real-time Arena update via WebSocket
  useEffect(() => {
    if (arenaSessionEvent) {
      setSession(arenaSessionEvent);
      // Only overwrite board states if tournament is NOT currently running
      if (arenaSessionEvent.status !== "running") {
        if (arenaSessionEvent.live_boards) {
          setLiveBoards((prev) => ({
            ...prev,
            ...arenaSessionEvent.live_boards,
          }));
        }
        if (arenaSessionEvent.live_board) {
          setLiveBoard((prev) => ({
            ...(prev || {}),
            ...arenaSessionEvent.live_board,
          }));
        }
      }
    }
  }, [arenaSessionEvent]);

  // Poll Arena status as fallback
  useEffect(() => {
    let timer: any;
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/arena/status");
        if (res.ok) {
          const data = await res.json();
          if (data.session) {
            setSession(data.session);
            // Only overwrite board state if tournament is NOT currently running
            if (data.session.status !== "running" && data.session.live_board && !liveBoard) {
              setLiveBoard(data.session.live_board);
            }
          }
        }
      } catch (e) {
        // ignore
      }
    };
    fetchStatus();
    timer = setInterval(fetchStatus, 1500);
    return () => clearInterval(timer);
  }, []);

  const handleToggleBot = (botId: string) => {
    setSelectedBotIds((prev) => {
      const next = new Set(prev);
      if (next.has(botId)) next.delete(botId);
      else next.add(botId);
      return next;
    });
  };

  const handleSelectAllBots = () => {
    setSelectedBotIds(new Set(bots.map((b) => b.id)));
  };

  const handleDeselectAllBots = () => {
    setSelectedBotIds(new Set());
  };

  const handleTogglePos = (posId: string) => {
    setSelectedPosIds((prev) => {
      const next = new Set(prev);
      if (next.has(posId)) next.delete(posId);
      else next.add(posId);
      return next;
    });
  };

  const handleSelectAllPositions = () => {
    setSelectedPosIds(new Set(positions.map((p) => p.id)));
  };

  const handleDeselectAllPositions = () => {
    setSelectedPosIds(new Set());
  };

  const handlePickRandomPositions = () => {
    if (positions.length === 0) return;
    const shuffled = [...positions].sort(() => 0.5 - Math.random());
    const count = Math.min(randomCount, positions.length);
    const next = new Set<string>();
    shuffled.slice(0, count).forEach((p) => next.add(p.id));
    setSelectedPosIds(next);
    if (shuffled.length > 0) setPreviewPos(shuffled[0]);
  };

  const handleGenerateOpening = async () => {
    try {
      const res = await fetch("/api/positions/generate-opening", { method: "POST" });
      if (res.ok) {
        const newPos = await res.json();
        if (onRefreshPositions) onRefreshPositions();
        setSelectedPosIds((prev) => new Set(prev).add(newPos.id));
        setPreviewPos(newPos);
      }
    } catch (e) {
      // ignore
    }
  };

  const handleStartArena = async () => {
    if (selectedBotIds.size < 2) {
      setErrorMsg("Vui lòng chọn tối thiểu 2 bot để tổ chức đấu trường");
      return;
    }
    if (selectedPosIds.size === 0) {
      setErrorMsg("Vui lòng chọn ít nhất 1 thế cờ để thi đấu");
      return;
    }

    setLoading(true);
    setErrorMsg(null);

    const payload = {
      bot_ids: Array.from(selectedBotIds),
      position_ids: Array.from(selectedPosIds),
      game_time_seconds: gameTimeSec,
      increment_seconds: incrementSec,
      format_type: formatType,
      concurrency: concurrency,
    };

    try {
      const res = await fetch("/api/arena/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Không thể khởi động đấu trường");
      }
      const data = await res.json();
      setSession(data);
      if (data.live_board) {
        setLiveBoard(data.live_board);
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Lỗi khi bắt đầu đấu trường");
    } finally {
      setLoading(false);
    }
  };

  const handleStopArena = async () => {
    try {
      await fetch("/api/arena/stop", { method: "POST" });
    } catch (e) {
      // ignore
    }
  };

  const filteredPositions = positions.filter((p) => {
    if (posFilter === "all") return true;
    return p.category === posFilter;
  });

  const isRunning = session?.status === "running";

  // Calculate estimated match count
  const nBots = selectedBotIds.size;
  const nPositions = selectedPosIds.size;
  const estimatedMatches =
    formatType === "round_robin"
      ? nBots >= 2
        ? nBots * (nBots - 1) * nPositions
        : 0
      : nBots >= 2
      ? 2 * nPositions
      : 0;

  // Active match info for live board
  const currMatch = session?.current_match;
  const whiteBotName = liveBoard?.white_bot_name || currMatch?.white_bot_name || "White Bot";
  const blackBotName = liveBoard?.black_bot_name || currMatch?.black_bot_name || "Black Bot";
  const posName = liveBoard?.position_name || currMatch?.position_name || (previewPos ? previewPos.name : "Chess960");
  const isTurnWhite = liveBoard?.turn === "white";
  const isTurnBlack = liveBoard?.turn === "black";
  const activeBoardFen =
    liveBoard?.fen || (currMatch ? currMatch.fen : previewPos ? previewPos.fen : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  return (
    <div className="arena-container">
      {/* SECTION HEADER */}
      <div className="section-header">
        <div>
          <h2>Đấu Trường Tự Động Nội Bộ (Bot-vs-Bot Self-Play Arena)</h2>
          <p className="section-desc">
            Chọn từ 2 bot trở lên và các thế cờ Chess960 mong muốn để thi đấu trực tiếp. Quan sát trực tiếp từng nước cờ thực tế trên bàn cờ với đồng hồ bấm giờ song hành và cập nhật hệ số Elo nội bộ FIDE.
          </p>
        </div>
        {isRunning && (
          <button className="btn-danger" onClick={handleStopArena}>
            ⏹ Dừng Đấu Trường
          </button>
        )}
      </div>

      {errorMsg && <div className="alert-box alert-danger">{errorMsg}</div>}

      {/* ========================================================================= */}
      {/* 1. RUNNING TOURNAMENT VIEW: LIVE CHESSBOARD + LEADERBOARD & STATS         */}
      {/* ========================================================================= */}
      {isRunning ? (
        <div className="arena-running-dashboard">
          {/* TOURNAMENT PROGRESS TOP BANNER */}
          <div className="arena-top-banner">
            <div className="arena-banner-header">
              <div className="arena-title-wrap">
                <span className="live-badge">🔴 Trận Đấu Trực Tiếp</span>
                <strong>
                  {whiteBotName} (Trắng) vs {blackBotName} (Đen)
                </strong>
                <span className="text-muted" style={{ fontSize: "0.86rem" }}>
                  • {posName}
                </span>
              </div>
              <div className="flex-center" style={{ gap: 12 }}>
                <span style={{ fontSize: "0.85rem" }}>
                  Tiến độ: <strong>{session?.completed_matches || 0}</strong> / {session?.total_matches || 0} trận
                </span>
                <button className="btn-danger btn-sm" onClick={handleStopArena}>
                  ⏹ Dừng
                </button>
              </div>
            </div>

            {/* PROGRESS BAR */}
            <div className="progress-bar-wrap" style={{ marginTop: 8 }}>
              <div
                className="progress-bar-fill"
                style={{
                  width: `${session?.total_matches ? ((session.completed_matches || 0) / session.total_matches) * 100 : 0}%`,
                }}
              />
            </div>
          </div>

          {/* CONCURRENCY AWARE ARENA RUNNING VIEW */}
          {(session?.concurrency || concurrency) > 1 ? (
            <div>
              {/* MULTI-BOARD GRID (WHEN CONCURRENCY > 1) */}
              <div className="multi-board-grid" style={{ marginTop: 14 }}>
                {Array.from({ length: session?.concurrency || concurrency }).map((_, slotId) => {
                  const b = liveBoards[String(slotId)];
                  const boardFen =
                    b?.fen || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
                  const isWhiteTurn = b?.turn === "white";
                  const isBlackTurn = b?.turn === "black";

                  return (
                    <div
                      key={slotId}
                      className={`mini-board-card ${b?.status === "playing" ? "active-slot" : ""}`}
                    >
                      <div className="mini-card-header">
                        <div className="flex-center" style={{ gap: 6 }}>
                          <span className="slot-badge">Bàn #{slotId + 1}</span>
                          {b?.match_id && <span className="mini-run-tag">Trận #{b.match_id}</span>}
                        </div>
                        <span className="mini-pos-info" title={b?.position_name}>
                          {b?.position_name || "Thế cờ"}
                        </span>
                      </div>

                      <div className={`mini-clock ${isBlackTurn && b?.status === "playing" ? "mini-clock-active" : ""}`}>
                        <span className="mini-clock-player">
                          ⚫ {b?.black_bot_name || "Đen"}
                          {isBlackTurn && b?.status === "playing" && <span className="pulse-dot"> •</span>}
                        </span>
                        <strong className="mini-clock-val">
                          {formatClock(b?.black_ms ?? gameTimeSec * 1000)}
                        </strong>
                      </div>

                      <div className="mini-board-wrapper">
                        <Chessboard
                          position={boardFen}
                          boardOrientation="white"
                          arePiecesDraggable={false}
                        />
                      </div>

                      <div className={`mini-clock ${isWhiteTurn && b?.status === "playing" ? "mini-clock-active" : ""}`}>
                        <span className="mini-clock-player">
                          ⚪ {b?.white_bot_name || "Trắng"}
                          {isWhiteTurn && b?.status === "playing" && <span className="pulse-dot"> •</span>}
                        </span>
                        <strong className="mini-clock-val">
                          {formatClock(b?.white_ms ?? gameTimeSec * 1000)}
                        </strong>
                      </div>

                      <div className="mini-card-footer">
                        <div className="mini-last-move">
                          <span>Nước: </span>
                          <strong>
                            {b?.last_move_san ? `${b.last_move_san} (${b.last_move})` : "Chưa có"}
                          </strong>
                        </div>
                        <div className="mini-moves-count">
                          {b?.moves_count ? `${b.moves_count} nước` : "0 nước"}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* BELOW MULTI-BOARD: LEADERBOARD & RECENT MATCHES */}
              <div className="arena-bottom-two-col" style={{ marginTop: 16 }}>
                {/* LEADERBOARD TABLE */}
                {session?.leaderboard && session.leaderboard.length > 0 && (
                  <div className="panel bench-result-compact">
                    <h3 className="m-0 mb-8">🏆 Bảng Xếp Hạng & Hệ Số Elo Nội Bộ</h3>
                    <div className="scrollable-table-box" style={{ maxHeight: 240 }}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Hạng</th>
                            <th>Tên Bot</th>
                            <th>Hệ Số Elo</th>
                            <th>Điểm Số</th>
                            <th>Thắng</th>
                            <th>Hòa</th>
                            <th>Thua</th>
                            <th>Timeout</th>
                            <th>Tỷ Lệ Thắng</th>
                          </tr>
                        </thead>
                        <tbody>
                          {session.leaderboard.map((row, idx) => (
                            <tr key={row.bot_id} className={idx === 0 ? "leader-row" : ""}>
                              <td>
                                <strong className="rank-num">
                                  {idx === 0 ? "🥇 1" : idx === 1 ? "🥈 2" : idx === 2 ? "🥉 3" : idx + 1}
                                </strong>
                              </td>
                              <td><strong>{row.bot_name}</strong></td>
                              <td><strong className="elo-badge">{row.elo}</strong></td>
                              <td><strong className="points-text">{row.points} pts</strong></td>
                              <td className="text-success">{row.wins}</td>
                              <td>{row.draws}</td>
                              <td className="text-danger">{row.losses}</td>
                              <td>
                                {row.timeouts > 0 ? (
                                  <span className="text-danger">⚠️ {row.timeouts}</span>
                                ) : (
                                  <span className="text-muted">0</span>
                                )}
                              </td>
                              <td>{row.winrate}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* COMPLETED MATCHES TABLE IN MULTI-BOARD VIEW */}
                {session?.matches && session.matches.filter((m) => m.status === "finished").length > 0 && (
                  <div className="panel bench-result-compact">
                    <div className="flex-between mb-8">
                      <h3 className="m-0">
                        Lịch Sử Trận Đã Đấu ({session.matches.filter((m) => m.status === "finished").length})
                      </h3>
                      {onViewReplay && (
                        <button
                          className="btn-secondary btn-sm"
                          onClick={() => onViewReplay("")}
                        >
                          📜 Xem Biên Bản Tất Cả
                        </button>
                      )}
                    </div>
                    <div className="scrollable-table-box" style={{ maxHeight: 240 }}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Trận</th>
                            <th>Trắng</th>
                            <th>Đen</th>
                            <th>Thế Cờ</th>
                            <th>Tỷ Số</th>
                            <th>Kết Thúc</th>
                            <th>Biên Bản</th>
                          </tr>
                        </thead>
                        <tbody>
                          {session.matches
                            .filter((m) => m.status === "finished")
                            .map((m) => (
                              <tr key={m.match_id}>
                                <td>#{m.match_id}</td>
                                <td>{m.white_bot_name}</td>
                                <td>{m.black_bot_name}</td>
                                <td>{m.position_name}</td>
                                <td><span className="outcome-pill">{m.result}</span></td>
                                <td>
                                  {m.is_timeout ? (
                                    <strong className="text-danger">⚠️ Timeout</strong>
                                  ) : (
                                    m.termination
                                  )}
                                </td>
                                <td>
                                  {onViewReplay && (
                                    <button
                                      className="btn-secondary btn-sm"
                                      onClick={() => onViewReplay("")}
                                    >
                                      👁️ Xem
                                    </button>
                                  )}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* SINGLE BOARD CLASSIC VIEW (WHEN CONCURRENCY === 1) */
            <div className="benchmark-two-col" style={{ marginTop: 14 }}>
              {/* LEFT COLUMN: LIVE CHESSBOARD WITH REAL-TIME CLOCKS */}
              <div className="bench-left-panel panel">
                <div className="live-board-header">
                  <span className="live-badge">♟️ Bàn Cờ Thi Đấu Thực Tế</span>
                  <strong className="live-pos-title" title={posName}>
                    {posName}
                  </strong>
                </div>

                {/* BLACK BOT CLOCK */}
                <div className={`bench-clock ${isTurnBlack ? "clock-running" : ""}`}>
                  <div className="clock-user-info">
                    <span>⚫</span>
                    <strong>{blackBotName}</strong>
                    {isTurnBlack && <span className="pulse-dot">Đang nghĩ...</span>}
                  </div>
                  <strong>{formatClock(liveBoard?.black_ms ?? gameTimeSec * 1000)}</strong>
                </div>

                {/* CHESSBOARD */}
                <div className="bench-chessboard-box">
                  <Chessboard
                    position={activeBoardFen}
                    boardOrientation="white"
                    arePiecesDraggable={false}
                  />
                </div>

                {/* WHITE BOT CLOCK */}
                <div className={`bench-clock ${isTurnWhite ? "clock-running" : ""}`}>
                  <div className="clock-user-info">
                    <span>⚪</span>
                    <strong>{whiteBotName}</strong>
                    {isTurnWhite && <span className="pulse-dot">Đang nghĩ...</span>}
                  </div>
                  <strong>{formatClock(liveBoard?.white_ms ?? gameTimeSec * 1000)}</strong>
                </div>

                {/* MOVE TICKER */}
                <div className="live-moves-ticker">
                  <span className="ticker-label">Nước đi vừa thực hiện:</span>
                  <strong className="ticker-val">
                    {liveBoard?.last_move_san ? `${liveBoard.last_move_san} (${liveBoard.last_move})` : "Bắt đầu"}
                  </strong>
                  <span className="ticker-count">
                    {liveBoard?.moves_count ? `${liveBoard.moves_count} nước` : "0 nước"}
                  </span>
                </div>

                {/* RECENT MOVES CHIPS */}
                {liveBoard?.history && liveBoard.history.length > 0 && (
                  <div className="scrollable-table-box" style={{ maxHeight: 90, padding: 6, fontSize: "0.78rem" }}>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {liveBoard.history.map((h, idx) => (
                        <span
                          key={idx}
                          className={`status-pill ${h.turn === "white" ? "pill-running" : ""}`}
                          style={{ fontSize: "0.72rem", padding: "2px 6px" }}
                        >
                          {h.turn === "white" ? "⚪" : "⚫"} {h.san} ({h.elapsed_ms ?? 0}ms)
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* RIGHT COLUMN: ACTIVE LEADERBOARD & RECENT MATCHES */}
              <div className="bench-right-panel">
              {/* CURRENT MATCH BANNER */}
              {currMatch && (
                <div className="active-match-banner" style={{ margin: 0 }}>
                  <div className="match-title">Đang Thi Đấu Ván #{currMatch.match_id}</div>
                  <div className="match-participants">
                    <span className="player white-player">⚪ {currMatch.white_bot_name}</span>
                    <span className="vs-badge">VS</span>
                    <span className="player black-player">⚫ {currMatch.black_bot_name}</span>
                  </div>
                  <div className="match-pos">Thế cờ: {currMatch.position_name}</div>
                </div>
              )}

              {/* LEADERBOARD TABLE */}
              {session?.leaderboard && session.leaderboard.length > 0 && (
                <div className="panel bench-result-compact">
                  <h3 className="m-0 mb-8">🏆 Bảng Xếp Hạng & Hệ Số Elo Nội Bộ (Cập nhật thời gian thực)</h3>
                  <div className="scrollable-table-box" style={{ maxHeight: 220 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Hạng</th>
                          <th>Tên Bot</th>
                          <th>Hệ Số Elo</th>
                          <th>Điểm Số</th>
                          <th>Thắng</th>
                          <th>Hòa</th>
                          <th>Thua</th>
                          <th>Timeout</th>
                          <th>Tỷ Lệ Thắng</th>
                        </tr>
                      </thead>
                      <tbody>
                        {session.leaderboard.map((row, idx) => (
                          <tr key={row.bot_id} className={idx === 0 ? "leader-row" : ""}>
                            <td>
                              <strong className="rank-num">
                                {idx === 0 ? "🥇 1" : idx === 1 ? "🥈 2" : idx === 2 ? "🥉 3" : idx + 1}
                              </strong>
                            </td>
                            <td><strong>{row.bot_name}</strong></td>
                            <td><strong className="elo-badge">{row.elo}</strong></td>
                            <td><strong className="points-text">{row.points} pts</strong></td>
                            <td className="text-success">{row.wins}</td>
                            <td>{row.draws}</td>
                            <td className="text-danger">{row.losses}</td>
                            <td>
                              {row.timeouts > 0 ? (
                                <span className="text-danger">⚠️ {row.timeouts}</span>
                              ) : (
                                <span className="text-muted">0</span>
                              )}
                            </td>
                            <td>{row.winrate}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* COMPLETED MATCHES TABLE */}
              {session?.matches && session.matches.filter((m) => m.status === "finished").length > 0 && (
                <div className="panel bench-result-compact">
                  <div className="flex-between mb-8">
                    <h3 className="m-0">
                      Lịch Sử Trận Đã Đấu ({session.matches.filter((m) => m.status === "finished").length})
                    </h3>
                  </div>
                  <div className="scrollable-table-box" style={{ maxHeight: 180 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Trận</th>
                          <th>Trắng (White)</th>
                          <th>Đen (Black)</th>
                          <th>Thế Cờ</th>
                          <th>Tỷ Số</th>
                          <th>Kết Thúc</th>
                          <th>Biên Bản</th>
                        </tr>
                      </thead>
                      <tbody>
                        {session.matches
                          .filter((m) => m.status === "finished")
                          .map((m) => (
                            <tr key={m.match_id}>
                              <td>#{m.match_id}</td>
                              <td>{m.white_bot_name}</td>
                              <td>{m.black_bot_name}</td>
                              <td>{m.position_name}</td>
                              <td><span className="outcome-pill">{m.result}</span></td>
                              <td>
                                {m.is_timeout ? (
                                  <strong className="text-danger">⚠️ Timeout</strong>
                                ) : (
                                  m.termination
                                )}
                              </td>
                              <td>
                                {onViewReplay && (
                                  <button
                                    className="btn-secondary btn-sm"
                                    onClick={() => onViewReplay("")}
                                  >
                                    👁️ Xem
                                  </button>
                                )}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        </div>
      ) : (
        /* ========================================================================= */
        /* 2. SETUP & CONFIGURATION VIEW (TWO-COLUMN SETUP WITH PREVIEW BOARD)      */
        /* ========================================================================= */
        <div>
          {/* PREVIOUS COMPLETED TOURNAMENT RESULTS (IF ANY) */}
          {session && (session.status === "finished" || session.status === "stopped") && (
            <div className="panel session-result-panel" style={{ marginBottom: 16 }}>
              <div className="session-header-row">
                <div>
                  <h3 className="m-0">Kết Quả Đấu Trường Vừa Diễn Ra</h3>
                  <small className="text-muted">
                    Hoàn tất {session.completed_matches} / {session.total_matches} trận đấu
                  </small>
                </div>
                <span className={`status-pill pill-${session.status}`}>
                  {session.status === "finished" ? "🏆 Đã hoàn tất giải đấu" : "⏹ Đã dừng"}
                </span>
              </div>

              {/* LEADERBOARD TABLE */}
              {session.leaderboard.length > 0 && (
                <div className="leaderboard-section">
                  <h4 style={{ margin: "10px 0 6px 0" }}>🏆 Bảng Xếp Hạng & Hệ Số Elo Nội Bộ</h4>
                  <div className="scrollable-table-box" style={{ maxHeight: 200 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Hạng</th>
                          <th>Tên Bot</th>
                          <th>Hệ Số Elo</th>
                          <th>Điểm Số</th>
                          <th>Thắng</th>
                          <th>Hòa</th>
                          <th>Thua</th>
                          <th>Timeout</th>
                          <th>Tỷ Lệ Thắng</th>
                        </tr>
                      </thead>
                      <tbody>
                        {session.leaderboard.map((row, idx) => (
                          <tr key={row.bot_id} className={idx === 0 ? "leader-row" : ""}>
                            <td>
                              <strong className="rank-num">
                                {idx === 0 ? "🥇 1" : idx === 1 ? "🥈 2" : idx === 2 ? "🥉 3" : idx + 1}
                              </strong>
                            </td>
                            <td><strong>{row.bot_name}</strong></td>
                            <td><strong className="elo-badge">{row.elo}</strong></td>
                            <td><strong className="points-text">{row.points} pts</strong></td>
                            <td className="text-success">{row.wins}</td>
                            <td>{row.draws}</td>
                            <td className="text-danger">{row.losses}</td>
                            <td>
                              {row.timeouts > 0 ? (
                                <span className="text-danger">⚠️ {row.timeouts}</span>
                              ) : (
                                <span className="text-muted">0</span>
                              )}
                            </td>
                            <td>{row.winrate}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* MATCH HISTORY LOG */}
              {session.matches.filter((m) => m.status === "finished").length > 0 && (
                <div className="games-table-wrap" style={{ marginTop: 12 }}>
                  <h4 style={{ margin: "0 0 6px 0" }}>Lịch Sử Các Trận Đã Đấu ({session.matches.filter((m) => m.status === "finished").length})</h4>
                  <div className="scrollable-table-box" style={{ maxHeight: 160 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Trận</th>
                          <th>Trắng (White)</th>
                          <th>Đen (Black)</th>
                          <th>Thế Cờ</th>
                          <th>Tỷ Số</th>
                          <th>Chi Tiết Kết Thúc</th>
                        </tr>
                      </thead>
                      <tbody>
                        {session.matches
                          .filter((m) => m.status === "finished")
                          .map((m) => (
                            <tr key={m.match_id}>
                              <td>#{m.match_id}</td>
                              <td>{m.white_bot_name}</td>
                              <td>{m.black_bot_name}</td>
                              <td>{m.position_name}</td>
                              <td><span className="outcome-pill">{m.result}</span></td>
                              <td>
                                {m.is_timeout ? (
                                  <strong className="text-danger">⚠️ Thua do Tràn Giờ</strong>
                                ) : (
                                  m.termination
                                )}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TWO-COLUMN SETUP DASHBOARD */}
          <div className="benchmark-two-col">
            {/* LEFT COLUMN: PREVIEW CHESSBOARD */}
            <div className="bench-left-panel panel">
              <div className="live-board-header">
                <span className="live-badge">👁 Xem Trước Thế Cờ Thi Đấu</span>
                <strong className="live-pos-title" title={previewPos?.name}>
                  {previewPos ? previewPos.name : "Thế cờ Chess960"}
                </strong>
              </div>

              {/* CHESSBOARD COMPONENT */}
              <div className="bench-chessboard-box">
                <Chessboard
                  position={previewPos?.fen || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
                  boardOrientation="white"
                  arePiecesDraggable={false}
                />
              </div>

              {/* POSITION DESCRIPTION BOX */}
              {previewPos && (
                <div className="preview-desc-box">
                  <span className="cat-pill cat-preview">{previewPos.category.toUpperCase()}</span>
                  <p>{previewPos.description}</p>
                </div>
              )}
            </div>

            {/* RIGHT COLUMN: CONFIGURATION & POSITION SELECTION */}
            <div className="bench-right-panel">
              {/* CARD 1: SELECT BOTS */}
              <div className="panel bench-config-compact">
                <div className="flex-between mb-8">
                  <h3 className="m-0">1. Chọn Các Bot Tham Gia (Tối thiểu 2 bot)</h3>
                  <div style={{ display: "flex", gap: 6 }}>
                    <button className="btn-sm btn-outline" onClick={handleSelectAllBots}>Chọn Tất Cả</button>
                    <button className="btn-sm btn-outline" onClick={handleDeselectAllBots}>Bỏ Hết</button>
                  </div>
                </div>
                <div className="bot-selector-list" style={{ maxHeight: 150, overflowY: "auto" }}>
                  {bots.map((b) => {
                    const isChecked = selectedBotIds.has(b.id);
                    return (
                      <label
                        key={b.id}
                        className={`bot-checkbox-card ${isChecked ? "card-selected" : ""}`}
                        style={{ padding: "6px 10px", margin: "2px 0" }}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          disabled={isRunning}
                          onChange={() => handleToggleBot(b.id)}
                        />
                        <div className="bot-card-info">
                          <strong>{b.name}</strong>
                          <span className="bot-id-text" style={{ fontSize: "0.75rem" }}>ID: {b.id}</span>
                        </div>
                        {b.is_default && <span className="badge badge-default">Mặc định</span>}
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* CARD 2: TIME CONTROLS & FORMAT */}
              <div className="panel bench-config-compact">
                <h3 className="m-0 mb-8">2. Thể Thức & Cài Đặt Thời Gian Ván Đấu</h3>
                <div className="config-row-flex">
                  <div className="form-group flex-1">
                    <label>Tổng Thời Gian Ván Đấu (Mỗi Bên):</label>
                    <select
                      value={gameTimeSec}
                      disabled={isRunning}
                      onChange={(e) => setGameTimeSec(Number(e.target.value))}
                    >
                      <option value={30}>30 Giây / bên (Bullet cực nhanh)</option>
                      <option value={60}>60 Giây / bên (1 phút - Chuẩn test)</option>
                      <option value={120}>120 Giây / bên (2 phút Blitz)</option>
                      <option value={180}>180 Giây / bên (3 phút Blitz)</option>
                      <option value={300}>300 Giây / bên (5 phút)</option>
                    </select>
                  </div>

                  <div className="form-group flex-1">
                    <label>Cộng Thêm / Nước Đi (Increment):</label>
                    <select
                      value={incrementSec}
                      disabled={isRunning}
                      onChange={(e) => setIncrementSec(Number(e.target.value))}
                    >
                      <option value={0}>+0s (Chuẩn BTL: Không cộng thời gian)</option>
                      <option value={1}>+1s / nước đi</option>
                      <option value={2}>+2s / nước đi</option>
                    </select>
                  </div>

                  <div className="form-group flex-1">
                    <label>Thể Thức Thi Đấu:</label>
                    <select
                      value={formatType}
                      disabled={isRunning}
                      onChange={(e) => setFormatType(e.target.value as any)}
                    >
                      <option value="round_robin">Vòng Tròn (Round-Robin 2 lượt W/B)</option>
                      <option value="1v1">Đấu Tay Đôi (1v1 Match)</option>
                    </select>
                  </div>

                  <div className="form-group flex-1">
                    <label>⚡ Bàn Cờ Song Song:</label>
                    <select
                      value={concurrency}
                      disabled={isRunning}
                      onChange={(e) => setConcurrency(Number(e.target.value))}
                    >
                      <option value={1}>1 Bàn Cờ (Tuần tự - Mặc định)</option>
                      <option value={2}>2 Bàn Cờ Song Song</option>
                      <option value={3}>3 Bàn Cờ Song Song</option>
                      <option value={4}>4 Bàn Cờ Song Song</option>
                    </select>
                  </div>
                </div>

                {/* ACTION & SUMMARY ROW */}
                <div className="flex-between bench-start-row">
                  <div className="bench-plan-info">
                    Đã chọn: <strong>{selectedBotIds.size} bot</strong> × <strong>{selectedPosIds.size} thế cờ</strong> ={" "}
                    <strong className="text-highlight">{estimatedMatches} trận đấu</strong>{" "}
                    (Chạy song song: <strong>{concurrency} bàn cờ đồng thời</strong>)
                  </div>
                  <button
                    className="btn-primary btn-large"
                    disabled={isRunning || loading || selectedBotIds.size < 2 || selectedPosIds.size === 0}
                    onClick={handleStartArena}
                  >
                    {loading
                      ? "⏳ Đang Khởi Động..."
                      : selectedBotIds.size < 2
                      ? "⚠️ Chọn tối thiểu 2 bot"
                      : selectedPosIds.size === 0
                      ? "⚠️ Hãy chọn ít nhất 1 thế cờ"
                      : `⚔️ Bắt Đầu Đấu Trường (${estimatedMatches} Trận)`}
                  </button>
                </div>
              </div>

              {/* CARD 3: 40-POSITION BANK SELECTOR */}
              <div className="panel bench-positions-box">
                <div className="positions-header-compact">
                  <div>
                    <h3 className="m-0">3. Chọn Thế Cờ Thi Đấu (Ngân Hàng 40 Thế Cờ)</h3>
                    <small className="text-muted">Nhấp vào dòng bất kỳ để xem trước bàn cờ ở cột bên trái</small>
                  </div>
                  <div className="positions-actions-compact">
                    <button className="btn-sm btn-outline" onClick={handleSelectAllPositions}>Chọn Tất Cả</button>
                    <button className="btn-sm btn-outline" onClick={handleDeselectAllPositions}>Bỏ Hết</button>
                    <div className="quick-random-group">
                      <input
                        type="number"
                        min="1"
                        max="40"
                        value={randomCount}
                        onChange={(e) => setRandomCount(Number(e.target.value))}
                        className="input-inline-num"
                      />
                      <button className="btn-sm btn-secondary" onClick={handlePickRandomPositions}>
                        Chọn N Ngẫu Nhiên
                      </button>
                    </div>
                    <button className="btn-sm btn-primary" onClick={handleGenerateOpening}>
                      + Sinh 960 Mới
                    </button>
                  </div>
                </div>

                {/* FILTER TABS */}
                <div className="subtabs-compact">
                  <button
                    className={`subtab-btn ${posFilter === "all" ? "active" : ""}`}
                    onClick={() => setPosFilter("all")}
                  >
                    Tất cả ({positions.length})
                  </button>
                  <button
                    className={`subtab-btn ${posFilter === "opening" ? "active" : ""}`}
                    onClick={() => setPosFilter("opening")}
                  >
                    Khai Cuộc ({positions.filter((p) => p.category === "opening").length})
                  </button>
                  <button
                    className={`subtab-btn ${posFilter === "midgame" ? "active" : ""}`}
                    onClick={() => setPosFilter("midgame")}
                  >
                    Trung Cuộc ({positions.filter((p) => p.category === "midgame").length})
                  </button>
                  <button
                    className={`subtab-btn ${posFilter === "endgame" ? "active" : ""}`}
                    onClick={() => setPosFilter("endgame")}
                  >
                    Tàn Cuộc ({positions.filter((p) => p.category === "endgame").length})
                  </button>
                </div>

                {/* SCROLLABLE TABLE */}
                <div className="scrollable-table-box" style={{ maxHeight: 240 }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th style={{ width: 36 }}>Chọn</th>
                        <th>Mã ID</th>
                        <th>Tên Thế Cờ</th>
                        <th>Phân Loại</th>
                        <th>Mô Tả Chiến Thuật</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredPositions.map((pos) => {
                        const isChecked = selectedPosIds.has(pos.id);
                        return (
                          <tr
                            key={pos.id}
                            className={`pos-row ${isChecked ? "pos-selected" : ""} ${previewPos?.id === pos.id ? "previewing-row" : ""}`}
                            onClick={() => setPreviewPos(pos)}
                          >
                            <td onClick={(e) => e.stopPropagation()}>
                              <input
                                type="checkbox"
                                checked={isChecked}
                                disabled={isRunning}
                                onChange={() => handleTogglePos(pos.id)}
                              />
                            </td>
                            <td><code>{pos.id}</code></td>
                            <td><strong>{pos.name}</strong></td>
                            <td>
                              <span className={`cat-pill cat-${pos.category}`}>
                                {pos.category}
                              </span>
                            </td>
                            <td>{pos.description}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
