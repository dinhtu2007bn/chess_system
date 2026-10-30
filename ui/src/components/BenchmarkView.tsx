import React, { useEffect, useState } from "react";
import { Chessboard } from "react-chessboard";
import { BenchmarkSession, ChessPosition, LiveBoardState, ManagedBot } from "../types";

type Props = {
  bots: ManagedBot[];
  positions: ChessPosition[];
  selectedBotIdDefault?: string | null;
  onRefreshPositions: () => void;
  benchmarkLiveEvent?: any;
  benchmarkSessionEvent?: any;
};

function formatClock(milliseconds: number) {
  const total = Math.max(0, Math.floor(milliseconds));
  const minutes = Math.floor(total / 60000);
  const seconds = Math.floor((total % 60000) / 1000);
  const millis = total % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(Math.floor(millis / 100))}`;
}

export const BenchmarkView: React.FC<Props> = ({
  bots,
  positions,
  selectedBotIdDefault,
  onRefreshPositions,
  benchmarkLiveEvent,
  benchmarkSessionEvent,
}) => {
  const [selectedBotId, setSelectedBotId] = useState<string>(
    selectedBotIdDefault || (bots[0]?.id ?? "")
  );
  const [opponentType, setOpponentType] = useState<"stockfish" | "managed_bot">("stockfish");
  const [opponentElo, setOpponentElo] = useState<number>(1700);
  const [opponentBotId, setOpponentBotId] = useState<string>(
    bots.length > 1 ? bots[1].id : bots[0]?.id ?? ""
  );

  // Position selection: initialize with default positions immediately
  const [selectedPosIds, setSelectedPosIds] = useState<Set<string>>(() => {
    const init = new Set<string>();
    if (positions && positions.length > 0) {
      positions.filter((p) => p.category === "opening").slice(0, 2).forEach((p) => init.add(p.id));
      positions.filter((p) => p.category === "midgame").slice(0, 3).forEach((p) => init.add(p.id));
      positions.filter((p) => p.category === "endgame").slice(0, 1).forEach((p) => init.add(p.id));
    }
    return init;
  });
  const [posFilter, setPosFilter] = useState<"all" | "opening" | "midgame" | "endgame">("all");
  const [randomCount, setRandomCount] = useState<number>(5);

  // Preview position board (when not running or inspecting)
  const [previewPos, setPreviewPos] = useState<ChessPosition | null>(
    positions && positions.length > 0 ? positions[0] : null
  );

  // Time controls: Full Game Clock per player
  const [gameTimeSec, setGameTimeSec] = useState<number>(60); // 60s per player
  const [incrementSec, setIncrementSec] = useState<number>(1);  // +1s increment per move
  const [runsPerPosition, setRunsPerPosition] = useState<number>(3);
  const [concurrency, setConcurrency] = useState<number>(3); // Multi-board concurrent workers

  // Live session state & multi-board tracking
  const [session, setSession] = useState<BenchmarkSession | null>(null);
  const [liveBoard, setLiveBoard] = useState<LiveBoardState | null>(null);
  const [liveBoards, setLiveBoards] = useState<Record<string, LiveBoardState>>({});
  const [runningTab, setRunningTab] = useState<"arena" | "positions">("arena");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (selectedBotIdDefault) {
      setSelectedBotId(selectedBotIdDefault);
    } else if (!selectedBotId && bots.length > 0) {
      setSelectedBotId(bots[0].id);
    }
  }, [selectedBotIdDefault, bots]);

  // Default select first 6 positions when positions are loaded if none selected
  useEffect(() => {
    if (positions.length > 0 && selectedPosIds.size === 0) {
      const initial = new Set<string>();
      positions.filter((p) => p.category === "opening").slice(0, 2).forEach((p) => initial.add(p.id));
      positions.filter((p) => p.category === "midgame").slice(0, 3).forEach((p) => initial.add(p.id));
      positions.filter((p) => p.category === "endgame").slice(0, 1).forEach((p) => initial.add(p.id));
      setSelectedPosIds(initial);
      if (!previewPos) setPreviewPos(positions[0]);
    }
  }, [positions]);

  // Real-time live board move updates via WebSocket for all concurrent slots
  useEffect(() => {
    if (benchmarkLiveEvent) {
      const slotKey = String(benchmarkLiveEvent.slot_id ?? 0);
      setLiveBoards((prev) => ({
        ...prev,
        [slotKey]: {
          slot_id: benchmarkLiveEvent.slot_id ?? 0,
          position_id: benchmarkLiveEvent.position_id,
          position_name: benchmarkLiveEvent.position_name,
          run_number: benchmarkLiveEvent.run_number,
          bot_color: benchmarkLiveEvent.bot_color,
          fen: benchmarkLiveEvent.fen,
          turn: benchmarkLiveEvent.turn,
          white_ms: benchmarkLiveEvent.white_ms,
          black_ms: benchmarkLiveEvent.black_ms,
          last_move: benchmarkLiveEvent.last_move,
          last_move_san: benchmarkLiveEvent.last_move_san,
          moves_count: benchmarkLiveEvent.moves_count,
          history: benchmarkLiveEvent.history,
          status: "playing",
        },
      }));

      // Maintain legacy single board
      if (benchmarkLiveEvent.slot_id === 0 || !liveBoard) {
        setLiveBoard({
          fen: benchmarkLiveEvent.fen,
          turn: benchmarkLiveEvent.turn,
          white_ms: benchmarkLiveEvent.white_ms,
          black_ms: benchmarkLiveEvent.black_ms,
          last_move: benchmarkLiveEvent.last_move,
          last_move_san: benchmarkLiveEvent.last_move_san,
          moves_count: benchmarkLiveEvent.moves_count,
          history: benchmarkLiveEvent.history,
        });
      }
    }
  }, [benchmarkLiveEvent]);

  // Real-time benchmark session updates via WebSocket
  useEffect(() => {
    if (benchmarkSessionEvent) {
      setSession(benchmarkSessionEvent);
      if (benchmarkSessionEvent.live_boards) {
        setLiveBoards((prev) => ({
          ...prev,
          ...benchmarkSessionEvent.live_boards,
        }));
      }
      if (benchmarkSessionEvent.live_board) {
        setLiveBoard(benchmarkSessionEvent.live_board);
      }
    }
  }, [benchmarkSessionEvent]);

  // Active game clock countdown ticker for all active slots
  useEffect(() => {
    if (!session || session.status !== "running") return;
    const ticker = setInterval(() => {
      setLiveBoards((prev) => {
        let changed = false;
        const next: Record<string, LiveBoardState> = {};
        for (const [key, b] of Object.entries(prev)) {
          if (b.status === "finished") {
            next[key] = b;
            continue;
          }
          if (b.turn === "white" && b.white_ms > 0) {
            next[key] = { ...b, white_ms: Math.max(0, b.white_ms - 100) };
            changed = true;
          } else if (b.turn === "black" && b.black_ms > 0) {
            next[key] = { ...b, black_ms: Math.max(0, b.black_ms - 100) };
            changed = true;
          } else {
            next[key] = b;
          }
        }
        return changed ? next : prev;
      });

      // Also tick single board
      setLiveBoard((prev) => {
        if (!prev) return prev;
        if (prev.turn === "white") {
          return { ...prev, white_ms: Math.max(0, prev.white_ms - 100) };
        } else {
          return { ...prev, black_ms: Math.max(0, prev.black_ms - 100) };
        }
      });
    }, 100);
    return () => clearInterval(ticker);
  }, [session?.status]);

  // Fallback Polling (1200ms) to ensure sync if WebSocket drops
  useEffect(() => {
    let timer: any;
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/benchmark/status");
        if (res.ok) {
          const data = await res.json();
          if (data.session) {
            setSession(data.session);
            if (data.session.live_boards) {
              setLiveBoards((prev) => ({
                ...prev,
                ...data.session.live_boards,
              }));
            }
            if (data.session.live_board && !benchmarkLiveEvent) {
              setLiveBoard(data.session.live_board);
            }
          }
        }
      } catch (e) {
        // ignore
      }
    };
    fetchStatus();
    timer = setInterval(fetchStatus, 1200);
    return () => clearInterval(timer);
  }, [benchmarkLiveEvent]);

  const handleTogglePos = (pos: ChessPosition) => {
    setPreviewPos(pos);
    setSelectedPosIds((prev) => {
      const next = new Set(prev);
      if (next.has(pos.id)) next.delete(pos.id);
      else next.add(pos.id);
      return next;
    });
  };

  const handleSelectAll = () => {
    const next = new Set<string>();
    positions.forEach((p) => next.add(p.id));
    setSelectedPosIds(next);
  };

  const handleDeselectAll = () => {
    setSelectedPosIds(new Set());
  };

  const handlePickRandom = () => {
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
        onRefreshPositions();
        setSelectedPosIds((prev) => new Set(prev).add(newPos.id));
        setPreviewPos(newPos);
      }
    } catch (e) {
      // ignore
    }
  };

  const handleStartBenchmark = async () => {
    if (!selectedBotId) {
      setErrorMsg("Vui lòng chọn bot cần benchmark");
      return;
    }
    if (selectedPosIds.size === 0) {
      setErrorMsg("Vui lòng chọn ít nhất 1 thế cờ để kiểm thử");
      return;
    }

    setLoading(true);
    setErrorMsg(null);

    const payload = {
      bot_id: selectedBotId,
      position_ids: Array.from(selectedPosIds),
      opponent_type: opponentType,
      opponent_elo: opponentElo,
      opponent_bot_id: opponentType === "managed_bot" ? opponentBotId : null,
      time_mode: "global",
      game_time_seconds: gameTimeSec,
      increment_seconds: incrementSec,
      runs_per_position: runsPerPosition,
      concurrency: concurrency,
    };

    try {
      const res = await fetch("/api/benchmark/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Không thể bắt đầu benchmark");
      }
      const data = await res.json();
      setSession(data);
      if (data.live_boards) {
        setLiveBoards(data.live_boards);
      }
      if (data.live_board) {
        setLiveBoard(data.live_board);
      }
      setRunningTab("arena");
    } catch (err: any) {
      setErrorMsg(err.message || "Lỗi khi khởi động benchmark");
    } finally {
      setLoading(false);
    }
  };

  const handleStopBenchmark = async () => {
    try {
      await fetch("/api/benchmark/stop", { method: "POST" });
    } catch (e) {
      // ignore
    }
  };

  const filteredPositions = positions.filter((p) => {
    if (posFilter === "all") return true;
    return p.category === posFilter;
  });

  const isRunning = session?.status === "running";
  const metrics = session?.metrics;
  const activeConcurrency = session?.concurrency || concurrency;

  // Render slots array for multi-board cards
  const slotIndices = Array.from({ length: activeConcurrency }, (_, i) => i);

  return (
    <div className="benchmark-app-wrapper">
      {/* COMPACT TOP HEADER */}
      <div className="section-header-compact">
        <div>
          <h2>Đánh Giá & Chấm Điểm Thuật Toán Bot (Benchmark Đa Bàn Cờ Song Song)</h2>
          <p className="section-desc">
            Chạy đồng thời nhiều bàn cờ song song ({activeConcurrency} luồng độc lập), đảo màu quân, đo tỷ lệ Timeout, tính toán Điểm Tối Ưu Hóa (%) và chẩn đoán <code>time_manager.py</code>.
          </p>
        </div>
        {isRunning && (
          <button className="btn-danger btn-sm" onClick={handleStopBenchmark}>
            ⏹ Dừng Benchmark
          </button>
        )}
      </div>

      {errorMsg && <div className="alert-box alert-danger">{errorMsg}</div>}

      {/* ========================================================================= */}
      {/* 1. RUNNING MODE: MULTI-BOARD CONCURRENT ARENA                            */}
      {/* ========================================================================= */}
      {isRunning ? (
        <div>
          {/* TOP ARENA BANNER */}
          <div className="arena-top-banner">
            <div className="arena-banner-header">
              <div className="arena-title-wrap">
                <span className="live-badge">🔴 Đang Thi Đấu Trực Tiếp</span>
                <span className="concurrency-badge">
                  ⚡ Song song: {activeConcurrency} Bàn Cờ
                </span>
                <strong>
                  {session?.bot_name} vs {session?.opponent_name}
                </strong>
              </div>

              <div className="flex-center" style={{ gap: 8 }}>
                <button
                  className={`arena-tab-btn ${runningTab === "arena" ? "active" : ""}`}
                  onClick={() => setRunningTab("arena")}
                >
                  ♟️ Đấu Trường {activeConcurrency} Bàn Cờ
                </button>
                <button
                  className={`arena-tab-btn ${runningTab === "positions" ? "active" : ""}`}
                  onClick={() => setRunningTab("positions")}
                >
                  📋 Xem Ngân Hàng Thế Cờ
                </button>
                <button className="btn-danger btn-sm" onClick={handleStopBenchmark}>
                  ⏹ Dừng
                </button>
              </div>
            </div>

            {/* PROGRESS BAR */}
            <div>
              <div className="flex-between mb-8" style={{ fontSize: "0.82rem" }}>
                <span>
                  Tiến độ: <strong>{session?.games_completed || 0}</strong> / {session?.total_games || 0} ván hoàn thành
                </span>
                <span className="text-highlight">
                  {session?.total_games ? Math.round(((session.games_completed || 0) / session.total_games) * 100) : 0}%
                </span>
              </div>
              <div className="progress-bar-wrap">
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${session?.total_games ? ((session.games_completed || 0) / session.total_games) * 100 : 0}%`,
                  }}
                />
              </div>
            </div>
          </div>

          {runningTab === "arena" ? (
            <div>
              {/* MULTI-BOARD GRID (Cards for each concurrent slot) */}
              <div className="multi-board-grid">
                {slotIndices.map((slotId) => {
                  const b = liveBoards[String(slotId)];
                  const isTurnWhite = b?.turn === "white";
                  const isTurnBlack = b?.turn === "black";
                  const botColor = b?.bot_color || "white";
                  const isBotTurn = (botColor === "white" && isTurnWhite) || (botColor === "black" && isTurnBlack);
                  const isOppTurn = (botColor === "white" && isTurnBlack) || (botColor === "black" && isTurnWhite);

                  const boardFen = b?.fen || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

                  return (
                    <div
                      key={slotId}
                      className={`mini-board-card ${b?.status === "playing" ? "active-slot" : ""}`}
                    >
                      {/* HEADER */}
                      <div className="mini-card-header">
                        <div className="mini-slot-badge">
                          <span className="slot-num">Bàn #{slotId + 1}</span>
                          <span className={`slot-status slot-${b?.status || "waiting"}`}>
                            {b?.status === "playing"
                              ? "🔴 Đang đấu"
                              : b?.status === "finished"
                              ? `✓ ${b.result || "Xong"}`
                              : "⏳ Chờ"}
                          </span>
                        </div>
                        <div className="mini-pos-info" title={b?.position_name}>
                          <strong>{b?.position_name || "Đang xếp lịch..."}</strong>
                          {b?.run_number && <small> (Lượt {b.run_number})</small>}
                        </div>
                      </div>

                      {/* OPPONENT CLOCK */}
                      <div className={`mini-clock ${isOppTurn && b?.status === "playing" ? "mini-clock-active" : ""}`}>
                        <span className="mini-clock-player">
                          {botColor === "white" ? "⚫" : "⚪"} {session?.opponent_name}
                          {isOppTurn && b?.status === "playing" && <span className="pulse-dot"> •</span>}
                        </span>
                        <strong className="mini-clock-val">
                          {formatClock(
                            b ? (botColor === "white" ? b.black_ms : b.white_ms) : gameTimeSec * 1000
                          )}
                        </strong>
                      </div>

                      {/* MINI CHESSBOARD */}
                      <div className="mini-board-wrapper">
                        <Chessboard
                          position={boardFen}
                          boardOrientation={botColor}
                          arePiecesDraggable={false}
                        />
                      </div>

                      {/* BOT CLOCK */}
                      <div className={`mini-clock ${isBotTurn && b?.status === "playing" ? "mini-clock-active" : ""}`}>
                        <span className="mini-clock-player">
                          {botColor === "white" ? "⚪" : "⚫"} {session?.bot_name}
                          {isBotTurn && b?.status === "playing" && <span className="pulse-dot"> •</span>}
                        </span>
                        <strong className="mini-clock-val">
                          {formatClock(
                            b ? (botColor === "white" ? b.white_ms : b.black_ms) : gameTimeSec * 1000
                          )}
                        </strong>
                      </div>

                      {/* MINI FOOTER */}
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

              {/* LIVE SCOREBOARD & COMPLETED GAMES (TWO-COLUMN DASHBOARD) */}
              <div className="arena-bottom-two-col">
                {/* METRICS & VERDICT */}
                <div className="panel bench-result-compact">
                  <h3 className="m-0 mb-8">Điểm Tối Ưu Hóa & Đánh Giá</h3>
                  {metrics && (
                    <div>
                      <div className="bench-metrics-compact-row" style={{ gridTemplateColumns: "140px repeat(3, 1fr)" }}>
                        <div className="score-badge-box">
                          <span className="score-mini-title">Điểm Tối Ưu</span>
                          <strong className="score-big-num">{metrics.optimization_score}%</strong>
                          <small className="score-mini-hint">WinRate × Time</small>
                        </div>

                        <div className="metric-mini-card">
                          <span className="m-label">Thắng / Hòa / Thua</span>
                          <strong className="m-val text-success">{metrics.win_rate}%</strong>
                          <span className="m-sub">{metrics.wins}T - {metrics.draws}H - {metrics.losses}B</span>
                        </div>

                        <div className="metric-mini-card">
                          <span className="m-label">Thua Tràn Giờ</span>
                          <strong className={`m-val ${metrics.timeout_losses > 0 ? "text-danger" : "text-success"}`}>
                            {metrics.timeout_losses} ván
                          </strong>
                          <span className="m-sub">Tỷ lệ: {(metrics.timeout_penalty * 100).toFixed(1)}%</span>
                        </div>

                        <div className="metric-mini-card">
                          <span className="m-label">Thời Gian TB</span>
                          <strong className="m-val">{metrics.avg_move_time_ms} ms</strong>
                          <span className="m-sub">Đỉnh: {metrics.peak_move_time_ms} ms</span>
                        </div>
                      </div>

                      <div className="verdict-banner-compact">
                        <strong>📋 Nhận xét thuật toán: </strong>
                        <span>{metrics.verdict}</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* COMPLETED GAMES TABLE */}
                <div className="panel bench-result-compact">
                  <div className="flex-between mb-8">
                    <h3 className="m-0">Ván Đấu Đã Hoàn Thành ({session?.games.length || 0})</h3>
                    <small className="text-muted">Cập nhật tức thì khi mỗi bàn kết thúc</small>
                  </div>

                  <div className="scrollable-table-box" style={{ maxHeight: 220 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Mã</th>
                          <th>Thế Cờ</th>
                          <th>Lượt</th>
                          <th>Quân</th>
                          <th>Kết Quả</th>
                          <th>Thời Gian TB</th>
                          <th>Lý Do</th>
                        </tr>
                      </thead>
                      <tbody>
                        {session?.games.map((g, i) => (
                          <tr key={i} className={`row-outcome-${g.bot_outcome}`}>
                            <td><code>{g.position_id}</code></td>
                            <td><strong>{g.position_name}</strong></td>
                            <td>Run {g.run_number}</td>
                            <td>{g.bot_color === "white" ? "⚪ Trắng" : "⚫ Đen"}</td>
                            <td>
                              <span className={`outcome-pill outcome-${g.bot_outcome}`}>
                                {g.result}
                              </span>
                            </td>
                            <td>{g.avg_move_time_ms} ms</td>
                            <td>
                              {g.is_timeout ? (
                                <strong className="text-danger">⚠️ Timeout</strong>
                              ) : (
                                g.termination
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* POSITIONS INSPECTION WHILE RUNNING */
            <div className="benchmark-two-col">
              <div className="bench-left-panel panel">
                <div className="live-board-header">
                  <span className="live-badge">👁 Xem Trước Thế Cờ</span>
                  <strong className="live-pos-title">{previewPos?.name || "Thế cờ Chess960"}</strong>
                </div>
                <div className="bench-chessboard-box">
                  <Chessboard
                    position={previewPos?.fen || "start"}
                    boardOrientation="white"
                    arePiecesDraggable={false}
                  />
                </div>
                {previewPos && (
                  <div className="preview-desc-box">
                    <span className="cat-pill cat-preview">{previewPos.category.toUpperCase()}</span>
                    <p>{previewPos.description}</p>
                  </div>
                )}
              </div>

              <div className="bench-right-panel">
                <div className="panel bench-positions-box">
                  <h3>Ngân Hàng Thế Cờ</h3>
                  <div className="scrollable-table-box" style={{ maxHeight: 380 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Mã</th>
                          <th>Tên Thế Cờ</th>
                          <th>Phân Loại</th>
                          <th>Mô Tả</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredPositions.map((pos) => (
                          <tr
                            key={pos.id}
                            className={`pos-row ${previewPos?.id === pos.id ? "previewing-row" : ""}`}
                            onClick={() => setPreviewPos(pos)}
                          >
                            <td><code>{pos.id}</code></td>
                            <td><strong>{pos.name}</strong></td>
                            <td>
                              <span className={`cat-pill cat-${pos.category}`}>
                                {pos.category}
                              </span>
                            </td>
                            <td>{pos.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* ========================================================================= */
        /* 2. IDLE / FINISHED SETUP MODE                                            */
        /* ========================================================================= */
        <div>
          {/* COMPLETED RESULTS CARD (IF FINISHED) */}
          {session && (session.status === "finished" || session.status === "stopped") && (
            <div className="panel bench-result-compact" style={{ marginBottom: 16 }}>
              <div className="flex-between mb-8">
                <div>
                  <h3 className="m-0">Kết Quả Benchmark Vừa Hoàn Thành</h3>
                  <small className="text-muted">
                    Hoàn tất {session.games_completed} / {session.total_games} ván đấu ({session.concurrency || 3} bàn cờ song song)
                  </small>
                </div>
                <span className={`status-pill pill-${session.status}`}>
                  {session.status === "finished" ? "✓ Hoàn Tất" : "⏹ Đã Dừng"}
                </span>
              </div>

              {metrics && (
                <div>
                  <div className="bench-metrics-compact-row">
                    <div className="score-badge-box">
                      <span className="score-mini-title">Điểm Tối Ưu Hóa</span>
                      <strong className="score-big-num">{metrics.optimization_score}%</strong>
                      <small className="score-mini-hint">WinRate × (1 - Timeout Penalty) × Time Efficiency</small>
                    </div>

                    <div className="metric-mini-card">
                      <span className="m-label">Tỷ Lệ Thắng</span>
                      <strong className="m-val text-success">{metrics.win_rate}%</strong>
                      <span className="m-sub">{metrics.wins}T - {metrics.draws}H - {metrics.losses}B</span>
                    </div>

                    <div className="metric-mini-card">
                      <span className="m-label">Thua Tràn Giờ</span>
                      <strong className={`m-val ${metrics.timeout_losses > 0 ? "text-danger" : "text-success"}`}>
                        {metrics.timeout_losses} ván
                      </strong>
                      <span className="m-sub">Tỷ lệ: {(metrics.timeout_penalty * 100).toFixed(1)}%</span>
                    </div>

                    <div className="metric-mini-card">
                      <span className="m-label">Nước Đi TB</span>
                      <strong className="m-val">{metrics.avg_move_time_ms} ms</strong>
                      <span className="m-sub">Đỉnh: {metrics.peak_move_time_ms} ms</span>
                    </div>
                  </div>

                  <div className="verdict-banner-compact">
                    <strong>📋 Nhận xét thuật toán: </strong>
                    <span>{metrics.verdict}</span>
                  </div>
                </div>
              )}

              {/* SCROLLABLE TABLE OF COMPLETED GAMES */}
              {session.games.length > 0 && (
                <div className="scrollable-table-box" style={{ maxHeight: 180 }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Mã</th>
                        <th>Thế Cờ</th>
                        <th>Lượt</th>
                        <th>Quân</th>
                        <th>Kết Quả</th>
                        <th>Thời Gian TB</th>
                        <th>Lý Do Kết Thúc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {session.games.map((g, i) => (
                        <tr key={i} className={`row-outcome-${g.bot_outcome}`}>
                          <td><code>{g.position_id}</code></td>
                          <td><strong>{g.position_name}</strong></td>
                          <td>Run {g.run_number}</td>
                          <td>{g.bot_color === "white" ? "⚪ Trắng" : "⚫ Đen"}</td>
                          <td>
                            <span className={`outcome-pill outcome-${g.bot_outcome}`}>
                              {g.result} ({g.bot_outcome.toUpperCase()})
                            </span>
                          </td>
                          <td>{g.avg_move_time_ms} ms</td>
                          <td>
                            {g.is_timeout ? (
                              <strong className="text-danger">⚠️ Tràn Giờ (Timeout)</strong>
                            ) : (
                              g.termination
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* TWO-COLUMN SETUP DASHBOARD */}
          <div className="benchmark-two-col">
            {/* LEFT COLUMN: PREVIEW CHESSBOARD */}
            <div className="bench-left-panel panel">
              <div className="live-board-header">
                <span className="live-badge">👁 Xem Trước Thế Cờ</span>
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
              {/* CONFIGURATION CARD */}
              <div className="panel bench-config-compact">
                <div className="config-row-flex">
                  <div className="form-group flex-1">
                    <label>1. Bot Cần Kiểm Thử (Student Bot):</label>
                    <select
                      value={selectedBotId}
                      disabled={isRunning}
                      onChange={(e) => setSelectedBotId(e.target.value)}
                    >
                      {bots.map((b) => (
                        <option key={b.id} value={b.id}>
                          {b.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group flex-1">
                    <label>2. Đối Thủ So Tài:</label>
                    <select
                      value={opponentType}
                      disabled={isRunning}
                      onChange={(e) => setOpponentType(e.target.value as any)}
                    >
                      <option value="stockfish">Stockfish 19 UCI (Giới hạn Elo)</option>
                      <option value="managed_bot">Bot khác trong hệ thống</option>
                    </select>
                  </div>

                  {opponentType === "stockfish" ? (
                    <div className="form-group flex-1">
                      <div className="flex-between">
                        <label>Elo Đối Thủ:</label>
                        <strong className="elo-tag">{opponentElo} Elo</strong>
                      </div>
                      <input
                        type="range"
                        min="1000"
                        max="2500"
                        step="50"
                        value={opponentElo}
                        disabled={isRunning}
                        onChange={(e) => setOpponentElo(Number(e.target.value))}
                      />
                    </div>
                  ) : (
                    <div className="form-group flex-1">
                      <label>Chọn bot đối thủ:</label>
                      <select
                        value={opponentBotId}
                        disabled={isRunning}
                        onChange={(e) => setOpponentBotId(e.target.value)}
                      >
                        {bots.filter((b) => b.id !== selectedBotId).map((b) => (
                          <option key={b.id} value={b.id}>{b.name}</option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>

                {/* GAME CLOCK & CONCURRENCY CONFIGURATION */}
                <div className="config-row-flex">
                  <div className="form-group flex-1">
                    <label>3. Tổng Thời Gian Ván Đấu (Mỗi Bên):</label>
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
                    <label>Cộng Thêm / Nước Đi:</label>
                    <select
                      value={incrementSec}
                      disabled={isRunning}
                      onChange={(e) => setIncrementSec(Number(e.target.value))}
                    >
                      <option value={0}>+0s (Không cộng)</option>
                      <option value={1}>+1s / nước đi (Khuyến nghị)</option>
                      <option value={2}>+2s / nước đi</option>
                    </select>
                  </div>

                  <div className="form-group flex-1">
                    <label>Số Lượt Chạy / Thế Cờ:</label>
                    <select
                      value={runsPerPosition}
                      disabled={isRunning}
                      onChange={(e) => setRunsPerPosition(Number(e.target.value))}
                    >
                      <option value={1}>1 Lượt (Test nhanh)</option>
                      <option value={2}>2 Lượt (Đổi màu Trắng - Đen chuẩn)</option>
                      <option value={3}>3 Lượt (Chuẩn khách quan W-B-W)</option>
                    </select>
                  </div>

                  {/* CONCURRENCY MULTI-BOARD SELECTOR */}
                  <div className="form-group flex-1">
                    <label>⚡ Bàn Cờ Song Song (Concurrency):</label>
                    <select
                      value={concurrency}
                      disabled={isRunning}
                      onChange={(e) => setConcurrency(Number(e.target.value))}
                    >
                      <option value={1}>1 Bàn Cờ (Tuần tự)</option>
                      <option value={2}>2 Bàn Cờ Song Song</option>
                      <option value={3}>3 Bàn Cờ Song Song (Khuyến nghị)</option>
                      <option value={4}>4 Bàn Cờ Song Song (Nhanh)</option>
                      <option value={6}>6 Bàn Cờ Song Song (Cực nhanh)</option>
                    </select>
                  </div>
                </div>

                {/* ACTION ROW */}
                <div className="flex-between bench-start-row">
                  <div className="bench-plan-info">
                    Đã chọn: <strong>{selectedPosIds.size} thế cờ</strong> × {runsPerPosition} lượt ={" "}
                    <strong className="text-highlight">{selectedPosIds.size * runsPerPosition} ván</strong>{" "}
                    (Chạy song song: <strong>{concurrency} bàn cờ đồng thời</strong>)
                  </div>
                  <button
                    className="btn-primary btn-large"
                    disabled={isRunning || loading}
                    onClick={handleStartBenchmark}
                  >
                    {loading
                      ? "⏳ Đang Khởi Động..."
                      : selectedPosIds.size === 0
                      ? "⚠️ Hãy chọn ít nhất 1 thế cờ"
                      : `▶ Chạy Benchmark (${selectedPosIds.size * runsPerPosition} Ván - ${concurrency} Bàn Cờ)`}
                  </button>
                </div>
              </div>

              {/* NGÂN HÀNG 40 THẾ CỜ */}
              <div className="panel bench-positions-box">
                <div className="positions-header-compact">
                  <div>
                    <h3 className="m-0">Ngân Hàng 40 Thế Cờ Chess960 Thực Tế</h3>
                    <small className="text-muted">Nhấp vào dòng bất kỳ để xem trước bàn cờ ở cột bên trái</small>
                  </div>
                  <div className="positions-actions-compact">
                    <button className="btn-sm btn-outline" onClick={handleSelectAll}>Chọn Tất Cả</button>
                    <button className="btn-sm btn-outline" onClick={handleDeselectAll}>Bỏ Hết</button>
                    <div className="quick-random-group">
                      <input
                        type="number"
                        min="1"
                        max="40"
                        value={randomCount}
                        onChange={(e) => setRandomCount(Number(e.target.value))}
                        className="input-inline-num"
                      />
                      <button className="btn-sm btn-secondary" onClick={handlePickRandom}>
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

                {/* FIXED-HEIGHT SCROLLABLE TABLE */}
                <div className="scrollable-table-box" style={{ maxHeight: 260 }}>
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
                        const isPreviewing = previewPos?.id === pos.id;
                        return (
                          <tr
                            key={pos.id}
                            className={`pos-row ${isChecked ? "selected-row" : ""} ${isPreviewing ? "previewing-row" : ""}`}
                            onClick={() => handleTogglePos(pos)}
                          >
                            <td>
                              <input
                                type="checkbox"
                                checked={isChecked}
                                onChange={() => {}} // handled by row click
                              />
                            </td>
                            <td><code>{pos.id}</code></td>
                            <td><strong>{pos.name}</strong></td>
                            <td>
                              <span className={`cat-pill cat-${pos.category}`}>
                                {pos.category === "opening" && "Khai Cuộc"}
                                {pos.category === "midgame" && "Trung Cuộc"}
                                {pos.category === "endgame" && "Tàn Cuộc"}
                              </span>
                            </td>
                            <td className="pos-desc-cell">{pos.description}</td>
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
