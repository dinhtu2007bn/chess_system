import React, { useEffect, useState, useRef } from "react";
import { Chessboard } from "react-chessboard";

type HistoryMatchSummary = {
  filename: string;
  session_type: string;
  event: string;
  date: string;
  time: string;
  created_at: string;
  round: string;
  white: string;
  black: string;
  result: string;
  termination: string;
  chess960_id: number;
  initial_fen: string;
  time_control: string;
  session_id: string;
  file_size: number;
};

type MoveStep = {
  ply: number;
  move_number: number;
  turn: "white" | "black";
  san: string;
  uci: string;
  fen: string;
  elapsed_ms: number | null;
  comment: string;
};

type HistoryMatchDetail = {
  filename: string;
  headers: Record<string, string>;
  session_type: string;
  event: string;
  date: string;
  time: string;
  white: string;
  black: string;
  result: string;
  termination: string;
  time_control: string;
  chess960_id: number;
  initial_fen: string;
  final_fen: string;
  total_plies: number;
  total_moves: number;
  moves: MoveStep[];
  pgn_raw: string;
};

type Props = {
  initialFilename?: string | null;
  onClearInitialFilename?: () => void;
};

export const HistoryView: React.FC<Props> = ({
  initialFilename,
  onClearInitialFilename,
}) => {
  const [matches, setMatches] = useState<HistoryMatchSummary[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [modeFilter, setModeFilter] = useState<string>("all");
  const [botFilter, setBotFilter] = useState<string>("");

  const [selectedFilename, setSelectedFilename] = useState<string | null>(initialFilename || null);
  const [detail, setDetail] = useState<HistoryMatchDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Replayer state
  const [currentPly, setCurrentPly] = useState<number>(0);
  const [boardOrientation, setBoardOrientation] = useState<"white" | "black">("white");
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [copied, setCopied] = useState(false);
  const playIntervalRef = useRef<any>(null);

  // Fetch list of saved matches
  const fetchMatches = async () => {
    setLoadingList(true);
    try {
      const params = new URLSearchParams();
      if (modeFilter !== "all") params.set("mode", modeFilter);
      if (botFilter.trim()) params.set("bot", botFilter.trim());
      params.set("limit", "150");

      const res = await fetch(`/api/history?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setMatches(data);
        if (!selectedFilename && data.length > 0) {
          setSelectedFilename(data[0].filename);
        }
      }
    } catch (e: any) {
      setErrorMsg("Không thể tải danh sách lịch sử: " + e.message);
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    fetchMatches();
  }, [modeFilter, botFilter]);

  // Load detail when selectedFilename changes
  useEffect(() => {
    if (!selectedFilename) return;
    setLoadingDetail(true);
    setErrorMsg(null);
    setIsPlaying(false);

    fetch(`/api/history/${encodeURIComponent(selectedFilename)}`)
      .then(async (res) => {
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Lỗi tải ván đấu");
        }
        return res.json();
      })
      .then((data: HistoryMatchDetail) => {
        setDetail(data);
        // Start at initial position
        setCurrentPly(0);
      })
      .catch((err) => {
        setErrorMsg(err.message);
      })
      .finally(() => {
        setLoadingDetail(false);
      });
  }, [selectedFilename]);

  // Keyboard navigation: Left/Right arrows to step
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (!detail) return;

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setCurrentPly((prev) => Math.max(0, prev - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setCurrentPly((prev) => Math.min(detail.total_plies, prev + 1));
      } else if (e.key === "Home") {
        e.preventDefault();
        setCurrentPly(0);
      } else if (e.key === "End") {
        e.preventDefault();
        setCurrentPly(detail.total_plies);
      } else if (e.key === " ") {
        e.preventDefault();
        setIsPlaying((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [detail]);

  // Auto-play timer
  useEffect(() => {
    if (isPlaying && detail) {
      playIntervalRef.current = setInterval(() => {
        setCurrentPly((prev) => {
          if (prev >= detail.total_plies) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 900);
    } else {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, detail]);

  const handleCopyPgn = () => {
    if (!detail) return;
    navigator.clipboard.writeText(detail.pgn_raw);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDeleteMatch = async (filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(`Bạn có chắc muốn xóa file ghi ván đấu '${filename}'?`)) return;
    try {
      const res = await fetch(`/api/history/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        if (selectedFilename === filename) {
          setSelectedFilename(null);
          setDetail(null);
        }
        fetchMatches();
      }
    } catch (e) {
      // ignore
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm("Bạn có chắc muốn xóa TOÀN BỘ lịch sử ván đấu đã lưu?")) return;
    try {
      const res = await fetch("/api/history/clear", { method: "POST" });
      if (res.ok) {
        setSelectedFilename(null);
        setDetail(null);
        fetchMatches();
      }
    } catch (e) {
      // ignore
    }
  };

  // Compute current display FEN
  const currentFen =
    detail && currentPly > 0 && currentPly <= detail.moves.length
      ? detail.moves[currentPly - 1].fen
      : detail?.initial_fen || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  const currentMove =
    detail && currentPly > 0 && currentPly <= detail.moves.length
      ? detail.moves[currentPly - 1]
      : null;

  return (
    <div className="history-container">
      {/* HEADER */}
      <div className="section-header">
        <div>
          <h2>📜 Lịch Sử Ván Đấu & Trình Xem Lại (Match History & Replayer)</h2>
          <p className="section-desc">
            Toàn bộ ván đấu từ Đấu Trường Nội Bộ và Benchmark được tự động lưu chuẩn PGN Chess960. Bạn có thể tua xem lại từng nước đi, kiểm tra thời gian tính toán của bot (ms) và tải file về máy.
          </p>
        </div>
        <div className="flex-center" style={{ gap: 10 }}>
          <button className="btn-secondary btn-sm" onClick={fetchMatches} disabled={loadingList}>
            🔄 Làm mới
          </button>
          {matches.length > 0 && (
            <button className="btn-danger btn-sm" onClick={handleClearAll}>
              🗑️ Xóa tất cả
            </button>
          )}
        </div>
      </div>

      {errorMsg && <div className="alert-box alert-danger">{errorMsg}</div>}

      {/* MAIN TWO-COLUMN REPLAYER & MATCH LIST */}
      <div className="history-main-layout">
        {/* LEFT COLUMN: INTERACTIVE BOARD REPLAYER */}
        <div className="history-replayer-panel panel">
          {detail ? (
            <div>
              {/* REPLAYER TOP HEADER */}
              <div className="replayer-header-bar flex-between mb-8">
                <div>
                  <div className="flex-center" style={{ gap: 8 }}>
                    <span className={`outcome-pill ${detail.session_type === "arena" ? "pill-arena" : "pill-bench"}`}>
                      {detail.session_type === "arena" ? "⚔️ Arena" : "⚡ Benchmark"}
                    </span>
                    <strong style={{ fontSize: "1.05rem" }}>
                      {detail.white} vs {detail.black}
                    </strong>
                    <span className="badge-result">{detail.result}</span>
                  </div>
                  <div className="text-muted" style={{ fontSize: "0.82rem", marginTop: 4 }}>
                    Thế cờ Chess960 #{detail.chess960_id} • Thời gian: {detail.time_control} • Lý do: {detail.termination} • {detail.date} {detail.time}
                  </div>
                </div>

                <div className="flex-center" style={{ gap: 8 }}>
                  <button className="btn-secondary btn-sm" onClick={() => setBoardOrientation((prev) => (prev === "white" ? "black" : "white"))}>
                    🔄 Đổi góc ({boardOrientation === "white" ? "Trắng" : "Đen"})
                  </button>
                  <button className="btn-secondary btn-sm" onClick={handleCopyPgn} title="Copy nội dung PGN vào clipboard">
                    {copied ? "✓ Đã copy!" : "📋 Copy PGN"}
                  </button>
                  <a
                    href={`/api/history/${encodeURIComponent(detail.filename)}/download`}
                    download={detail.filename}
                    className="btn-primary btn-sm"
                    style={{ textDecoration: "none", display: "inline-flex", alignItems: "center" }}
                  >
                    📥 Tải PGN
                  </a>
                </div>
              </div>

              {/* TWO SUB-COLUMNS: CHESSBOARD + MOVE TIMELINE */}
              <div className="replayer-board-and-moves">
                {/* SUB-LEFT: CHESSBOARD & CONTROLS */}
                <div className="replayer-board-col">
                  {/* PLAYER CLOCK / MOVE HEADER */}
                  <div className="replayer-clock-banner flex-between">
                    <span>
                      {boardOrientation === "white" ? "⚫ " + detail.black : "⚪ " + detail.white}
                    </span>
                    <span className="text-muted" style={{ fontSize: "0.82rem" }}>
                      Nước {currentPly}/{detail.total_plies}
                    </span>
                  </div>

                  <div className="replayer-chessboard-wrapper">
                    <Chessboard
                      position={currentFen}
                      boardOrientation={boardOrientation}
                      arePiecesDraggable={false}
                    />
                  </div>

                  <div className="replayer-clock-banner flex-between">
                    <span>
                      {boardOrientation === "white" ? "⚪ " + detail.white : "⚫ " + detail.black}
                    </span>
                    {currentMove && (
                      <span className="replayer-move-badge">
                        Nước vừa đi: <strong>{currentMove.san}</strong> ({currentMove.uci})
                        {currentMove.elapsed_ms !== null && (
                          <span className="text-success" style={{ marginLeft: 6 }}>
                            ⏱️ {currentMove.elapsed_ms}ms
                          </span>
                        )}
                      </span>
                    )}
                  </div>

                  {/* NAVIGATION CONTROLS */}
                  <div className="replayer-controls flex-center">
                    <button
                      className="btn-ctrl"
                      onClick={() => setCurrentPly(0)}
                      disabled={currentPly === 0}
                      title="Về đầu ván (Phím Home)"
                    >
                      ⏮ Đầu
                    </button>
                    <button
                      className="btn-ctrl"
                      onClick={() => setCurrentPly((prev) => Math.max(0, prev - 1))}
                      disabled={currentPly === 0}
                      title="Nước trước (Phím Mũi tên Trái)"
                    >
                      ◀ Trước
                    </button>
                    <button
                      className={`btn-ctrl ${isPlaying ? "btn-ctrl-active" : ""}`}
                      onClick={() => setIsPlaying((prev) => !prev)}
                      title="Tự động tua ván đấu (Phím Space)"
                    >
                      {isPlaying ? "⏸ Tạm Dừng" : "▶ Tự Động"}
                    </button>
                    <button
                      className="btn-ctrl"
                      onClick={() => setCurrentPly((prev) => Math.min(detail.total_plies, prev + 1))}
                      disabled={currentPly >= detail.total_plies}
                      title="Nước kế tiếp (Phím Mũi tên Phải)"
                    >
                      Sau ▶
                    </button>
                    <button
                      className="btn-ctrl"
                      onClick={() => setCurrentPly(detail.total_plies)}
                      disabled={currentPly >= detail.total_plies}
                      title="Đến cuối ván (Phím End)"
                    >
                      Cuối ⏭
                    </button>
                  </div>
                  <div className="replayer-key-hint text-center text-muted">
                    💡 Mẹo: Dùng phím <strong>←</strong> / <strong>→</strong> trên bàn phím hoặc <strong>Space</strong> để tua nhanh
                  </div>
                </div>

                {/* SUB-RIGHT: MOVE LIST / NOTATION */}
                <div className="replayer-moves-col">
                  <div className="flex-between mb-8">
                    <strong>Biên Bản Nước Đi ({detail.total_moves} nước)</strong>
                    <button
                      className="btn-link-subtle"
                      onClick={() => setCurrentPly(0)}
                      style={{ fontSize: "0.8rem", cursor: "pointer" }}
                    >
                      Thế cờ đầu
                    </button>
                  </div>

                  <div className="replayer-move-list-scroll">
                    {Array.from({ length: Math.ceil(detail.moves.length / 2) }).map((_, moveIdx) => {
                      const whiteStep = detail.moves[moveIdx * 2];
                      const blackStep = detail.moves[moveIdx * 2 + 1];
                      const isWhiteActive = currentPly === whiteStep?.ply;
                      const isBlackActive = blackStep && currentPly === blackStep.ply;

                      return (
                        <div key={moveIdx} className="move-record-row flex-between">
                          <span className="move-index-tag">{moveIdx + 1}.</span>

                          {/* WHITE MOVE */}
                          <div
                            className={`move-pill ${isWhiteActive ? "move-pill-active" : ""}`}
                            onClick={() => setCurrentPly(whiteStep.ply)}
                          >
                            <strong>{whiteStep.san}</strong>
                            {whiteStep.elapsed_ms !== null && (
                              <small className="move-time-tag">{whiteStep.elapsed_ms}ms</small>
                            )}
                          </div>

                          {/* BLACK MOVE */}
                          {blackStep ? (
                            <div
                              className={`move-pill ${isBlackActive ? "move-pill-active" : ""}`}
                              onClick={() => setCurrentPly(blackStep.ply)}
                            >
                              <strong>{blackStep.san}</strong>
                              {blackStep.elapsed_ms !== null && (
                                <small className="move-time-tag">{blackStep.elapsed_ms}ms</small>
                              )}
                            </div>
                          ) : (
                            <div className="move-pill-empty" />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          ) : loadingDetail ? (
            <div className="empty-state-card text-center" style={{ padding: "60px 20px" }}>
              <div className="spinner mb-8" />
              <strong>Đang tải biên bản ván cờ...</strong>
            </div>
          ) : (
            <div className="empty-state-card text-center" style={{ padding: "60px 20px" }}>
              <span style={{ fontSize: "2.5rem" }}>♟️</span>
              <h3 className="m-0 mt-8">Chưa Chọn Ván Đấu</h3>
              <p className="text-muted">
                Hãy nhấp vào một ván đấu ở bảng danh sách bên phải để tải biên bản và xem lại từng nước đi.
              </p>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: SAVED MATCHES LIST & FILTERS */}
        <div className="history-list-panel panel">
          <div className="flex-between mb-8">
            <h3 className="m-0">Ván Đã Đấu ({matches.length})</h3>
            <span className="text-muted" style={{ fontSize: "0.82rem" }}>
              {loadingList ? "Đang tải..." : "Chuẩn PGN"}
            </span>
          </div>

          {/* FILTERS */}
          <div className="history-filter-bar mb-8">
            <select
              value={modeFilter}
              onChange={(e) => setModeFilter(e.target.value)}
              className="filter-select"
            >
              <option value="all">Tất cả chế độ</option>
              <option value="arena">⚔️ Đấu Trường Nội Bộ</option>
              <option value="benchmark">⚡ Benchmark</option>
            </select>

            <input
              type="text"
              placeholder="Lọc theo tên Bot..."
              value={botFilter}
              onChange={(e) => setBotFilter(e.target.value)}
              className="filter-input"
            />
          </div>

          {/* MATCH TABLE / LIST */}
          <div className="history-table-scroll">
            {matches.length === 0 ? (
              <div className="text-center text-muted" style={{ padding: "40px 10px" }}>
                {loadingList ? "Đang tải lịch sử..." : "Chưa có ván đấu nào được lưu."}
              </div>
            ) : (
              <table className="data-table history-table">
                <thead>
                  <tr>
                    <th>Chế độ</th>
                    <th>Cặp Đấu</th>
                    <th>Thế Cờ</th>
                    <th>Tỷ Số</th>
                    <th>Thời Gian</th>
                    <th style={{ width: 40 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {matches.map((m) => {
                    const isSelected = selectedFilename === m.filename;
                    return (
                      <tr
                        key={m.filename}
                        className={`history-row ${isSelected ? "history-row-selected" : ""}`}
                        onClick={() => setSelectedFilename(m.filename)}
                      >
                        <td>
                          <span
                            className={`badge-sm ${m.session_type === "arena" ? "badge-arena" : "badge-bench"}`}
                          >
                            {m.session_type === "arena" ? "Arena" : "Bench"}
                          </span>
                        </td>
                        <td>
                          <div className="history-match-pair">
                            <strong>{m.white}</strong> vs <strong>{m.black}</strong>
                          </div>
                        </td>
                        <td>
                          <span className="pos-badge">#{m.chess960_id}</span>
                        </td>
                        <td>
                          <strong className="result-tag">{m.result}</strong>
                        </td>
                        <td>
                          <small className="text-muted">{m.created_at.slice(5, 16)}</small>
                        </td>
                        <td>
                          <button
                            className="btn-icon-danger"
                            onClick={(e) => handleDeleteMatch(m.filename, e)}
                            title="Xóa ván này"
                          >
                            ✕
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

