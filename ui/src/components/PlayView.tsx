import React, { useMemo, useState } from "react";
import { Chessboard } from "react-chessboard";
import { Color, GameState, LegalMove, ManagedBot } from "../types";

type Props = {
  game: GameState;
  bots: ManagedBot[];
  selectedBotId: string;
  onSelectBotId: (id: string) => void;
  onStartGame: (color: Color, timeSec: number, positionId: number, botId?: string) => void;
  onStopGame: () => void;
  onMove: (token: string) => void;
};

function formatClock(milliseconds: number) {
  const total = Math.max(0, Math.floor(milliseconds));
  const minutes = Math.floor(total / 60000);
  const seconds = Math.floor((total % 60000) / 1000);
  const millis = total % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

export const PlayView: React.FC<Props> = ({
  game,
  bots,
  selectedBotId,
  onSelectBotId,
  onStartGame,
  onStopGame,
  onMove,
}) => {
  const [botColor, setBotColor] = useState<Color>("black");
  const [timeSeconds, setTimeSeconds] = useState("240");
  const [positionId, setPositionId] = useState(518);
  const [selectedSquare, setSelectedSquare] = useState<string | null>(null);
  const [promotionChoices, setPromotionChoices] = useState<LegalMove[]>([]);

  const legalBySource = useMemo(() => {
    const map = new Map<string, LegalMove[]>();
    for (const move of game.legal_moves) {
      const list = map.get(move.from) ?? [];
      list.push(move);
      map.set(move.from, list);
    }
    return map;
  }, [game.legal_moves]);

  const customSquareStyles = useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};
    if (selectedSquare) {
      styles[selectedSquare] = { backgroundColor: "rgba(255, 255, 0, 0.4)" };
      const targets = legalBySource.get(selectedSquare) ?? [];
      for (const target of targets) {
        styles[target.to] = {
          background: "radial-gradient(circle, rgba(0,0,0,.5) 25%, transparent 25%)",
          borderRadius: "50%",
        };
      }
    }
    return styles;
  }, [selectedSquare, legalBySource]);

  const handleSquareClick = (square: string) => {
    if (game.status !== "running") return;
    const humanTurn = game.turn !== game.bot_color;
    if (!humanTurn) return;

    if (!selectedSquare) {
      if (legalBySource.has(square)) setSelectedSquare(square);
      return;
    }

    const available = legalBySource.get(selectedSquare) ?? [];
    const matched = available.filter((m) => m.to === square);

    if (matched.length === 1 && !matched[0].promotion) {
      onMove(matched[0].token);
      setSelectedSquare(null);
    } else if (matched.length > 0 && matched.some((m) => m.promotion)) {
      setPromotionChoices(matched);
    } else if (legalBySource.has(square)) {
      setSelectedSquare(square);
    } else {
      setSelectedSquare(null);
    }
  };

  const handlePieceDrop = (source: string, target: string) => {
    if (game.status !== "running" || game.turn === game.bot_color) return false;
    const available = legalBySource.get(source) ?? [];
    const matched = available.filter((m) => m.to === target);
    if (matched.length === 1 && !matched[0].promotion) {
      onMove(matched[0].token);
      return true;
    }
    if (matched.length > 0 && matched.some((m) => m.promotion)) {
      setPromotionChoices(matched);
      return true;
    }
    return false;
  };

  const handleRandomPosition = () => {
    setPositionId(Math.floor(Math.random() * 960));
  };

  const isHumanTurn = game.status === "running" && game.turn !== game.bot_color;

  return (
    <div className="play-layout">
      {/* BOARD COLUMN */}
      <div className="board-column">
        {/* TOP CLOCK (Opponent / Bot) */}
        <div className={`clock ${game.status === "running" && game.turn === game.bot_color ? "active" : ""}`}>
          <div className="clock-label">
            <span>{game.bot_color === "white" ? "⚪ Bot (Trắng)" : "⚫ Bot (Đen)"}</span>
            <small className="bot-tag">
              {bots.find((b) => b.id === selectedBotId)?.name || "Bot cờ vua"}
            </small>
          </div>
          <strong>{formatClock(game.bot_color === "white" ? game.white_ms : game.black_ms)}</strong>
        </div>

        {/* CHESSBOARD */}
        <div className="board-wrap">
          <Chessboard
            position={game.fen || "start"}
            boardOrientation={game.bot_color === "white" ? "black" : "white"}
            onPieceDrop={handlePieceDrop}
            onSquareClick={handleSquareClick}
            customSquareStyles={customSquareStyles}
            arePiecesDraggable={isHumanTurn}
          />
        </div>

        {/* BOTTOM CLOCK (Human) */}
        <div className={`clock ${game.status === "running" && game.turn !== game.bot_color ? "active" : ""}`}>
          <div className="clock-label">
            <span>{game.bot_color === "white" ? "⚫ Bạn (Đen)" : "⚪ Bạn (Trắng)"}</span>
            {isHumanTurn && <small className="turn-tag">Đến lượt bạn đi</small>}
          </div>
          <strong>{formatClock(game.bot_color === "white" ? game.black_ms : game.white_ms)}</strong>
        </div>
      </div>

      {/* SIDEBAR PANEL */}
      <aside className="play-sidebar">
        {/* GAME CONTROLS PANEL */}
        <div className="panel config">
          <h3>Cài Đặt Ván Đấu</h3>

          <div className="form-group">
            <label>Chọn bot đối thủ:</label>
            <select
              value={selectedBotId}
              disabled={game.status === "running"}
              onChange={(e) => onSelectBotId(e.target.value)}
            >
              {bots.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Quân cờ của Bot:</label>
            <div className="radio-group">
              <label className="radio-label">
                <input
                  type="radio"
                  name="botcolor"
                  value="black"
                  checked={botColor === "black"}
                  disabled={game.status === "running"}
                  onChange={() => setBotColor("black")}
                />
                Bot cầm Đen (Bạn đi trước)
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="botcolor"
                  value="white"
                  checked={botColor === "white"}
                  disabled={game.status === "running"}
                  onChange={() => setBotColor("white")}
                />
                Bot cầm Trắng (Bot đi trước)
              </label>
            </div>
          </div>

          <div className="form-group">
            <label>Thời gian ván đấu:</label>
            <select
              value={timeSeconds}
              disabled={game.status === "running"}
              onChange={(e) => setTimeSeconds(e.target.value)}
            >
              <option value="60">1 Phút (Bullet chớp nhoáng)</option>
              <option value="180">3 Phút (Blitz nhanh)</option>
              <option value="240">4 Phút (Mặc định)</option>
              <option value="300">5 Phút (Blitz chuẩn)</option>
              <option value="600">10 Phút (Rapid)</option>
            </select>
          </div>

          <div className="form-group">
            <div className="flex-between">
              <label>Thế cờ xuất phát Chess960 (0 - 959):</label>
              <button
                type="button"
                className="btn-link"
                disabled={game.status === "running"}
                onClick={handleRandomPosition}
              >
                🎲 Ngẫu nhiên
              </button>
            </div>
            <input
              type="number"
              min="0"
              max="959"
              value={positionId}
              disabled={game.status === "running"}
              onChange={(e) => setPositionId(Number(e.target.value))}
            />
            <small className="input-hint">#518 là thế cờ vua tiêu chuẩn (Classical Chess).</small>
          </div>

          <div className="game-action-buttons">
            {game.status === "running" ? (
              <button className="btn-danger btn-full" onClick={onStopGame}>
                ⏹ Dừng / Đầu Hàng
              </button>
            ) : (
              <button
                className="btn-primary btn-full btn-large"
                onClick={() =>
                  onStartGame(botColor, Number(timeSeconds), positionId, selectedBotId)
                }
              >
                ▶ Bắt Đầu Ván Đấu
              </button>
            )}
          </div>
        </div>

        {/* STATUS & OUTCOME PANEL */}
        {game.result && (
          <div className={`panel outcome-card outcome-${game.result === "1/2-1/2" ? "draw" : "win"}`}>
            <h3>Kết Quả: {game.result}</h3>
            <p>
              {game.termination === "timeout"
                ? "Thua do hết giờ (Timeout)"
                : game.termination === "checkmate"
                ? "Chiếu bí (Checkmate)"
                : `Kết thúc: ${game.termination}`}
            </p>
          </div>
        )}

        {/* MOVE HISTORY PANEL */}
        <div className="panel history-panel">
          <h3>Biên Bản Nước Đi ({game.history.length} nước)</h3>
          <div className="history-list">
            {game.history.length === 0 ? (
              <p className="text-muted">Chưa có nước đi nào.</p>
            ) : (
              <div className="moves-grid">
                {game.history.map((h, i) => (
                  <div key={i} className="move-item">
                    <span className="move-ply">{Math.floor(i / 2) + 1}{i % 2 === 0 ? "." : "..."}</span>
                    <strong className="move-san">{h.san}</strong>
                    {h.elapsed_ms !== undefined && (
                      <span className="move-time">({(h.elapsed_ms / 1000).toFixed(1)}s)</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* PROMOTION MODAL */}
      {promotionChoices.length > 0 && (
        <div className="modal-backdrop">
          <div className="modal-content modal-sm">
            <h3>Chọn Quân Phong Cấp</h3>
            <div className="promotion-button-grid">
              {promotionChoices.map((p) => (
                <button
                  key={p.token}
                  className="btn-promotion"
                  onClick={() => {
                    onMove(p.token);
                    setPromotionChoices([]);
                    setSelectedSquare(null);
                  }}
                >
                  {p.promotion === "queen" && "♕ Hậu (Queen)"}
                  {p.promotion === "rook" && "♖ Xe (Rook)"}
                  {p.promotion === "bishop" && "♗ Tượng (Bishop)"}
                  {p.promotion === "knight" && "♘ Mã (Knight)"}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

