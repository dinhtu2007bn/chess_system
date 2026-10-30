import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { ArenaView } from "./components/ArenaView";
import { BenchmarkView } from "./components/BenchmarkView";
import { BotManagerView } from "./components/BotManagerView";
import { HistoryView } from "./components/HistoryView";
import { PlayView } from "./components/PlayView";
import { ChessPosition, Color, GameState, ManagedBot } from "./types";
import "./styles.css";

const emptyGame: GameState = {
  status: "idle",
  result: null,
  termination: null,
  chess960_id: 518,
  fen: "",
  turn: "white",
  bot_color: "black",
  white_ms: 240000,
  black_ms: 240000,
  legal_moves: [],
  history: [],
};

export function App() {
  const [activeTab, setActiveTab] = useState<"benchmark" | "bots" | "arena" | "play" | "history">("benchmark");
  const [historyReplayFile, setHistoryReplayFile] = useState<string | null>(null);
  const [bots, setBots] = useState<ManagedBot[]>([]);
  const [positions, setPositions] = useState<ChessPosition[]>([]);
  const [selectedBotId, setSelectedBotId] = useState<string>("");
  const [game, setGame] = useState<GameState>(emptyGame);
  const [engineStatus, setEngineStatus] = useState<any>(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Fetch Bots
  const refreshBots = async () => {
    try {
      const res = await fetch("/api/bots");
      if (res.ok) {
        const data = await res.json();
        setBots(data);
        if (data.length > 0 && !selectedBotId) {
          setSelectedBotId(data[0].id);
        }
      }
    } catch (e) {
      // ignore
    }
  };

  // Fetch Positions
  const refreshPositions = async () => {
    try {
      const res = await fetch("/api/positions");
      if (res.ok) {
        const data = await res.json();
        setPositions(data);
      }
    } catch (e) {
      // ignore
    }
  };

  // Fetch Engine Status
  const refreshEngine = async () => {
    try {
      const res = await fetch("/api/engine/status");
      if (res.ok) {
        const data = await res.json();
        setEngineStatus(data);
      }
    } catch (e) {
      // ignore
    }
  };

  // Fetch Initial Game State
  const refreshGameState = async () => {
    try {
      const res = await fetch("/api/state");
      if (res.ok) {
        const data = await res.json();
        setGame(data);
      }
    } catch (e) {
      // ignore
    }
  };

  useEffect(() => {
    refreshBots();
    refreshPositions();
    refreshEngine();
    refreshGameState();
  }, []);

  const [benchmarkLiveEvent, setBenchmarkLiveEvent] = useState<any>(null);
  const [benchmarkSessionEvent, setBenchmarkSessionEvent] = useState<any>(null);
  const [arenaSessionEvent, setArenaSessionEvent] = useState<any>(null);
  const [arenaLiveEvent, setArenaLiveEvent] = useState<any>(null);

  // WebSocket Connection
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    let socket: WebSocket | null = null;
    let reconnectTimeout: any;

    const connect = () => {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        setWsConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === "benchmark_live_move") {
            setBenchmarkLiveEvent(payload);
          } else if (payload.type === "benchmark_update") {
            setBenchmarkSessionEvent(payload.session);
          } else if (payload.type === "arena_live_move") {
            setArenaLiveEvent(payload);
          } else if (payload.type === "arena_live_clock") {
            setArenaLiveEvent((prev: any) => ({
              ...(prev || {}),
              white_ms: payload.white_ms,
              black_ms: payload.black_ms,
              turn: payload.turn,
            }));
          } else if (payload.type === "arena_update") {
            setArenaSessionEvent(payload.session);
          } else if (payload.status !== undefined && payload.turn !== undefined) {
            setGame(payload);
          }
        } catch (e) {
          // ignore
        }
      };

      socket.onclose = () => {
        setWsConnected(false);
        reconnectTimeout = setTimeout(connect, 2000);
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      socket?.close();
    };
  }, []);

  // Play Actions
  const handleStartGame = async (
    botColor: Color,
    timeSec: number,
    positionId: number,
    botId?: string
  ) => {
    try {
      const res = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bot_color: botColor,
          time_seconds: timeSec,
          chess960_id: positionId,
          bot_id: botId || selectedBotId,
        }),
      });
      if (res.ok) {
        const state = await res.json();
        setGame(state);
      }
    } catch (e) {
      // ignore
    }
  };

  const handleStopGame = async () => {
    try {
      const res = await fetch("/api/stop", { method: "POST" });
      if (res.ok) {
        const state = await res.json();
        setGame(state);
      }
    } catch (e) {
      // ignore
    }
  };

  const handleMove = async (token: string) => {
    try {
      const res = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      if (res.ok) {
        const state = await res.json();
        setGame(state);
      }
    } catch (e) {
      // ignore
    }
  };

  return (
    <div className="app-shell">
      {/* GLOBAL TOP NAVIGATION */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-logo">♔</div>
          <div className="brand-text">
            <h1>Chess960 Arena & Benchmark</h1>
            <span className="brand-subtitle">Hệ Thống Đánh Giá & Quản Lý Bot Cờ Vua</span>
          </div>
        </div>

        {/* TABS */}
        <nav className="header-nav">
          <button
            className={`nav-tab ${activeTab === "benchmark" ? "active" : ""}`}
            onClick={() => setActiveTab("benchmark")}
          >
            ⚡ Benchmark (3 Runs/Thế Cờ)
          </button>
          <button
            className={`nav-tab ${activeTab === "bots" ? "active" : ""}`}
            onClick={() => setActiveTab("bots")}
          >
            🤖 Quản Lý Bot ({bots.length})
          </button>
          <button
            className={`nav-tab ${activeTab === "arena" ? "active" : ""}`}
            onClick={() => setActiveTab("arena")}
          >
            ⚔️ Đấu Trường Nội Bộ
          </button>
          <button
            className={`nav-tab ${activeTab === "play" ? "active" : ""}`}
            onClick={() => setActiveTab("play")}
          >
            ♟️ Đấu Với Bot
          </button>
          <button
            className={`nav-tab ${activeTab === "history" ? "active" : ""}`}
            onClick={() => setActiveTab("history")}
          >
            📜 Lịch Sử Ván Đấu
          </button>
        </nav>

        {/* STATUS INDICATORS */}
        <div className="header-status">
          {engineStatus?.is_installed && (
            <span className="status-badge badge-engine" title={engineStatus.path}>
              ✓ Stockfish 19 UCI
            </span>
          )}
          <span className={`status-badge ${wsConnected ? "badge-connected" : "badge-disconnected"}`}>
            {wsConnected ? "● Live Sync" : "○ Disconnected"}
          </span>
        </div>
      </header>

      {/* MAIN VIEW CONTENT */}
      <main className="app-content">
        {activeTab === "benchmark" && (
          <BenchmarkView
            bots={bots}
            positions={positions}
            selectedBotIdDefault={selectedBotId}
            onRefreshPositions={refreshPositions}
            benchmarkLiveEvent={benchmarkLiveEvent}
            benchmarkSessionEvent={benchmarkSessionEvent}
          />
        )}

        {activeTab === "bots" && (
          <BotManagerView
            bots={bots}
            onRefresh={refreshBots}
            onSelectForBenchmark={(id) => {
              setSelectedBotId(id);
              setActiveTab("benchmark");
            }}
            onSelectForPlay={(id) => {
              setSelectedBotId(id);
              setActiveTab("play");
            }}
          />
        )}

        {activeTab === "arena" && (
          <ArenaView
            bots={bots}
            positions={positions}
            arenaSessionEvent={arenaSessionEvent}
            arenaLiveEvent={arenaLiveEvent}
            onRefreshPositions={refreshPositions}
            onViewReplay={(filename) => {
              setHistoryReplayFile(filename);
              setActiveTab("history");
            }}
          />
        )}

        {activeTab === "play" && (
          <PlayView
            game={game}
            bots={bots}
            selectedBotId={selectedBotId}
            onSelectBotId={setSelectedBotId}
            onStartGame={handleStartGame}
            onStopGame={handleStopGame}
            onMove={handleMove}
          />
        )}

        {activeTab === "history" && (
          <HistoryView
            initialFilename={historyReplayFile}
            onClearInitialFilename={() => setHistoryReplayFile(null)}
          />
        )}
      </main>
    </div>
  );
}

const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
}
