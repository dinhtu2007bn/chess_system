export type Color = "white" | "black";

export type LegalMove = {
  token: string;
  from: string;
  to: string;
  promotion: string | null;
  castling: "kingside" | "queenside" | null;
};

export type GameState = {
  status: string;
  result: string | null;
  termination: string | null;
  chess960_id: number;
  fen: string;
  turn: Color;
  bot_color: Color;
  white_ms: number;
  black_ms: number;
  legal_moves: LegalMove[];
  history: Array<{ ply: number; color: string; san: string; uci?: string; elapsed_ms?: number }>;
};

export type ManagedBot = {
  id: string;
  name: string;
  dir_name: string;
  dir_path: string | null;
  created_at: string;
  is_default: boolean;
  description: string;
  is_valid: boolean;
};

export type ChessPosition = {
  id: string;
  name: string;
  category: "opening" | "midgame" | "endgame";
  fen: string;
  chess960_id?: number;
  description: string;
  recommended_time_ms: number;
};

export type BenchmarkGame = {
  game_id: string;
  position_id: string;
  position_name: string;
  run_number: number;
  bot_color: Color;
  result: string;
  bot_outcome: "win" | "draw" | "loss";
  termination: string;
  is_timeout: boolean;
  moves_count: number;
  bot_moves_count: number;
  avg_move_time_ms: number;
  peak_move_time_ms: number;
  total_time_ms: number;
  final_fen?: string;
};

export type OptimizationMetrics = {
  wins: number;
  draws: number;
  losses: number;
  timeout_losses: number;
  checkmate_losses: number;
  win_rate: number;
  timeout_penalty: number;
  time_efficiency_factor: number;
  optimization_score: number;
  avg_move_time_ms: number;
  peak_move_time_ms: number;
  verdict: string;
};

export type LiveBoardState = {
  match_id?: number;
  slot_id?: number;
  position_id?: string;
  position_name?: string;
  white_bot_name?: string;
  black_bot_name?: string;
  run_number?: number;
  bot_color?: Color;
  fen: string;
  turn: Color;
  white_ms: number;
  black_ms: number;
  last_move?: string | null;
  last_move_san?: string | null;
  moves_count?: number;
  history?: Array<{ ply: number; san: string; uci: string; turn: string; elapsed_ms?: number }>;
  status?: "waiting" | "playing" | "finished" | "stopped";
  result?: string;
  termination?: string;
};

export type BenchmarkSession = {
  id: string;
  bot_id: string;
  bot_name: string;
  opponent_type: string;
  opponent_elo: number;
  opponent_bot_id?: string;
  opponent_name: string;
  time_mode: "global" | "per_position";
  game_time_seconds: number;
  increment_seconds: number;
  global_time_seconds?: number;
  per_position_times: Record<string, number>;
  runs_per_position: number;
  concurrency?: number;
  selected_positions: ChessPosition[];
  total_games: number;
  games_completed: number;
  current_game_index: number;
  current_position_name: string;
  current_position_fen?: string;
  current_run: number;
  status: "idle" | "running" | "stopped" | "finished" | "error";
  start_time: string;
  end_time?: string;
  error?: string;
  games: BenchmarkGame[];
  metrics: OptimizationMetrics;
  live_board?: LiveBoardState;
  live_boards?: Record<string, LiveBoardState>;
};

export type ArenaLeaderboardEntry = {
  bot_id: string;
  bot_name: string;
  elo: number;
  points: number;
  wins: number;
  draws: number;
  losses: number;
  timeouts: number;
  games_played: number;
  winrate: number;
};

export type ArenaMatch = {
  match_id: number;
  position_id: string;
  position_name: string;
  fen: string;
  white_bot_id: string;
  white_bot_name: string;
  black_bot_id: string;
  black_bot_name: string;
  status: "pending" | "playing" | "finished";
  result?: string;
  termination?: string;
  moves_count?: number;
  is_timeout?: boolean;
};

export type ArenaSession = {
  id: string;
  format: "round_robin" | "1v1";
  bots: ManagedBot[];
  positions: ChessPosition[];
  game_time_seconds: number;
  increment_seconds: number;
  total_matches: number;
  completed_matches: number;
  current_match: ArenaMatch | null;
  status: "idle" | "running" | "stopped" | "finished" | "error";
  matches: ArenaMatch[];
  elo_ratings: Record<string, number>;
  score_table: Record<string, { wins: number; draws: number; losses: number; points: number; timeouts: number }>;
  leaderboard: ArenaLeaderboardEntry[];
  error?: string;
  live_board?: LiveBoardState;
};
