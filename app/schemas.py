from __future__ import annotations

from pydantic import BaseModel, Field


class StartGameRequest(BaseModel):
    bot_color: str = Field(pattern="^(white|black)$")
    time_seconds: int = Field(default=240, ge=5, le=3600)
    chess960_id: int | None = Field(default=None, ge=0, le=959)
    bot_id: str | None = Field(default=None)


class MoveRequest(BaseModel):
    token: str = Field(min_length=4, max_length=8)


class BotRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class StartBenchmarkRequest(BaseModel):
    bot_id: str
    position_ids: list[str] = Field(min_length=1)
    opponent_type: str = Field(default="stockfish", pattern="^(stockfish|managed_bot)$")
    opponent_elo: int = Field(default=1700, ge=1000, le=2500)
    opponent_bot_id: str | None = None
    time_mode: str = Field(default="global", pattern="^(global|per_position)$")
    game_time_seconds: int = Field(default=60, ge=5, le=3600)
    increment_seconds: int = Field(default=1, ge=0, le=30)
    global_time_seconds: int | None = None
    per_position_times: dict[str, int] = Field(default_factory=dict)
    runs_per_position: int = Field(default=3, ge=1, le=10)
    concurrency: int = Field(default=3, ge=1, le=8)


class StartArenaRequest(BaseModel):
    bot_ids: list[str] = Field(min_length=2)
    position_ids: list[str] = Field(min_length=1)
    game_time_seconds: int = Field(default=60, ge=5, le=3600)
    increment_seconds: int = Field(default=1, ge=0, le=30)
    time_seconds: int | None = None
    format_type: str = Field(default="round_robin", pattern="^(round_robin|1v1)$")
