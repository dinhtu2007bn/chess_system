@echo off
title Chess960 Arena & Benchmark
cd /d "D:\chess_system"
set "UV_CACHE_DIR=D:\.uv_cache"
set "PNPM_HOME=D:\.pnpm"
set "npm_config_cache=D:\.npm_cache"

echo ================================================================
echo   DANG KHOI DONG HE THONG CHESS960 ARENA & BENCHMARK
echo   Cong ket noi moi (chong cache cu): http://127.0.0.1:8888
echo ================================================================

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8888 ^| findstr LISTENING') do taskkill /f /t /pid %%a >nul 2>&1

start "" "http://127.0.0.1:8888"
".venv\Scripts\python.exe" -m uvicorn app.server:app --host 0.0.0.0 --port 8888

pause
