@echo off
title Chess960 Arena & Benchmark
cd /d "%~dp0"

echo ================================================================
echo   DANG KHOI DONG HE THONG CHESS960 ARENA & BENCHMARK
echo   Cong ket noi: http://127.0.0.1:8888
echo ================================================================

rem Kiem tra neu chua co virtualenv thi tu dong khoi tao
if not exist ".venv\Scripts\python.exe" (
    echo [.venv chua ton tai - Dang tu dong khoi tao moi truong...]
    where uv >nul 2>&1
    if %errorlevel% equ 0 (
        uv sync
    ) else (
        python -m venv .venv
        call ".venv\Scripts\activate.bat"
        python -m pip install -r requirements.txt
    )
)

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8888 ^| findstr LISTENING') do taskkill /f /t /pid %%a >nul 2>&1

start "" "http://127.0.0.1:8888"
".venv\Scripts\python.exe" -m uvicorn app.server:app --host 0.0.0.0 --port 8888

pause
