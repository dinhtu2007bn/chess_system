@echo off
title Chia Se Online - Chess960 Arena & Benchmark
cd /d "D:\chess_system"

echo ================================================================
echo   DANG TAO DUONG LINK ONLINE CHO BAN BE (CLOUDFLARE TUNNEL)
echo   Yeu cau: File CHAY_HET_THONG.bat phai dang duoc bat!
echo ================================================================
echo.
echo Hay tim dong link co dang:
echo    https://xxxx.trycloudflare.com
echo.
echo Copy duong link do gui cho ban be la ho xem va dung duoc ngay!
echo ================================================================
echo.

cloudflared.exe tunnel --url http://127.0.0.1:8888

pause
