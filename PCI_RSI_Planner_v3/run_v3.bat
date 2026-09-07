@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  TürkTelekom PCI/RSI Planner  --  v3
echo ============================================================
echo.

rem Bu klasordeki sanal ortami kullan (setup.bat olusturur),
rem yoksa sistem python'una dus.
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Uygulama baslatiliyor (v3)...
echo Tarayicinizda http://localhost:8502 adresini acin.
echo Kapatmak icin bu pencerede Ctrl+C'ye basin.
echo.

"%PY%" -m streamlit run app.py
pause
