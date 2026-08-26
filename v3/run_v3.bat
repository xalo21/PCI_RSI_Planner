@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  TürkTelekom PCI/RSI Planner  --  v3
echo ============================================================
echo.

if not exist ..\.venv\Scripts\python.exe (
    echo [HATA] Sanal ortam bulunamadi! Once ust klasordeki setup.bat calistirin.
    pause
    exit /b 1
)

echo Uygulama baslatiliyor (v3)...
echo Tarayicinizda http://localhost:8502 adresini acin.
echo v2 ayni anda 8501'de calisabilir.
echo Kapatmak icin bu pencerede Ctrl+C'ye basin.
echo.

..\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8502
pause
