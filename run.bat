@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  TürkTelekom PCI/RSI Planner
echo ============================================================
echo.

if not exist .venv\Scripts\python.exe (
    echo [HATA] Sanal ortam bulunamadi! Once setup.bat calistirin.
    pause
    exit /b 1
)

echo Uygulama baslatiliyor...
echo Tarayicinizda http://localhost:8501 adresini acin.
echo Kapatmak icin bu pencerede Ctrl+C'ye basin.
echo.

.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
pause
