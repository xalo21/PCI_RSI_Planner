@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  TürkTelekom PCI/RSI Planner  --  v3  (Kapali Ag / Kurum Ici)
echo ============================================================
echo.

rem Kapali ag modu: harita sekmesi devre disi birakilir.  Folium haritasi
rem Leaflet'i ve altlik karolari CDN'den ceker; internet yoksa iframe bos
rem beyaz bir kutu olur ve calismadigi belli bile olmaz.  Diger tum
rem sekmeler (analiz, planlama, raporlar, Excel, Nokia XML) tam calisir.
set "PCI_OFFLINE=1"

rem Ust klasordeki sanal ortami tercih et, yoksa sistem python'una dus.
set "PY=..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Erisim adresleri:
echo   Bu makineden       : http://localhost:8502
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /C:"IPv4"') do (
    for /f "tokens=* delims= " %%B in ("%%A") do echo   Agdaki makinelerden: http://%%B:8502
)
echo.
echo Ilk kurulumda 8502 portu icin guvenlik duvari kurali gerekir.
echo Yonetici komut isteminde bir kez calistirin:
echo.
echo   netsh advfirewall firewall add rule name="PCI RSI Planner" dir=in action=allow protocol=TCP localport=8502
echo.
echo Kapatmak icin bu pencerede Ctrl+C'ye basin.
echo.

"%PY%" -m streamlit run app.py
pause
