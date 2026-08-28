@echo off
chcp 65001 >nul
echo ============================================================
echo  TürkTelekom PCI/RSI Planner - Otomatik Kurulum (Offline)
echo  Python 3.12 veya 3.14 — Windows x64
echo ============================================================
echo.

REM Bat dosyasinin bulundugu klasore gec
cd /d "%~dp0"

REM Python kontrolu — PATH'te veya standart konumlarda ariyoruz
set PYTHON_EXE=
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_EXE=python
    goto :check_version
)
REM Standart kurulum konumlari (wheels klasorunde 3.12 ve 3.14 icin paket var)
if exist "C:\Python312\python.exe" (
    set PYTHON_EXE=C:\Python312\python.exe
    goto :check_version
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :check_version
)
if exist "C:\Python314\python.exe" (
    set PYTHON_EXE=C:\Python314\python.exe
    goto :check_version
)
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe
    goto :check_version
)
echo [HATA] Python bulunamadi!
echo   Lutfen Python 3.12 (onerilen) veya 3.14 x64 yukleyin, ya da PATH'e ekleyin.
echo   Beklenen konumlar:
echo     - PATH'te python komutu
echo     - C:\Python312\python.exe
echo     - %LOCALAPPDATA%\Programs\Python\Python312\python.exe
echo     - C:\Python314\python.exe
echo     - %LOCALAPPDATA%\Programs\Python\Python314\python.exe
pause
exit /b 1

:check_version
echo Python bulundu: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

REM Versiyon kontrolu. wheels klasorunde derlenmis paketler (numpy, pandas,
REM pyarrow, pillow...) Python surumune KILITLIDIR: cp312 ve cp314 var.
REM Baska bir surumde offline kurulum kacinilmaz olarak basarisiz olur --
REM internet olmadigi icin pip alternatif indiremez.
for /f "tokens=2 delims= " %%V in ('%PYTHON_EXE% --version 2^>^&1') do set PYVER=%%V
echo Tespit edilen versiyon: %PYVER%
echo %PYVER% | findstr /B "3.12 3.14" >nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [UYARI] Python %PYVER% tespit edildi.
    echo   Bu paketteki hazir kutuphaneler yalnizca 3.12 ve 3.14 icindir.
    echo   Bu surumle offline kurulum BUYUK OLASILIKLA BASARISIZ olacak.
    echo   Onerilen: Python 3.12 x64 kurun.
    echo.
    echo   Yine de denemek istiyor musunuz?
    choice /C EH /M "E=Evet, H=Hayir"
    if %ERRORLEVEL% EQU 2 exit /b 1
)

REM Eski venv varsa sil
if exist .venv (
    echo Eski sanal ortam siliniyor...
    rmdir /s /q .venv
)

REM Venv olustur
echo.
echo [1/3] Sanal ortam olusturuluyor...
%PYTHON_EXE% -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [HATA] Sanal ortam olusturulamadi!
    pause
    exit /b 1
)

REM pip'i guncelle (venv icindeki pip eski olabilir)
echo [2/3] pip guncelleniyor...
.venv\Scripts\python.exe -m pip install --no-index --find-links=wheels pip 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   pip wheel bulunamadi, mevcut pip ile devam ediliyor...
)

REM Kutuphaneleri offline wheel'lardan yukle
echo [3/3] Kutuphaneler offline yukleniyor...
.venv\Scripts\python.exe -m pip install --no-index --find-links=wheels -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [HATA] Bazi kutuphaneler yuklenemedi!
    echo   Olasiliklar:
    echo     1. wheels klasorunde eksik paket olabilir
    echo     2. Python versiyonu uyumsuz olabilir (3.12.x veya 3.14.x gerekli)
    echo   Detayli hata icin yukardaki ciktiyi inceleyin.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Kurulum tamamlandi!
echo.
echo  Uygulamayi calistirmak icin: run.bat
echo ============================================================
pause
