@echo off
setlocal ENABLEDELAYEDEXPANSION
title Voice Agent Launcher

REM --- Go to project root (this script's directory) ---
cd /d "%~dp0"

echo =====================================================
echo ===           Starting Voice Agent Demo...         ===
echo =====================================================
echo.

REM --- Sanity checks ---
if not exist "app.py" (
  echo [ERROR] app.py not found in %cd%
  echo Please run this script in the project root.
  pause
  goto :END
)

REM --- Quick UI marker check (helps verify which index.html is served) ---
findstr /C:"(NEW UI)" "client\index.html" >nul 2>&1
if errorlevel 1 (
  echo [WARN] client\index.html does NOT contain the marker "(NEW UI)".
  echo        This is optional, but adding it helps verify the correct file is served.
  echo        Example: change H1 to "Voice Agent (NEW UI)"
  echo.
)

REM --- Pick Python (prefer venv) ---
set "PY_BIN="
set "USING_VENV="

if exist ".venv\Scripts\python.exe" (
  set "PY_BIN=.venv\Scripts\python.exe"
  set "USING_VENV=1"
) else (
  where python >nul 2>&1 && set "PY_BIN=python"
  if not defined PY_BIN (
    where py >nul 2>&1 && set "PY_BIN=py"
  )
)

if not defined PY_BIN (
  echo [ERROR] Python not found in PATH.
  echo Please install Python 3.10+ and/or create .venv first.
  pause
  goto :END
)

REM --- Define pip command bound to chosen Python ---
set "PIP_CMD=%PY_BIN% -m pip"

REM --- If using venv, try to activate for user comfort (optional) ---
if "%USING_VENV%"=="1" (
  if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
)

REM --- Check essential deps; install if missing ---
echo [INFO] Checking dependencies...
"%PY_BIN%" -c "import fastapi,uvicorn" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Dependencies missing. Installing from requirements.txt...
  %PIP_CMD% install --upgrade pip
  %PIP_CMD% install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo [ERROR] pip installation failed. Please check your network/permissions.
    pause
    goto :END
  )
  echo [INFO] Dependencies installed successfully.
)

REM --- Kill ports if already in use (prevents old UI being served) ---
echo [INFO] Ensuring ports 8000 and 8080 are free...

for %%P in (8000 8080) do (
  for /f "tokens=5" %%a in ('netstat -aon ^| findstr /R /C:":%%P .*LISTENING"') do (
    echo [INFO] Port %%P in use by PID %%a. Killing...
    taskkill /PID %%a /F >nul 2>&1
  )
)

REM --- Launch Backend (FastAPI + Uvicorn) ---
echo [INFO] Launching backend server...
start "Backend" cmd /k "%PY_BIN% -m uvicorn app:app --host 127.0.0.1 --port 8000"

REM --- Wait until backend is ready (poll /ping) ---
echo [INFO] Waiting for backend to be ready...
set "READY="
for /L %%i in (1,1,30) do (
  powershell -Command "try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/ping).StatusCode } catch { 0 }" | findstr "200" >nul
  if not errorlevel 1 (
    set "READY=1"
    goto :BACKEND_OK
  )
  timeout /t 1 >nul
)

:BACKEND_OK
if not defined READY (
  echo [WARN] Backend not ready after 30s. UI will open anyway; chat may fail until backend finishes startup.
)

REM --- Launch Frontend (static files on 8080) ---
echo [INFO] Launching frontend server on port 8080 from: %cd%
start "Frontend - python -m http.server" cmd /k "%PY_BIN% -m http.server 8080"

REM --- Small wait so frontend starts ---
timeout /t 1 >nul

REM --- Auto open browser to client page ---
set "TARGET_URL=http://127.0.0.1:8080/client/index.html"
echo [INFO] Opening %TARGET_URL%
start "" "%TARGET_URL%"

echo.
echo Voice Agent ready! If the browser didn't open, visit:
echo   %TARGET_URL%
echo.
pause


:END
endlocal
