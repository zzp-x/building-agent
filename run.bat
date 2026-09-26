@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo   Yunxi Garden Bldg 3 - Project Assistant
echo ========================================
echo.

REM Prefer the dedicated conda env (ai_agent); fall back to PATH python
set "PY=D:\anaconda3\envs\ai_agent\python.exe"
if not exist "%PY%" set "PY=python"

REM 1. Check Python
"%PY%" --version
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add to PATH.
    pause
    exit /b 1
)
echo [1/4] Python check: OK

REM 2. Install dependencies
echo [2/4] Installing dependencies (flask / requests / jieba / langgraph)...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       done.

REM 3. Check .env (create from example if missing)
if not exist "app\.env" (
    if exist "app\.env.example" (
        echo [3/4] app\.env not found, copying from .env.example...
        copy /y "app\.env.example" "app\.env"
        echo       Please edit app\.env and fill in your LLM API Key.
        echo       Without a key the assistant runs in degraded mode.
    ) else (
        echo [3/4] app\.env not found, will run in degraded mode.
    )
) else (
    echo [3/4] app\.env found: OK
)

REM 4. Start server
echo [4/4] Starting server...
echo.
echo ========================================
echo   Server started. Open your browser:
echo   http://127.0.0.1:5000
echo   Press Ctrl+C to stop.
echo ========================================
echo.
cd app
"%PY%" app.py

endlocal
