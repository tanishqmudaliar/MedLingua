@echo off
setlocal
title MedLingua Orchestrator
set "ROOT=%~dp0"
pushd "%ROOT%"

echo ===============================================================================
echo            MedLingua - Clinical NLP, Translation and AI System
echo            Hardware-Optimized Orchestrator (Windows 11 / GTX 1650)
echo ===============================================================================
echo.

:: -----------------------------------------------------------------------------
:: Step 1: Core System Prerequisite Checks
:: -----------------------------------------------------------------------------
echo [1/6] Checking system prerequisites...

:: 1.1 Docker Desktop / CLI Check
where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop or the Docker CLI is not installed or not in PATH.
    echo Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    goto :failed
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is installed, but the Docker daemon is not running.
    echo Please start Docker Desktop and wait for the engine to initialize before retrying.
    goto :failed
)
echo  - Docker engine: Active and responsive.

:: 1.2 Python 3.11 Check
set "SYS_PYTHON="
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "SYS_PYTHON=py -3.11"
    goto :python_found
)

python --version >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "SYS_PYTHON=python"
        goto :python_found
    )
)

:no_python
echo [ERROR] Python 3.11 is required.
echo Please install Python 3.11 64-bit from https://www.python.org/downloads/
echo Ensure Add Python to PATH is checked during installation.
goto :failed

:python_found
echo  - Python runtime: Verified Python 3.11.

:: 1.3 Node.js and npm Check
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js and npm are required for the Next.js frontend.
    echo Please install Node.js LTS from https://nodejs.org/
    goto :failed
)
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js binary was not found in PATH.
    goto :failed
)
echo  - Node.js and npm: Detected.

:: 1.4 GPU and NVIDIA Driver Inspection
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader 2^>nul') do (
        echo  - Dedicated GPU: %%g
    )
) else (
    echo  - Dedicated GPU: Not detected via nvidia-smi. CPU inference mode will be used.
)

:: -----------------------------------------------------------------------------
:: Step 2: Virtual Environment and Configuration Files
:: -----------------------------------------------------------------------------
echo.
echo [2/6] Configuring virtual environment and environment variables...

if not exist "backend\.venv\Scripts\python.exe" (
    echo Creating Python virtual environment in backend\.venv...
    %SYS_PYTHON% -m venv "backend\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        goto :failed
    )
)
set "PYTHON=%ROOT%backend\.venv\Scripts\python.exe"

if not exist "backend\.env" (
    if exist "backend\.env.example" (
        copy "backend\.env.example" "backend\.env" >nul
        echo Created backend\.env from example.
    ) else (
        echo [ERROR] backend\.env.example is missing.
        goto :failed
    )
)

if not exist "frontend\.env.local" (
    if exist "frontend\.env.local.example" (
        copy "frontend\.env.local.example" "frontend\.env.local" >nul
        echo Created frontend\.env.local from example.
    )
)

:: Ensure PyTorch and core requirements are installed
echo Verifying backend dependencies...
"%PYTHON%" -c "import sys; sys.path.insert(0, 'backend'); import fastapi, sqlalchemy, asyncpg, torch; from app import translation" >nul 2>&1
if errorlevel 1 (
    echo Installing backend dependencies from requirements.txt...
    "%PYTHON%" -m pip install --upgrade pip
    "%PYTHON%" -m pip install -r "backend\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Backend dependency installation failed.
        goto :failed
    )
) else (
    echo  - Backend packages: Verified and ready.
)

:: -----------------------------------------------------------------------------
:: Step 3: Ollama Runtime and LLM Model Verification
:: -----------------------------------------------------------------------------
echo.
echo [3/6] Verifying local AI inference engine (Ollama)...

:: Locate Ollama executable
set "OLLAMA_CMD="
where ollama >nul 2>&1
if not errorlevel 1 (
    set "OLLAMA_CMD=ollama"
) else if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_CMD=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
)

:: Check if Ollama service is responding
"%PYTHON%" -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)" >nul 2>&1
if errorlevel 1 (
    if not "%OLLAMA_CMD%"=="" (
        echo Starting Ollama background service...
        start "Ollama Service" /min "%OLLAMA_CMD%" serve
        timeout /t 3 /nobreak >nul
    ) else (
        echo [WARNING] Ollama HTTP service is not responding on http://localhost:11434.
        echo If using local LLM summarization, start Ollama before submitting requests.
    )
)

:: Verify Llama 3.2 3B model availability
"%PYTHON%" -c "import urllib.request, json; data = json.loads(urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3).read()); assert any('llama3.2' in m['name'] for m in data.get('models', []))" >nul 2>&1
if errorlevel 1 (
    if not "%OLLAMA_CMD%"=="" (
        echo [PULLING] llama3.2:3b model not found in Ollama cache. Pulling now...
        "%OLLAMA_CMD%" pull llama3.2:3b
        if errorlevel 1 (
            echo [WARNING] Failed to auto-pull llama3.2:3b. Extractive fallback will be used.
        ) else (
            echo  - llama3.2:3b: Successfully cached.
        )
    ) else (
        echo [NOTICE] llama3.2:3b not found. Ensure it is pulled using: ollama pull llama3.2:3b
    )
) else (
    echo  - Verified LLM model: llama3.2:3b is cached and ready.
)

:: -----------------------------------------------------------------------------
:: Step 4: PostgreSQL Database and Migrations
:: -----------------------------------------------------------------------------
echo.
echo [4/6] Initializing PostgreSQL database container...

docker compose up -d postgres
if errorlevel 1 (
    echo [ERROR] Could not start PostgreSQL container via Docker Compose.
    goto :failed
)

echo Waiting for PostgreSQL to accept connections...
set /a ATTEMPTS=0
:wait_for_postgres
docker compose exec -T postgres pg_isready -U medlingua -d medlingua >nul 2>&1
if not errorlevel 1 goto postgres_ready
set /a ATTEMPTS+=1
if %ATTEMPTS% GEQ 30 goto postgres_timeout
timeout /t 2 /nobreak >nul
goto wait_for_postgres

:postgres_timeout
echo [ERROR] PostgreSQL did not become ready within 60 seconds.
goto :failed

:postgres_ready
echo  - PostgreSQL database is ready.

echo Applying Alembic migrations...
pushd "backend"
"%PYTHON%" -m alembic upgrade head
if errorlevel 1 (
    popd
    echo [ERROR] Alembic database migration failed.
    goto :failed
)
popd
echo  - Database schema is up to date (Queue and Chat tables applied).

:: -----------------------------------------------------------------------------
:: Step 5: Frontend Dependencies
:: -----------------------------------------------------------------------------
echo.
echo [5/6] Verifying frontend dependencies...

if not exist "frontend\node_modules" (
    echo Installing Next.js frontend dependencies via npm install...
    pushd "frontend"
    call npm install
    if errorlevel 1 (
        popd
        echo [ERROR] Frontend dependency installation failed.
        goto :failed
    )
    popd
) else (
    echo  - Frontend dependencies: Already installed.
)

:: -----------------------------------------------------------------------------
:: Step 6: Launch Backend and Frontend Services
:: -----------------------------------------------------------------------------
echo.
echo [6/6] Launching MedLingua services...

echo Starting FastAPI Backend (with integrated serialized translation worker)...
start "MedLingua Backend (FastAPI + Worker)" /D "%ROOT%backend" cmd /k ""%PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo Starting Next.js Frontend...
start "MedLingua Frontend (Next.js)" /D "%ROOT%frontend" cmd /k "npm run dev"

echo.
echo ===============================================================================
echo                MedLingua is successfully up and running!
echo ===============================================================================
echo.
echo   * Web Application:        http://localhost:3000
echo   * API Documentation:      http://localhost:8000/docs
echo   * Hardware Diagnostics:   http://localhost:8000/system/hardware
echo.
echo   [Tip] Keep the Backend and Frontend terminal windows open while using the app.
echo   [Tip] To stop all services, close the terminal windows and run:
echo         docker compose stop
echo.
popd
exit /b 0

:failed
echo.
echo ===============================================================================
echo   Setup encountered an error. Review the error message above and try again.
echo ===============================================================================
pause
popd
exit /b 1
