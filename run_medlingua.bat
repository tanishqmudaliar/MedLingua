@echo off
setlocal EnableDelayedExpansion
title MedLingua Orchestrator - Hardware-Optimized Clinical AI
set "ROOT=%~dp0"
pushd "%ROOT%"

:: Enable UTF-8 codepage for clean UI rendering
chcp 65001 >nul 2>&1

echo ===============================================================================
echo            MedLingua - Clinical NLP, Translation and AI System
echo       Hardware-Optimized Orchestrator (Windows 11 / NVIDIA GTX 1650)
echo ===============================================================================
echo.

:: -----------------------------------------------------------------------------
:: Step 1: Core System Prerequisite Checks
:: -----------------------------------------------------------------------------
echo [1/7] Inspecting system prerequisites and hardware environment...

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
echo  - Docker Engine: Active and responsive.

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
echo [ERROR] Python 3.11 (64-bit) is required for PyTorch CUDA and Transformers compatibility.
echo Please install Python 3.11 64-bit from https://www.python.org/downloads/
echo Ensure 'Add Python to PATH' is checked during installation.
goto :failed

:python_found
echo  - Python Runtime: Verified Python 3.11 64-bit.

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
echo  - Node.js and npm: Detected and ready.

:: 1.4 Dedicated GPU and NVIDIA Driver Check
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name^,memory.total^,driver_version --format^=csv^,noheader 2^>nul') do (
        echo  - Dedicated GPU: %%g
    )
) else (
    echo  - Dedicated GPU: nvidia-smi not found. CPU inference fallback will be utilized.
)

:: -----------------------------------------------------------------------------
:: Step 2: Virtual Environment, Configuration and Dependencies
:: -----------------------------------------------------------------------------
echo.
echo [2/7] Verifying Python virtual environment and dependencies...

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
        echo Created backend\.env from example template.
    ) else (
        echo [ERROR] backend\.env.example is missing.
        goto :failed
    )
)

if not exist "frontend\.env.local" (
    if exist "frontend\.env.local.example" (
        copy "frontend\.env.local.example" "frontend\.env.local" >nul
        echo Created frontend\.env.local from example template.
    )
)

:: Verify core packages
echo Verifying backend package dependencies...
"%PYTHON%" -c "import sys; sys.path.insert(0, 'backend'); import fastapi, uvicorn, sqlalchemy, asyncpg, alembic, torch, transformers, IndicTransToolkit" >nul 2>&1
if errorlevel 1 (
    echo Installing / updating backend dependencies from backend\requirements.txt...
    "%PYTHON%" -m pip install --upgrade pip
    "%PYTHON%" -m pip install -r "backend\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Backend dependency installation failed.
        goto :failed
    )
) else (
    echo  - Backend packages: Verified and up to date.
)

:: Verify PyTorch CUDA Acceleration
"%PYTHON%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 (
    echo  - PyTorch CUDA: Active and accelerated on NVIDIA hardware.
) else (
    echo  - PyTorch CUDA: Running in CPU mode.
)

:: -----------------------------------------------------------------------------
:: Step 3: Local AI Inference Runtime (Ollama)
:: -----------------------------------------------------------------------------
echo.
echo [3/7] Verifying local LLM engine (Ollama + Llama 3.2 3B)...

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
        timeout /t 4 /nobreak >nul
    ) else (
        echo [WARNING] Ollama HTTP service is not responding on http://localhost:11434.
        echo Please ensure Ollama is installed from https://ollama.ai and running.
    )
)

:: Verify Llama 3.2 3B model availability in Ollama
"%PYTHON%" -c "import urllib.request, json; data = json.loads(urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3).read()); assert any('llama3.2' in m['name'] for m in data.get('models', []))" >nul 2>&1
if errorlevel 1 (
    if not "%OLLAMA_CMD%"=="" (
        echo [PULLING] llama3.2:3b model not found in Ollama cache. Pulling now...
        "%OLLAMA_CMD%" pull llama3.2:3b
        if errorlevel 1 (
            echo [WARNING] Failed to pull llama3.2:3b. Extractive fallback will be used.
        ) else (
            echo  - Local LLM: llama3.2:3b cached successfully.
        )
    ) else (
        echo [NOTICE] llama3.2:3b not found. Ensure it is pulled using: ollama pull llama3.2:3b
    )
) else (
    echo  - Local LLM: llama3.2:3b is verified and ready.
)

:: -----------------------------------------------------------------------------
:: Step 4: Neural Translation Models (IndicTrans2 & M2M100)
:: -----------------------------------------------------------------------------
echo.
echo [4/7] Verifying neural translation models and Hugging Face cache...

pushd "backend"
"%PYTHON%" -m app.download_translation_model --check-only
popd

:: If user passes --download-indictrans or wants to download now
if "%~1"=="--download-indictrans" (
    echo Downloading IndicTrans2 weights per command line argument...
    pushd "backend"
    "%PYTHON%" -m app.download_translation_model --model indictrans2
    popd
)

:: -----------------------------------------------------------------------------
:: Step 5: PostgreSQL Database Container and Migrations
:: -----------------------------------------------------------------------------
echo.
echo [5/7] Starting PostgreSQL database container and running migrations...

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
echo  - PostgreSQL database is ready and accepting queries.

echo Applying Alembic migrations...
pushd "backend"
"%PYTHON%" -m alembic upgrade head
if errorlevel 1 (
    popd
    echo [ERROR] Alembic database migration failed.
    goto :failed
)
popd
echo  - Database schema is fully migrated (users, documents, translation_jobs, conversations, chat_messages).

:: -----------------------------------------------------------------------------
:: Step 6: Frontend Dependencies and Setup
:: -----------------------------------------------------------------------------
echo.
echo [6/7] Verifying Next.js frontend dependencies...

if not exist "frontend\node_modules" (
    echo Installing Next.js dependencies via npm install...
    pushd "frontend"
    call npm install
    if errorlevel 1 (
        popd
        echo [ERROR] Frontend dependency installation failed.
        goto :failed
    )
    popd
) else (
    echo  - Frontend dependencies: Verified and present.
)

:: -----------------------------------------------------------------------------
:: Step 7: System Preflight Diagnostics & Service Launch
:: -----------------------------------------------------------------------------
echo.
echo [7/7] Running automated preflight diagnostics...

pushd "backend"
"%PYTHON%" -m app.preflight
if errorlevel 1 (
    popd
    echo [ERROR] Preflight audit encountered critical errors.
    goto :failed
)
popd

echo.
echo Launching MedLingua services...

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
echo   [Shortcuts ^& Tools]
echo   - Preflight Diagnostic:   cd backend ^& .venv\Scripts\python -m app.preflight
echo   - Download Translation:   cd backend ^& .venv\Scripts\python -m app.download_translation_model
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
