@echo off
setlocal
set "ROOT=%~dp0"
pushd "%ROOT%"

where docker >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop and the Docker CLI are required.
    goto :failed
)
where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher is required. Install Python 3.11 and enable the launcher.
    goto :failed
)
where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js and npm are required.
    goto :failed
)
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 was not found. Install Python 3.11 and try again.
    goto :failed
)
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is installed but its engine is not running. Start Docker Desktop and retry.
    goto :failed
)

if not exist "backend\.venv\Scripts\python.exe" (
    echo Creating the backend virtual environment...
    py -3.11 -m venv "backend\.venv"
    if errorlevel 1 (
        echo ERROR: Could not create the backend virtual environment.
        goto :failed
    )
)
set "PYTHON=%ROOT%backend\.venv\Scripts\python.exe"

if not exist "backend\.env" (
    if not exist "backend\.env.example" (
        echo ERROR: backend\.env.example is missing.
        goto :failed
    )
    copy "backend\.env.example" "backend\.env" >nul
    if errorlevel 1 (
        echo ERROR: Could not create backend\.env.
        goto :failed
    )
    echo Created backend\.env from the example. Review its local settings if needed.
)
if not exist "frontend\.env.local" (
    if not exist "frontend\.env.local.example" (
        echo ERROR: frontend\.env.local.example is missing.
        goto :failed
    )
    copy "frontend\.env.local.example" "frontend\.env.local" >nul
    if errorlevel 1 (
        echo ERROR: Could not create frontend\.env.local.
        goto :failed
    )
)

echo Installing backend dependencies...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Could not upgrade pip.
    goto :failed
)
"%PYTHON%" -m pip install -r "backend\requirements.txt"
if errorlevel 1 (
    echo ERROR: Backend dependency installation failed.
    goto :failed
)

echo Starting PostgreSQL with Docker Compose...
docker compose up -d postgres
if errorlevel 1 (
    echo ERROR: Could not start PostgreSQL.
    goto :failed
)

echo Waiting for PostgreSQL...
set /a ATTEMPTS=0
:wait_for_postgres
docker compose exec -T postgres pg_isready -U medlingua -d medlingua >nul 2>&1
if not errorlevel 1 goto postgres_ready
set /a ATTEMPTS+=1
if %ATTEMPTS% GEQ 30 goto postgres_timeout
timeout /t 2 /nobreak >nul
goto wait_for_postgres

:postgres_timeout
echo ERROR: PostgreSQL did not become ready within 60 seconds.
goto :failed

:postgres_ready
echo Applying database migrations...
pushd "backend"
"%PYTHON%" -m alembic upgrade head
if errorlevel 1 (
    popd
    echo ERROR: Database migration failed.
    goto :failed
)

echo Checking/downloading the local translation model...
"%PYTHON%" -m app.download_translation_model
if errorlevel 1 (
    popd
    echo ERROR: Translation model setup failed. Check your internet connection and disk space.
    goto :failed
)
popd

echo Installing frontend dependencies...
pushd "frontend"
call npm ci
if errorlevel 1 (
    popd
    echo ERROR: Frontend dependency installation failed.
    goto :failed
)
popd

echo Starting MedLingua backend and frontend...
start "MedLingua Backend" /D "%ROOT%backend" cmd /k ""%PYTHON%" -m uvicorn app.main:app --reload --port 8000"
start "MedLingua Frontend" /D "%ROOT%frontend" cmd /k "npm run dev"
echo.
echo MedLingua is starting:
echo   Frontend: http://localhost:3000
echo   API docs: http://localhost:8000/docs
echo Keep both server windows open while using the app.
popd
exit /b 0

:failed
echo.
echo Setup did not finish. Fix the error above and run this file again.
pause
popd
exit /b 1
