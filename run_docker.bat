@echo off
setlocal
title MedLingua - Docker Orchestrator
set "ROOT=%~dp0"
pushd "%ROOT%"

chcp 65001 >nul 2>&1

echo ===============================================================================
echo            MedLingua - Full Stack Containerized Deployment
echo ===============================================================================
echo.
echo This orchestrator will build and run all 4 components inside Docker:
echo   [1/4] PostgreSQL 16 Database
echo   [2/4] Ollama Container + Llama 3.2 3B Clinical LLM (~2.0 GB model download)
echo   [3/4] FastAPI Backend + PyTorch and Neural Translation Pipeline
echo   [4/4] Next.js 15 Web Frontend
echo.
echo -------------------------------------------------------------------------------
echo  [DOWNLOAD AND DISK NOTICE]:
echo  - First-time startup downloads full AI container runtimes (~10 to 12 GB total).
echo  - Download time depends on your internet speed (typically 5 to 15 minutes).
echo  - All models and database records will be permanently cached in Docker volumes.
echo  - Subsequent runs will start in under 10 seconds.
echo  - To completely remove everything later, run: clean_docker.bat
echo -------------------------------------------------------------------------------
echo.

where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not found in PATH.
    echo Please install and start Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    popd
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running.
    echo Please start Docker Desktop and wait for the engine to initialize.
    pause
    popd
    exit /b 1
)

echo Starting Docker Compose build and execution...
echo.
docker compose up --build

popd
