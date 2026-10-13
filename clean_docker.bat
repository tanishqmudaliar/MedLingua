@echo off
setlocal
title MedLingua - Docker Zero-Trace Reset and Cleanup
set "ROOT=%~dp0"
pushd "%ROOT%"

chcp 65001 >nul 2>&1

echo ===============================================================================
echo            MedLingua - Complete Docker Zero-Trace Cleanup
echo ===============================================================================
echo.
echo This script will purge all Docker resources created for MedLingua:
echo   [x] Stop and remove all MedLingua containers (frontend, backend, postgres, ollama)
echo   [x] Delete all persistent Docker volumes (database data, model weights, uploads)
echo   [x] Remove all Docker images and build caches
echo.
echo NOTE: Your host machine's local Python virtual environment (.venv), local model cache (.cache),
echo       local Ollama, and local node_modules will NOT be affected.
echo.
set "CONFIRM="
set /p CONFIRM="Proceed with complete Docker purge? [y/N]: "
if /i not "%CONFIRM%"=="y" (
    echo.
    echo [INFO] Cleanup cancelled by user. Exiting.
    popd
    exit /b 0
)

echo.
echo [1/4] Stopping and removing containers and persistent volumes...
docker compose down -v --remove-orphans

echo.
echo [2/4] Removing built MedLingua images and base images...
docker compose down --rmi all 2>nul
docker rmi postgres:16 ollama/ollama:latest 2>nul

echo.
echo [3/4] Purging project-specific named volumes if any remain...
docker volume rm medlingua_postgres medlingua_hf_cache medlingua_uploads medlingua_ollama_models 2>nul

echo.
echo [4/4] Pruning Docker build cache and dangling layers...
docker builder prune -a -f
docker system prune -f

echo.
echo ===============================================================================
echo  [SUCCESS] All MedLingua Docker containers, models, and images are purged!
echo ===============================================================================
echo.
echo ===============================================================================
echo  CRITICAL FINAL STEP: RECLAIM 100%% DISK SPACE IN DOCKER DESKTOP
echo ===============================================================================
echo  Docker Desktop stores virtual disk data inside a virtual disk file (vhdx).
echo  When models or images are deleted inside Docker, Windows does NOT shrink
echo  this virtual disk file automatically until you clear storage in Docker:
echo.
echo    Step 1: Open the Docker Desktop window.
echo    Step 2: Click the Troubleshoot icon (the bug icon in top-right header).
echo    Step 3: Click "Clean / Purge data".
echo    Step 4: Check all boxes and click "Delete" (or click "Compact disk").
echo.
echo  This completely frees all disk gigabytes back to your physical C: drive.
echo ===============================================================================
echo.
pause
popd
