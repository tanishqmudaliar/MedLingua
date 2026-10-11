#!/usr/bin/env bash
set -e

echo "==============================================================================="
echo "           MedLingua - Complete Docker Zero-Trace Cleanup"
echo "==============================================================================="
echo ""
echo "This script will purge all Docker resources created for MedLingua:"
echo "  [x] Stop and remove all MedLingua containers (frontend, backend, postgres)"
echo "  [x] Delete all persistent Docker volumes (database data, model weights, uploads)"
echo "  [x] Remove all Docker images and build caches"
echo ""
echo "NOTE: Your host machine's local Python virtual environment (.venv), local Ollama,"
echo "      and local node_modules will NOT be affected."
echo ""
read -p "Proceed with complete Docker purge? [y/N]: " -r CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo ""
    echo "[INFO] Cleanup cancelled by user. Exiting."
    exit 0
fi

echo ""
echo "[1/4] Stopping and removing containers and persistent volumes..."
docker compose down -v --remove-orphans

echo ""
echo "[2/4] Removing built MedLingua images and base images..."
docker compose down --rmi all 2>/dev/null || true
docker rmi postgres:16 ollama/ollama:latest 2>/dev/null || true

echo ""
echo "[3/4] Purging project-specific named volumes if any remain..."
docker volume rm medlingua_postgres medlingua_hf_cache medlingua_uploads medlingua_ollama_models 2>/dev/null || true

echo ""
echo "[4/4] Pruning Docker build cache..."
docker builder prune -a -f
docker system prune -f

echo ""
echo "==============================================================================="
echo " [SUCCESS] All MedLingua Docker containers, models, and images have been removed!"
echo "==============================================================================="
echo ""
echo "==============================================================================="
echo " CRITICAL FINAL STEP: RECLAIM 100% DISK SPACE IN DOCKER DESKTOP"
echo "==============================================================================="
echo " Docker Desktop stores virtual disk data inside a virtual disk file (raw / vhdx)."
echo " To shrink and recover this disk space back to your host system:"
echo ""
echo "   1. Open Docker Desktop."
echo "   2. Click the Troubleshoot icon (the bug icon 🐞 in top-right header)."
echo "   3. Click 'Clean / Purge data' (or 'Compact disk')."
echo "   4. Confirm to clear storage."
echo "==============================================================================="
echo ""
