#!/usr/bin/env bash
set -e

echo "==============================================================================="
echo "           MedLingua - Full Stack Containerized Deployment"
echo "==============================================================================="
echo ""
echo "This orchestrator will build and run all 4 components inside Docker:"
echo "  [1/4] PostgreSQL 16 Database"
echo "  [2/4] Ollama Container + Llama 3.2 3B Clinical LLM (~2.0 GB model download)"
echo "  [3/4] FastAPI Backend + PyTorch & Neural Translation Pipeline"
echo "  [4/4] Next.js 15 Web Frontend"
echo ""
echo "-------------------------------------------------------------------------------"
echo " [DOWNLOAD & DISK NOTICE]:"
echo " - First-time startup downloads full AI container runtimes (~10 to 12 GB total)."
echo " - Download time depends on your internet speed (typically 5 to 15 minutes)."
echo " - All models and database records will be permanently cached in Docker volumes."
echo " - Subsequent runs will start in under 10 seconds!"
echo " - To completely remove everything later, run: ./clean_docker.sh"
echo "-------------------------------------------------------------------------------"
echo ""

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "[ERROR] Docker daemon is not running. Please start Docker."
    exit 1
fi

echo "Starting Docker Compose build and execution..."
echo ""
docker compose up --build
