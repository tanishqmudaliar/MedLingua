#!/bin/sh
set -e

echo ""
echo "==============================================================================="
echo "                MedLingua Backend Container Initialization"
echo "==============================================================================="
echo ""

echo "[1/4] Checking PostgreSQL database connection..."
python - << 'EOF'
import os
import sys
import time
from sqlalchemy import create_engine, text

db_url = os.getenv("DATABASE_URL", "")
if not db_url:
    print("  [WARNING] DATABASE_URL not set.")
    sys.exit(0)

sync_url = db_url.replace("+asyncpg", "+psycopg")
max_retries = 30
for attempt in range(1, max_retries + 1):
    try:
        engine = create_engine(sync_url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"  -> PostgreSQL is responsive and ready (connected on attempt {attempt}).")
        sys.exit(0)
    except Exception:
        if attempt % 5 == 0:
            print(f"  -> Still waiting for PostgreSQL to initialize ({attempt}/{max_retries})...")
        time.sleep(1)

print("  [ERROR] PostgreSQL did not become ready in time.")
sys.exit(1)
EOF

echo ""
echo "[2/4] Applying database schema migrations (Alembic)..."
alembic upgrade head
echo "  -> Database tables and schema verified."

echo ""
echo "[3/4] Verifying Ollama container LLM service..."
python - << 'EOF'
import json
import os
import urllib.request

llm_base = os.getenv("LLM_API_BASE", "http://ollama:11434")
model_name = os.getenv("LLM_MODEL", "llama3.2:3b")

print(f"  -> Connecting to containerized Ollama at: {llm_base}")
try:
    req = urllib.request.Request(f"{llm_base}/api/tags", headers={"User-Agent": "MedLingua-Docker"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        found = any(model_name.split(":")[0] in m for m in models)
        if found:
            print(f"  -> SUCCESS: Ollama is running and model '{model_name}' is loaded!")
        else:
            print(f"  -> Ollama is active; 'ollama-puller' container is currently caching '{model_name}'.")
except Exception as e:
    print(f"  -> [NOTICE] Ollama service initializing at {llm_base}. Waiting for model puller.")
EOF

echo ""
echo "[4/4] Starting FastAPI production server with Uvicorn..."
echo "==============================================================================="
echo " MedLingua Backend is LIVE at http://0.0.0.0:8000"
echo " Interactive Swagger API Documentation: http://localhost:8000/docs"
echo " Web Frontend URL: http://localhost:3000"
echo "==============================================================================="
echo ""

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
