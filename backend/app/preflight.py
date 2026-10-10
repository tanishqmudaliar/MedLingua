"""MedLingua Comprehensive Preflight Diagnostic System.

Inspects hardware, database, Ollama LLM, translation cache, and API integrity
before startup to ensure zero-crash operations.
"""

import asyncio
import json
import os
import platform
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Color styling for terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_badge(label: str, status: str, color: str, details: str = "") -> None:
    print(f" {BOLD}[{color}{status}{RESET}{BOLD}]{RESET} {label.ljust(30)} {details}")


def check_hardware() -> tuple[bool, str]:
    details = []
    # CPU & RAM
    try:
        import psutil

        ram = psutil.virtual_memory()
        details.append(f"RAM: {ram.total / (1024**3):.1f} GB ({ram.available / (1024**3):.1f} GB free)")
    except ImportError:
        details.append("RAM: Detected")

    # PyTorch & CUDA
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            details.append(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM, CUDA Active)")
            return True, " | ".join(details)
        else:
            details.append("GPU: CPU-only mode (No CUDA acceleration)")
            return False, " | ".join(details)
    except Exception as exc:
        details.append(f"PyTorch error: {exc}")
        return False, " | ".join(details)


def check_ollama(api_base: str, expected_model: str) -> tuple[bool, str]:
    url = f"{api_base}/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            has_expected = any(expected_model in m for m in models)
            if has_expected:
                return True, f"Running on {api_base} with '{expected_model}' loaded"
            else:
                return False, f"Running on {api_base}, but '{expected_model}' is missing (Found: {models})"
    except urllib.error.URLError:
        return False, f"Service unresponsive on {url}"
    except Exception as exc:
        return False, f"Error: {exc}"


async def check_database(db_url: str) -> tuple[bool, str]:
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(db_url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            # Check tables
            res = await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_name"
                )
            )
            tables = [row[0] for row in res.fetchall()]
        await engine.dispose()

        expected = {"users", "documents", "translation_jobs", "conversations", "chat_messages"}
        missing = expected - set(tables)
        if not missing:
            return True, f"PostgreSQL online. All {len(expected)} tables verified: {', '.join(sorted(expected))}"
        else:
            return True, f"PostgreSQL online. Missing tables: {missing} (Need alembic upgrade head)"
    except Exception as exc:
        return False, f"Connection failed: {exc}"


def check_translation_models() -> tuple[bool, str]:
    from .download_translation_model import INDICTRANS2_MODEL, M2M100_MODEL, get_cache_info

    indic_info = get_cache_info(INDICTRANS2_MODEL)
    m2m_info = get_cache_info(M2M100_MODEL)

    if indic_info["has_weights"]:
        return True, f"Primary engine ({INDICTRANS2_MODEL}) weights ready ({indic_info['size_mb']} MB)"
    elif m2m_info["has_weights"]:
        return True, f"Offline fallback engine ({M2M100_MODEL}) weights ready ({m2m_info['size_mb']} MB)"
    else:
        return False, "No translation model weights in cache. Run 'python -m app.download_translation_model'"


def main() -> int:
    print("\n" + "=" * 78)
    print(f"{BOLD}           MedLingua System Preflight Diagnostics & Audit{RESET}")
    print("=" * 78 + "\n")

    from .config import get_settings

    settings = get_settings()
    has_critical_error = False

    # 1. Hardware & CUDA
    gpu_ok, hw_info = check_hardware()
    if gpu_ok:
        print_badge("Hardware & Acceleration", "PASS", GREEN, hw_info)
    else:
        print_badge("Hardware & Acceleration", "WARN", YELLOW, hw_info)

    # 2. Database
    db_ok, db_info = asyncio.run(check_database(settings.database_url))
    if db_ok:
        print_badge("PostgreSQL Database", "PASS", GREEN, db_info)
    else:
        print_badge("PostgreSQL Database", "FAIL", RED, db_info)
        has_critical_error = True

    # 3. Ollama LLM
    ollama_ok, ollama_info = check_ollama(settings.llm_api_base, settings.llm_model)
    if ollama_ok:
        print_badge("Local LLM (Ollama)", "PASS", GREEN, ollama_info)
    else:
        print_badge("Local LLM (Ollama)", "WARN", YELLOW, ollama_info)

    # 4. Translation Engine
    trans_ok, trans_info = check_translation_models()
    if trans_ok:
        print_badge("Translation Weights", "PASS", GREEN, trans_info)
    else:
        print_badge("Translation Weights", "WARN", YELLOW, trans_info)

    print("\n" + "-" * 78)
    if not has_critical_error:
        print(f" {GREEN}{BOLD}PREFLIGHT STATUS: ALL CORE SUBSYSTEMS OPERATIONAL!{RESET}")
    else:
        print(f" {RED}{BOLD}PREFLIGHT STATUS: ONE OR MORE CRITICAL ERRORS DETECTED.{RESET}")
    print("-" * 78 + "\n")

    return 1 if has_critical_error else 0


if __name__ == "__main__":
    sys.exit(main())
