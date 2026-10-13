import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_HF_HOME = DEFAULT_CACHE_DIR / "huggingface"
DEFAULT_PADDLEOCR_DIR = DEFAULT_CACHE_DIR / "paddleocr"
DEFAULT_TORCH_HOME = DEFAULT_CACHE_DIR / "torch"


class Settings(BaseSettings):
    database_url: str
    jwt_secret_key: str
    jwt_expire_minutes: int = 1440
    upload_dir: Path = Path("./uploads")
    max_upload_size_mb: int = 20
    max_pdf_pages: int = 20
    frontend_origin: str = "http://localhost:3000"

    # Local LLM and Hardware configuration
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    llm_num_ctx: int = 2048
    llm_max_output_tokens: int = 768
    llm_gpu_required: bool = False
    llm_api_base: str = "http://localhost:11434"
    summary_fallback_enabled: bool = True

    # Translation configuration
    hf_token: str | None = None
    indictrans_model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M"
    indictrans_device: str = "auto"  # "auto", "cuda", "cpu"

    # Project-local model & cache directories
    project_root: Path = PROJECT_ROOT
    cache_dir: Path = DEFAULT_CACHE_DIR
    hf_home: Path = DEFAULT_HF_HOME
    paddleocr_dir: Path = DEFAULT_PADDLEOCR_DIR
    torch_home: Path = DEFAULT_TORCH_HOME

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def resolved_hf_home(self) -> Path:
        p = self.hf_home
        if not p.is_absolute():
            if (PROJECT_ROOT / p).exists():
                return (PROJECT_ROOT / p).resolve()
            if (BACKEND_DIR / p).exists():
                return (BACKEND_DIR / p).resolve()
            return (PROJECT_ROOT / p).resolve()
        return p

    @property
    def resolved_paddleocr_dir(self) -> Path:
        p = self.paddleocr_dir
        if not p.is_absolute():
            if (PROJECT_ROOT / p).exists():
                return (PROJECT_ROOT / p).resolve()
            if (BACKEND_DIR / p).exists():
                return (BACKEND_DIR / p).resolve()
            return (PROJECT_ROOT / p).resolve()
        return p

    @property
    def resolved_torch_home(self) -> Path:
        p = self.torch_home
        if not p.is_absolute():
            if (PROJECT_ROOT / p).exists():
                return (PROJECT_ROOT / p).resolve()
            if (BACKEND_DIR / p).exists():
                return (BACKEND_DIR / p).resolve()
            return (PROJECT_ROOT / p).resolve()
        return p


def configure_cache_environment() -> None:
    """Configures project-local model cache directories across the process environment."""
    settings = get_settings()
    hf_path = str(settings.resolved_hf_home)
    torch_path = str(settings.resolved_torch_home)
    paddle_path = str(settings.resolved_paddleocr_dir)

    os.environ["HF_HOME"] = hf_path
    os.environ["TRANSFORMERS_CACHE"] = hf_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(settings.resolved_hf_home / "hub")
    os.environ["TORCH_HOME"] = torch_path
    os.environ["PADDLEOCR_BASE_DIR"] = paddle_path

    # Ensure cache directories exist
    settings.resolved_hf_home.mkdir(parents=True, exist_ok=True)
    settings.resolved_paddleocr_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_torch_home.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Export cache directories immediately
    hf_path = str(settings.resolved_hf_home)
    torch_path = str(settings.resolved_torch_home)
    paddle_path = str(settings.resolved_paddleocr_dir)

    os.environ.setdefault("HF_HOME", hf_path)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_path)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(settings.resolved_hf_home / "hub"))
    os.environ.setdefault("TORCH_HOME", torch_path)
    os.environ.setdefault("PADDLEOCR_BASE_DIR", paddle_path)
    return settings


def cleanup_user_cache_artifacts() -> None:
    """Removes any empty legacy directories created by libraries in user profile ~/.cache."""
    import shutil

    user_paddle = Path.home() / ".cache" / "paddle"
    if user_paddle.exists():
        try:
            shutil.rmtree(user_paddle, ignore_errors=True)
        except Exception:
            pass


import atexit

# Ensure environment variables are registered at module load
configure_cache_environment()
atexit.register(cleanup_user_cache_artifacts)
cleanup_user_cache_artifacts()
