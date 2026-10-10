from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
