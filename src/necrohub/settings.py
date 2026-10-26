from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from NECROHUB_* (and the backends' own) environment variables when
    instantiated."""

    model_config = SettingsConfigDict(env_prefix="NECROHUB_", extra="ignore")

    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    ollama_host: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_HOST")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    database_url: SecretStr | None = Field(default=None, validation_alias="DATABASE_URL")
    store: Literal["memory", "pgvector"] = "memory"
    index_dir: Path = Path("data/serving/index")
    index_version: str | None = None  # default: the version in the index directory
    llm: Literal["extractive", "ollama", "gemini"] = "extractive"
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"  # NECROHUB_LLM_MODEL, ollama and gemini


def resolve_device(device: str = "auto") -> str:
    """Return cpu or cuda. auto follows NECROHUB_DEVICE, then CUDA availability (imports torch)."""
    if device == "auto":
        device = Settings().device
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unknown device: {device}")
    return device
