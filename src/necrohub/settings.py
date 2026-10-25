from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from NECROHUB_* environment variables when instantiated."""

    model_config = SettingsConfigDict(env_prefix="NECROHUB_", extra="ignore")

    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


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
