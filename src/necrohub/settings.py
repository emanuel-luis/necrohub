from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from NECROHUB_* environment variables when instantiated."""

    model_config = SettingsConfigDict(env_prefix="NECROHUB_", extra="ignore")

    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
