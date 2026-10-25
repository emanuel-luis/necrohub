"""Typed sections of params.yaml. Each phase adds its own section; unknown sections are ignored."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LlmParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["ollama", "gemini"] = "ollama"
    model: str = "qwen3:4b-instruct-2507-q4_K_M"
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_attempts: int = Field(default=3, ge=1)


class GenerateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_cases: int = Field(ge=1)
    realizer: Literal["template", "llm"]
    holdout_templates: list[str]
    holdout_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    max_reject_fraction: float = Field(ge=0.0, le=1.0)
    pdf_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    llm: LlmParams = Field(default_factory=LlmParams)


class Params(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed: int
    generate: GenerateParams


def load_params(path: Path) -> Params:
    return Params.model_validate(yaml.safe_load(path.read_text("utf-8")))
