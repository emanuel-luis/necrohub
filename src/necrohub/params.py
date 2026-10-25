"""Typed sections of params.yaml. Each phase adds its own section; unknown sections are ignored."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from necrohub.preprocess.normalize import DEFAULT_STEPS


class LlmParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["ollama", "gemini"] = "ollama"
    model: str = "qwen3:4b-instruct-2507-q4_K_M"
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_attempts: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)


class GenerateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_cases: int = Field(ge=1)
    realizer: Literal["template", "llm"]
    holdout_templates: list[str]
    holdout_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    max_reject_fraction: float = Field(ge=0.0, le=1.0)
    pdf_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    llm: LlmParams = Field(default_factory=LlmParams)


class IngestParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # a case with a PDF in formats is read from the PDF, otherwise from its txt
    formats: list[Literal["txt", "pdf"]] = ["txt", "pdf"]  # pydantic copies list defaults
    max_pdf_length_drift: float = Field(default=0.15, ge=0.0)


class HfNerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class PreprocessParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalize: list[str] = Field(default_factory=lambda: list(DEFAULT_STEPS))
    deidentifiers: list[Literal["regex", "hf_ner"]] = ["regex"]
    hf_ner: HfNerParams = Field(default_factory=HfNerParams)

    @model_validator(mode="after")
    def _hf_ner_needs_a_model(self) -> Self:
        if "hf_ner" in self.deidentifiers and self.hf_ner.model is None:
            raise ValueError(
                "preprocess.deidentifiers has hf_ner but preprocess.hf_ner.model is null"
            )
        return self


class Params(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed: int
    generate: GenerateParams
    ingest: IngestParams = Field(default_factory=IngestParams)
    preprocess: PreprocessParams = Field(default_factory=PreprocessParams)


def load_params(path: Path) -> Params:
    return Params.model_validate(yaml.safe_load(path.read_text("utf-8")))
