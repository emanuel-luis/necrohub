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


class SplitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: float = Field(default=0.8, ge=0.0, le=1.0)
    validation: float = Field(default=0.1, ge=0.0, le=1.0)
    test: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> Self:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("dataset.splits must sum to 1")
        return self


class DatasetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    splits: SplitParams = Field(default_factory=SplitParams)
    tasks: list[Literal["ner", "sections", "extraction", "qa"]] = [
        "ner",
        "sections",
        "extraction",
        "qa",
    ]
    max_align_reject_fraction: float = Field(default=0.05, ge=0.0, le=1.0)


class TinyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocab_size: int = Field(default=2000, ge=100)
    hidden_size: int = Field(default=64, ge=2)
    num_layers: int = Field(default=2, ge=1)
    num_heads: int = Field(default=2, ge=1)
    intermediate_size: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def _heads_divide_hidden(self) -> Self:
        if self.hidden_size % self.num_heads:
            raise ValueError("train.tiny.hidden_size must be a multiple of num_heads")
        return self


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str = "tiny-scratch"  # or a Hub id / local dir, e.g. pucpr/biobertpt-clin
    tasks: list[Literal["ner", "sections"]] = ["ner", "sections"]
    max_length: int = Field(default=256, ge=8)
    stride: int = Field(default=64, ge=0)  # token overlap between windows of a long section
    epochs: int = Field(default=8, ge=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=5.0e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    fp16: bool = False
    gradient_checkpointing: bool = False
    save_steps: int = Field(default=200, ge=1)
    keep_checkpoints: bool = False
    tiny: TinyParams = Field(default_factory=TinyParams)

    @model_validator(mode="after")
    def _stride_below_length(self) -> Self:
        if self.stride >= self.max_length - 2:
            raise ValueError("train.stride must be smaller than train.max_length - 2")
        return self


class EvaluateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baselines: list[Literal["regex"]] = ["regex"]


class HfEmbedParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "pucpr/biobertpt-clin"
    layer_weighting: Literal["last", "linear_decay"] = "last"
    max_length: int = Field(default=512, ge=8)
    stride: int = Field(default=64, ge=0)  # token overlap between windows of a long text


class EmbedParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["hashing", "hf"] = "hashing"
    dim: int = Field(default=256, ge=1)  # hashing only; hf takes the model's hidden size
    hf: HfEmbedParams = Field(default_factory=HfEmbedParams)


class IndexParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_tokens: int = Field(default=128, ge=2)
    chunk_stride: int = Field(default=32, ge=0)

    @model_validator(mode="after")
    def _stride_below_tokens(self) -> Self:
        if self.chunk_stride >= self.chunk_tokens:
            raise ValueError("index.chunk_stride must be smaller than index.chunk_tokens")
        return self


class Params(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed: int
    generate: GenerateParams
    ingest: IngestParams = Field(default_factory=IngestParams)
    preprocess: PreprocessParams = Field(default_factory=PreprocessParams)
    dataset: DatasetParams = Field(default_factory=DatasetParams)
    train: TrainParams = Field(default_factory=TrainParams)
    evaluate: EvaluateParams = Field(default_factory=EvaluateParams)
    embed: EmbedParams = Field(default_factory=EmbedParams)
    index: IndexParams = Field(default_factory=IndexParams)


def load_params(path: Path) -> Params:
    return Params.model_validate(yaml.safe_load(path.read_text("utf-8")))
