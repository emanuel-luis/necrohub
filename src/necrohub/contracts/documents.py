from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from necrohub.contracts.spans import PredictedSpan, SectionSpan


class ProcessedText(BaseModel):
    """What preprocess.pipeline.process returns. The text keeps its PII (the corpus is synthetic)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    text: str
    normalize_steps: list[str]
    sections: list[SectionSpan]
    entities: list[PredictedSpan]
    redacted_text: str
    deidentifiers: list[str]

    @model_validator(mode="after")
    def _offsets_fit_text(self) -> Self:
        for entity in self.entities:
            entity.verify(self.text)
        end = 0
        for section in self.sections:
            if section.start < end or section.end > len(self.text):
                raise ValueError("sections must be ordered, non-overlapping and inside the text")
            end = section.end
        return self


class ProcessedDocument(ProcessedText):
    """One record of processed/documents.jsonl: the pipeline output of one ingested document."""

    doc_id: str


class IngestedDocument(BaseModel):
    """One record of interim/ingested.jsonl: the text of one report as read from the raw layer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    doc_id: str
    source_path: str  # relative to the raw layer, forward slashes
    source_format: Literal["txt", "pdf"]
    source_sha256: str
    extractor: str  # name@version
    text: str
    n_chars: int
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _n_chars_is_len(self) -> Self:
        if self.n_chars != len(self.text):
            raise ValueError("n_chars must equal len(text)")
        return self


class DocReject(BaseModel):
    """A document an ingest or preprocess run could not accept; any reject fails the run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    doc_id: str
    reason: str
    detail: str = ""
