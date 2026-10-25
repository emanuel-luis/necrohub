from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

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
