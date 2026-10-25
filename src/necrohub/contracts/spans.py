from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntityLabel(StrEnum):
    PER = "PER"
    CPF = "CPF"
    RG = "RG"
    CRM = "CRM"
    CNPJ = "CNPJ"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"
    DATE = "DATE"
    DOC_NUMBER = "DOC_NUMBER"
    LOC = "LOC"
    ORGAN = "ORGAN"
    INJURY = "INJURY"
    FINDING = "FINDING"
    CAUSE_OF_DEATH = "CAUSE_OF_DEATH"
    MANNER_OF_DEATH = "MANNER_OF_DEATH"
    SUBSTANCE = "SUBSTANCE"

    @property
    def is_pii(self) -> bool:
        return self in _PII


_PII = frozenset(
    EntityLabel(n)
    for n in (
        [
            "PER",
            "CPF",
            "RG",
            "CRM",
            "CNPJ",
            "PHONE",
            "EMAIL",
            "ADDRESS",
            "DATE",
            "DOC_NUMBER",
            "LOC",
        ]
    )
)


class SectionLabel(StrEnum):
    PREAMBLE = "PREAMBLE"
    IDENTIFICATION = "IDENTIFICATION"
    HISTORY = "HISTORY"
    EXTERNAL_EXAM = "EXTERNAL_EXAM"
    INTERNAL_EXAM = "INTERNAL_EXAM"
    COMPLEMENTARY_EXAMS = "COMPLEMENTARY_EXAMS"
    CONCLUSION = "CONCLUSION"
    SIGNATURE = "SIGNATURE"


class _Range(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self


class Span(_Range):
    label: EntityLabel
    text: str

    def verify(self, source: str) -> None:
        """Raise ValueError unless this span points at its own text inside source."""
        if self.end > len(source) or source[self.start : self.end] != self.text:
            raise ValueError(f"span {self.label}[{self.start}:{self.end}] does not match the text")


class PredictedSpan(Span):
    score: float = Field(ge=0.0, le=1.0)
    source: Literal["regex", "hf_ner", "llm"]


class SectionSpan(_Range):
    label: SectionLabel
