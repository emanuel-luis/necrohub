"""Records of the datasets layer: one model per task, one JSON object per line."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from necrohub.contracts.gold import Injury, OrganFinding, Scenario
from necrohub.contracts.spans import SectionLabel, Span


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1


class NerRecord(_Record):
    """One section of a processed document with its gold spans (offsets relative to `text`)."""

    id: str
    doc_id: str
    case_id: str
    section: SectionLabel
    text: str
    spans: list[Span]

    @model_validator(mode="after")
    def _spans_match_text(self) -> Self:
        for span in self.spans:
            span.verify(self.text)
        return self


class SectionRecord(_Record):
    """A section body without its header line, labelled with its section."""

    id: str
    doc_id: str
    text: str
    label: SectionLabel


class CaseExtraction(BaseModel):
    """The non-PII facts of a case that a report states: the extraction target."""

    model_config = ConfigDict(extra="forbid")

    scenario: Scenario
    cause_of_death: str
    mechanism: str
    manner_of_death: str
    time_since_death_h: tuple[int, int]
    injuries: list[Injury]
    organ_findings: list[OrganFinding]


class ExtractionRecord(_Record):
    id: str
    doc_id: str
    input_text: str  # the redacted processed text
    target: CaseExtraction


class QaRecord(_Record):
    id: str
    doc_id: str
    question: str
    answer: str
    section: SectionLabel  # where the answer is stated
