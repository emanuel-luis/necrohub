import pytest
from pydantic import ValidationError

from necrohub.contracts.documents import IngestedDocument, ProcessedDocument
from necrohub.preprocess.pipeline import process


def test_ingested_document_round_trips_and_checks_n_chars() -> None:
    doc = IngestedDocument(
        doc_id="c7-00000",
        source_path="reports/c7-00000.pdf",
        source_format="pdf",
        source_sha256="0" * 64,
        extractor="pypdf@6",
        text="Laudo\n",
        n_chars=6,
        warnings=["w"],
    )
    assert IngestedDocument.model_validate_json(doc.model_dump_json()) == doc
    with pytest.raises(ValidationError):
        IngestedDocument.model_validate(doc.model_dump() | {"n_chars": 5})
    with pytest.raises(ValidationError):
        IngestedDocument.model_validate(doc.model_dump() | {"extra": 1})


def test_processed_document_round_trips_and_validates_like_processed_text(
    report: tuple[str, str],
) -> None:
    doc = ProcessedDocument(doc_id="c7-00000", **process(report[1]).model_dump())
    assert ProcessedDocument.model_validate_json(doc.model_dump_json()) == doc
    bad = doc.model_dump()
    bad["entities"][0]["end"] = len(doc.text) + 5
    with pytest.raises(ValidationError):
        ProcessedDocument.model_validate(bad)
