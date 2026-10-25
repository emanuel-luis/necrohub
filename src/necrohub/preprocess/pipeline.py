from collections.abc import Sequence

from necrohub.contracts.documents import ProcessedText
from necrohub.preprocess.deid.base import Deidentifier, merge_spans, redact
from necrohub.preprocess.deid.regex import RegexDeidentifier
from necrohub.preprocess.normalize import DEFAULT_STEPS, normalize
from necrohub.preprocess.sections import split_sections


def process(
    text: str,
    deidentifiers: Sequence[Deidentifier] = (RegexDeidentifier(),),
    steps: Sequence[str] = DEFAULT_STEPS,
) -> ProcessedText:
    """Normalize, split into sections, detect PII on the normalized text, merge and redact once.

    `entities` keeps the longest of overlapping spans, but redaction covers every detected span:
    redacting only the merged spans would leave the uncovered tail of a partial overlap readable.
    """
    normalized = normalize(text, steps)
    detected = [d.detect(normalized) for d in deidentifiers]
    return ProcessedText(
        text=normalized,
        normalize_steps=list(steps),
        sections=split_sections(normalized),
        entities=merge_spans(detected),
        redacted_text=redact(normalized, [s for group in detected for s in group]),
        deidentifiers=[d.name for d in deidentifiers],
    )
