import re
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from necrohub.contracts.spans import EntityLabel, PredictedSpan
from necrohub.settings import resolve_device

# Entity groups of Portuguese NER models mapped to our labels. Groups not listed (for example
# ORGANIZACAO: EntityLabel.ORGAN means an anatomical organ, not an organization) are dropped.
DEFAULT_LABEL_MAP: Mapping[str, str] = {
    "PESSOA": "PER",
    "PER": "PER",
    "LOCAL": "LOC",
    "LOC": "LOC",
}

_WORD_BEFORE = re.compile(r"\w*$")
_WORD_AFTER = re.compile(r"\w*")


class NerEntity(TypedDict):
    entity_group: str
    score: float
    word: str
    start: int
    end: int


def merge_subwords(entities: Sequence[NerEntity], text: str) -> list[NerEntity]:
    """Glue "##" pieces onto the previous entity of the same group, then trim each entity and
    widen it to whole words; word becomes the text slice. Pure: the input is not changed.
    """
    merged: list[NerEntity] = []
    for ent in sorted(entities, key=lambda e: e["start"]):
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and ent["word"].startswith("##")
            and ent["entity_group"] == prev["entity_group"]
            and ent["start"] == prev["end"]
        ):
            prev["end"] = ent["end"]
            prev["score"] = max(prev["score"], ent["score"])
        else:
            merged.append(NerEntity(**ent))
    for ent in merged:
        start, end = ent["start"], ent["end"]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        before = _WORD_BEFORE.search(text, 0, start)
        after = _WORD_AFTER.match(text, end)
        start -= len(before.group()) if before else 0
        end += len(after.group()) if after else 0
        ent["start"], ent["end"], ent["word"] = start, end, text[start:end]
    return merged


class HfNerDeidentifier:
    name = "hf_ner"

    def __init__(
        self,
        model: str,
        threshold: float = 0.7,
        device: str = "auto",
        label_map: Mapping[str, str] | None = None,
        stride: int = 64,
    ) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        self._threshold = threshold
        self._labels = {
            k.upper(): EntityLabel(v) for k, v in (label_map or DEFAULT_LABEL_MAP).items()
        }
        tokenizer = AutoTokenizer.from_pretrained(model)
        if not tokenizer.is_fast:
            # a slow tokenizer cannot stride, and truncating long reports would leak the tail
            raise ValueError(f"{model}: a fast tokenizer is required for windowed NER")
        tokenizer.model_max_length = min(tokenizer.model_max_length, 512)
        self._pipe: Any = pipeline(
            "token-classification",
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            stride=stride,
            device=resolve_device(device),
        )

    def detect(self, text: str) -> list[PredictedSpan]:
        if not text.strip():
            return []
        raw = [
            NerEntity(
                entity_group=str(e["entity_group"]),
                score=float(e["score"]),
                word=str(e["word"]),
                start=int(e["start"]),
                end=int(e["end"]),
            )
            for e in self._pipe(text)
        ]
        spans: list[PredictedSpan] = []
        for ent in merge_subwords(raw, text):
            label = self._labels.get(ent["entity_group"].upper())
            if label is None or ent["score"] < self._threshold or ent["end"] <= ent["start"]:
                continue
            span = PredictedSpan(
                start=ent["start"],
                end=ent["end"],
                label=label,
                text=text[ent["start"] : ent["end"]],
                score=min(ent["score"], 1.0),
                source="hf_ner",
            )
            span.verify(text)
            spans.append(span)
        return spans
