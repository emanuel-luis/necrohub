"""Extraction scoring: parse raw model output into CaseExtraction and compare with the gold.

Pure: no torch. All floats are rounded to 4 decimals. An invalid prediction counts as wrong on
every field, so a model cannot raise its accuracy by failing to answer.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from necrohub.contracts.datasets import CaseExtraction, ExtractionRecord
from necrohub.evaluate.metrics import prf
from necrohub.runs import canonical_json, read_jsonl, write_text_atomic

_ND = 4
_SCALARS = ("scenario", "cause_of_death", "mechanism", "manner_of_death")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _validate(obj: Any) -> CaseExtraction | None:
    try:
        return CaseExtraction.model_validate(obj)
    except ValidationError:
        return None


def parse_extraction(text: str) -> CaseExtraction | None:
    """The CaseExtraction in a model's raw output, or None.

    Accepts the bare JSON object, one inside a ```json fence, or one surrounded by prose: the
    first object that validates wins.
    """
    decoder = json.JSONDecoder()
    for chunk in [text, *_FENCE.findall(text)]:
        for m in re.finditer(r"\{", chunk):
            try:
                obj, _ = decoder.raw_decode(chunk, m.start())
            except ValueError:
                continue
            if (found := _validate(obj)) is not None:
                return found
    return None


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _scalars(e: CaseExtraction) -> dict[str, Any]:
    out: dict[str, Any] = {f: _norm(getattr(e, f)) for f in _SCALARS}
    out["time_since_death_h"] = tuple(e.time_since_death_h)
    return out


def _injuries(e: CaseExtraction) -> set[tuple[Any, ...]]:
    return {
        (
            _norm(i.kind),
            _norm(i.region),
            _norm(i.side or ""),
            None if i.size_cm is None else round(i.size_cm, _ND),
            tuple(sorted(_norm(f) for f in i.features)),
        )
        for i in e.injuries
    }


def _organs(e: CaseExtraction) -> set[tuple[Any, ...]]:
    return {(_norm(o.organ), _norm(o.finding), o.abnormal) for o in e.organ_findings}


def score_extraction(
    gold: Sequence[CaseExtraction], pred: Sequence[CaseExtraction | None]
) -> dict[str, Any]:
    """Validity rate, exact accuracy per scalar field, set P/R/F1 for injuries and organ findings
    (duplicates collapse), and the rate of cases with every field exact."""
    n = len(gold)
    hits = {f: 0 for f in (*_SCALARS, "time_since_death_h")}
    counts = {"injuries": [0, 0, 0], "organ_findings": [0, 0, 0]}
    valid = exact = 0
    for g, p in zip(gold, pred, strict=True):
        gs = {"injuries": _injuries(g), "organ_findings": _organs(g)}
        ps: dict[str, set[tuple[Any, ...]]] = {"injuries": set(), "organ_findings": set()}
        field_ok = dict.fromkeys(hits, False)
        if p is not None:
            valid += 1
            ps = {"injuries": _injuries(p), "organ_findings": _organs(p)}
            gv, pv = _scalars(g), _scalars(p)
            field_ok = {f: gv[f] == pv[f] for f in hits}
        for f, ok in field_ok.items():
            hits[f] += ok
        for f, c in counts.items():
            c[0] += len(gs[f] & ps[f])
            c[1] += len(ps[f] - gs[f])
            c[2] += len(gs[f] - ps[f])
        exact += all(field_ok.values()) and all(gs[f] == ps[f] for f in counts)

    def rate(k: int) -> float:
        return round(k / n, _ND) if n else 0.0

    return {
        "n": n,
        "valid_json_rate": rate(valid),
        "field_accuracy": {f: rate(k) for f, k in hits.items()},
        "injuries": prf(*counts["injuries"]),
        "organ_findings": prf(*counts["organ_findings"]),
        "all_fields_exact": rate(exact),
    }


def score_predictions_file(
    data_dir: Path, split: str, predictions: Path, out: Path
) -> dict[str, Any]:
    """Score a JSONL of {"id", "text"} (raw model output) against <data_dir>/extraction/<split>.

    A gold record without a prediction line counts as invalid. The result is also written to out.
    """
    gold = read_jsonl(data_dir / "extraction" / f"{split}.jsonl", ExtractionRecord)
    texts: dict[str, str] = {}
    for line in predictions.read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            texts[str(row["id"])] = str(row["text"])
    parsed = [parse_extraction(texts[r.id]) if r.id in texts else None for r in gold]
    scores = score_extraction([r.target for r in gold], parsed)
    scores["split"] = split
    scores["missing_predictions"] = sum(1 for r in gold if r.id not in texts)
    write_text_atomic(out, canonical_json(scores))
    return scores
