"""Span and section metrics. Pure: no torch. All floats are rounded to 4 decimals."""

from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from necrohub.contracts.spans import Span

Mode = Literal["strict", "partial"]
_ND = 4


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, _ND),
        "recall": round(r, _ND),
        "f1": round(f1, _ND),
    }


def _matches(gold: Sequence[Span], pred: Sequence[Span], mode: Mode) -> int:
    """Number of one-to-one matches between same-label gold and predicted spans of one record."""
    if mode == "strict":
        g = Counter((s.start, s.end) for s in gold)
        p = Counter((s.start, s.end) for s in pred)
        return sum((g & p).values())
    pairs = sorted(
        (
            (-(min(g.end, p.end) - max(g.start, p.start)), i, j)
            for i, g in enumerate(gold)
            for j, p in enumerate(pred)
            if min(g.end, p.end) > max(g.start, p.start)
        ),
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    for _, i, j in pairs:  # greedy by overlap length, then position
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
    return len(used_g)


def span_scores(
    gold: Sequence[Sequence[Span]], pred: Sequence[Sequence[Span]], mode: Mode
) -> dict[str, Any]:
    """Per-label, micro, macro (labels with gold support) and micro_pii counts and P/R/F1.

    `gold[i]` and `pred[i]` are the spans of record i. A predicted span counts only against
    gold spans of its own label.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must hold the same number of records")
    counts: dict[str, list[int]] = {}
    pii: set[str] = set()
    for g_rec, p_rec in zip(gold, pred, strict=True):
        labels = {s.label for s in g_rec} | {s.label for s in p_rec}
        for label in labels:
            g = [s for s in g_rec if s.label == label]
            p = [s for s in p_rec if s.label == label]
            tp = _matches(g, p, mode)
            c = counts.setdefault(label.value, [0, 0, 0])
            c[0] += tp
            c[1] += len(p) - tp
            c[2] += len(g) - tp
            if label.is_pii:
                pii.add(label.value)
    per_label = {k: prf(*v) for k, v in sorted(counts.items())}

    def total(keys: Sequence[str]) -> dict[str, Any]:
        return prf(*(sum(counts[k][i] for k in keys) for i in range(3)))

    supported = [k for k, v in per_label.items() if v["tp"] + v["fn"] > 0]
    macro = {
        m: round(sum(per_label[k][m] for k in supported) / len(supported), _ND)
        if supported
        else 0.0
        for m in ("precision", "recall", "f1")
    }
    return {
        "per_label": per_label,
        "micro": total(list(counts)),
        "macro": macro,
        "micro_pii": total(sorted(pii)),
    }


def _union(spans: Sequence[Span]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s in sorted(spans, key=lambda s: (s.start, s.end)):
        if merged and s.start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s.end))
        else:
            merged.append((s.start, s.end))
    return merged


def pii_coverage(
    gold: Sequence[Sequence[Span]], pred: Sequence[Sequence[Span]]
) -> tuple[float, int]:
    """(pii_char_recall, pii_spans_missed). The share of gold PII characters covered by any
    predicted PII span, whatever its label, and the gold PII spans with zero coverage. A corpus
    without gold PII has recall 1.0."""
    total = covered = missed = 0
    for g_rec, p_rec in zip(gold, pred, strict=True):
        gp = [s for s in g_rec if s.label.is_pii]
        pp = _union([s for s in p_rec if s.label.is_pii])
        total += sum(e - s for s, e in _union(gp))
        covered += sum(max(0, min(e, pe) - max(s, ps)) for s, e in _union(gp) for ps, pe in pp)
        missed += sum(1 for g in gp if not any(min(g.end, pe) > max(g.start, ps) for ps, pe in pp))
    return (round(covered / total, _ND) if total else 1.0), missed


def ner_scores(gold: Sequence[Sequence[Span]], pred: Sequence[Sequence[Span]]) -> dict[str, Any]:
    recall, missed = pii_coverage(gold, pred)
    return {
        "strict": span_scores(gold, pred, "strict"),
        "partial": span_scores(gold, pred, "partial"),
        "pii_char_recall": recall,
        "pii_spans_missed": missed,
    }


def section_scores(gold: Sequence[str], pred: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro-F1 over labels with gold support, and per-label P/R/F1."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    counts: dict[str, list[int]] = {}
    for g, p in zip(gold, pred, strict=True):
        for label in {g, p}:
            c = counts.setdefault(label, [0, 0, 0])
            if g == p:
                c[0] += 1
            elif label == p:
                c[1] += 1
            else:
                c[2] += 1
    per_label = {k: prf(*v) for k, v in sorted(counts.items())}
    supported = [v["f1"] for v in per_label.values() if v["tp"] + v["fn"] > 0]
    return {
        "accuracy": round(sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold), _ND)
        if gold
        else 0.0,
        "macro_f1": round(sum(supported) / len(supported), _ND) if supported else 0.0,
        "per_label": per_label,
    }
