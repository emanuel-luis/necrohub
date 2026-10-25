"""generate -> raw boundary checks, run on the files written to the stage's temp output."""

from pathlib import Path

from pydantic import ValidationError

from necrohub.contracts.gold import GoldRecord, Reject
from necrohub.contracts.spans import EntityLabel, Span
from necrohub.ids import IDENTIFIER_PATTERNS
from necrohub.runs import read_jsonl

CheckResult = tuple[str, bool, str]
_IDENTIFIER_KINDS = ("CPF", "CNPJ", "RG", "CRM", "PHONE", "EMAIL")
_MARKERS = ("{{", "[[", "<<")


def unlabeled_identifiers(text: str, entities: list[Span]) -> list[str]:
    """Identifier-shaped values that no gold PII span covers, as "<kind>@<offset>" (no values)."""
    pii = [(s.start, s.end) for s in entities if s.label.is_pii]
    return [
        f"{kind}@{m.start('value')}"
        for kind in _IDENTIFIER_KINDS
        for pattern in IDENTIFIER_PATTERNS[kind]
        for m in pattern.finditer(text)
        if not any(a <= m.start("value") and m.end("value") <= b for a, b in pii)
    ]


def _missing_required(gold: GoldRecord) -> list[str]:
    found = {(s.label, s.text) for s in gold.entities}
    labels = {s.label for s in gold.entities}
    missing = []
    if (EntityLabel.PER, gold.case.decedent.name) not in found:
        missing.append("decedent name")
    if (EntityLabel.CRM, gold.case.examiner.crm) not in found:
        missing.append("examiner CRM")
    missing += [lbl for lbl in ("CAUSE_OF_DEATH", "MANNER_OF_DEATH") if lbl not in labels]
    return missing


def _first(problems: list[str]) -> str:
    return f"{len(problems)} problem(s); first: {problems[0]}" if problems else ""


def check_raw(raw: Path, n_cases: int, max_reject_fraction: float) -> list[CheckResult]:
    """Every boundary check on a raw layer directory; any failure fails the stage."""
    invalid: list[str] = []
    mismatched: list[str] = []
    missing: list[str] = []
    leftovers: list[str] = []
    leaks: list[str] = []
    for path in sorted((raw / "gold").glob("*.json")):
        try:
            gold = GoldRecord.model_validate_json(path.read_text("utf-8"))
        except ValidationError as exc:
            invalid.append(f"{path.name}: {exc.error_count()} error(s)")
            continue
        text = (raw / gold.files.txt).read_text("utf-8")
        try:
            gold.verify(text)
        except ValueError as exc:
            mismatched.append(str(exc))
        missing += [f"{gold.case.case_id}: {m}" for m in _missing_required(gold)]
        leftovers += [f"{gold.case.case_id}: {m!r}" for m in _MARKERS if m in text]
        leaks += [f"{gold.case.case_id}: {x}" for x in unlabeled_identifiers(text, gold.entities)]
    rejected = len(read_jsonl(raw / "rejects.jsonl", Reject))
    fraction = rejected / n_cases
    return [
        ("gold_validates", not invalid, _first(invalid)),
        ("spans_match_text", not mismatched, _first(mismatched)),
        ("required_slots_present", not missing, _first(missing)),
        ("no_leftover_markers", not leftovers, _first(leftovers)),
        ("no_unlabeled_identifiers", not leaks, _first(leaks)),
        (
            "reject_fraction_within_budget",
            fraction <= max_reject_fraction,
            f"{rejected}/{n_cases} rejected, budget {max_reject_fraction}",
        ),
    ]
