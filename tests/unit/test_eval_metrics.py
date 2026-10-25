from necrohub.contracts.spans import EntityLabel, Span
from necrohub.evaluate.metrics import ner_scores, section_scores, span_scores

TEXT = "x" * 100


def sp(start: int, end: int, label: EntityLabel = EntityLabel.PER) -> Span:
    return Span(start=start, end=end, label=label, text=TEXT[start:end])


def test_perfect_prediction() -> None:
    gold = [[sp(0, 5), sp(10, 20, EntityLabel.INJURY)]]
    s = ner_scores(gold, gold)
    for mode in ("strict", "partial"):
        assert s[mode]["micro"]["f1"] == 1.0 and s[mode]["macro"]["f1"] == 1.0
        assert s[mode]["micro_pii"]["tp"] == 1  # INJURY is not PII
    assert s["pii_char_recall"] == 1.0 and s["pii_spans_missed"] == 0


def test_empty_prediction() -> None:
    s = ner_scores([[sp(0, 5)]], [[]])
    assert s["strict"]["micro"] == {
        "tp": 0, "fp": 0, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0,
    }  # fmt: skip
    assert s["pii_char_recall"] == 0.0 and s["pii_spans_missed"] == 1


def test_boundary_off_by_one_is_a_strict_miss_and_a_partial_hit() -> None:
    gold, pred = [[sp(10, 20)]], [[sp(10, 21)]]
    strict = span_scores(gold, pred, "strict")["micro"]
    partial = span_scores(gold, pred, "partial")["micro"]
    assert (strict["tp"], strict["fp"], strict["fn"]) == (0, 1, 1)
    assert (partial["tp"], partial["fp"], partial["fn"]) == (1, 0, 0)
    assert ner_scores(gold, pred)["pii_char_recall"] == 1.0


def test_label_mismatch_counts_against_both_labels_but_pii_recall_is_label_agnostic() -> None:
    gold, pred = [[sp(0, 5, EntityLabel.PER)]], [[sp(0, 5, EntityLabel.LOC)]]
    s = ner_scores(gold, pred)
    per = s["strict"]["per_label"]
    assert per["PER"]["fn"] == 1 and per["LOC"]["fp"] == 1
    assert s["pii_char_recall"] == 1.0 and s["pii_spans_missed"] == 0
    # a clinical prediction does not cover PII
    assert ner_scores(gold, [[sp(0, 5, EntityLabel.INJURY)]])["pii_spans_missed"] == 1


def test_partial_matching_is_one_to_one() -> None:
    gold = [[sp(0, 10)]]
    pred = [[sp(0, 5), sp(5, 10)]]  # two predictions overlap one gold span
    m = span_scores(gold, pred, "partial")["micro"]
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_macro_averages_only_labels_with_gold_support() -> None:
    gold = [[sp(0, 5)]]
    pred = [[sp(0, 5), sp(10, 15, EntityLabel.LOC)]]  # LOC has no gold: a false positive only
    s = span_scores(gold, pred, "strict")
    assert s["macro"]["f1"] == 1.0 and s["micro"]["precision"] == 0.5


def test_aggregation_over_records() -> None:
    gold = [[sp(0, 5)], [sp(0, 5)]]
    pred = [[sp(0, 5)], []]
    m = span_scores(gold, pred, "strict")["micro"]
    assert (m["tp"], m["fn"], m["recall"]) == (1, 1, 0.5)


def test_sections() -> None:
    s = section_scores(["A", "A", "B", "B"], ["A", "B", "B", "B"])
    assert s["accuracy"] == 0.75
    assert s["per_label"]["A"]["recall"] == 0.5 and s["per_label"]["B"]["precision"] == 0.6667
    assert s["macro_f1"] == round((0.6667 + 0.8) / 2, 4)
    assert section_scores([], [])["accuracy"] == 0.0
