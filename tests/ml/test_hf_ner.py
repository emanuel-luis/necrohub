from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from conftest import fixture_text
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from necrohub.contracts.spans import EntityLabel
from necrohub.preprocess.deid.hf_ner import HfNerDeidentifier

pytestmark = pytest.mark.ml

LABELS = ["O", "B-PESSOA", "I-PESSOA", "B-LOCAL", "I-LOCAL"]


@pytest.fixture(scope="module")
def make_model(
    tiny_tokenizer_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[list[float]], str]:
    """Tiny token classifier whose logits ignore the input: classifier bias = the given logits."""
    tokenizer = BertTokenizerFast.from_pretrained(tiny_tokenizer_dir)

    def make(bias: list[float]) -> str:
        torch.manual_seed(0)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
            id2label=dict(enumerate(LABELS)),
            label2id={v: k for k, v in enumerate(LABELS)},
        )
        model = BertForTokenClassification(config)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor(bias))
        out = tmp_path_factory.mktemp("tiny-ner")
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        return str(out)

    return make


def test_spans_in_bounds_and_labels_mapped(make_model: Callable[[list[float]], str]) -> None:
    # p(B-PESSOA) = e^6 / (e^6 + 4) ~ 0.99 on every token
    det = HfNerDeidentifier(make_model([0, 6, 0, 0, 0]), threshold=0.7, device="cpu", stride=4)
    text = fixture_text("firearm_a")
    spans = det.detect(text)
    assert spans
    for s in spans:
        assert 0 <= s.start < s.end <= len(text)
        assert text[s.start : s.end] == s.text
        assert s.label is EntityLabel.PER
        assert s.source == "hf_ner" and 0.7 <= s.score <= 1.0
        s.verify(text)


def test_local_maps_to_loc(make_model: Callable[[list[float]], str]) -> None:
    det = HfNerDeidentifier(make_model([0, 0, 0, 6, 0]), device="cpu", stride=4)
    spans = det.detect(fixture_text("natural"))
    assert spans and {s.label for s in spans} == {EntityLabel.LOC}


def test_threshold_respected(make_model: Callable[[list[float]], str]) -> None:
    model = make_model([0, 2, 0, 0, 0])  # p(B-PESSOA) = e^2 / (e^2 + 4) ~ 0.65
    text = fixture_text("firearm_b")
    assert HfNerDeidentifier(model, threshold=0.7, device="cpu", stride=4).detect(text) == []
    assert HfNerDeidentifier(model, threshold=0.5, device="cpu", stride=4).detect(text)


def test_unmapped_groups_dropped_and_custom_map(make_model: Callable[[list[float]], str]) -> None:
    model = make_model([0, 6, 0, 0, 0])
    text = fixture_text("firearm_a")
    assert (
        HfNerDeidentifier(model, device="cpu", label_map={"LOCAL": "LOC"}, stride=4).detect(text)
        == []
    )
    custom = HfNerDeidentifier(model, device="cpu", label_map={"PESSOA": "ADDRESS"}, stride=4)
    assert {s.label for s in custom.detect(text)} == {EntityLabel.ADDRESS}


def test_empty_text(make_model: Callable[[list[float]], str]) -> None:
    assert HfNerDeidentifier(make_model([0, 6, 0, 0, 0]), device="cpu", stride=4).detect("  ") == []


def test_loads_a_p4_checkpoint_and_returns_only_pii_labels(p4_layers: Path, tmp_path: Path) -> None:
    from conftest import p4_params

    from necrohub.train.stage import run_train

    out = tmp_path / "models" / "encoder-ner"
    run_train(
        "ner",
        p4_params(),
        p4_layers / "datasets",
        out,
        metrics_dir=tmp_path / "metrics",
        runs_dir=tmp_path / "runs",
    )
    # our id2label names are EntityLabel values, so the default map has PER and LOC only and
    # every clinical group is dropped; an identity map keeps all PII labels
    pii = {label.value: label.value for label in EntityLabel if label.is_pii}
    det = HfNerDeidentifier(str(out), threshold=0.0, device="cpu", label_map=pii, stride=8)
    text = fixture_text("firearm_a")
    spans = det.detect(text)
    for s in spans:
        assert s.label.is_pii and 0 <= s.start < s.end <= len(text)
        s.verify(text)
    default = HfNerDeidentifier(str(out), threshold=0.0, device="cpu", stride=8).detect(text)
    assert {s.label for s in default} <= {EntityLabel.PER, EntityLabel.LOC}
