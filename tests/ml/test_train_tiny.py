import json
import shutil
from pathlib import Path

import pytest
from conftest import p4_params
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.contracts.datasets import NerRecord, SectionRecord
from necrohub.contracts.spans import SectionLabel
from necrohub.evaluate.stage import run_evaluate
from necrohub.runs import read_jsonl
from necrohub.train.encoder import load_trained, predict_ner, predict_sections, train_task
from necrohub.train.stage import run_train

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def trained(p4_layers: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """<root>/models/encoder-{ner,sections} and <root>/metrics from one train run per task."""
    root = tmp_path_factory.mktemp("trained")
    params = p4_params()
    for task in ("ner", "sections"):
        run_train(
            task,  # type: ignore[arg-type]
            params,
            p4_layers / "datasets",
            root / "models" / f"encoder-{task}",
            metrics_dir=root / "metrics",
            runs_dir=root / "runs",
        )
    return root


def test_out_dir_layout_and_train_info(trained: Path) -> None:
    for task in ("ner", "sections"):
        out = trained / "models" / f"encoder-{task}"
        names = {p.name for p in out.iterdir()}
        assert {"config.json", "model.safetensors", "necrohub_train.json"} <= names
        assert "tokenizer.json" in names
        info = json.loads((out / "necrohub_train.json").read_text("utf-8"))
        assert info["task"] == task and info["base_model"] == "tiny-scratch" and info["seed"] == 7
        assert len(info["dataset_sha256"]) == 64 and info["labels"]
        metrics = json.loads((trained / "metrics" / f"train_{task}.json").read_text("utf-8"))
        assert metrics["n_steps"] > 0 and metrics["resumed_from_step"] == 0
        assert metrics["dataset_sha256"] == info["dataset_sha256"]
        assert metrics["train_loss"] > 0 and metrics["val_loss"] is not None
        left = trained / "models" / ".checkpoints" / task
        assert not left.exists() or not any(left.iterdir())  # removed after a good run


def test_same_machine_training_is_deterministic(
    trained: Path, p4_layers: Path, tmp_path: Path
) -> None:
    run_train(
        "ner",
        p4_params(),
        p4_layers / "datasets",
        tmp_path / "models" / "encoder-ner",
        metrics_dir=tmp_path / "metrics",
        runs_dir=tmp_path / "runs",
    )
    assert (tmp_path / "metrics" / "train_ner.json").read_bytes() == (
        trained / "metrics" / "train_ner.json"
    ).read_bytes()


def test_resume_from_the_last_checkpoint(p4_layers: Path, tmp_path: Path) -> None:
    t = p4_params().train.model_copy(update={"save_steps": 1})
    train = read_jsonl(p4_layers / "datasets" / "sections" / "train.jsonl", SectionRecord)
    val = read_jsonl(p4_layers / "datasets" / "sections" / "validation.jsonl", SectionRecord)
    ckpt = tmp_path / "ckpt"
    first = train_task("sections", train, val, t, 7, tmp_path / "a", ckpt, "cpu")
    assert first.resumed_from_step == 0 and any(ckpt.glob("checkpoint-*"))
    second = train_task("sections", train, val, t, 7, tmp_path / "b", ckpt, "cpu")
    assert second.resumed_from_step == first.n_steps == second.n_steps


def test_fp16_needs_cuda(p4_layers: Path, tmp_path: Path) -> None:
    t = p4_params().train.model_copy(update={"fp16": True})
    with pytest.raises(ValueError, match="CUDA"):
        train_task("sections", [], [], t, 7, tmp_path / "o", tmp_path / "c", "cpu")


def test_predictions_are_in_bounds(trained: Path, p4_layers: Path) -> None:
    t = p4_params().train
    ner = read_jsonl(p4_layers / "datasets" / "ner" / "test.jsonl", NerRecord)[:6]
    model, tok = load_trained(trained / "models" / "encoder-ner", "ner")
    texts = [r.text for r in ner]
    # a short window forces several windows per text
    for spans, text in zip(predict_ner(model, tok, texts, 32, 8, "cpu"), texts, strict=True):
        for s in spans:
            assert 0 <= s.start < s.end <= len(text)
            s.verify(text)
    assert predict_ner(model, tok, [], t.max_length, t.stride, "cpu") == []
    sections = read_jsonl(p4_layers / "datasets" / "sections" / "test.jsonl", SectionRecord)
    model, tok = load_trained(trained / "models" / "encoder-sections", "sections")
    labels = predict_sections(model, tok, [r.text for r in sections], t.max_length, "cpu")
    assert len(labels) == len(sections) and all(isinstance(x, SectionLabel) for x in labels)


def test_evaluate_end_to_end(trained: Path, p4_layers: Path, tmp_path: Path) -> None:
    out = tmp_path / "eval.json"
    run_evaluate(
        p4_params(),
        trained / "models",
        p4_layers / "datasets",
        p4_layers / "raw",
        out,
        runs_dir=tmp_path / "runs",
    )
    ev = json.loads(out.read_text("utf-8"))
    assert ev["schema_version"] == 1
    assert ev["base_model"] == {"ner": "tiny-scratch", "sections": "tiny-scratch"}
    n_test = len(read_jsonl(p4_layers / "datasets" / "ner" / "test.jsonl", NerRecord))
    assert sum(ev["ner"][s]["n_records"] for s in ev["ner"]) == n_test
    held = ev["ner"]["heldout_templates"]
    assert held["n_records"] > 0
    for system in ("regex", "encoder"):
        m = held[system]
        assert 0.0 <= m["strict"]["micro"]["f1"] <= m["partial"]["micro"]["f1"] <= 1.0
        assert 0.0 <= m["pii_char_recall"] <= 1.0
    n_sections = len(read_jsonl(p4_layers / "datasets" / "sections" / "test.jsonl", SectionRecord))
    assert sum(ev["sections"][s]["n_records"] for s in ev["sections"]) == n_sections


def test_cli_train_and_evaluate(p4_layers: Path, tmp_path: Path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text(
        "seed: 7\ngenerate: {n_cases: 16, realizer: template, holdout_templates: [t3],"
        " max_reject_fraction: 0.2}\ntrain: {epochs: 1, tiny: {vocab_size: 600}}\n",
        encoding="utf-8",
    )
    data = p4_layers / "datasets"
    common = ["--params", str(params), "--runs-dir", str(tmp_path / "runs")]
    runner = CliRunner()
    for task in ("ner", "sections"):
        args = ["train", "--task", task, "--data", str(data), *common]
        args += ["--out", str(tmp_path / "models" / f"encoder-{task}")]
        args += ["--metrics-dir", str(tmp_path / "metrics")]
        assert runner.invoke(app, args).exit_code == 0
    args = ["evaluate", "--models", str(tmp_path / "models"), "--data", str(data), *common]
    args += ["--raw", str(p4_layers / "raw"), "--out", str(tmp_path / "eval.json")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["train", "--task", "qa", *common]).exit_code == 2
    shutil.rmtree(tmp_path / "models" / "encoder-ner")
    assert runner.invoke(app, args).exit_code != 0


def test_fingerprint_covers_code_and_library_versions(
    p4_layers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib import metadata

    from necrohub.train import stage

    params = p4_params()
    base = stage.fingerprint("ner", params, "a" * 64)
    assert base == stage.fingerprint("ner", params, "a" * 64) and len(base) == 16
    assert base != stage.fingerprint("sections", params, "a" * 64)
    assert base != stage.fingerprint("ner", params, "b" * 64)
    monkeypatch.setattr(stage, "_code_sha256", lambda: "changed")
    assert base != stage.fingerprint("ner", params, "a" * 64)
    monkeypatch.undo()
    real = metadata.version
    monkeypatch.setattr(
        stage.metadata, "version", lambda n: "0.0" if n == "transformers" else real(n)
    )
    assert base != stage.fingerprint("ner", params, "a" * 64)
