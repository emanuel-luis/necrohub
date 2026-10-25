"""The train stage: datasets/<task> -> models/encoder-<task> and metrics/train_<task>.json."""

import hashlib
import shutil
from pathlib import Path
from typing import Any, Literal

from necrohub.contracts.datasets import NerRecord, SectionRecord
from necrohub.params import Params
from necrohub.runs import (
    Counts,
    FileRef,
    RunManifest,
    StageFailed,
    atomic_dir,
    canonical_json,
    read_jsonl,
    sha256_tree,
    stage_run,
    write_text_atomic,
)
from necrohub.settings import resolve_device
from necrohub.train.encoder import TrainResult, train_task
from necrohub.train.labels import NER_LABELS, SECTION_LABELS

Task = Literal["ner", "sections"]


def fingerprint(task: str, params: Params, dataset_sha256: str) -> str:
    """16 hex characters over everything that decides the trained weights."""
    payload = {
        "task": task,
        "train": params.train.model_dump(),
        "base_model": params.train.base_model,
        "seed": params.seed,
        "dataset_sha256": dataset_sha256,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def run_train(
    task: Task,
    params: Params,
    data_dir: Path,
    out: Path,
    *,
    metrics_dir: Path = Path("metrics"),
    runs_dir: Path = Path("runs"),
) -> RunManifest:
    """Fine-tune the encoder for `task` on train, validate on validation; the test split is
    never opened. A label id out of range makes train_task raise. Checkpoints go to <out's parent>/.checkpoints/<task>/<fingerprint> (decision
    16) and are removed after a successful run unless train.keep_checkpoints is set.
    """
    t = params.train
    stage_params = {"task": task, "seed": params.seed, "train": t.model_dump()}
    with stage_run(f"train_{task}", stage_params, runs_dir) as run:
        task_dir = data_dir / task
        record_type = NerRecord if task == "ner" else SectionRecord
        train = read_jsonl(task_dir / "train.jsonl", record_type)
        val = read_jsonl(task_dir / "validation.jsonl", record_type)
        dataset_sha = sha256_tree(task_dir)
        run.inputs = [FileRef(path=task_dir.as_posix(), sha256=dataset_sha, records=len(train))]

        def group(r: Any) -> str:
            return str(r.case_id if task == "ner" else r.doc_id)

        overlap = {group(r) for r in train} & {group(r) for r in val}
        run.check("train_validation_disjoint", not overlap, f"{len(overlap)} shared case(s)")
        run.check("has_training_records", bool(train), f"{len(train)} record(s)")
        run.raise_if_failed()

        device = resolve_device("auto")
        if device == "cuda":
            from necrohub.train.gpu import gpu_problem, gpu_report

            problem = gpu_problem(gpu_report())
            if problem:
                raise StageFailed(f"train_{task}: {problem}")
        ckpt_dir = out.parent / ".checkpoints" / task / fingerprint(task, params, dataset_sha)
        with atomic_dir(out, run.run_id) as tmp:
            result: TrainResult = train_task(
                task,
                train,
                val,
                t,
                params.seed,
                tmp,
                ckpt_dir,
                device,
            )
            write_text_atomic(
                tmp / "necrohub_train.json",
                canonical_json(
                    {
                        "task": task,
                        "base_model": t.base_model,
                        "train": t.model_dump(),
                        "seed": params.seed,
                        "dataset_sha256": dataset_sha,
                        "labels": NER_LABELS if task == "ner" else [str(s) for s in SECTION_LABELS],
                        "counts": {
                            "train_records": result.n_train_records,
                            "validation_records": result.n_val_records,
                            "train_windows": result.n_train_windows,
                        },
                    }
                ),
            )
            run.counts = Counts(read=len(train) + len(val), written=1)
            run.outputs = [FileRef(path=out.as_posix(), sha256=sha256_tree(tmp))]
        if not t.keep_checkpoints:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    metrics = {
        "task": task,
        "base_model": result.base_model,
        "dataset_sha256": dataset_sha,
        "n_records": {"train": result.n_train_records, "validation": result.n_val_records},
        "n_windows": result.n_train_windows,
        "n_steps": result.n_steps,
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
        "resumed_from_step": result.resumed_from_step,
    }
    write_text_atomic(metrics_dir / f"train_{task}.json", canonical_json(metrics))
    return run
