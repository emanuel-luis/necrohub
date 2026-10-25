"""Fine-tune and run an encoder (tiny-scratch, a Hub model or a local dir) for NER and sections."""

import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from necrohub.contracts.datasets import NerRecord, SectionRecord
from necrohub.contracts.spans import SectionLabel, Span
from necrohub.params import TrainParams
from necrohub.train.labels import (
    IGNORE,
    NER_LABELS,
    SECTION_LABELS,
    bio_labels,
    decode_bio,
    merge_windows,
)
from necrohub.train.tiny import build_tiny

Task = Literal["ner", "sections"]
TINY = "tiny-scratch"
_PREDICT_BATCH = 32


class Rows(Dataset[dict[str, Any]]):
    """A list of tokenized examples."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass(frozen=True)
class TrainResult:
    n_train_records: int
    n_val_records: int
    n_train_windows: int  # NER windows (sections: one example per record)
    n_steps: int
    train_loss: float
    val_loss: float | None
    base_model: str
    resumed_from_step: int


def id2label(task: Task) -> dict[int, str]:
    names = NER_LABELS if task == "ner" else [str(s) for s in SECTION_LABELS]
    return dict(enumerate(names))


def ner_rows(
    records: Sequence[NerRecord], tokenizer: PreTrainedTokenizerBase, max_length: int, stride: int
) -> list[dict[str, Any]]:
    """One example per window of each record: ids, mask and BIO labels (-100 on special tokens)."""
    if not records:
        return []
    enc = tokenizer(
        [r.text for r in records],
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    rows = []
    for i, offsets in enumerate(enc["offset_mapping"]):
        record = records[enc["overflow_to_sample_mapping"][i]]
        labels = bio_labels(offsets, record.spans)
        if any(x != IGNORE and not 0 <= x < len(NER_LABELS) for x in labels):
            raise ValueError(f"label id out of range in {record.id}")
        rows.append(
            {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": labels,
            }
        )
    return rows


def section_rows(
    records: Sequence[SectionRecord], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> list[dict[str, Any]]:
    if not records:
        return []
    enc = tokenizer([r.text for r in records], truncation=True, max_length=max_length)
    return [
        {
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": SECTION_LABELS.index(r.label),
        }
        for i, r in enumerate(records)
    ]


def _fast(tokenizer: PreTrainedTokenizerBase, name: str) -> None:
    if not tokenizer.is_fast:
        raise ValueError(f"{name}: a fast tokenizer is required (character offsets)")


def load_base(
    task: Task, params: TrainParams, train_texts: Sequence[str], seed: int
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """The model to fine-tune: tiny-scratch (built offline from the train texts) or a Hub id or
    local directory, with our labels."""
    labels = id2label(task)
    if params.base_model == TINY:
        return build_tiny(train_texts, params.tiny, seed, len(labels), task, labels)
    tokenizer = AutoTokenizer.from_pretrained(params.base_model)
    _fast(tokenizer, params.base_model)
    auto = AutoModelForTokenClassification if task == "ner" else AutoModelForSequenceClassification
    model = auto.from_pretrained(
        params.base_model,
        num_labels=len(labels),
        id2label=labels,
        label2id={v: k for k, v in labels.items()},
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def load_trained(path: Path, task: Task) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """A model directory written by `train_task`."""
    auto = AutoModelForTokenClassification if task == "ner" else AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(path)
    _fast(tokenizer, str(path))
    return auto.from_pretrained(path), tokenizer


def train_task(
    task: Task,
    train_records: Sequence[Any],  # NerRecord for ner, SectionRecord for sections
    val_records: Sequence[Any],
    params: TrainParams,
    seed: int,
    out_dir: Path,
    ckpt_dir: Path,
    device: str,
) -> TrainResult:
    """Fine-tune with the Trainer, resume from the last checkpoint in ckpt_dir if there is one,
    and save the model and tokenizer to out_dir. Checkpoints are left for the caller to clean."""
    if params.fp16 and device != "cuda":
        raise ValueError("train.fp16 needs a CUDA device (NECROHUB_DEVICE=cuda); set it to false")
    set_seed(seed)
    if device == "cuda":
        torch.use_deterministic_algorithms(True, warn_only=True)
    model, tokenizer = load_base(task, params, [r.text for r in train_records], seed)
    if task == "ner":
        train_rows = ner_rows(train_records, tokenizer, params.max_length, params.stride)
        val_rows = ner_rows(val_records, tokenizer, params.max_length, params.stride)
        collator: Any = DataCollatorForTokenClassification(tokenizer)
    else:
        train_rows = section_rows(train_records, tokenizer, params.max_length)
        val_rows = section_rows(val_records, tokenizer, params.max_length)
        collator = DataCollatorWithPadding(tokenizer)
    if not train_rows:
        raise ValueError(f"no training examples for {task}")
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=params.epochs,
        per_device_train_batch_size=params.batch_size,
        per_device_eval_batch_size=params.batch_size,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        warmup_steps=min(params.warmup_ratio, 0.999),  # below 1 it is a share of the steps
        fp16=params.fp16,
        gradient_checkpointing=params.gradient_checkpointing,
        eval_strategy="epoch" if val_rows else "no",
        save_strategy="steps",
        save_steps=params.save_steps,
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=device == "cpu",
        seed=seed,
        data_seed=seed,
        full_determinism=device == "cpu",
        disable_tqdm=True,
    )
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Rows(train_rows),
        eval_dataset=Rows(val_rows) if val_rows else None,
        data_collator=collator,
        processing_class=tokenizer,
    )
    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None  # type: ignore[no-untyped-call]
    trainer.train(resume_from_checkpoint=last)
    val_loss = float(trainer.evaluate()["eval_loss"]) if val_rows else None
    losses = [float(e["loss"]) for e in trainer.state.log_history if "loss" in e]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return TrainResult(
        n_train_records=len(train_records),
        n_val_records=len(val_records),
        n_train_windows=len(train_rows),
        n_steps=int(trainer.state.global_step),
        train_loss=round(sum(losses) / len(losses), 4) if losses else 0.0,
        val_loss=round(val_loss, 4) if val_loss is not None else None,
        base_model=params.base_model,
        resumed_from_step=int(last.rsplit("-", 1)[1]) if last else 0,
    )


def _logits(
    model: Any, tokenizer: PreTrainedTokenizerBase, ids: list[list[int]], device: str
) -> list[Any]:
    """argmax per token (token classification) or per example, in batches, eval mode, no grad."""
    model.to(device).eval()
    out: list[Any] = []
    for lo in range(0, len(ids), _PREDICT_BATCH):
        chunk = ids[lo : lo + _PREDICT_BATCH]
        batch = tokenizer.pad({"input_ids": chunk}, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**batch).logits
        out += logits.argmax(-1).tolist()
    return out


def predict_ner(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int,
    stride: int,
    device: str,
) -> list[list[Span]]:
    """Spans per text. Long texts are windowed and the windows merged (`merge_windows`)."""
    if not texts:
        return []
    enc = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    argmax = _logits(model, tokenizer, enc["input_ids"], device)
    windows: list[list[tuple[Any, Any]]] = [[] for _ in texts]
    for i, labels in enumerate(argmax):
        n = len(enc["input_ids"][i])
        windows[enc["overflow_to_sample_mapping"][i]].append((enc["offset_mapping"][i], labels[:n]))
    result = []
    for text, parts in zip(texts, windows, strict=True):
        offsets, labels = merge_windows(parts)
        result.append(decode_bio(offsets, labels, text))
    return result


def predict_sections(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int,
    device: str,
) -> list[SectionLabel]:
    if not texts:
        return []
    enc = tokenizer(list(texts), truncation=True, max_length=max_length)
    return [SECTION_LABELS[i] for i in _logits(model, tokenizer, enc["input_ids"], device)]
