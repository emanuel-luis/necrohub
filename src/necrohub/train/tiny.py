"""A tiny BERT trained from scratch: the demo model. Offline, no Hub access."""

import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
import transformers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    BertConfig,
    PreTrainedModel,
)

from necrohub.params import TinyParams

# a fast tokenizer in every supported transformers version (v5 renamed the class)
BertTokenizerFast: Any = transformers.BertTokenizerFast  # type: ignore[attr-defined]
_SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
MAX_POSITIONS = 512


def build_tokenizer(texts: Iterable[str], vocab_size: int) -> Any:
    """WordPiece trained on `texts` only. No lowercasing and no accent stripping: the reports
    are pt-BR and case carries information (names, acronyms)."""
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=False, lowercase=False
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(
        texts, trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=_SPECIAL)
    )
    with tempfile.TemporaryDirectory() as tmp:
        files = tok.model.save(tmp)
        return BertTokenizerFast(
            vocab_file=str(Path(files[0])),
            do_lower_case=False,
            strip_accents=False,
            model_max_length=MAX_POSITIONS,
        )


def build_tiny(
    texts: Iterable[str],
    tiny: TinyParams,
    seed: int,
    num_labels: int,
    task: Literal["ner", "sections"],
    id2label: Mapping[int, str] | None = None,
) -> tuple[PreTrainedModel, Any]:
    """The tokenizer (from the train texts) and a freshly initialised model, seeded."""
    tokenizer = build_tokenizer(texts, tiny.vocab_size)
    labels = dict(id2label) if id2label else {i: f"LABEL_{i}" for i in range(num_labels)}
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=tiny.hidden_size,
        num_hidden_layers=tiny.num_layers,
        num_attention_heads=tiny.num_heads,
        intermediate_size=tiny.intermediate_size,
        max_position_embeddings=MAX_POSITIONS,
        pad_token_id=tokenizer.pad_token_id,
        id2label=labels,
        label2id={v: k for k, v in labels.items()},
    )
    torch.manual_seed(seed)
    auto = AutoModelForTokenClassification if task == "ner" else AutoModelForSequenceClassification
    model: PreTrainedModel = auto.from_config(config)  # type: ignore[no-untyped-call]
    return model, tokenizer
