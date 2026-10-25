from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from necrohub.embed.chunking import windows
from necrohub.settings import resolve_device

_MAX_POSITIONS = 512  # BERT-style encoders; the two specials are subtracted below


def window_ids(
    ids: Sequence[int], size: int, stride: int, cls_id: int, sep_id: int
) -> list[list[int]]:
    """Token-id windows of at most `size` ids, each wrapped as [cls, ..., sep]."""
    return [[cls_id, *ids[s:e], sep_id] for s, e in windows(len(ids), size, stride)]


class HfEmbedder:
    """Mean-pooled encoder embeddings of arbitrarily long text, windowed over token ids."""

    def __init__(
        self,
        model: str,
        layer_weighting: Literal["last", "linear_decay"] = "last",
        max_length: int = 512,
        stride: int = 64,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = resolve_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).to(self._device).eval()
        self._weighting = layer_weighting
        self._stride = stride
        tok = self._tokenizer
        self._size = min(tok.model_max_length, max_length, _MAX_POSITIONS) - 2
        cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
        if cls is None or sep is None:
            raise ValueError(f"tokenizer of {model} has no cls/sep (or bos/eos) token")
        self._cls: int = cls
        self._sep: int = sep
        self._pad: int = tok.pad_token_id or 0
        self.id = f"hf:{model}:{layer_weighting}"
        self.dim: int = self._model.config.hidden_size

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts])

    def _embed_one(self, text: str) -> npt.NDArray[np.float32]:
        torch = self._torch
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError("text has no tokens, cannot embed it")
        chunks = window_ids(ids, self._size, self._stride, self._cls, self._sep)
        width = max(len(c) for c in chunks)
        input_ids = torch.tensor([c + [self._pad] * (width - len(c)) for c in chunks])
        mask = torch.tensor([[1] * len(c) + [0] * (width - len(c)) for c in chunks])
        with torch.inference_mode():
            out = self._model(
                input_ids=input_ids.to(self._device),
                attention_mask=mask.to(self._device),
                output_hidden_states=True,
            )
            if self._weighting == "last":
                hidden = out.last_hidden_state
            else:  # earlier layers weigh more: 1.0 down to 0.3 over the hidden states
                states = torch.stack(list(out.hidden_states))
                w = torch.linspace(1.0, 0.3, states.size(0), device=states.device)
                hidden = (states * w.view(-1, 1, 1, 1)).sum(0) / w.sum()
            m = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
            per_chunk = (hidden * m).sum(1) / m.sum(1)  # (n_chunks, H)
            doc = torch.nn.functional.normalize(per_chunk.mean(dim=0), p=2, dim=0)
        return np.asarray(doc.cpu().numpy(), dtype=np.float32)
