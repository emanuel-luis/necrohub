from necrohub.embed.base import Embedder
from necrohub.embed.hashing import HashingEmbedder
from necrohub.params import EmbedParams


def build_embedder(p: EmbedParams, device: str = "auto") -> Embedder:
    """The configured embedder. The HF backend imports torch and loads the model."""
    if p.backend == "hashing":
        return HashingEmbedder(p.dim)
    from necrohub.embed.hf import HfEmbedder

    return HfEmbedder(p.hf.model, p.hf.layer_weighting, p.hf.max_length, p.hf.stride, device=device)
