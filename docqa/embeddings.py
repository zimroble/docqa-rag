"""Text embeddings.

The default model is WordLlama (256-d static token embeddings distilled from
LLM embedding tables). It runs on CPU in ~1 ms per chunk, needs no GPU or
network access, and ships inside the pip package, so the app and its tests
work fully offline. Any other model can be plugged in by implementing
`Embedder`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""


class WordLlamaEmbedder:
    dim = 256

    def __init__(self) -> None:
        import wordllama
        from wordllama import WordLlama

        # Load the weights and tokenizer bundled in the wheel; never download.
        pkg = Path(wordllama.__file__).parent
        self._model = WordLlama.load(dim=self.dim, cache_dir=pkg, disable_download=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.asarray(self._model.embed(texts, norm=True), dtype=np.float32)
        return vecs


@lru_cache
def get_embedder() -> Embedder:
    return WordLlamaEmbedder()
