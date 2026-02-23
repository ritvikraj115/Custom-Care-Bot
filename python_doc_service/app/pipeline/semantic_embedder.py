from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from app.pipeline.logger import get_logger

log = get_logger("semantic-embedder")


class HashingSentenceEmbedder:
    """
    Lightweight, dependency-stable embedder with SentenceTransformer-like API.
    Uses hashed character n-grams to produce dense fixed-size vectors.
    """

    def __init__(
        self,
        n_features: int = 768,
        ngram_min: int = 3,
        ngram_max: int = 5,
    ) -> None:
        self.n_features = max(128, int(n_features))
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            analyzer="char_wb",
            ngram_range=(max(1, int(ngram_min)), max(1, int(ngram_max))),
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        log.info(
            "Semantic embedder initialized | backend=hashing | n_features=%d | ngram=%d-%d",
            self.n_features,
            int(ngram_min),
            int(ngram_max),
        )

    def encode(
        self,
        inputs: Any,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> np.ndarray:
        if isinstance(inputs, str):
            rows = [inputs]
            single = True
        else:
            rows = [str(x or "") for x in (inputs or [])]
            single = False

        if not rows:
            empty = np.zeros((0, self.n_features), dtype=np.float32)
            return empty[0] if single else empty

        sparse = self._vectorizer.transform(rows)
        dense = sparse.astype(np.float32).toarray()

        if bool(normalize_embeddings):
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            dense = dense / norms

        return dense[0] if single else dense


def get_embedder() -> HashingSentenceEmbedder:
    n_features = int(os.getenv("EMBEDDING_DIM", "768") or 768)
    ngram_min = int(os.getenv("EMBEDDING_NGRAM_MIN", "3") or 3)
    ngram_max = int(os.getenv("EMBEDDING_NGRAM_MAX", "5") or 5)
    return HashingSentenceEmbedder(
        n_features=n_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
