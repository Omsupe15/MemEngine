"""Local embedding engine for MemEngine.

Wraps Sentence Transformers for generating local semantic embeddings on CPU.
Includes lazy model initialization, L2 normalization, LRU caching, and cosine similarity.
"""

from __future__ import annotations

import functools
import threading
from typing import List, Sequence, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingEngine:
    """Thread-safe local embedding generator using Sentence Transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model_name
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}
        self._cache_capacity = 2048

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Load model onto CPU locally
                    self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized 384-dim float embedding for a single string.

        Uses in-memory LRU cache to avoid recomputing embeddings for repeated strings.
        """
        cleaned = text.strip()
        if not cleaned:
            # Return zero vector if string is empty
            return [0.0] * self.settings.embedding_dimension

        with self._lock:
            if cleaned in self._cache:
                return self._cache[cleaned]

        # Generate normalized embedding
        vector = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vector_list = vector.astype(float).tolist()

        with self._lock:
            if len(self._cache) >= self._cache_capacity:
                # Evict first inserted key (FIFO fallback for simple LRU)
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[cleaned] = vector_list

        return vector_list

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch-generate normalized embeddings for multiple strings."""
        if not texts:
            return []

        # Find items already cached vs uncached
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        with self._lock:
            for idx, text in enumerate(texts):
                cleaned = text.strip()
                if cleaned in self._cache:
                    results[idx] = self._cache[cleaned]
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(cleaned)

        if uncached_texts:
            encoded_vectors = self.model.encode(
                uncached_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            with self._lock:
                for idx, vector in zip(uncached_indices, encoded_vectors):
                    vec_list = vector.astype(float).tolist()
                    results[idx] = vec_list
                    if len(self._cache) < self._cache_capacity:
                        self._cache[texts[idx].strip()] = vec_list

        return [r if r is not None else [0.0] * self.settings.embedding_dimension for r in results]

    @staticmethod
    def cosine_similarity(
        vec_a: Union[Sequence[float], np.ndarray],
        vec_b: Union[Sequence[float], np.ndarray],
    ) -> float:
        """Compute cosine similarity between two vectors.

        Because vectors are L2-normalized upon encoding, cosine similarity equals dot product.
        """
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Global default instance
_embedding_engine: EmbeddingEngine | None = None


def get_embedding_engine() -> EmbeddingEngine:
    """Return default singleton EmbeddingEngine instance."""
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine
