"""Retrieval and Hybrid Recall Engine for MemEngine.

Combines semantic nearest-neighbor search from ChromaDB with BM25 lexical keyword matching,
calibrated relevance cutoff thresholding, and importance score ranking.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.embeddings import EmbeddingEngine, get_embedding_engine
from app.schemas import RecallItem, RecallRequest, RecallResponse
from app.storage import StorageManager, get_storage_manager


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 indexing and querying."""
    return re.findall(r"\b[A-Za-z0-9_+\-.]+\b", text.lower())


class RecallEngine:
    """Hybrid retrieval engine combining semantic vector search with BM25 lexical search."""

    def __init__(
        self,
        storage: Optional[StorageManager] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage = storage or get_storage_manager()
        self.embeddings = embedding_engine or get_embedding_engine()

    def recall(self, request: RecallRequest) -> RecallResponse:
        """Execute hybrid recall search for active memories matching query."""
        query = request.query.strip()
        top_k = min(request.top_k, self.settings.max_top_k)
        threshold = (
            request.threshold
            if request.threshold is not None
            else self.settings.default_similarity_threshold
        )

        # 1. Fetch all active memories from SQLite
        active_records = self.storage.get_active_memories()
        if not active_records:
            return RecallResponse(memories=[], query=query, count=0)

        # Map active records by ID for fast lookup
        record_map = {rec.id: rec for rec in active_records}

        # 2. Semantic search using ChromaDB
        query_embedding = self.embeddings.embed_text(query)
        semantic_results = self.storage.vector_store.search(
            query_embedding=query_embedding,
            top_k=len(active_records),
            status_filter="active",
        )

        # Build map of semantic similarity scores {memory_id: cosine_sim}
        semantic_scores: Dict[int, float] = {}
        for r in semantic_results:
            mem_id = r["memory_id"]
            if mem_id in record_map:
                semantic_scores[mem_id] = float(r["score"])

        for rec in active_records:
            if rec.id not in semantic_scores:
                rec_emb = self.embeddings.embed_text(rec.text)
                sim = self.embeddings.cosine_similarity(query_embedding, rec_emb)
                semantic_scores[rec.id] = max(0.0, min(1.0, sim))

        # 3. Lexical BM25 search
        query_tokens = tokenize(query)
        corpus_ids = [rec.id for rec in active_records]
        tokenized_corpus = [tokenize(rec.text) for rec in active_records]

        bm25_scores: Dict[int, float] = {rec_id: 0.0 for rec_id in corpus_ids}
        if query_tokens and tokenized_corpus:
            try:
                bm25 = BM25Okapi(tokenized_corpus)
                raw_bm25 = bm25.get_scores(query_tokens)
                max_bm25 = float(np.max(raw_bm25)) if len(raw_bm25) > 0 else 0.0
                if max_bm25 > 0.0:
                    for idx, rec_id in enumerate(corpus_ids):
                        bm25_scores[rec_id] = float(raw_bm25[idx] / max_bm25)
            except Exception:
                pass

        # 4. Hybrid score combination & importance weighting
        alpha = self.settings.hybrid_alpha
        beta = self.settings.importance_weight

        scored_candidates: List[RecallItem] = []
        for rec in active_records:
            s_sem = semantic_scores.get(rec.id, 0.0)
            s_bm25 = bm25_scores.get(rec.id, 0.0)
            importance_mod = beta * (rec.importance - 0.5)

            # Hybrid combination: semantic similarity forms the baseline, BM25 provides lexical boost
            lexical_boost = (1.0 - alpha) * s_bm25
            base_score = max(s_sem, (alpha * s_sem) + lexical_boost)
            hybrid_score = base_score + importance_mod
            hybrid_score = max(0.0, min(1.0, hybrid_score))

            if hybrid_score >= threshold:
                item = RecallItem(
                    id=rec.id,
                    text=rec.text,
                    type=rec.type.value if hasattr(rec.type, "value") else str(rec.type),
                    score=round(hybrid_score, 4),
                    subject=rec.subject,
                    attribute=rec.attribute,
                    value=rec.value,
                    importance=rec.importance,
                )
                scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        final_results = scored_candidates[:top_k]

        return RecallResponse(
            memories=final_results,
            query=query,
            count=len(final_results),
        )


# Global default instance
_recall_engine: Optional[RecallEngine] = None


def get_recall_engine() -> RecallEngine:
    """Return default singleton RecallEngine."""
    global _recall_engine
    if _recall_engine is None:
        _recall_engine = RecallEngine()
    return _recall_engine
