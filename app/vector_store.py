"""ChromaDB vector store adapter for MemEngine.

Provides persistent vector indexing and fast HNSW nearest-neighbor semantic search
with metadata filtering (e.g. status='active') and cosine distance calibration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.config import get_settings


class ChromaVectorStore:
    """Manages ChromaDB vector collection and semantic nearest neighbor searches."""

    def __init__(
        self,
        persist_directory: Optional[str | Path] = None,
        collection_name: str = "memengine_memories",
        client: Optional[ClientAPI] = None,
    ) -> None:
        self.settings = get_settings()
        self.collection_name = collection_name

        if client is not None:
            self._client = client
        elif persist_directory == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            path = Path(persist_directory or self.settings.resolved_chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))

        # Create or fetch collection using cosine similarity metric
        self._collection: Collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Collection:
        """Return the underlying ChromaDB Collection."""
        return self._collection

    def add_memory(
        self,
        memory_id: int,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Upsert a memory vector and its metadata into ChromaDB."""
        doc_id = str(memory_id)
        clean_metadata: Dict[str, Any] = {
            "memory_id": int(memory_id),
            "status": str(metadata.get("status", "active")),
            "type": str(metadata.get("type", "fact")),
            "subject": str(metadata.get("subject", "")),
            "attribute": str(metadata.get("attribute", "")),
            "importance": float(metadata.get("importance", 0.5)),
        }

        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[clean_metadata],
        )

    def update_status(self, memory_id: int, new_status: str) -> None:
        """Update the lifecycle status in ChromaDB metadata (e.g. 'superseded' or 'deleted')."""
        doc_id = str(memory_id)
        existing = self._collection.get(ids=[doc_id], include=["metadatas"])
        if existing and existing["metadatas"] and len(existing["metadatas"]) > 0:
            current_meta = dict(existing["metadatas"][0])
            current_meta["status"] = str(new_status)
            self._collection.update(
                ids=[doc_id],
                metadatas=[current_meta],
            )

    def delete_memory(self, memory_id: int) -> None:
        """Remove a memory vector from ChromaDB."""
        doc_id = str(memory_id)
        self._collection.delete(ids=[doc_id])

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        status_filter: Optional[str] = "active",
        type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform nearest-neighbor vector search with optional status/type filtering.

        Returns list of dicts:
        [{"memory_id": int, "text": str, "score": float, "distance": float, "metadata": dict}]
        """
        where_clause: Dict[str, Any] = {}
        conditions = []
        if status_filter:
            conditions.append({"status": status_filter})
        if type_filter:
            conditions.append({"type": type_filter})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        total_count = self._collection.count()
        if total_count == 0:
            return []

        actual_k = min(top_k, total_count)
        query_args: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            query_args["where"] = where_clause

        try:
            results = self._collection.query(**query_args)
        except Exception:
            return []

        retrieved: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for idx, doc_id in enumerate(ids):
            dist = float(distances[idx]) if idx < len(distances) else 1.0
            cos_sim = max(0.0, min(1.0, 1.0 - dist))
            meta = metadatas[idx] if idx < len(metadatas) else {}
            doc_text = documents[idx] if idx < len(documents) else ""
            mem_id = int(meta.get("memory_id", doc_id))

            retrieved.append(
                {
                    "memory_id": mem_id,
                    "text": doc_text,
                    "score": cos_sim,
                    "distance": dist,
                    "metadata": meta,
                }
            )

        retrieved.sort(key=lambda x: x["score"], reverse=True)
        return retrieved

    def count(self) -> int:
        """Return total number of vectors in collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Reset or clear all vectors in collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
