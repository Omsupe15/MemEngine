"""Configuration settings for MemEngine.

Manages application parameters, persistence paths, local embedding models,
similarity thresholds, and hybrid search weights. Supports environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Immutable application settings with environment variable fallbacks."""

    # Storage settings
    db_path: str = field(
        default_factory=lambda: os.getenv("MEMENGINE_DB_PATH", "memengine.db")
    )
    chroma_path: str = field(
        default_factory=lambda: os.getenv("MEMENGINE_CHROMA_PATH", "chroma_db")
    )

    # Embedding settings
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "MEMENGINE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("MEMENGINE_EMBEDDING_DIM", "384"))
    )

    # Retrieval settings
    default_similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("MEMENGINE_SIMILARITY_THRESHOLD", "0.52")
        )
    )
    hybrid_alpha: float = field(
        default_factory=lambda: float(os.getenv("MEMENGINE_HYBRID_ALPHA", "0.75"))
    )
    importance_weight: float = field(
        default_factory=lambda: float(os.getenv("MEMENGINE_IMPORTANCE_WEIGHT", "0.10"))
    )
    default_top_k: int = field(
        default_factory=lambda: int(os.getenv("MEMENGINE_DEFAULT_TOP_K", "5"))
    )
    max_top_k: int = field(
        default_factory=lambda: int(os.getenv("MEMENGINE_MAX_TOP_K", "50"))
    )

    # Server settings
    host: str = field(
        default_factory=lambda: os.getenv("MEMENGINE_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("MEMENGINE_PORT", "8000"))
    )

    @property
    def resolved_db_path(self) -> Path:
        """Return absolute Path for database file."""
        return Path(self.db_path).resolve()

    @property
    def resolved_chroma_path(self) -> Path:
        """Return absolute Path for ChromaDB directory."""
        return Path(self.chroma_path).resolve()


# Singleton instance
settings = Settings()


def get_settings() -> Settings:
    """Return active settings instance."""
    return settings
