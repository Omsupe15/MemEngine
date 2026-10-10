"""MemEngine: A local memory engine for AI chatbots.

Runs locally on your laptop, extracts durable facts, detects replacements and
contradictions, and retrieves relevant memories via hybrid semantic search with ChromaDB.
"""

from app.config import Settings, get_settings, settings
from app.embeddings import EmbeddingEngine, get_embedding_engine
from app.ingestion import (
    ExtractedCandidate,
    IngestionEngine,
    get_ingestion_engine,
)
from app.retrieval import RecallEngine, get_recall_engine
from app.schemas import (
    DeleteResponse,
    IngestRequest,
    IngestResponse,
    ListMemoriesResponse,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    RecallItem,
    RecallRequest,
    RecallResponse,
)
from app.storage import SQLiteStorage, StorageManager, get_storage_manager
from app.updates import ConflictResolver, get_conflict_resolver
from app.vector_store import ChromaVectorStore

__version__ = "0.1.0"
__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "MemoryType",
    "MemoryStatus",
    "MemoryRecord",
    "IngestRequest",
    "IngestResponse",
    "RecallRequest",
    "RecallItem",
    "RecallResponse",
    "ListMemoriesResponse",
    "DeleteResponse",
    "EmbeddingEngine",
    "get_embedding_engine",
    "ChromaVectorStore",
    "SQLiteStorage",
    "StorageManager",
    "get_storage_manager",
    "ExtractedCandidate",
    "IngestionEngine",
    "get_ingestion_engine",
    "ConflictResolver",
    "get_conflict_resolver",
    "RecallEngine",
    "get_recall_engine",
]
