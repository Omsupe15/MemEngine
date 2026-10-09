"""MemEngine: A local memory engine for AI chatbots.

Runs locally on your laptop, extracts durable facts, detects replacements and
contradictions, and retrieves relevant memories via hybrid semantic search.
"""

from memengine.config import Settings, get_settings, settings
from memengine.schemas import (
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
]
