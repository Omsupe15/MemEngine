"""Unit tests for MemEngine configuration and schemas."""

from datetime import datetime, timezone
import pytest
from app.config import Settings, get_settings
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


def test_settings_defaults():
    settings = get_settings()
    assert settings.db_path.endswith("memengine.db")
    assert "MiniLM" in settings.embedding_model_name
    assert settings.default_similarity_threshold == 0.52
    assert settings.hybrid_alpha == 0.85
    assert settings.default_top_k == 5


def test_memory_record_creation():
    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id=1,
        text="User uses Arch Linux.",
        type=MemoryType.FACT,
        subject="os",
        attribute="user_os",
        value="Arch Linux",
        created_at=now,
        updated_at=now,
        source_message="I use Arch Linux.",
        status=MemoryStatus.ACTIVE,
        replaced_by=None,
        importance=0.8,
    )
    assert record.id == 1
    assert record.text == "User uses Arch Linux."
    assert record.status == MemoryStatus.ACTIVE
    assert record.importance == 0.8


def test_memory_record_superseded_validation():
    now = datetime.now(timezone.utc)
    old_record = MemoryRecord(
        id=10,
        text="User's project deadline is October 15.",
        type=MemoryType.PROJECT_DETAIL,
        subject="project",
        attribute="deadline",
        value="October 15",
        created_at=now,
        updated_at=now,
        source_message="My project deadline is October 15.",
        status=MemoryStatus.SUPERSEDED,
        replaced_by=11,
        importance=0.9,
    )
    assert old_record.status == MemoryStatus.SUPERSEDED
    assert old_record.replaced_by == 11


def test_ingest_request_cleaning():
    req = IngestRequest(message="   I am building a Next.js app.   ", conflict_threshold=0.65)
    assert req.message == "I am building a Next.js app."
    assert req.conflict_threshold == 0.65

    blank_req = IngestRequest(message="    ")
    assert blank_req.message is None


def test_recall_request_and_response():
    req = RecallRequest(query="When is the project deadline?", top_k=3, alpha=0.9)
    assert req.top_k == 3
    assert req.alpha == 0.9

    item = RecallItem(
        id=11,
        text="Project deadline is November 2.",
        type=MemoryType.PROJECT_DETAIL.value,
        score=0.94,
        subject="project",
        attribute="deadline",
        value="November 2",
    )
    resp = RecallResponse(memories=[item], query=req.query, count=1)
    assert len(resp.memories) == 1
    assert resp.memories[0].score == 0.94
    assert resp.memories[0].text == "Project deadline is November 2."


def test_app_exports():
    import app

    assert hasattr(app, "settings")
    assert hasattr(app, "MemoryRecord")
    assert hasattr(app, "StorageManager")
    assert hasattr(app, "ConflictResolver")
    assert hasattr(app, "RecallEngine")

