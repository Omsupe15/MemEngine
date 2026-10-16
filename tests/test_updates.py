"""Unit tests for Component 4: Updates, Contradiction Resolution, and History Lineage."""

import pytest
from app.embeddings import EmbeddingEngine
from app.ingestion import IngestionEngine
from app.schemas import IngestRequest, MemoryStatus
from app.storage import StorageManager
from app.updates import ConflictResolver


@pytest.fixture
def test_resolver(tmp_path):
    db_file = tmp_path / "test_updates.db"
    chroma_dir = tmp_path / "test_updates_chroma"
    storage = StorageManager(
        db_path=db_file,
        chroma_path=chroma_dir,
        embedding_engine=EmbeddingEngine(),
    )
    return ConflictResolver(storage=storage, ingestion=IngestionEngine())


def test_small_talk_is_ignored(test_resolver):
    resp = test_resolver.process_ingest_request(IngestRequest(message="ok thanks"))
    assert resp.status == "ignored"
    assert resp.memory is None
    assert len(test_resolver.storage.get_active_memories()) == 0


def test_canonical_four_messages_lifecycle(test_resolver):
    # 1. remember("I use Arch Linux.")
    r1 = test_resolver.process_ingest_request(IngestRequest(message="I use Arch Linux."))
    assert r1.status == "created"
    assert r1.memory is not None
    assert r1.memory.id == 1
    assert "Arch Linux" in r1.memory.text

    # 2. remember("I'm building a Next.js app.")
    r2 = test_resolver.process_ingest_request(IngestRequest(message="I'm building a Next.js app."))
    assert r2.status == "created"
    assert r2.memory is not None
    assert r2.memory.id == 2
    assert "Next.js" in r2.memory.text

    # 3. remember("My project deadline is October 15.")
    r3 = test_resolver.process_ingest_request(IngestRequest(message="My project deadline is October 15."))
    assert r3.status == "created"
    assert r3.memory is not None
    assert r3.memory.id == 3
    assert "October 15" in r3.memory.text
    assert r3.memory.status == MemoryStatus.ACTIVE

    # 4. remember("Actually, the deadline moved to November 2.")
    r4 = test_resolver.process_ingest_request(
        IngestRequest(message="Actually, the deadline moved to November 2.")
    )
    assert r4.status == "superseded"
    assert r4.superseded_id == 3
    assert r4.memory is not None
    assert r4.memory.id == 4
    assert "November 2" in r4.memory.text
    assert r4.memory.status == MemoryStatus.ACTIVE

    # Check SQLite historical integrity
    old_mem = test_resolver.storage.get_memory(3)
    assert old_mem is not None
    assert old_mem.status == MemoryStatus.SUPERSEDED
    assert old_mem.replaced_by == 4

    new_mem = test_resolver.storage.get_memory(4)
    assert new_mem is not None
    assert new_mem.status == MemoryStatus.ACTIVE
    assert new_mem.replaced_by is None

    # Total active memories should be exactly 3 (Arch Linux, Next.js, November 2)
    active_mems = test_resolver.storage.get_active_memories()
    assert len(active_mems) == 3
    active_ids = [m.id for m in active_mems]
    assert 3 not in active_ids
    assert 4 in active_ids


def test_idempotent_duplicate_message(test_resolver):
    r1 = test_resolver.process_ingest_request(IngestRequest(message="I use Arch Linux."))
    assert r1.status == "created"
    mem_id = r1.memory.id

    # Re-sending identical statement should not duplicate
    r2 = test_resolver.process_ingest_request(IngestRequest(message="I use Arch Linux."))
    assert r2.memory.id == mem_id
    assert len(test_resolver.storage.get_active_memories()) == 1


def test_direct_structured_fact_insertion(test_resolver):
    req = IngestRequest(
        text="User prefers PostgreSQL.",
        type="preference",
        subject="database",
        attribute="preference",
        value="PostgreSQL",
        importance=0.9,
    )
    resp = test_resolver.process_ingest_request(req)
    assert resp.status == "created"
    assert resp.memory is not None
    assert resp.memory.value == "PostgreSQL"
    assert resp.memory.importance == 0.9


def test_unrelated_messages_do_not_supersede(test_resolver):
    """Ensure unrelated messages under generic categories are not mistakenly superseded."""
    # 1. Ingest Italian food preference
    r1 = test_resolver.process_ingest_request(IngestRequest(message="I prefer Italian food."))
    assert r1.status == "created"
    assert r1.memory is not None
    assert r1.memory.id == 1
    assert "Italian food" in r1.memory.text

    # 2. Ingest completely unrelated rock music preference
    r2 = test_resolver.process_ingest_request(IngestRequest(message="I prefer rock music."))
    assert r2.status == "created"
    assert r2.memory is not None
    assert r2.memory.id == 2
    assert "rock music" in r2.memory.text

    # Verify both distinct preferences remain ACTIVE
    active_mems = test_resolver.storage.get_active_memories()
    assert len(active_mems) == 2
    active_ids = {m.id for m in active_mems}
    assert active_ids == {1, 2}

    # 3. Ingest an actual update to Italian food (Mexican food)
    r3 = test_resolver.process_ingest_request(
        IngestRequest(message="Actually, I prefer Mexican food.")
    )
    assert r3.status == "superseded"
    assert r3.superseded_id == 1  # Italian food superseded
    assert r3.memory is not None
    assert r3.memory.id == 3

    # Rock music (id 2) must remain active; Italian food (id 1) superseded; Mexican food (id 3) active
    active_mems_after = test_resolver.storage.get_active_memories()
    assert len(active_mems_after) == 2
    active_ids_after = {m.id for m in active_mems_after}
    assert active_ids_after == {2, 3}


def test_custom_conflict_threshold_behavior(test_resolver):
    """Validate that custom conflict_threshold dynamically controls superseding sensitivity."""
    # Ingest base OS fact
    r1 = test_resolver.process_ingest_request(IngestRequest(message="I use Arch Linux."))
    assert r1.status == "created"

    # With very strict threshold (0.95), "I switched to Fedora." (similarity ~0.66) does NOT supersede
    r2 = test_resolver.process_ingest_request(
        IngestRequest(message="I switched to Fedora.", conflict_threshold=0.95)
    )
    assert r2.status == "created"
    assert len(test_resolver.storage.get_active_memories()) == 2

    # Ingest another related fact with permissive threshold (0.50), which triggers supersede
    r3 = test_resolver.process_ingest_request(
        IngestRequest(message="Actually, I moved to Debian.", conflict_threshold=0.50)
    )
    assert r3.status == "superseded"
    assert r3.superseded_id in [1, 2]
