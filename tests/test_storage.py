"""Tests for Component 3: Embeddings, ChromaDB Vector Store, and SQLite Storage."""

from datetime import datetime, timezone
import pytest
import numpy as np

from memengine.embeddings import EmbeddingEngine
from memengine.schemas import MemoryRecord, MemoryStatus, MemoryType
from memengine.storage import SQLiteStorage, StorageManager
from memengine.vector_store import ChromaVectorStore


@pytest.fixture
def embedding_engine():
    return EmbeddingEngine()


@pytest.fixture
def temp_storage(tmp_path, embedding_engine):
    db_file = tmp_path / "test_memengine.db"
    chroma_dir = tmp_path / "test_chroma"
    return StorageManager(
        db_path=db_file,
        chroma_path=chroma_dir,
        embedding_engine=embedding_engine,
    )


def test_embedding_generation_and_normalization(embedding_engine):
    vec = embedding_engine.embed_text("I use Arch Linux.")
    assert len(vec) == 384
    # Verify L2 normalization
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_embedding_batch_and_cache(embedding_engine):
    texts = ["I use Arch Linux.", "I'm building a Next.js app."]
    vectors = embedding_engine.embed_texts(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384

    # Cached lookup should be identical
    cached_vec = embedding_engine.embed_text("I use Arch Linux.")
    assert vectors[0] == cached_vec


def test_semantic_similarity_discrimination(embedding_engine):
    os_fact = embedding_engine.embed_text("User uses Arch Linux.")
    os_query = embedding_engine.embed_text("What OS does the user use?")
    food_query = embedding_engine.embed_text("What is the user's favourite food?")

    sim_relevant = embedding_engine.cosine_similarity(os_fact, os_query)
    sim_unrelated = embedding_engine.cosine_similarity(os_fact, food_query)

    assert sim_relevant > 0.60
    assert sim_unrelated < 0.35
    assert sim_relevant > sim_unrelated + 0.30


def test_chroma_vector_store_operations(tmp_path, embedding_engine):
    chroma_dir = tmp_path / "chroma_unit_test"
    store = ChromaVectorStore(persist_directory=chroma_dir)

    vec1 = embedding_engine.embed_text("User uses Arch Linux.")
    vec2 = embedding_engine.embed_text("Project deadline is October 15.")

    store.add_memory(
        memory_id=1,
        text="User uses Arch Linux.",
        embedding=vec1,
        metadata={"status": "active", "type": "fact", "subject": "os", "attribute": "user_os"},
    )
    store.add_memory(
        memory_id=2,
        text="Project deadline is October 15.",
        embedding=vec2,
        metadata={"status": "active", "type": "project_detail", "subject": "project", "attribute": "deadline"},
    )

    assert store.count() == 2

    # Query OS
    q_vec = embedding_engine.embed_text("What OS does the user use?")
    results = store.search(q_vec, top_k=2, status_filter="active")
    assert len(results) == 2
    assert results[0]["memory_id"] == 1
    assert results[0]["score"] > 0.60

    # Mark memory 1 as superseded
    store.update_status(memory_id=1, new_status="superseded")
    active_results = store.search(q_vec, top_k=2, status_filter="active")
    # Only memory 2 is active now
    assert len(active_results) == 1
    assert active_results[0]["memory_id"] == 2


def test_sqlite_storage_crud(tmp_path):
    db_file = tmp_path / "sqlite_test.db"
    sqlite_store = SQLiteStorage(db_path=db_file)

    now = datetime.now(timezone.utc)
    rec1 = MemoryRecord(
        id=0,
        text="User's project deadline is October 15.",
        type=MemoryType.PROJECT_DETAIL,
        subject="project",
        attribute="deadline",
        value="October 15",
        created_at=now,
        updated_at=now,
        source_message="My project deadline is October 15.",
        status=MemoryStatus.ACTIVE,
        replaced_by=None,
        importance=0.9,
    )

    mem_id1 = sqlite_store.insert_memory(rec1)
    assert mem_id1 == 1

    fetched = sqlite_store.get_memory(1)
    assert fetched is not None
    assert fetched.value == "October 15"
    assert fetched.status == MemoryStatus.ACTIVE

    # Find active candidate
    candidate = sqlite_store.find_active_candidate("project", "deadline")
    assert candidate is not None
    assert candidate.id == 1

    # Insert updated memory and supersede old
    rec2 = MemoryRecord(
        id=0,
        text="User's project deadline is November 2.",
        type=MemoryType.PROJECT_DETAIL,
        subject="project",
        attribute="deadline",
        value="November 2",
        created_at=now,
        updated_at=now,
        source_message="Actually, the deadline moved to November 2.",
        status=MemoryStatus.ACTIVE,
        replaced_by=None,
        importance=0.9,
    )
    mem_id2 = sqlite_store.insert_memory(rec2)
    assert mem_id2 == 2

    sqlite_store.supersede_memory(old_id=mem_id1, new_id=mem_id2)

    old_fetched = sqlite_store.get_memory(mem_id1)
    assert old_fetched.status == MemoryStatus.SUPERSEDED
    assert old_fetched.replaced_by == mem_id2

    # Active memories should only return mem_id2
    active = sqlite_store.get_active_memories()
    assert len(active) == 1
    assert active[0].id == mem_id2


def test_storage_manager_end_to_end(temp_storage):
    now = datetime.now(timezone.utc)
    rec1 = MemoryRecord(
        id=0,
        text="User's project deadline is October 15.",
        type=MemoryType.PROJECT_DETAIL,
        subject="project",
        attribute="deadline",
        value="October 15",
        created_at=now,
        updated_at=now,
        source_message="My project deadline is October 15.",
        status=MemoryStatus.ACTIVE,
        replaced_by=None,
        importance=0.9,
    )
    saved1 = temp_storage.save_memory(rec1)
    assert saved1.id == 1

    # Search query
    res1 = temp_storage.search_active_vectors("When is the project deadline?", top_k=5)
    assert len(res1) == 1
    assert res1[0]["memory_id"] == 1
    assert res1[0]["score"] > 0.60

    # Save new update and supersede
    rec2 = MemoryRecord(
        id=0,
        text="User's project deadline is November 2.",
        type=MemoryType.PROJECT_DETAIL,
        subject="project",
        attribute="deadline",
        value="November 2",
        created_at=now,
        updated_at=now,
        source_message="Actually, the deadline moved to November 2.",
        status=MemoryStatus.ACTIVE,
        replaced_by=None,
        importance=0.9,
    )
    saved2 = temp_storage.save_memory(rec2)
    assert saved2.id == 2
    temp_storage.supersede_memory(old_id=saved1.id, new_id=saved2.id)

    # Search again: Only November 2 should be returned in active search!
    res2 = temp_storage.search_active_vectors("When is the project deadline?", top_k=5)
    assert len(res2) == 1
    assert res2[0]["memory_id"] == 2
    assert "November 2" in res2[0]["text"]

    # Historical retrieval in SQLite still retains memory 1 with status='superseded'
    old_record = temp_storage.get_memory(1)
    assert old_record.status == MemoryStatus.SUPERSEDED
    assert old_record.replaced_by == 2
