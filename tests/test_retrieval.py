"""Unit tests for Component 5: Retrieval, Hybrid Ranking, and Calibrated Thresholding."""

import pytest
from app.embeddings import EmbeddingEngine
from app.ingestion import IngestionEngine
from app.retrieval import RecallEngine
from app.schemas import IngestRequest, RecallRequest
from app.storage import StorageManager
from app.updates import ConflictResolver


@pytest.fixture
def populated_engine(tmp_path):
    """Fixture that initializes isolated storage and populates the canonical 4 messages."""
    db_file = tmp_path / "test_retrieval.db"
    chroma_dir = tmp_path / "test_retrieval_chroma"
    storage = StorageManager(
        db_path=db_file,
        chroma_path=chroma_dir,
        embedding_engine=EmbeddingEngine(),
    )
    resolver = ConflictResolver(storage=storage, ingestion=IngestionEngine())

    # Ingest the 4 canonical messages
    resolver.process_ingest_request(IngestRequest(message="I use Arch Linux."))
    resolver.process_ingest_request(IngestRequest(message="I'm building a Next.js app."))
    resolver.process_ingest_request(IngestRequest(message="My project deadline is October 15."))
    resolver.process_ingest_request(
        IngestRequest(message="Actually, the deadline moved to November 2.")
    )

    return RecallEngine(storage=storage)


def test_canonical_recall_os_query(populated_engine):
    # Query: "What OS does the user use?" -> "User uses Arch Linux."
    resp = populated_engine.recall(RecallRequest(query="What OS does the user use?", top_k=5))
    assert len(resp.memories) >= 1
    top_memory = resp.memories[0]
    assert "Arch Linux" in top_memory.text
    assert top_memory.score > 0.60
    assert top_memory.type == "fact"


def test_canonical_recall_deadline_query(populated_engine):
    # Query: "When is the project deadline?" -> "Deadline is November 2."
    resp = populated_engine.recall(RecallRequest(query="When is the project deadline?", top_k=5))
    assert len(resp.memories) >= 1
    top_memory = resp.memories[0]
    assert "November 2" in top_memory.text
    assert top_memory.score > 0.60

    # CRITICAL: October 15 must NOT be in the retrieved results!
    all_texts = [m.text for m in resp.memories]
    assert not any("October 15" in t for t in all_texts)


def test_canonical_recall_unknown_query_rejected(populated_engine):
    # Query: "What is the user's favourite food?" -> no relevant memory, return empty list!
    resp = populated_engine.recall(
        RecallRequest(query="What is the user's favourite food?", top_k=5)
    )
    assert len(resp.memories) == 0
    assert resp.count == 0


def test_custom_top_k_parameter(populated_engine):
    resp = populated_engine.recall(
        RecallRequest(query="Tell me about user's system and project", top_k=1)
    )
    assert len(resp.memories) <= 1


def test_strict_custom_threshold(populated_engine):
    # Extremely high threshold should filter out everything
    resp = populated_engine.recall(
        RecallRequest(query="What OS does the user use?", top_k=5, threshold=0.99)
    )
    assert len(resp.memories) == 0


def test_hybrid_search_boosts_exact_keywords(populated_engine):
    # Query with exact keyword "Next.js"
    resp = populated_engine.recall(
        RecallRequest(query="Next.js application framework", top_k=3)
    )
    assert len(resp.memories) >= 1
    assert "Next.js" in resp.memories[0].text
    assert resp.memories[0].score > 0.70


def test_semantic_weight_priority(populated_engine):
    # Paraphrased query with no exact keyword overlap with "User uses Arch Linux."
    # With high semantic weight (alpha=0.95), dense semantic similarity dominates
    resp = populated_engine.recall(
        RecallRequest(query="Which operating system platform is the person running?", top_k=3, alpha=0.95)
    )
    assert len(resp.memories) >= 1
    assert "Arch Linux" in resp.memories[0].text
    assert resp.memories[0].score > 0.60

