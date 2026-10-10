"""Integration tests for Component 6: FastAPI Endpoints and Web Dashboard."""

import pytest
from fastapi.testclient import TestClient

from app.embeddings import EmbeddingEngine
from app.ingestion import IngestionEngine
from app.main import app
from app.retrieval import RecallEngine
from app.storage import StorageManager
from app.updates import ConflictResolver
import app.main as main_module


@pytest.fixture
def client(tmp_path):
    """Fixture providing isolated TestClient with isolated temporary databases."""
    db_file = tmp_path / "test_api.db"
    chroma_dir = tmp_path / "test_api_chroma"

    # Create isolated instances
    test_storage = StorageManager(
        db_path=db_file,
        chroma_path=chroma_dir,
        embedding_engine=EmbeddingEngine(),
    )
    test_resolver = ConflictResolver(
        storage=test_storage,
        ingestion=IngestionEngine(),
    )
    test_recall = RecallEngine(
        storage=test_storage,
        embedding_engine=test_storage.embedding_engine,
    )

    # Patch singletons in main module
    main_module.get_storage_manager = lambda: test_storage
    main_module.get_conflict_resolver = lambda: test_resolver
    main_module.get_recall_engine = lambda: test_recall

    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_dashboard_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "MemEngine" in res.text


def test_post_memory_conversational(client):
    res = client.post("/memory", json={"message": "I use Arch Linux."})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "created"
    assert data["memory"]["id"] == 1
    assert "Arch Linux" in data["memory"]["text"]


def test_post_memory_small_talk_ignored(client):
    res = client.post("/memory", json={"message": "ok thanks"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ignored"
    assert data["memory"] is None


def test_canonical_full_api_flow(client):
    # 1. Ingest Arch Linux
    c1 = client.post("/memory", json={"message": "I use Arch Linux."})
    assert c1.status_code == 200
    assert c1.json()["status"] == "created"

    # 2. Ingest Next.js
    c2 = client.post("/memory", json={"message": "I'm building a Next.js app."})
    assert c2.status_code == 200
    assert c2.json()["status"] == "created"

    # 3. Ingest October 15 deadline
    c3 = client.post("/memory", json={"message": "My project deadline is October 15."})
    assert c3.status_code == 200
    assert c3.json()["status"] == "created"
    id_oct15 = c3.json()["memory"]["id"]

    # 4. Ingest update: November 2
    c4 = client.post(
        "/memory", json={"message": "Actually, the deadline moved to November 2."}
    )
    assert c4.status_code == 200
    assert c4.json()["status"] == "superseded"
    assert c4.json()["superseded_id"] == id_oct15
    id_nov2 = c4.json()["memory"]["id"]

    # Recall 1: "What OS does the user use?" -> "User uses Arch Linux."
    r1 = client.post("/recall", json={"query": "What OS does the user use?", "top_k": 5})
    assert r1.status_code == 200
    mems1 = r1.json()["memories"]
    assert len(mems1) >= 1
    assert "Arch Linux" in mems1[0]["text"]
    assert mems1[0]["score"] > 0.60

    # Recall 2: "When is the project deadline?" -> "Deadline is November 2."
    r2 = client.post("/recall", json={"query": "When is the project deadline?", "top_k": 5})
    assert r2.status_code == 200
    mems2 = r2.json()["memories"]
    assert len(mems2) >= 1
    assert "November 2" in mems2[0]["text"]
    # Verify October 15 is NOT returned
    assert not any("October 15" in m["text"] for m in mems2)

    # Recall 3: "What is the user's favourite food?" -> no memories returned!
    r3 = client.post(
        "/recall", json={"query": "What is the user's favourite food?", "top_k": 5}
    )
    assert r3.status_code == 200
    assert len(r3.json()["memories"]) == 0

    # GET /memories filtering
    all_res = client.get("/memories?status=all")
    assert all_res.status_code == 200
    assert all_res.json()["total"] == 4

    active_res = client.get("/memories?status=active")
    assert active_res.status_code == 200
    assert active_res.json()["total"] == 3

    superseded_res = client.get("/memories?status=superseded")
    assert superseded_res.status_code == 200
    assert superseded_res.json()["total"] == 1
    assert superseded_res.json()["memories"][0]["id"] == id_oct15
    assert superseded_res.json()["memories"][0]["replaced_by"] == id_nov2


def test_delete_memory_endpoint(client):
    # Ingest a memory
    c = client.post("/memory", json={"message": "I live in Berlin."})
    mem_id = c.json()["memory"]["id"]

    # Delete it
    del_res = client.delete(f"/memory/{mem_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Subsequent recall should return nothing
    rec_res = client.post("/recall", json={"query": "Where does the user live?"})
    assert len(rec_res.json()["memories"]) == 0

    # Deleting nonexistent memory returns 404
    del_404 = client.delete("/memory/99999")
    assert del_404.status_code == 404


def test_direct_structured_memory_creation(client):
    req = {
        "text": "User prefers PostgreSQL.",
        "type": "preference",
        "subject": "database",
        "attribute": "preference",
        "value": "PostgreSQL",
        "importance": 0.9,
    }
    res = client.post("/memory", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "created"
    assert data["memory"]["value"] == "PostgreSQL"
