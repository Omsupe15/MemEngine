"""FastAPI REST API server and Interactive Dashboard for MemEngine.

Exposes REST endpoints:
- POST /memory : Ingest conversation messages or structured facts
- POST /recall : Query active memories via hybrid semantic search
- GET /memories : List memories with status filtering and pagination
- DELETE /memory/{id} : Soft-delete memory
- GET /health : Health check
- GET / : Interactive web dashboard
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.retrieval import get_recall_engine
from app.schemas import (
    DeleteResponse,
    IngestRequest,
    IngestResponse,
    ListMemoriesResponse,
    RecallRequest,
    RecallResponse,
)
from app.storage import get_storage_manager
from app.updates import get_conflict_resolver

app = FastAPI(
    title="MemEngine API",
    description="Local-first cognitive memory and semantic recall engine for AI chatbots",
    version="0.1.0",
)

# Enable CORS for local client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and Web Dashboard
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the interactive web UI dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse(
        """<!DOCTYPE html>
<html>
<head><title>MemEngine</title></head>
<body>
<h1>MemEngine API</h1>
<p>MemEngine API is online. Interactive UI is available via Streamlit: <code>uv run streamlit run streamlit_app.py</code></p>
<p>API documentation available at <a href="/docs">/docs</a>.</p>
</body>
</html>"""
    )


@app.get("/health", tags=["System"])
async def health_check():
    """Return health status and count of active memories."""
    storage = get_storage_manager()
    active_mems = storage.get_active_memories()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "active_memories": len(active_mems),
        "total_memories": storage.sqlite.count_memories(),
    }


@app.post("/memory", response_model=IngestResponse, tags=["Memory"])
async def ingest_memory(request: IngestRequest) -> IngestResponse:
    """Ingest a message or memory.

    Accepts conversational turns, filters small talk, extracts structured facts,
    and automatically supersedes conflicting historical facts.
    """
    resolver = get_conflict_resolver()
    return resolver.process_ingest_request(request)


@app.post("/recall", response_model=RecallResponse, tags=["Memory"])
async def recall_memories(request: RecallRequest) -> RecallResponse:
    """Retrieve top relevant active memories for a query.

    Uses hybrid semantic (ChromaDB) + lexical (BM25) search with a calibrated
    relevance threshold to avoid returning unrelated facts.
    """
    recall_engine = get_recall_engine()
    return recall_engine.recall(request)


@app.get("/memories", response_model=ListMemoriesResponse, tags=["Memory"])
async def list_memories(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: 'active', 'superseded', 'deleted', or 'all'",
    ),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ListMemoriesResponse:
    """List stored memories with optional status filter and pagination."""
    storage = get_storage_manager()
    status_filter = None if status in (None, "all") else status
    records = storage.list_memories(status=status_filter, limit=limit, offset=offset)
    total_count = storage.sqlite.count_memories(status=status_filter)
    return ListMemoriesResponse(
        memories=records,
        total=total_count,
        status_filter=status,
    )


@app.delete("/memory/{memory_id}", response_model=DeleteResponse, tags=["Memory"])
async def delete_memory(memory_id: int) -> DeleteResponse:
    """Remove a memory by ID.

    Marks the memory as deleted in SQLite and ChromaDB so it is excluded from search.
    """
    storage = get_storage_manager()
    existing = storage.get_memory(memory_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=f"Memory with ID {memory_id} not found.",
        )

    storage.delete_memory(memory_id)
    return DeleteResponse(
        success=True,
        id=memory_id,
        message=f"Memory #{memory_id} successfully deleted.",
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
