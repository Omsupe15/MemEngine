"""Storage layer for MemEngine.

Combines SQLite for durable relational memory records, lineage, and messages
with ChromaDB for fast nearest-neighbor vector retrieval.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.embeddings import EmbeddingEngine, get_embedding_engine
from app.schemas import MemoryRecord, MemoryStatus, MemoryType
from app.vector_store import ChromaVectorStore


class SQLiteStorage:
    """Manages SQLite schema, transactions, and relational queries."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.settings = get_settings()
        self.db_path = str(db_path or self.settings.resolved_db_path)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize database tables and indexes."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            # Memories table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    replaced_by INTEGER,
                    importance REAL NOT NULL DEFAULT 0.5,
                    FOREIGN KEY (replaced_by) REFERENCES memories(id)
                );
                """
            )

            # Indexes for high performance querying
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_subject_attr ON memories(subject, attribute);"
            )

            # Messages log table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    memories_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.commit()

    def insert_memory(self, record: MemoryRecord) -> int:
        """Insert a memory record into SQLite and return its auto-increment ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    text, type, subject, attribute, value,
                    created_at, updated_at, source_message,
                    status, replaced_by, importance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.text,
                    record.type.value if hasattr(record.type, "value") else str(record.type),
                    record.subject,
                    record.attribute,
                    record.value,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.source_message,
                    record.status.value if hasattr(record.status, "value") else str(record.status),
                    record.replaced_by,
                    record.importance,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_memory(self, memory_id: int) -> Optional[MemoryRecord]:
        """Fetch a memory by its ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?;", (memory_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def find_active_candidate(self, subject: str, attribute: str) -> Optional[MemoryRecord]:
        """Find an existing active memory matching (subject, attribute)."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM memories 
                WHERE LOWER(subject) = LOWER(?) 
                  AND LOWER(attribute) = LOWER(?) 
                  AND status = 'active'
                ORDER BY id DESC LIMIT 1;
                """,
                (subject, attribute),
            ).fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def supersede_memory(self, old_id: int, new_id: int) -> None:
        """Mark an old memory as superseded and link replaced_by to new_id."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE memories 
                SET status = 'superseded', 
                    replaced_by = ?, 
                    updated_at = ? 
                WHERE id = ?;
                """,
                (new_id, now, old_id),
            )
            conn.commit()

    def delete_memory(self, memory_id: int) -> bool:
        """Soft-delete a memory by setting status='deleted'."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE memories 
                SET status = 'deleted', 
                    updated_at = ? 
                WHERE id = ?;
                """,
                (now, memory_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_memories(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """List memories with optional status filter and pagination."""
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM memories 
                    WHERE status = ? 
                    ORDER BY id ASC 
                    LIMIT ? OFFSET ?;
                    """,
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories 
                    ORDER BY id ASC 
                    LIMIT ? OFFSET ?;
                    """,
                    (limit, offset),
                ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_active_memories(self) -> List[MemoryRecord]:
        """Return all active memories."""
        return self.list_memories(status="active", limit=1000, offset=0)

    def log_message(self, content: str, memories_count: int = 0) -> int:
        """Log incoming user message."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (content, created_at, memories_count) 
                VALUES (?, ?, ?);
                """,
                (content, now, memories_count),
            )
            conn.commit()
            return cursor.lastrowid

    def count_memories(self, status: Optional[str] = None) -> int:
        """Count memories in database."""
        with self._get_connection() as conn:
            if status:
                res = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE status = ?;", (status,)
                ).fetchone()
            else:
                res = conn.execute("SELECT COUNT(*) FROM memories;").fetchone()
            return res[0] if res else 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        """Convert a database row into a MemoryRecord Pydantic model."""
        return MemoryRecord(
            id=row["id"],
            text=row["text"],
            type=MemoryType(row["type"]),
            subject=row["subject"],
            attribute=row["attribute"],
            value=row["value"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_message=row["source_message"],
            status=MemoryStatus(row["status"]),
            replaced_by=row["replaced_by"],
            importance=row["importance"],
        )


class StorageManager:
    """High-level storage coordinator managing SQLite records and ChromaDB vectors."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        chroma_path: Optional[str | Path] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        self.sqlite = SQLiteStorage(db_path=db_path)
        self.vector_store = ChromaVectorStore(persist_directory=chroma_path)
        self.embedding_engine = embedding_engine or get_embedding_engine()

    def save_memory(
        self,
        record: MemoryRecord,
        embedding: Optional[List[float]] = None,
    ) -> MemoryRecord:
        """Save a new memory to SQLite and index its embedding in ChromaDB."""
        # 1. Insert into SQLite
        assigned_id = self.sqlite.insert_memory(record)
        saved_record = record.model_copy(update={"id": assigned_id})

        # 2. Compute embedding if not provided
        if embedding is None:
            embedding = self.embedding_engine.embed_text(saved_record.text)

        # 3. Insert into ChromaDB
        self.vector_store.add_memory(
            memory_id=assigned_id,
            text=saved_record.text,
            embedding=embedding,
            metadata={
                "status": saved_record.status.value,
                "type": saved_record.type.value,
                "subject": saved_record.subject,
                "attribute": saved_record.attribute,
                "importance": saved_record.importance,
            },
        )
        return saved_record

    def supersede_memory(self, old_id: int, new_id: int) -> None:
        """Supersede old memory in both SQLite and ChromaDB."""
        # 1. Update SQLite
        self.sqlite.supersede_memory(old_id=old_id, new_id=new_id)

        # 2. Update ChromaDB metadata to status='superseded'
        self.vector_store.update_status(memory_id=old_id, new_status="superseded")

    def delete_memory(self, memory_id: int) -> bool:
        """Soft-delete memory in SQLite and update ChromaDB."""
        success = self.sqlite.delete_memory(memory_id=memory_id)
        if success:
            self.vector_store.update_status(memory_id=memory_id, new_status="deleted")
        return success

    def get_memory(self, memory_id: int) -> Optional[MemoryRecord]:
        """Retrieve memory by ID."""
        return self.sqlite.get_memory(memory_id)

    def find_active_candidate(self, subject: str, attribute: str) -> Optional[MemoryRecord]:
        """Find active memory candidate by subject and attribute."""
        return self.sqlite.find_active_candidate(subject=subject, attribute=attribute)

    def list_memories(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """List memories."""
        return self.sqlite.list_memories(status=status, limit=limit, offset=offset)

    def get_active_memories(self) -> List[MemoryRecord]:
        """Get all active memories."""
        return self.sqlite.get_active_memories()

    def search_active_vectors(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB for active memories matching query."""
        query_embedding = self.embedding_engine.embed_text(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            status_filter="active",
        )


# Global storage manager singleton
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Return default singleton StorageManager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager
