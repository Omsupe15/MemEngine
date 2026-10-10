"""Update and Contradiction Resolution Engine for MemEngine.

Detects conflicts, replacements, and revisions to existing memories.
When new information contradicts or updates an old memory, the new memory
becomes active while the old memory is marked as superseded with full lineage preserved.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from app.ingestion import ExtractedCandidate, IngestionEngine, get_ingestion_engine
from app.schemas import (
    IngestRequest,
    IngestResponse,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
)
from app.storage import StorageManager, get_storage_manager


class ConflictResolver:
    """Detects and resolves memory updates, replacements, and contradictions."""

    def __init__(
        self,
        storage: Optional[StorageManager] = None,
        ingestion: Optional[IngestionEngine] = None,
    ) -> None:
        self.storage = storage or get_storage_manager()
        self.ingestion = ingestion or get_ingestion_engine()

    def process_ingest_request(self, req: IngestRequest) -> IngestResponse:
        """Process an incoming IngestRequest (message or direct fact) through conflict resolution."""
        # 1. Direct structured fact insertion path
        if req.text is not None and req.subject is not None and req.attribute is not None:
            candidate = ExtractedCandidate(
                text=req.text,
                type=req.type or MemoryType.FACT,
                subject=req.subject,
                attribute=req.attribute,
                value=req.value or req.text,
                importance=req.importance if req.importance is not None else 0.5,
                is_update_cue=False,
            )
            source_msg = req.text
            return self._resolve_and_save(candidate, source_msg)

        # 2. Conversational message extraction path
        if req.message:
            # Check for small talk / conversational filler
            if self.ingestion.is_small_talk(req.message):
                # Log message to database but produce no memory
                self.storage.sqlite.log_message(req.message, memories_count=0)
                return IngestResponse(
                    status="ignored",
                    memory=None,
                    superseded_id=None,
                    message="Conversational filler or small talk ignored. No memory created.",
                )

            candidate = self.ingestion.extract_candidate(req.message)
            if candidate is None:
                self.storage.sqlite.log_message(req.message, memories_count=0)
                return IngestResponse(
                    status="ignored",
                    memory=None,
                    superseded_id=None,
                    message="No durable factual information detected in message.",
                )

            # Apply importance override if supplied in request
            if req.importance is not None:
                candidate.importance = req.importance

            return self._resolve_and_save(candidate, req.message)

        return IngestResponse(
            status="ignored",
            memory=None,
            superseded_id=None,
            message="Empty request. Please provide 'message' or structured memory fields.",
        )

    def _resolve_and_save(
        self,
        candidate: ExtractedCandidate,
        source_message: str,
    ) -> IngestResponse:
        """Evaluate candidate against active memories, supersede conflicts, and persist."""
        now = datetime.now(timezone.utc)

        # Step 1: Look for exact active candidate match on (subject, attribute)
        existing = self.storage.find_active_candidate(
            subject=candidate.subject,
            attribute=candidate.attribute,
        )

        # Step 2: If no exact match on (subject, attribute), check semantic collision if update cue present
        if existing is None and candidate.is_update_cue:
            active_matches = self.storage.search_active_vectors(
                query=candidate.text,
                top_k=3,
            )
            for match in active_matches:
                meta = match.get("metadata", {})
                if (
                    meta.get("subject") == candidate.subject
                    or match.get("score", 0.0) >= 0.82
                ):
                    matched_id = match["memory_id"]
                    found_rec = self.storage.get_memory(matched_id)
                    if found_rec and found_rec.status == MemoryStatus.ACTIVE:
                        existing = found_rec
                        break

        # Step 3: Handle detected existing active memory
        if existing is not None:
            # If the value is identical, treat as idempotent confirmation
            if existing.value.strip().lower() == candidate.value.strip().lower():
                self.storage.sqlite.log_message(source_message, memories_count=0)
                return IngestResponse(
                    status="created",
                    memory=existing,
                    superseded_id=None,
                    message=f"Memory already active with identical value: {existing.text}",
                )

            # Contradiction / Update detected!
            new_record = MemoryRecord(
                id=0,  # Auto-generated by SQLite
                text=candidate.text,
                type=candidate.type,
                subject=candidate.subject,
                attribute=candidate.attribute,
                value=candidate.value,
                created_at=now,
                updated_at=now,
                source_message=source_message,
                status=MemoryStatus.ACTIVE,
                replaced_by=None,
                importance=candidate.importance,
            )

            # Persist new memory in SQLite and ChromaDB
            saved_new_record = self.storage.save_memory(new_record)

            # Supersede old memory in both SQLite and ChromaDB
            self.storage.supersede_memory(
                old_id=existing.id,
                new_id=saved_new_record.id,
            )

            self.storage.sqlite.log_message(source_message, memories_count=1)

            return IngestResponse(
                status="superseded",
                memory=saved_new_record,
                superseded_id=existing.id,
                message=(
                    f"Updated existing memory #{existing.id} ('{existing.text}'). "
                    f"New memory #{saved_new_record.id} is now active."
                ),
            )

        # Step 4: No conflict -> Create brand new active memory
        new_record = MemoryRecord(
            id=0,
            text=candidate.text,
            type=candidate.type,
            subject=candidate.subject,
            attribute=candidate.attribute,
            value=candidate.value,
            created_at=now,
            updated_at=now,
            source_message=source_message,
            status=MemoryStatus.ACTIVE,
            replaced_by=None,
            importance=candidate.importance,
        )

        saved_record = self.storage.save_memory(new_record)
        self.storage.sqlite.log_message(source_message, memories_count=1)

        return IngestResponse(
            status="created",
            memory=saved_record,
            superseded_id=None,
            message=f"Created new active memory #{saved_record.id}.",
        )


# Global default instance
_conflict_resolver: Optional[ConflictResolver] = None


def get_conflict_resolver() -> ConflictResolver:
    """Return default singleton ConflictResolver."""
    global _conflict_resolver
    if _conflict_resolver is None:
        _conflict_resolver = ConflictResolver()
    return _conflict_resolver
