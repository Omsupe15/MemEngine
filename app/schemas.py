"""Data models and schemas for MemEngine.

Defines Pydantic representations for memories, ingestion requests, recall queries,
and API responses with strict validation and serializable types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryType(str, Enum):
    """Categorical classification of memories."""

    FACT = "fact"
    PREFERENCE = "preference"
    PROJECT_DETAIL = "project_detail"
    EVENT = "event"
    GOAL = "goal"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class MemoryRecord(BaseModel):
    """Canonical representation of a memory record."""

    id: int = Field(description="Unique identifier for the memory")
    text: str = Field(description="Extracted or declared memory text")
    type: MemoryType = Field(description="Classification of the memory")
    subject: str = Field(description="Subject entity of the fact (e.g. 'os', 'project')")
    attribute: str = Field(description="Attribute or property (e.g. 'user_os', 'deadline')")
    value: str = Field(description="Extracted value (e.g. 'Arch Linux', 'November 2')")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the memory was initially created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the memory was last modified or superseded",
    )
    source_message: str = Field(
        description="Original conversation message that produced this memory"
    )
    status: MemoryStatus = Field(
        default=MemoryStatus.ACTIVE,
        description="Current lifecycle status of the memory",
    )
    replaced_by: Optional[int] = Field(
        default=None,
        description="ID of the newer memory that superseded this one, if any",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relevance importance weight between 0.0 and 1.0",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 42,
                "text": "User's project deadline is November 2.",
                "type": "project_detail",
                "subject": "project",
                "attribute": "deadline",
                "value": "November 2",
                "created_at": "2026-10-09T16:00:00Z",
                "updated_at": "2026-10-09T16:00:00Z",
                "source_message": "Actually, the deadline moved to November 2.",
                "status": "active",
                "replaced_by": None,
                "importance": 0.9,
            }
        }
    )


class IngestRequest(BaseModel):
    """Request payload for POST /memory.

    Supports either natural language conversation messages or direct structured facts.
    """

    # Option A: Conversational message (auto-extraction)
    message: Optional[str] = Field(
        default=None,
        description="Conversational message from the user (e.g. 'I use Arch Linux.')",
    )

    # Option B: Direct structured fact insertion
    text: Optional[str] = Field(
        default=None,
        description="Explicit memory text if adding pre-formulated memory directly",
    )
    type: Optional[MemoryType] = Field(
        default=None,
        description="Memory type if inserting directly",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Entity subject if inserting directly",
    )
    attribute: Optional[str] = Field(
        default=None,
        description="Attribute name if inserting directly",
    )
    value: Optional[str] = Field(
        default=None,
        description="Attribute value if inserting directly",
    )
    importance: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Importance score override (0.0 to 1.0)",
    )

    @field_validator("message", "text")
    @classmethod
    def clean_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class IngestResponse(BaseModel):
    """Response returned after processing POST /memory."""

    status: str = Field(
        description="Outcome status: 'created', 'superseded', or 'ignored'"
    )
    memory: Optional[MemoryRecord] = Field(
        default=None,
        description="Created memory record if ingested",
    )
    superseded_id: Optional[int] = Field(
        default=None,
        description="ID of the previous memory superseded by this new one",
    )
    message: str = Field(
        description="Human-readable explanation of the ingestion result"
    )


class RecallRequest(BaseModel):
    """Request payload for POST /recall."""

    query: str = Field(
        min_length=1,
        description="Natural language question or query (e.g. 'What OS does the user use?')",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant memories to return",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance cutoff threshold; scores below this are discarded",
    )


class RecallItem(BaseModel):
    """Individual retrieved memory with matching score."""

    id: int = Field(description="Database ID of the memory")
    text: str = Field(description="Stored memory text")
    type: str = Field(description="Memory type category")
    score: float = Field(description="Normalized similarity score (0.0 to 1.0)")
    subject: str = Field(description="Subject entity")
    attribute: str = Field(description="Attribute name")
    value: str = Field(description="Attribute value")
    importance: float = Field(default=0.5, description="Memory importance")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "text": "User prefers PostgreSQL.",
                "type": "preference",
                "score": 0.91,
                "subject": "database",
                "attribute": "preference",
                "value": "PostgreSQL",
                "importance": 0.8,
            }
        }
    )


class RecallResponse(BaseModel):
    """Response returned from POST /recall."""

    memories: List[RecallItem] = Field(
        default_factory=list,
        description="List of relevant active memories ranked by score",
    )
    query: str = Field(description="Original search query")
    count: int = Field(description="Total count of retrieved memories")


class ListMemoriesResponse(BaseModel):
    """Response returned from GET /memories."""

    memories: List[MemoryRecord] = Field(description="List of memories")
    total: int = Field(description="Total count of returned memories")
    status_filter: Optional[str] = Field(
        default=None, description="Status filter applied, if any"
    )


class DeleteResponse(BaseModel):
    """Response returned from DELETE /memory/{id}."""

    success: bool = Field(description="Whether the memory was removed")
    id: int = Field(description="ID of the deleted memory")
    message: str = Field(description="Summary message")
