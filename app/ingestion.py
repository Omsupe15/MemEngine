"""Ingestion and Classification Engine for MemEngine.

Filters conversational filler (small talk) and extracts structured facts,
preferences, project details, events, and goals into normalized records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple
from app.schemas import IngestRequest, MemoryRecord, MemoryStatus, MemoryType


@dataclass
class ExtractedCandidate:
    """Intermediate structured fact candidate extracted from raw text."""

    text: str
    type: MemoryType
    subject: str
    attribute: str
    value: str
    importance: float = 0.5
    is_update_cue: bool = False


# Common small talk patterns to discard
SMALL_TALK_PATTERNS = [
    r"^(?:ok|okay|k|cool|nice|great|awesome|sweet|alright|fine)[\s.!]*$",
    r"^(?:thanks|thank you|thx|ty|many thanks|thanks a lot)[\s.!]*$",
    r"^(?:ok thanks|okay thank you|cool thanks|got it thanks)[\s.!]*$",
    r"^(?:hello|hi|hey|greetings|howdy|good morning|good evening|good afternoon)[\s.!]*$",
    r"^(?:how are you|how's it going|what's up|sup|how are things)[\s.?!]*$",
    r"^(?:bye|goodbye|see ya|see you|see you later|cya|talk to you later)[\s.!]*$",
    r"^(?:yes|no|yep|nope|sure|definitely|absolutely|not really)[\s.!]*$",
    r"^(?:got it|understood|makes sense|sounds good|i see)[\s.!]*$",
    r"^(?:who are you|what can you do|help me|can you help)[\s.?!]*$",
]

# Update markers indicating changes to previous facts
UPDATE_CUES = [
    r"\bactually\b",
    r"\bmoved to\b",
    r"\bchanged to\b",
    r"\bswitched to\b",
    r"\bno longer\b",
    r"\bnot anymore\b",
    r"\binstead of\b",
    r"\breplaced by\b",
    r"\bcorrection\b",
    r"\bupdate\b",
]


class IngestionEngine:
    """Analyzes incoming conversational messages and extracts durable memories."""

    def __init__(self) -> None:
        self._small_talk_regexes = [
            re.compile(p, re.IGNORECASE) for p in SMALL_TALK_PATTERNS
        ]
        self._update_cue_regexes = [
            re.compile(p, re.IGNORECASE) for p in UPDATE_CUES
        ]

    def is_small_talk(self, message: str) -> bool:
        """Return True if message consists purely of small talk or greetings."""
        cleaned = message.strip()
        if len(cleaned) < 2:
            return True

        for regex in self._small_talk_regexes:
            if regex.match(cleaned):
                return True
        return False

    def has_update_cue(self, message: str) -> bool:
        """Check if message contains cues indicating an update to an existing fact."""
        return any(regex.search(message) for regex in self._update_cue_regexes)

    def extract_candidate(self, message: str) -> Optional[ExtractedCandidate]:
        """Extract a structured fact candidate from a message, or None if not memorable."""
        cleaned = message.strip()
        if self.is_small_talk(cleaned):
            return None

        is_update = self.has_update_cue(cleaned)

        # 1. Project deadline updates: e.g. "Actually, the deadline moved to November 2."
        m = re.search(
            r"(?:actually,?\s*)?(?:the\s*)?(?:project\s*)?deadline\s+(?:moved|changed|pushed|extended)\s+to\s+([^.!,]+)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"Deadline is {val}.",
                type=MemoryType.PROJECT_DETAIL,
                subject="project",
                attribute="deadline",
                value=val,
                importance=0.9,
                is_update_cue=True,
            )

        # 2. Project deadline declaration: e.g. "My project deadline is October 15."
        m = re.search(
            r"(?:my\s+project\s+deadline\s+is|the\s+deadline\s+is|project\s+deadline\s+is)\s+([^.!,]+)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"Deadline is {val}.",
                type=MemoryType.PROJECT_DETAIL,
                subject="project",
                attribute="deadline",
                value=val,
                importance=0.9,
                is_update_cue=is_update,
            )

        # 3. Operating System / Distro: e.g. "I use Arch Linux." or "I switched to Fedora."
        m = re.search(
            r"(?:i\s+switched\s+to|i\s+migrated\s+to|i\s+upgraded\s+to)\s+([A-Za-z0-9\s_\-\+]+?)(?:\s+from\s+[^.!,]+)?(?:[.!,]|$)",
            cleaned,
            re.IGNORECASE,
        )
        if m and any(os_term in cleaned.lower() for os_term in ["linux", "arch", "fedora", "ubuntu", "debian", "macos", "windows", "os"]):
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"User uses {val}.",
                type=MemoryType.FACT,
                subject="os",
                attribute="user_os",
                value=val,
                importance=0.85,
                is_update_cue=True,
            )

        m = re.search(
            r"(?:i\s+use|i'm\s+using|i\s+am\s+using|my\s+os\s+is|i\s+run)\s+([A-Za-z0-9\s_\-\+]+?)(?:[.!,]|$)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip()
            # If matches tech or os
            if any(term in val.lower() for term in ["linux", "arch", "ubuntu", "fedora", "macos", "windows", "debian", "gentoo", "bsd"]):
                return ExtractedCandidate(
                    text=f"User uses {val}.",
                    type=MemoryType.FACT,
                    subject="os",
                    attribute="user_os",
                    value=val,
                    importance=0.85,
                    is_update_cue=is_update,
                )

        # 4. Framework / Project building: e.g. "I'm building a Next.js app."
        m = re.search(
            r"(?:i'm\s+building|i\s+am\s+building|we\s+are\s+building|i'm\s+working\s+on)\s+(?:a|an)?\s*(.+?)(?:\s+(?:app|application|project|system|service))?[.!,]?$",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"User is building a {val} app.",
                type=MemoryType.PROJECT_DETAIL,
                subject="project",
                attribute="framework",
                value=val,
                importance=0.8,
                is_update_cue=is_update,
            )

        # 5. Preferences: e.g. "I prefer PostgreSQL." or "I love dark mode."
        m = re.search(
            r"(?:i\s+prefer|i\s+always\s+choose|my\s+preference\s+is)\s+([^.!,]+?)(?:\s+over\s+([^.!,]+))?(?:[.!,]|$)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip()
            subject = "preference"
            attr = "general"
            if any(db in val.lower() for db in ["postgres", "postgresql", "mysql", "sqlite", "mongodb", "redis"]):
                subject = "database"
                attr = "preference"
            elif any(theme in val.lower() for theme in ["dark mode", "light mode"]):
                subject = "theme"
                attr = "preference"
            elif any(ed in val.lower() for ed in ["vim", "neovim", "vscode", "vs code", "emacs"]):
                subject = "editor"
                attr = "preference"
            return ExtractedCandidate(
                text=f"User prefers {val}.",
                type=MemoryType.PREFERENCE,
                subject=subject,
                attribute=attr,
                value=val,
                importance=0.75,
                is_update_cue=is_update,
            )

        # 6. Favorite items: e.g. "My favorite editor is Neovim."
        m = re.search(
            r"my\s+favou?rite\s+([A-Za-z0-9_\-\s]+?)\s+is\s+([^.!,]+)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            category = m.group(1).strip()
            val = m.group(2).strip()
            return ExtractedCandidate(
                text=f"User's favorite {category} is {val}.",
                type=MemoryType.PREFERENCE,
                subject=category.lower().replace(" ", "_"),
                attribute="favorite",
                value=val,
                importance=0.7,
                is_update_cue=is_update,
            )

        # 7. Goals: e.g. "I want to learn Rust by December." or "My goal is to learn Rust."
        m = re.search(
            r"(?:my\s+goal\s+is\s+to|i\s+want\s+to\s+learn|i\s+plan\s+to\s+learn|i\s+aim\s+to)\s+([^.!,]+)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"User's goal is to {val}.",
                type=MemoryType.GOAL,
                subject="goal",
                attribute="learning",
                value=val,
                importance=0.7,
                is_update_cue=is_update,
            )

        # 8. Events: e.g. "I'm attending PyCon next month."
        m = re.search(
            r"(?:i'm\s+attending|i\s+am\s+attending|i'm\s+going\s+to|i\s+have\s+a\s+conference)\s+([^.!,]+)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"User is attending {val}.",
                type=MemoryType.EVENT,
                subject="event",
                attribute="attendance",
                value=val,
                importance=0.75,
                is_update_cue=is_update,
            )

        # 9. Generic durable personal declarations: e.g. "I live in Berlin." or "My name is Alice."
        m = re.search(r"i\s+live\s+in\s+([^.!,]+)", cleaned, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"User lives in {val}.",
                type=MemoryType.FACT,
                subject="location",
                attribute="city",
                value=val,
                importance=0.8,
                is_update_cue=is_update,
            )

        m = re.search(r"my\s+name\s+is\s+([^.!,]+)", cleaned, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"User's name is {val}.",
                type=MemoryType.FACT,
                subject="user",
                attribute="name",
                value=val,
                importance=0.9,
                is_update_cue=is_update,
            )

        m = re.search(r"i\s+work\s+as\s+(?:a|an)?\s*([^.!,]+)", cleaned, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            return ExtractedCandidate(
                text=f"User works as {val}.",
                type=MemoryType.FACT,
                subject="career",
                attribute="role",
                value=val,
                importance=0.8,
                is_update_cue=is_update,
            )

        # Fallback: if message is long enough and not small talk, create general fact
        if len(cleaned) >= 8 and not cleaned.endswith("?"):
            norm_text = cleaned.rstrip(".!?")
            return ExtractedCandidate(
                text=norm_text if norm_text.startswith("User") else f"User noted: {norm_text}.",
                type=MemoryType.FACT,
                subject="general",
                attribute="note",
                value=norm_text,
                importance=0.5,
                is_update_cue=is_update,
            )

        return None


# Global default instance
_ingestion_engine: Optional[IngestionEngine] = None


def get_ingestion_engine() -> IngestionEngine:
    """Return default singleton IngestionEngine."""
    global _ingestion_engine
    if _ingestion_engine is None:
        _ingestion_engine = IngestionEngine()
    return _ingestion_engine
