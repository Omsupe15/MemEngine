"""Unit tests for Component 4: Ingestion and Small Talk Filtering."""

import pytest
from app.ingestion import IngestionEngine, get_ingestion_engine
from app.schemas import MemoryType


@pytest.fixture
def engine():
    return IngestionEngine()


@pytest.mark.parametrize(
    "phrase",
    [
        "ok thanks",
        "okay",
        "thanks a lot",
        "cool",
        "sounds good",
        "hello",
        "good morning",
        "how are you?",
        "bye",
        "see you later",
        "got it",
    ],
)
def test_small_talk_detection(engine, phrase):
    assert engine.is_small_talk(phrase) is True
    candidate = engine.extract_candidate(phrase)
    assert candidate is None


def test_os_fact_extraction(engine):
    cand = engine.extract_candidate("I use Arch Linux.")
    assert cand is not None
    assert cand.type == MemoryType.FACT
    assert cand.subject == "os"
    assert cand.attribute == "user_os"
    assert "Arch Linux" in cand.value
    assert "User uses Arch Linux." in cand.text


def test_framework_project_detail_extraction(engine):
    cand = engine.extract_candidate("I'm building a Next.js app.")
    assert cand is not None
    assert cand.type == MemoryType.PROJECT_DETAIL
    assert cand.subject == "project"
    assert cand.attribute == "framework"
    assert "Next.js" in cand.value


def test_deadline_project_detail_extraction(engine):
    cand = engine.extract_candidate("My project deadline is October 15.")
    assert cand is not None
    assert cand.type == MemoryType.PROJECT_DETAIL
    assert cand.subject == "project"
    assert cand.attribute == "deadline"
    assert "October 15" in cand.value


def test_deadline_update_cue_extraction(engine):
    cand = engine.extract_candidate("Actually, the deadline moved to November 2.")
    assert cand is not None
    assert cand.type == MemoryType.PROJECT_DETAIL
    assert cand.subject == "project"
    assert cand.attribute == "deadline"
    assert "November 2" in cand.value
    assert cand.is_update_cue is True


def test_preference_extraction(engine):
    cand = engine.extract_candidate("I prefer PostgreSQL over MySQL.")
    assert cand is not None
    assert cand.type == MemoryType.PREFERENCE
    assert cand.subject == "database"
    assert cand.attribute == "preference"
    assert "PostgreSQL" in cand.value


def test_favorite_item_extraction(engine):
    cand = engine.extract_candidate("My favorite editor is Neovim.")
    assert cand is not None
    assert cand.type == MemoryType.PREFERENCE
    assert cand.subject == "editor"
    assert cand.attribute == "favorite"
    assert "Neovim" in cand.value


def test_goal_extraction(engine):
    cand = engine.extract_candidate("My goal is to learn Rust by December.")
    assert cand is not None
    assert cand.type == MemoryType.GOAL
    assert cand.subject == "goal"
    assert "Rust" in cand.value


def test_event_extraction(engine):
    cand = engine.extract_candidate("I'm attending PyCon next month.")
    assert cand is not None
    assert cand.type == MemoryType.EVENT
    assert cand.subject == "event"
    assert "PyCon" in cand.value
