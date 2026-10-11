"""Unit tests for Component 7: Evaluator and Benchmark Suite."""

import pytest
from app.evaluator import evaluate_single_case, load_evaluation_cases, run_full_evaluation


def test_load_evaluation_cases():
    cases = load_evaluation_cases()
    assert len(cases) >= 32
    # Verify required keys in each test case
    for c in cases:
        assert "id" in c
        assert "category" in c
        assert "conversation" in c
        assert "query" in c
        assert "is_unknown" in c


def test_evaluate_single_positive_case():
    case = {
        "id": "test_fact_arch",
        "category": "Direct Facts",
        "description": "OS fact test",
        "conversation": ["I use Arch Linux."],
        "query": "What OS does the user use?",
        "expected_positive": "Arch Linux",
        "expected_negative": None,
        "is_unknown": False,
        "min_expected_results": 1,
    }
    result = evaluate_single_case(case)
    assert result.passed is True
    assert result.top_1_match is True
    assert result.recall_at_5 is True
    assert result.retrieved_count >= 1
    assert "Arch Linux" in (result.top_text or "")


def test_evaluate_single_unknown_negative_case():
    case = {
        "id": "test_unknown_food",
        "category": "Unknown Answers",
        "description": "Unknown query test",
        "conversation": ["I use Arch Linux.", "I'm building a Next.js app."],
        "query": "What is the user's favourite food?",
        "expected_positive": None,
        "expected_negative": None,
        "is_unknown": True,
        "min_expected_results": 0,
    }
    result = evaluate_single_case(case)
    assert result.passed is True
    assert result.unknown_rejection is True
    assert result.retrieved_count == 0


def test_evaluate_single_supersede_case():
    case = {
        "id": "test_update_deadline",
        "category": "Updated Information",
        "description": "Deadline update test",
        "conversation": [
            "My project deadline is October 15.",
            "Actually, the deadline moved to November 2.",
        ],
        "query": "When is the project deadline?",
        "expected_positive": "November 2",
        "expected_negative": "October 15",
        "is_unknown": False,
        "min_expected_results": 1,
    }
    result = evaluate_single_case(case)
    assert result.passed is True
    assert result.top_1_match is True
    assert result.supersede_integrity is True
    assert "November 2" in (result.top_text or "")
    assert "October 15" not in (result.top_text or "")
