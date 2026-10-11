"""Automated Evaluation & Benchmark Suite for MemEngine.

Executes 35+ test cases across 7 categories to compute:
- Recall@5
- Top-1 Accuracy
- Unknown Rejection Rate
- Update & Supersede Integrity
- Category-by-category breakdown & Latency
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.embeddings import EmbeddingEngine, get_embedding_engine
from app.ingestion import IngestionEngine
from app.retrieval import RecallEngine
from app.schemas import IngestRequest, RecallRequest
from app.storage import StorageManager
from app.updates import ConflictResolver


@dataclass
class TestCaseResult:
    """Individual test case evaluation output."""

    case_id: str
    category: str
    description: str
    query: str
    passed: bool
    top_1_match: bool
    recall_at_5: bool
    supersede_integrity: bool
    unknown_rejection: bool
    retrieved_count: int
    top_text: Optional[str]
    top_score: Optional[float]
    latency_ms: float
    notes: str


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics across all test cases."""

    total_cases: int
    passed_cases: int
    overall_accuracy: float
    top_1_accuracy: float
    recall_at_5: float
    unknown_rejection_rate: float
    supersede_integrity_rate: float
    avg_latency_ms: float
    category_metrics: Dict[str, Dict[str, float]]
    case_results: List[TestCaseResult] = field(default_factory=list)


def load_evaluation_cases(cases_path: Optional[Path | str] = None) -> List[dict]:
    """Load evaluation test cases from JSON file."""
    if cases_path is None:
        cases_path = Path(__file__).parent.parent / "tests" / "evaluation_cases.json"
    p = Path(cases_path)
    if not p.exists():
        raise FileNotFoundError(f"Evaluation cases file not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single_case(
    case: dict,
    embedding_engine: Optional[EmbeddingEngine] = None,
) -> TestCaseResult:
    """Run an isolated test case with ephemeral database and evaluate retrieval."""
    if embedding_engine is None:
        embedding_engine = get_embedding_engine()

    storage = StorageManager(
        db_path=":memory:",
        chroma_path=":memory:",
        embedding_engine=embedding_engine,
    )
    resolver = ConflictResolver(storage=storage, ingestion=IngestionEngine())
    recall_engine = RecallEngine(storage=storage, embedding_engine=embedding_engine)

    # 1. Ingest conversation messages sequentially
    for msg in case.get("conversation", []):
        resolver.process_ingest_request(IngestRequest(message=msg))

    # 2. Query recall engine and measure latency
    t0 = time.perf_counter()
    query = case["query"]
    recall_resp = recall_engine.recall(RecallRequest(query=query, top_k=5))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    retrieved = recall_resp.memories
    retrieved_count = len(retrieved)
    top_text = retrieved[0].text if retrieved else None
    top_score = retrieved[0].score if retrieved else None

    # 3. Evaluate assertions
    is_unknown = case.get("is_unknown", False)
    expected_pos = case.get("expected_positive")
    expected_neg = case.get("expected_negative")
    min_expected = case.get("min_expected_results", 1)

    notes_parts = []

    if is_unknown:
        unknown_rejection = (retrieved_count == 0)
        top_1_match = True
        recall_at_5 = True
        supersede_integrity = True
        passed = unknown_rejection
        if passed:
            notes_parts.append("Correctly rejected unknown query without hallucination.")
        else:
            notes_parts.append(f"Failed to reject unknown query; retrieved {retrieved_count} memories.")
    else:
        unknown_rejection = True  # Not applicable to positive queries

        # Check positive match
        top_1_match = False
        recall_at_5 = False
        if expected_pos:
            ep_lower = expected_pos.lower()
            if retrieved:
                top_rec = retrieved[0]
                if ep_lower in top_rec.text.lower() or ep_lower in (top_rec.value or "").lower():
                    top_1_match = True

            for r in retrieved[:5]:
                if ep_lower in r.text.lower() or ep_lower in (r.value or "").lower():
                    recall_at_5 = True
                    break
        else:
            top_1_match = True
            recall_at_5 = True

        # Check negative match (superseded memory must not appear)
        supersede_integrity = True
        if expected_neg:
            en_lower = expected_neg.lower()
            for r in retrieved:
                if en_lower in r.text.lower() or en_lower in (r.value or "").lower():
                    supersede_integrity = False
                    notes_parts.append(f"Superseded info '{expected_neg}' leaked into recall!")
                    break

        # Multi-result check
        has_min_results = retrieved_count >= min_expected

        # Overall pass criteria
        if case.get("category") == "Multiple Relevant Memories":
            passed = recall_at_5 and has_min_results and supersede_integrity
        else:
            passed = top_1_match and supersede_integrity

        if passed:
            notes_parts.append(f"Matched expected '{expected_pos}' with score {top_score:.3f}." if top_score else "Passed.")
        else:
            if not top_1_match:
                notes_parts.append(f"Expected '{expected_pos}' not in top-1 match.")
            if not has_min_results:
                notes_parts.append(f"Expected at least {min_expected} results, got {retrieved_count}.")

    return TestCaseResult(
        case_id=case["id"],
        category=case.get("category", "General"),
        description=case.get("description", ""),
        query=query,
        passed=passed,
        top_1_match=top_1_match,
        recall_at_5=recall_at_5,
        supersede_integrity=supersede_integrity,
        unknown_rejection=unknown_rejection,
        retrieved_count=retrieved_count,
        top_text=top_text,
        top_score=top_score,
        latency_ms=round(latency_ms, 2),
        notes=" ".join(notes_parts),
    )


def run_full_evaluation(
    cases_path: Optional[Path | str] = None,
    progress_callback: Optional[Callable[[int, int, TestCaseResult], None]] = None,
) -> EvaluationSummary:
    """Execute evaluation on all test cases and compute aggregated metrics."""
    cases = load_evaluation_cases(cases_path)
    engine = get_embedding_engine()
    total = len(cases)

    results: List[TestCaseResult] = []
    category_data: Dict[str, Dict[str, int]] = {}

    for idx, case in enumerate(cases, 1):
        cat = case.get("category", "General")
        if cat not in category_data:
            category_data[cat] = {"total": 0, "passed": 0}
        category_data[cat]["total"] += 1

        res = evaluate_single_case(case, embedding_engine=engine)
        if res.passed:
            category_data[cat]["passed"] += 1

        results.append(res)
        if progress_callback:
            progress_callback(idx, total, res)

    # Compute overall rates
    passed_cases = sum(1 for r in results if r.passed)
    overall_accuracy = (passed_cases / total) * 100.0 if total > 0 else 0.0

    pos_cases = [r for r in results if not any(c.get("is_unknown") for c in cases if c["id"] == r.case_id)]
    top_1_accuracy = (sum(1 for r in pos_cases if r.top_1_match) / len(pos_cases)) * 100.0 if pos_cases else 100.0
    recall_at_5 = (sum(1 for r in pos_cases if r.recall_at_5) / len(pos_cases)) * 100.0 if pos_cases else 100.0

    unk_cases = [r for r in results if any(c.get("is_unknown") for c in cases if c["id"] == r.case_id)]
    unk_rate = (sum(1 for r in unk_cases if r.unknown_rejection) / len(unk_cases)) * 100.0 if unk_cases else 100.0

    neg_check_cases = [r for r in results if any(c.get("expected_negative") for c in cases if c["id"] == r.case_id)]
    supersede_rate = (sum(1 for r in neg_check_cases if r.supersede_integrity) / len(neg_check_cases)) * 100.0 if neg_check_cases else 100.0

    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0.0

    # Build category breakdown
    category_metrics = {}
    for cat, stats in category_data.items():
        tot = stats["total"]
        pas = stats["passed"]
        category_metrics[cat] = {
            "total": tot,
            "passed": pas,
            "accuracy": round((pas / tot) * 100.0, 1) if tot > 0 else 0.0,
        }

    return EvaluationSummary(
        total_cases=total,
        passed_cases=passed_cases,
        overall_accuracy=round(overall_accuracy, 2),
        top_1_accuracy=round(top_1_accuracy, 2),
        recall_at_5=round(recall_at_5, 2),
        unknown_rejection_rate=round(unk_rate, 2),
        supersede_integrity_rate=round(supersede_rate, 2),
        avg_latency_ms=round(avg_latency, 2),
        category_metrics=category_metrics,
        case_results=results,
    )
