"""Latency and Throughput Profiler for MemEngine.

Measures p50, p95, and p99 latency for:
- Message Ingestion & Conflict Detection
- Hybrid Semantic Recall
"""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Ensure root directory in sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Configure UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from rich.console import Console
from rich.table import Table

from app.embeddings import get_embedding_engine
from app.ingestion import IngestionEngine
from app.retrieval import RecallEngine
from app.schemas import IngestRequest, RecallRequest
from app.storage import StorageManager
from app.updates import ConflictResolver


@dataclass
class LatencyProfile:
    """Latency percentiles for an operation."""

    operation: str
    sample_size: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ops_sec: float


def run_latency_benchmark(n_samples: int = 100) -> Tuple[LatencyProfile, LatencyProfile]:
    """Execute latency benchmark over n_samples ingestion and recall queries."""
    engine = get_embedding_engine()

    test_messages = [
        "I use Arch Linux.",
        "I'm building a Next.js app.",
        "My project deadline is October 15.",
        "Actually, the deadline moved to November 2.",
        "I prefer PostgreSQL over MySQL.",
        "My favorite editor is Neovim.",
        "Our backend service is written in FastAPI.",
        "We deploy our infrastructure on AWS.",
        "I live in Berlin.",
        "My default shell is zsh.",
    ]

    test_queries = [
        "What OS does the user use?",
        "When is the project deadline?",
        "What database does the user prefer?",
        "What is the user's favorite editor?",
        "What backend framework is being used?",
        "Where is the infrastructure deployed?",
        "Where does the user live?",
        "What shell does the user use?",
        "What is the user's favourite food?",
        "What car does the user drive?",
    ]

    storage = StorageManager(
        db_path=":memory:",
        chroma_path=":memory:",
        embedding_engine=engine,
    )
    resolver = ConflictResolver(storage=storage, ingestion=IngestionEngine())
    recall_engine = RecallEngine(storage=storage, embedding_engine=engine)

    # 1. Warm-up
    resolver.process_ingest_request(IngestRequest(message="Warm-up message."))
    recall_engine.recall(RecallRequest(query="Warm-up query?"))

    # 2. Benchmark Ingestion
    ingest_latencies: List[float] = []
    for i in range(n_samples):
        msg = test_messages[i % len(test_messages)]
        t0 = time.perf_counter()
        resolver.process_ingest_request(IngestRequest(message=msg))
        ingest_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 3. Benchmark Recall
    recall_latencies: List[float] = []
    for i in range(n_samples):
        qry = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        recall_engine.recall(RecallRequest(query=qry, top_k=5))
        recall_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Calculate statistics
    ing_arr = np.array(ingest_latencies)
    rec_arr = np.array(recall_latencies)

    ing_profile = LatencyProfile(
        operation="Ingestion & Conflict Detection",
        sample_size=n_samples,
        mean_ms=float(np.mean(ing_arr)),
        p50_ms=float(np.percentile(ing_arr, 50)),
        p95_ms=float(np.percentile(ing_arr, 95)),
        p99_ms=float(np.percentile(ing_arr, 99)),
        throughput_ops_sec=float(1000.0 / np.mean(ing_arr)) if np.mean(ing_arr) > 0 else 0.0,
    )

    rec_profile = LatencyProfile(
        operation="Hybrid Semantic Recall",
        sample_size=n_samples,
        mean_ms=float(np.mean(rec_arr)),
        p50_ms=float(np.percentile(rec_arr, 50)),
        p95_ms=float(np.percentile(rec_arr, 95)),
        p99_ms=float(np.percentile(rec_arr, 99)),
        throughput_ops_sec=float(1000.0 / np.mean(rec_arr)) if np.mean(rec_arr) > 0 else 0.0,
    )

    return ing_profile, rec_profile


def main():
    console = Console()
    console.print("\n[bold cyan]⚡ Running MemEngine Latency Benchmark (100 Samples)...[/bold cyan]\n")

    ing_p, rec_p = run_latency_benchmark(n_samples=100)

    table = Table(title="MemEngine Latency & Throughput Profile", header_style="bold green")
    table.add_column("Operation", style="cyan")
    table.add_column("Samples", justify="center")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("p50 (ms)", justify="right", style="bold green")
    table.add_column("p95 (ms)", justify="right", style="bold yellow")
    table.add_column("p99 (ms)", justify="right", style="bold red")
    table.add_column("Throughput", justify="right", style="bold blue")

    for p in [ing_p, rec_p]:
        table.add_row(
            p.operation,
            str(p.sample_size),
            f"{p.mean_ms:.2f}",
            f"{p.p50_ms:.2f}",
            f"{p.p95_ms:.2f}",
            f"{p.p99_ms:.2f}",
            f"{p.throughput_ops_sec:.1f} ops/s",
        )

    console.print(table)


if __name__ == "__main__":
    main()
