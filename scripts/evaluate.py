"""CLI Runner for MemEngine Evaluation Suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory in sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Configure UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.table import Table
from app.evaluator import run_full_evaluation


def main():
    console = Console()
    console.print("\n[bold cyan]🧠 Running MemEngine Comprehensive Benchmark Suite (35 Cases)...[/bold cyan]\n")

    summary = run_full_evaluation(
        progress_callback=lambda idx, total, res: console.print(
            f"  [{idx:02d}/{total:02d}] [{'green' if res.passed else 'red'}]{'PASS' if res.passed else 'FAIL'}[/] "
            f"({res.category}) [dim]{res.query}[/] -> [bold]{res.top_score or 0.0:.3f}[/] ({res.latency_ms:.1f}ms)"
        )
    )

    console.print("\n[bold green]==================== EVALUATION SUMMARY ====================[/bold green]")
    table = Table(title="Overall Benchmark Metrics", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score / Value", style="bold green", justify="right")

    table.add_row("Total Test Cases", str(summary.total_cases))
    table.add_row("Passed Cases", f"{summary.passed_cases} / {summary.total_cases}")
    table.add_row("Overall Accuracy", f"{summary.overall_accuracy:.1f}%")
    table.add_row("Recall@5", f"{summary.recall_at_5:.1f}%")
    table.add_row("Top-1 Accuracy", f"{summary.top_1_accuracy:.1f}%")
    table.add_row("Unknown Rejection Rate", f"{summary.unknown_rejection_rate:.1f}%")
    table.add_row("Supersede Integrity Rate", f"{summary.supersede_integrity_rate:.1f}%")
    table.add_row("Average Recall Latency", f"{summary.avg_latency_ms:.2f} ms")
    console.print(table)

    cat_table = Table(title="Category Breakdown", header_style="bold blue")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Passed", justify="center")
    cat_table.add_column("Total", justify="center")
    cat_table.add_column("Accuracy", justify="right")

    for cat, stats in summary.category_metrics.items():
        color = "green" if stats["accuracy"] >= 90 else ("yellow" if stats["accuracy"] >= 75 else "red")
        cat_table.add_row(
            cat,
            str(stats["passed"]),
            str(stats["total"]),
            f"[{color}]{stats['accuracy']:.1f}%[/]",
        )
    console.print(cat_table)


if __name__ == "__main__":
    main()
