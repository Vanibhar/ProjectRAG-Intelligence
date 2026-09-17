"""JSON-ready summaries and CSV exports for the four-query-set experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def overall_summary(results: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten overall metrics by retriever and query set."""
    rows: list[dict[str, Any]] = []
    for retriever, by_set in sorted(results.items()):
        for query_set, metrics in sorted(by_set.items()):
            rows.append({"retriever": retriever, "query_set": query_set, **metrics["overall"]})
    return rows


def classwise_summary(results: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten classwise metrics by retriever, query set, and class."""
    rows: list[dict[str, Any]] = []
    for retriever, by_set in sorted(results.items()):
        for query_set, metrics in sorted(by_set.items()):
            for class_name, values in sorted(metrics["by_class"].items()):
                rows.append(
                    {"retriever": retriever, "query_set": query_set, "class": class_name, **values}
                )
    return rows


def delta_summary(deltas: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten aggregate Set-1 deltas; excludes the verbose per-query section."""
    rows: list[dict[str, Any]] = []
    for retriever, by_set in sorted(deltas.items()):
        for query_set, comparison in sorted(by_set.items()):
            rows.append({"retriever": retriever, "query_set": query_set, **comparison["overall"]})
    return rows


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flattened report rows as UTF-8 CSV, preserving all discovered columns."""
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
