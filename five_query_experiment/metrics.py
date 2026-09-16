"""Retrieval metrics and paired Set-1 comparison for four query conditions."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import fmean
from typing import Any


def compact(text: str) -> str:
    """Remove whitespace for the project's exact-gold-passage convention."""
    return re.sub(r"\s+", "", text)


def per_query_metrics(record: dict[str, Any], k: int = 5) -> dict[str, Any]:
    """Calculate Recall, reciprocal rank, and exact-gold hit for one query."""
    retrieved = record["retrieval"][:k]
    target_article = record["article_reference"]
    ranks = [
        item.get("rank", position)
        for position, item in enumerate(retrieved, start=1)
        if item["article_reference"] == target_article
    ]
    first_rank = min(ranks) if ranks else None
    merged_text = compact("".join(item["text"] for item in retrieved))
    gold_passages = record["document"]
    exact_gold_hit = all(
        compact(passage) in merged_text for passage in gold_passages if passage.strip()
    )
    return {
        "article_hit_at_k": int(first_rank is not None),
        "article_reciprocal_rank_at_k": 1.0 / first_rank if first_rank else 0.0,
        "first_relevant_rank": first_rank,
        "exact_gold_hit_at_k": int(exact_gold_hit),
    }


def aggregate_metrics(records: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """Calculate the existing project's Recall@k, MRR@k, and exact-gold Recall@k."""
    values = [per_query_metrics(record, k) for record in records]
    total = len(values)
    return {
        f"article_recall_at_{k}": sum(item["article_hit_at_k"] for item in values) / total if total else 0.0,
        f"article_mrr_at_{k}": (
            sum(item["article_reciprocal_rank_at_k"] for item in values) / total if total else 0.0
        ),
        f"legal_dc_exact_gold_recall_at_{k}": (
            sum(item["exact_gold_hit_at_k"] for item in values) / total if total else 0.0
        ),
        "queries": total,
    }


def classwise_metrics(records: list[dict[str, Any]], k: int = 5) -> dict[str, dict[str, Any]]:
    """Calculate the same metrics separately for each existing QA class."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["class"]].append(record)
    return {class_name: aggregate_metrics(group, k) for class_name, group in sorted(grouped.items())}


def evaluate_retrieval(records: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """Return the standard overall and classwise retrieval report."""
    return {"overall": aggregate_metrics(records, k), "by_class": classwise_metrics(records, k)}


def paired_delta_comparison(
    baseline_records: list[dict[str, Any]], variant_records: list[dict[str, Any]], k: int = 5
) -> dict[str, Any]:
    """Compare a transformed set against Set 1 using matched ``query_id`` values."""
    baseline_by_id = {record["query_id"]: record for record in baseline_records}
    variant_by_id = {record["query_id"]: record for record in variant_records}
    if set(baseline_by_id) != set(variant_by_id):
        raise ValueError("Paired comparison requires identical query_id sets.")

    paired: list[dict[str, Any]] = []
    for query_id in sorted(baseline_by_id):
        baseline = per_query_metrics(baseline_by_id[query_id], k)
        variant = per_query_metrics(variant_by_id[query_id], k)
        recall_delta = variant["article_hit_at_k"] - baseline["article_hit_at_k"]
        mrr_delta = variant["article_reciprocal_rank_at_k"] - baseline["article_reciprocal_rank_at_k"]
        gold_delta = variant["exact_gold_hit_at_k"] - baseline["exact_gold_hit_at_k"]
        if mrr_delta > 0:
            outcome = "improved"
        elif mrr_delta < 0:
            outcome = "worsened"
        else:
            outcome = "unchanged"
        paired.append(
            {
                "query_id": query_id,
                "class": variant_by_id[query_id]["class"],
                "baseline_first_relevant_rank": baseline["first_relevant_rank"],
                "variant_first_relevant_rank": variant["first_relevant_rank"],
                "baseline_article_hit_at_k": baseline["article_hit_at_k"],
                "variant_article_hit_at_k": variant["article_hit_at_k"],
                "baseline_exact_gold_hit_at_k": baseline["exact_gold_hit_at_k"],
                "variant_exact_gold_hit_at_k": variant["exact_gold_hit_at_k"],
                "recall_delta": recall_delta,
                "mrr_delta": mrr_delta,
                "exact_gold_recall_delta": gold_delta,
                "outcome": outcome,
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(items),
            "article_recall_delta": fmean(item["recall_delta"] for item in items) if items else 0.0,
            "article_mrr_delta": fmean(item["mrr_delta"] for item in items) if items else 0.0,
            "legal_dc_exact_gold_recall_delta": (
                fmean(item["exact_gold_recall_delta"] for item in items) if items else 0.0
            ),
            "improved_queries": sum(item["outcome"] == "improved" for item in items),
            "unchanged_queries": sum(item["outcome"] == "unchanged" for item in items),
            "worsened_queries": sum(item["outcome"] == "worsened" for item in items),
        }

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in paired:
        by_class[item["class"]].append(item)
    return {
        "overall": summarize(paired),
        "by_class": {class_name: summarize(items) for class_name, items in sorted(by_class.items())},
        "per_query": paired,
    }
