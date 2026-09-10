"""
Class-wise retrieval experiment for Adaptive RAG.

This experiment compares:

1. BM25-only
2. Dense-only
3. Hybrid retrieval

for the complete QA benchmark.

Metrics are reported:
- Overall Article Recall@5
- Overall Article MRR@5
- Overall Exact Gold Recall@5
- Per-class Recall@5
- Per-class MRR@5
- Per-class Exact Gold Recall@5

The QA 'class' field is used ONLY for analysis.
It is NOT used to choose the retriever.

This prevents label leakage.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from legal_dc_baseline.data import load_corpus, load_qa
from adaptive_rag.retrieval_variants import (
    RetrievalResources,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)


def compact(text: str) -> str:
    """
    Remove whitespace so that exact gold passage matching
    is not affected by formatting differences.
    """
    return "".join(text.split())


def evaluate_retrieval(
    records: list[dict],
    k: int = 5,
) -> dict:
    """
    Calculate retrieval metrics overall and separately
    for each QA class.
    """

    classes = sorted(
        {
            record["class"]
            for record in records
        }
    )

    # -----------------------------------------
    # Containers for overall metrics
    # -----------------------------------------

    overall = {
        "article_hits": 0,
        "reciprocal_ranks": [],
        "gold_hits": 0,
    }

    # -----------------------------------------
    # Containers for class-wise metrics
    # -----------------------------------------

    by_class = {
        class_name: {
            "queries": 0,
            "article_hits": 0,
            "reciprocal_ranks": [],
            "gold_hits": 0,
        }
        for class_name in classes
    }

    # -----------------------------------------
    # Evaluate every question
    # -----------------------------------------

    for record in records:

        retrieved = record["retrieval"][:k]

        target_article = record["article_reference"]

        query_class = record["class"]

        class_stats = by_class[query_class]

        class_stats["queries"] += 1

        # -------------------------------------
        # Article Recall + MRR
        # -------------------------------------

        ranks = [
            item["rank"]
            for item in retrieved
            if item["article_reference"] == target_article
        ]

        if ranks:

            overall["article_hits"] += 1
            class_stats["article_hits"] += 1

            reciprocal_rank = 1.0 / min(ranks)

            overall["reciprocal_ranks"].append(
                reciprocal_rank
            )

            class_stats["reciprocal_ranks"].append(
                reciprocal_rank
            )

        else:

            overall["reciprocal_ranks"].append(0.0)

            class_stats["reciprocal_ranks"].append(0.0)

        # -------------------------------------
        # Exact gold passage recall
        # -------------------------------------

        merged_text = compact(
            "".join(
                item["text"]
                for item in retrieved
            )
        )

        gold_passages = record["document"]

        gold_found = all(
            compact(passage) in merged_text
            for passage in gold_passages
            if passage.strip()
        )

        if gold_found:

            overall["gold_hits"] += 1
            class_stats["gold_hits"] += 1

    # -----------------------------------------
    # Build final metrics
    # -----------------------------------------

    total = len(records)

    overall_metrics = {
        "article_recall_at_5": (
            overall["article_hits"] / total
            if total
            else 0.0
        ),
        "article_mrr_at_5": (
            sum(overall["reciprocal_ranks"]) / total
            if total
            else 0.0
        ),
        "legal_dc_exact_gold_recall_at_5": (
            overall["gold_hits"] / total
            if total
            else 0.0
        ),
        "queries": total,
    }

    class_metrics = {}

    for class_name, stats in by_class.items():

        class_total = stats["queries"]

        class_metrics[class_name] = {
            "article_recall_at_5": (
                stats["article_hits"] / class_total
                if class_total
                else 0.0
            ),
            "article_mrr_at_5": (
                sum(stats["reciprocal_ranks"]) / class_total
                if class_total
                else 0.0
            ),
            "legal_dc_exact_gold_recall_at_5": (
                stats["gold_hits"] / class_total
                if class_total
                else 0.0
            ),
            "queries": class_total,
        }

    return {
        "overall": overall_metrics,
        "by_class": class_metrics,
    }


def run_experiment(
    corpus_path: Path,
    qa_path: Path,
    results_dir: Path,
    dense_model: str,
    reranker_model: str,
    dense_k: int,
    bm25_k: int,
    final_k: int,
    batch_size: int,
    device: str | None,
) -> Path:

    print("\nLoading corpus...")

    chunks = load_corpus(corpus_path)

    print(f"Loaded {len(chunks)} corpus chunks.")

    print("\nLoading QA dataset...")

    qa_records = load_qa(qa_path)

    print(f"Loaded {len(qa_records)} QA records.")

    # -----------------------------------------
    # Build shared retrieval resources
    # -----------------------------------------

    print("\nLoading retrieval models and indexes...")

    resources = RetrievalResources(
        chunks=chunks,
        dense_model=dense_model,
        reranker_model=reranker_model,
        batch_size=batch_size,
        device=device,
    )

    # -----------------------------------------
    # Create the three retrieval strategies
    # -----------------------------------------

    bm25_retriever = BM25Retriever(
        resources=resources,
        final_k=final_k,
    )

    dense_retriever = DenseRetriever(
        resources=resources,
        final_k=final_k,
    )

    hybrid_retriever = HybridRetriever(
        resources=resources,
        dense_k=dense_k,
        bm25_k=bm25_k,
        final_k=final_k,
    )

    retrievers = {
        "bm25": bm25_retriever,
        "dense": dense_retriever,
        "hybrid": hybrid_retriever,
    }

    all_results = {}

    # -----------------------------------------
    # Run every retrieval strategy
    # -----------------------------------------

    for strategy_name, retriever in retrievers.items():

        print("\n" + "=" * 60)
        print(f"Running retrieval strategy: {strategy_name.upper()}")
        print("=" * 60)

        outputs = []

        for source_record in tqdm(
            qa_records,
            desc=f"{strategy_name.upper()} retrieval",
        ):

            record = dict(source_record)

            record["retrieval"] = retriever.retrieve(
                record["query"]
            )

            outputs.append(record)

        # -------------------------------------
        # Evaluate
        # -------------------------------------

        metrics = evaluate_retrieval(
            outputs,
            k=final_k,
        )

        all_results[strategy_name] = metrics

    # -----------------------------------------
    # Save experiment
    # -----------------------------------------

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = (
        results_dir
        / f"adaptive_retrieval_analysis_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary = {
        "method": "Adaptive RAG retrieval-strategy analysis",
        "purpose": (
            "Compare BM25, dense, and hybrid retrieval "
            "overall and by QA class before building "
            "the adaptive router."
        ),
        "configuration": {
            "dense_model": dense_model,
            "reranker_model": reranker_model,
            "dense_k": dense_k,
            "bm25_k": bm25_k,
            "final_k": final_k,
            "batch_size": batch_size,
            "device": device,
        },
        "corpus_chunks": len(chunks),
        "queries": len(qa_records),
        "results": all_results,
    }

    output_path = output_dir / "classwise_metrics.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output:

        json.dump(
            summary,
            output,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print(f"\nResults saved to:")
    print(output_path)

    return output_dir


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare BM25, dense, and hybrid retrieval "
            "overall and by QA class."
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/document_corpus.json"),
    )

    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("data/qa_pairs.json"),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--dense-model",
        default="BAAI/bge-base-en-v1.5",
    )

    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-base",
    )

    parser.add_argument(
        "--dense-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--bm25-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="For example: cpu or cuda",
    )

    args = parser.parse_args()

    run_experiment(
        corpus_path=args.corpus,
        qa_path=args.qa,
        results_dir=args.results_dir,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        final_k=args.final_k,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()