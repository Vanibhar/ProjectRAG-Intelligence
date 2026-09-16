"""Runner for the controlled four-query-set retrieval experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from adaptive_rag.retrieval_variants import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalResources,
)
from legal_dc_baseline.data import load_corpus

from .data_validation import load_and_validate_query_sets
from .metrics import evaluate_retrieval, paired_delta_comparison
from .reporting import classwise_summary, delta_summary, export_csv, overall_summary


@dataclass(frozen=True)
class FourQueryExperimentConfig:
    corpus_path: Path = Path("data/document_corpus.json")
    query_set_dir: Path = Path("data/query_sets")
    results_dir: Path = Path("results")
    dense_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    dense_k: int = 10
    bm25_k: int = 10
    final_k: int = 5
    batch_size: int = 32
    device: str | None = None
    expected_query_count: int = 1071


def _write_json(path: Path, content: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(content, output, ensure_ascii=False, indent=2)


def _output_record(source_record: dict[str, Any], retriever_name: str, retrieval: list[dict]) -> dict[str, Any]:
    """Copy immutable input metadata and attach one strategy's retrieved passages."""
    record = dict(source_record)
    record["retriever"] = retriever_name
    record["retrieval"] = retrieval
    return record


def run_experiment(config: FourQueryExperimentConfig) -> Path:
    """Run all four validated query sets through unchanged retrieval implementations."""
    query_sets = load_and_validate_query_sets(
        config.query_set_dir, expected_count=config.expected_query_count
    )
    chunks = load_corpus(config.corpus_path)
    resources = RetrievalResources(
        chunks=chunks,
        dense_model=config.dense_model,
        reranker_model=config.reranker_model,
        batch_size=config.batch_size,
        device=config.device,
    )
    retrievers = {
        "bm25": BM25Retriever(resources=resources, final_k=config.final_k),
        "dense": DenseRetriever(resources=resources, final_k=config.final_k),
        "hybrid": HybridRetriever(
            resources=resources,
            dense_k=config.dense_k,
            bm25_k=config.bm25_k,
            final_k=config.final_k,
        ),
    }

    per_query_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metrics_by_retriever: dict[str, dict[str, dict[str, Any]]] = {}
    deltas_by_retriever: dict[str, dict[str, dict[str, Any]]] = {}

    for retriever_name, retriever in retrievers.items():
        per_query_results[retriever_name] = {}
        metrics_by_retriever[retriever_name] = {}
        for set_name, records in query_sets.items():
            outputs: list[dict[str, Any]] = []
            for source_record in tqdm(records, desc=f"{retriever_name}/{set_name}"):
                retrieval = retriever.retrieve(source_record["query"])
                outputs.append(_output_record(source_record, retriever_name, retrieval))
            per_query_results[retriever_name][set_name] = outputs
            metrics_by_retriever[retriever_name][set_name] = evaluate_retrieval(
                outputs, k=config.final_k
            )

        baseline = per_query_results[retriever_name]["set_1_original"]
        deltas_by_retriever[retriever_name] = {
            set_name: paired_delta_comparison(baseline, outputs, k=config.final_k)
            for set_name, outputs in per_query_results[retriever_name].items()
            if set_name != "set_1_original"
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.results_dir / f"four_query_formulation_experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    configuration = asdict(config)
    for key, value in configuration.items():
        if isinstance(value, Path):
            configuration[key] = str(value)

    metric_report = {
        "method": "Controlled four-query-set legal retrieval experiment",
        "configuration": configuration,
        "corpus_chunks": len(chunks),
        "results": metrics_by_retriever,
    }
    _write_json(output_dir / "experiment_config.json", {"configuration": configuration, "corpus_chunks": len(chunks)})
    _write_json(output_dir / "overall_and_classwise_metrics.json", metric_report)
    _write_json(output_dir / "per_query_results.json", per_query_results)
    _write_json(output_dir / "paired_deltas_vs_set_1.json", deltas_by_retriever)
    _write_json(output_dir / "overall_summary.json", overall_summary(metrics_by_retriever))
    _write_json(output_dir / "classwise_summary.json", classwise_summary(metrics_by_retriever))
    _write_json(output_dir / "delta_summary.json", delta_summary(deltas_by_retriever))
    export_csv(output_dir / "overall_summary.csv", overall_summary(metrics_by_retriever))
    export_csv(output_dir / "classwise_summary.csv", classwise_summary(metrics_by_retriever))
    export_csv(output_dir / "delta_summary.csv", delta_summary(deltas_by_retriever))
    return output_dir
