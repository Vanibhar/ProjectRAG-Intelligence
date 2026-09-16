"""CLI entry point for the controlled four-query-set retrieval experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from five_query_experiment.experiment import FourQueryExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate four controlled legal-query formulations with BM25, dense, and hybrid retrieval."
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/document_corpus.json"))
    parser.add_argument("--query-set-dir", type=Path, default=Path("data/query_sets"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--bm25-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="For example: cpu or cuda")
    parser.add_argument("--expected-query-count", type=int, default=1071)
    args = parser.parse_args()
    output = run_experiment(
        FourQueryExperimentConfig(
            corpus_path=args.corpus,
            query_set_dir=args.query_set_dir,
            results_dir=args.results_dir,
            dense_model=args.dense_model,
            reranker_model=args.reranker_model,
            dense_k=args.dense_k,
            bm25_k=args.bm25_k,
            final_k=args.final_k,
            batch_size=args.batch_size,
            device=args.device,
            expected_query_count=args.expected_query_count,
        )
    )
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
