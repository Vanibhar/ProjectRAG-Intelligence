#/legal_dc_baseline/experiment.py
"""Experiment orchestration, result persistence, and CLI."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .config import ExperimentConfig
from .data import load_corpus, load_qa
from .generation import LocalGenerator
from .metrics import generation_metrics, retrieval_metrics
from .retrieval import HybridRetriever


def save_json(path: Path, content: object) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(content, output, ensure_ascii=False, indent=2)


def run(config: ExperimentConfig) -> Path:
    chunks = load_corpus(config.corpus_path)
    qa_records = load_qa(config.qa_path)
    retriever = HybridRetriever(
        chunks, config.dense_model, config.reranker_model, config.dense_k,
        config.bm25_k, config.final_k, config.batch_size, config.device,
    )
    generator = None if config.generator_model.lower() == "none" else LocalGenerator(
        config.generator_model, config.max_new_tokens, config.device
    )

    outputs: list[dict] = []
    for source_record in tqdm(qa_records, desc="Running baseline"):
        record = dict(source_record)  # never mutate dataset objects
        record["retrieval"] = retriever.retrieve(record["query"])
        if generator:
            record["rag_answer"] = generator.answer(record["query"], record["retrieval"])
        outputs.append(record)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.results_dir / f"legal_dc_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    retrieval_result = retrieval_metrics(outputs, config.final_k)
    summary = {
        "method": "Legal-DC-inspired hybrid baseline",
        "configuration": {
            "dense_model": config.dense_model,
            "reranker_model": config.reranker_model,
            "generator_model": config.generator_model,
            "dense_k": config.dense_k,
            "bm25_k": config.bm25_k,
            "final_k": config.final_k,
        },
        "corpus_chunks": len(chunks),
        "retrieval_metrics": retrieval_result,
    }
    if generator:
        summary["generation_metrics"] = generation_metrics(outputs)

    save_json(output_dir / "predictions.json", outputs)
    save_json(output_dir / "metrics.json", summary)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Legal-DC baseline on the Indian Constitution dataset.")
    parser.add_argument("--corpus", type=Path, default=Path("data/document_corpus.json"))
    parser.add_argument("--qa", type=Path, default=Path("data/qa_pairs.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--generator-model", default="google/flan-t5-base", help="Use 'none' for retrieval-only.")
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--bm25-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None, help="For example: cpu or cuda")
    arguments = parser.parse_args()
    output = run(ExperimentConfig(
        corpus_path=arguments.corpus, qa_path=arguments.qa, results_dir=arguments.results_dir,
        dense_model=arguments.dense_model, reranker_model=arguments.reranker_model,
        generator_model=arguments.generator_model, dense_k=arguments.dense_k, bm25_k=arguments.bm25_k,
        final_k=arguments.final_k, batch_size=arguments.batch_size,
        max_new_tokens=arguments.max_new_tokens, device=arguments.device,
    ))
    print(f"Results saved to: {output}")
