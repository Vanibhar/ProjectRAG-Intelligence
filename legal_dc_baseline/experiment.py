#/legal_dc_baseline/experiment.py
"""Experiment orchestration, result persistence, and CLI."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .config import ExperimentConfig
from .data import load_corpus, load_qa
from .generation import LocalGenerator
from .metrics import generation_metrics, retrieval_metrics
from .retrieval import HybridRetriever
# Change 1: Import the exact metric functions present in metrics.py
from legal_dc_baseline.metrics import generation_metrics, retrieval_metrics


def run_single_experiment(
    retriever, generator, qa_record: dict, top_k: int = 5
) -> dict:
    """Executes retrieval, metric calculation, and generation for a SINGLE query using native metrics.py functions."""
    start_time = time.time()

    query_text = qa_record["question"]
    target_article = qa_record.get("article_reference", "")

    # Ensure gold passages are passed as a list of strings
    gold_document = qa_record.get("document", [])
    if isinstance(gold_document, str):
        gold_document = [gold_document]

    # 1. Retrieve top-k documents for the single query
    retrieved_docs = retriever.retrieve(query_text, top_k=top_k)

    # Change 2: Format retrieved docs to match expected keys in metrics.py
    formatted_retrieval = []
    for rank_idx, doc in enumerate(retrieved_docs, start=1):
        formatted_retrieval.append({
            "rank": rank_idx,
            "article_reference": doc.get(
                "article_reference", doc.get("id", "")
            ),
            "text": doc.get("text", doc.get("content", "")),
        })

    # 2. Construct record required by retrieval_metrics & generation_metrics
    eval_record = {
        "question": query_text,
        "article_reference": target_article,
        "document": gold_document,
        "retrieval": formatted_retrieval,
        "answer": qa_record.get("answer", ""),
    }

    # 3. Generate response using LLM
    response = generator.generate(query=query_text, context_docs=retrieved_docs)
    eval_record["rag_answer"] = response

    execution_time = round(time.time() - start_time, 2)

    # Change 3: Call native metrics.py functions on the single-item list
    r_metrics = retrieval_metrics([eval_record], k=top_k)
    g_metrics = generation_metrics([eval_record])

    return {
        "query": query_text,
        "article_reference": target_article,
        "retrieved_docs": formatted_retrieval,
        "generated_answer": response,
        "execution_time_seconds": execution_time,

        # Retrieval metrics from metrics.py
        "article_mrr": r_metrics.get(f"article_mrr_at_{top_k}", 0.0),
        "article_recall": r_metrics.get(f"article_recall_at_{top_k}", 0.0),
        "legal_dc_exact_gold_recall": r_metrics.get(
            f"legal_dc_exact_gold_recall_at_{top_k}", 0.0
        ),

        # Generation metrics from metrics.py (mutated directly onto eval_record)
        "bleu_2": eval_record.get("bleu_2", 0.0),
        "rouge_l_f1": eval_record.get("rouge_l_f1", 0.0),
        "is_accuracy_bleu": eval_record.get("is_accuracy_bleu", 0),
        "is_accuracy_rouge_l": eval_record.get("is_accuracy_rouge_l", 0),
    }

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
