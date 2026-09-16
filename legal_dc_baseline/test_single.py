#/legal_dc_baseline/test_single.py
import argparse
from pathlib import Path
import time

from legal_dc_baseline.config import ExperimentConfig
from legal_dc_baseline.data import load_corpus, load_qa
from legal_dc_baseline.generation import LocalGenerator
from legal_dc_baseline.metrics import generation_metrics, retrieval_metrics
from legal_dc_baseline.retrieval import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Single-Query RAG Evaluation")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of QA record from data/qa.json",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    corpus_file = data_path / "corpus.json"
    qa_file = data_path / "qa.json"

    # Fallback to alternative naming if standard names do not exist
    if not corpus_file.exists():
        corpus_file = data_path / "document_corpus.json"
    if not qa_file.exists():
        qa_file = data_path / "qa_pairs.json"

    # 1. Initialize Configuration matching config.py
    config = ExperimentConfig(
        corpus_path=corpus_file,
        qa_path=qa_file,
        results_dir=Path("results"),
    )

    print("Loading corpus and dataset...")
    chunks = load_corpus(config.corpus_path)
    qa_records = load_qa(config.qa_path)

    print("Initializing retriever and generator...")
    retriever = HybridRetriever(
        chunks=chunks,
        dense_model=config.dense_model,
        reranker_model=config.reranker_model,
        dense_k=config.dense_k,
        bm25_k=config.bm25_k,
        final_k=config.final_k,
        batch_size=config.batch_size,
        device=config.device,
    )
    generator = LocalGenerator(
        model_name=config.generator_model,
        max_new_tokens=config.max_new_tokens,
        device=config.device,
    )

    # 2. Extract Target Record
    target_record = qa_records[args.index]
    query_text = target_record["query"]
    target_article = target_record.get("article_reference", "")

    gold_document = target_record.get("document", [])
    if isinstance(gold_document, str):
        gold_document = [gold_document]

    print(f"\n--- Testing QA Record Index {args.index} ---")
    print(f"Query: {query_text}")
    print(f"Target Article: {target_article}")

    start_time = time.time()

    # 3. Retrieve Passages (Uses HybridRetriever.retrieve)
    retrieved_items = retriever.retrieve(query_text)

    # 4. Generate Answer (Uses LocalGenerator.answer)
    rag_answer = generator.answer(query=query_text, passages=retrieved_items)

    elapsed = round(time.time() - start_time, 2)

    # 5. Format evaluation record matching metrics.py expectations
    eval_record = {
        "query": query_text,
        "article_reference": target_article,
        "document": gold_document,
        "retrieval": retrieved_items,
        "answer": target_record.get("answer", ""),
        "rag_answer": rag_answer,
    }

    # 6. Compute Metrics via metrics.py
    r_metrics = retrieval_metrics([eval_record], k=config.final_k)
    g_metrics = generation_metrics([eval_record])

    # 7. Print Output
    print("\n" + "=" * 60)
    print("SINGLE QUERY RESULTS")
    print("=" * 60)
    print(f"Execution Time                     : {elapsed}s")
    print(
        f"Article Recall @ {config.final_k}              : {r_metrics.get(f'article_recall_at_{config.final_k}', 0.0):.4f}"
    )
    print(
        f"Article MRR @ {config.final_k}                 : {r_metrics.get(f'article_mrr_at_{config.final_k}', 0.0):.4f}"
    )
    print(
        f"Legal DC Exact Gold Recall @ {config.final_k}  : {r_metrics.get(f'legal_dc_exact_gold_recall_at_{config.final_k}', 0.0):.4f}"
    )
    print("-" * 60)
    print(f"BLEU-2 Score                       : {eval_record['bleu_2']:.4f}")
    print(
        f"ROUGE-L F1 Score                   : {eval_record['rouge_l_f1']:.4f}"
    )
    print("-" * 60)
    print(f"Generated RAG Answer:\n{rag_answer}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()