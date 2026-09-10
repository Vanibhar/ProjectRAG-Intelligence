from pathlib import Path

from adaptive_rag.classwise_experiment import run_experiment


if __name__ == "__main__":

    run_experiment(
        corpus_path=Path("data/document_corpus.json"),
        qa_path=Path("data/qa_pairs.json"),
        results_dir=Path("results"),
        dense_model="BAAI/bge-base-en-v1.5",
        reranker_model="BAAI/bge-reranker-base",
        dense_k=10,
        bm25_k=10,
        final_k=5,
        batch_size=32,
        device="cuda",
    )