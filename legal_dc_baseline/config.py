#/legal_dc_baseline/config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    corpus_path: Path
    qa_path: Path
    results_dir: Path
    dense_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    generator_model: str = "google/flan-t5-base"
    dense_k: int = 10
    bm25_k: int = 10
    final_k: int = 5
    batch_size: int = 32
    max_new_tokens: int = 128
    device: str | None = None
