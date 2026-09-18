"""Environment-driven runtime settings; no secrets are sent to the browser."""

from dataclasses import dataclass
import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    corpus_path: Path = Path(os.getenv("CORPUS_PATH", "data/document_corpus.json"))
    artifacts_dir: Path = Path(os.getenv("RAG_ARTIFACTS_DIR", "runtime_artifacts"))
    dense_model: str = os.getenv("DENSE_MODEL", "BAAI/bge-base-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    generator_model: str = os.getenv("GENERATOR_MODEL", "google/flan-t5-base")
    device: str | None = os.getenv("RAG_DEVICE") or None
    dense_k: int = _int("DENSE_K", 10)
    bm25_k: int = _int("BM25_K", 10)
    final_k: int = _int("FINAL_K", 5)
    reranker_score_threshold: float = _float("RERANKER_SCORE_THRESHOLD", 0.3)
    max_new_tokens: int = _int("MAX_NEW_TOKENS", 128)
    local_models_only: bool = _bool("RAG_LOCAL_MODELS_ONLY", True)
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")
