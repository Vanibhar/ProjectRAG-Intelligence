"""One-time artifact construction for the online legal assistant."""

import json
import pickle

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from legal_dc_baseline.data import load_corpus
from legal_dc_baseline.retrieval import tokenize
from runtime.settings import Settings


INDEX_FILE = "legal.faiss"
BM25_FILE = "bm25.pkl"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"


def build_index(settings: Settings, batch_size: int = 32) -> None:
    """Load corpus once, embed it, and persist immutable retrieval artifacts."""
    chunks = load_corpus(settings.corpus_path)
    if not chunks:
        raise ValueError("The legal corpus did not contain any retrievable articles.")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(settings.dense_model, device=settings.device)
    vectors = embedder.encode([chunk.text for chunk in chunks], batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(settings.artifacts_dir / INDEX_FILE))
    with (settings.artifacts_dir / BM25_FILE).open("wb") as handle:
        pickle.dump(BM25Okapi([tokenize(chunk.text) for chunk in chunks]), handle)
    metadata = [{"chunk_id": c.chunk_id, "article_reference": c.article_reference,
                 "title": c.title, "text": c.text} for c in chunks]
    (settings.artifacts_dir / METADATA_FILE).write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    (settings.artifacts_dir / MANIFEST_FILE).write_text(
        json.dumps({"dense_model": settings.dense_model, "chunks": len(metadata)}, indent=2), encoding="utf-8")
