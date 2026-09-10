"""
Retrieval variants for the Adaptive RAG experiment.

This file does NOT modify the original Legal-DC baseline.

We provide three retrieval strategies:

1. BM25-only
2. Dense-only
3. Hybrid = BM25 + Dense + Cross-Encoder reranking

All strategies operate on the same 342 Constitution article chunks.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from legal_dc_baseline.data import Chunk


def tokenize(text: str) -> list[str]:
    """
    Tokenize English legal text for BM25 retrieval.
    """
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class RetrievalResources:
    """
    Shared retrieval resources.

    We create the expensive models/indexes only once and allow
    different retrieval strategies to reuse them.

    This avoids loading separate BGE models for BM25, dense,
    and hybrid retrieval.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        dense_model: str,
        reranker_model: str,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:

        self.chunks = chunks

        # -----------------------------
        # Dense retrieval resources
        # -----------------------------

        self.embedder = SentenceTransformer(
            dense_model,
            device=device,
        )

        self.embeddings = self.embedder.encode(
            [chunk.text for chunk in chunks],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # -----------------------------
        # BM25 resources
        # -----------------------------

        tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        # -----------------------------
        # Cross-encoder reranker
        # -----------------------------

        self.reranker = CrossEncoder(
            reranker_model,
            device=device,
        )


class BM25Retriever:
    """
    Pure lexical retrieval.

    Query
       ↓
    BM25
       ↓
    Top-k articles
    """

    def __init__(
        self,
        resources: RetrievalResources,
        final_k: int = 5,
    ) -> None:

        self.resources = resources
        self.final_k = final_k

    def retrieve(self, query: str) -> list[dict]:

        scores = self.resources.bm25.get_scores(
            tokenize(query)
        )

        indices = np.argsort(-scores)[: self.final_k]

        results = []

        for rank, index in enumerate(indices, start=1):

            chunk = self.resources.chunks[int(index)]

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "article_reference": chunk.article_reference,
                    "title": chunk.title,
                    "text": chunk.text,
                    "retrieval_score": float(scores[index]),
                }
            )

        return results


class DenseRetriever:
    """
    Pure semantic retrieval.

    Query
       ↓
    BGE embedding
       ↓
    Cosine similarity
       ↓
    Top-k articles
    """

    def __init__(
        self,
        resources: RetrievalResources,
        final_k: int = 5,
    ) -> None:

        self.resources = resources
        self.final_k = final_k

    def retrieve(self, query: str) -> list[dict]:

        query_embedding = self.resources.embedder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = self.resources.embeddings @ query_embedding

        indices = np.argsort(-scores)[: self.final_k]

        results = []

        for rank, index in enumerate(indices, start=1):

            chunk = self.resources.chunks[int(index)]

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "article_reference": chunk.article_reference,
                    "title": chunk.title,
                    "text": chunk.text,
                    "retrieval_score": float(scores[index]),
                }
            )

        return results


class HybridRetriever:
    """
    Hybrid retrieval.

    BM25
      +
    Dense
      ↓
    Candidate union
      ↓
    Cross-encoder reranker
      ↓
    Top-k
    """

    def __init__(
        self,
        resources: RetrievalResources,
        dense_k: int = 10,
        bm25_k: int = 10,
        final_k: int = 5,
    ) -> None:

        self.resources = resources
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k

    def _dense_candidates(self, query: str) -> list[int]:

        query_embedding = self.resources.embedder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = self.resources.embeddings @ query_embedding

        indices = np.argsort(-scores)[: self.dense_k]

        return [int(index) for index in indices]

    def _bm25_candidates(self, query: str) -> list[int]:

        scores = self.resources.bm25.get_scores(
            tokenize(query)
        )

        indices = np.argsort(-scores)[: self.bm25_k]

        return [int(index) for index in indices]

    def retrieve(self, query: str) -> list[dict]:

        dense_indices = self._dense_candidates(query)
        bm25_indices = self._bm25_candidates(query)

        # ---------------------------------
        # Combine candidates
        # ---------------------------------

        candidate_indices = bm25_indices + dense_indices

        seen_chunks: set[str] = set()

        candidates: list[Chunk] = []

        for index in candidate_indices:

            chunk = self.resources.chunks[index]

            if chunk.text not in seen_chunks:

                candidates.append(chunk)
                seen_chunks.add(chunk.text)

        # ---------------------------------
        # Cross-encoder reranking
        # ---------------------------------

        pairs = [
            (query, chunk.text)
            for chunk in candidates
        ]

        scores = self.resources.reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[: self.final_k]

        # ---------------------------------
        # Final results
        # ---------------------------------

        results = []

        for rank, (chunk, score) in enumerate(
            ranked,
            start=1,
        ):

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "article_reference": chunk.article_reference,
                    "title": chunk.title,
                    "text": chunk.text,
                    "reranker_score": float(score),
                }
            )

        return results