"""Online-only RAG service: loads models/artifacts once and answers one question."""

import json
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from legal_dc_baseline.generation import LocalGenerator
from legal_dc_baseline.retrieval import tokenize
from runtime.indexing import BM25_FILE, INDEX_FILE, MANIFEST_FILE, METADATA_FILE
from runtime.relevance import UNSUPPORTED_ANSWER, evidence_is_supported
from runtime.settings import Settings


class LegalRAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._load_artifacts()
        # These objects are initialized exactly once, on backend startup.
        self.embedder = SentenceTransformer(settings.dense_model, device=settings.device,
                                            local_files_only=settings.local_models_only)
        self.reranker = CrossEncoder(settings.reranker_model, device=settings.device,
                                     local_files_only=settings.local_models_only)
        self.generator = LocalGenerator(settings.generator_model, settings.max_new_tokens,
                                        settings.device, local_files_only=settings.local_models_only)

    def _load_artifacts(self) -> None:
        paths = [self.settings.artifacts_dir / name for name in (INDEX_FILE, BM25_FILE, METADATA_FILE, MANIFEST_FILE)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError("Retrieval artifacts are missing. Run `python build_index.py` first: " + ", ".join(missing))
        manifest = json.loads((self.settings.artifacts_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest["dense_model"] != self.settings.dense_model:
            raise RuntimeError("DENSE_MODEL differs from the built index. Rebuild with `python build_index.py`.")
        self.index = faiss.read_index(str(self.settings.artifacts_dir / INDEX_FILE))
        with (self.settings.artifacts_dir / BM25_FILE).open("rb") as handle:
            self.bm25 = pickle.load(handle)
        self.metadata = json.loads((self.settings.artifacts_dir / METADATA_FILE).read_text(encoding="utf-8"))

    @staticmethod
    def transform_query(question: str) -> str:
        """Production query normalization, kept separate from experiment query sets."""
        return re.sub(r"\s+", " ", question).strip()

    @staticmethod
    def classify_question(question: str) -> str:
        """Return the project QA taxonomy: logic, concept, or generalization."""
        text = question.lower()
        if any(term in text for term in ("can ", "whether", "allowed", "valid", "without")):
            return "logic"
        if any(term in text for term in ("what is", "define", "meaning", "explain")):
            return "concept"
        return "generalization"

    def retrieve(self, query: str) -> list[dict]:
        vector = self.embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        _, dense_ids = self.index.search(np.asarray([vector], dtype=np.float32), self.settings.dense_k)
        bm25_ids = np.argsort(-self.bm25.get_scores(tokenize(query)))[: self.settings.bm25_k]
        seen: set[str] = set()
        candidates: list[dict] = []
        for index in [*bm25_ids.tolist(), *dense_ids[0].tolist()]:
            if index >= 0 and self.metadata[int(index)]["text"] not in seen:
                item = self.metadata[int(index)]
                seen.add(item["text"])
                candidates.append(item)
        scores = self.reranker.predict([(query, item["text"]) for item in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)[: self.settings.final_k]
        return [{**item, "rank": rank, "reranker_score": float(score)}
                for rank, (item, score) in enumerate(ranked, start=1)]

    def _format_sources(self, passages: list[dict]) -> list[dict]:
        return [{"text": item["text"],
                 "source": f"{item['title']} — {item['article_reference']}",
                 "score": item["reranker_score"]} for item in passages]

    def ask(self, question: str) -> dict:
        query = self.transform_query(question)
        classification = self.classify_question(query)
        passages = self.retrieve(query)
        if not evidence_is_supported(passages, self.settings.reranker_score_threshold):
            return {"answer": UNSUPPORTED_ANSWER, "classification": classification,
                    "supported": False, "sources": []}
        sources = self._format_sources(passages)
        # Keep the question and strongest evidence inside FLAN-T5's small input window.
        # Full passages remain available to the client as sources below.
        generation_passages = [{**item, "text": item["text"][:450]} for item in passages[:2]]
        answer = self.generator.answer(
            query,
            generation_passages,
            insufficient_evidence_message=UNSUPPORTED_ANSWER,
        )
        if answer.strip() == UNSUPPORTED_ANSWER:
            return {"answer": UNSUPPORTED_ANSWER, "classification": classification,
                    "supported": False, "sources": []}
        return {"answer": answer, "classification": classification,
                "supported": True, "sources": sources}
