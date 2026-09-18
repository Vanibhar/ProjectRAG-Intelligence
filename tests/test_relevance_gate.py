"""Unit and integration checks for retrieval relevance gating."""

import os
import unittest
from pathlib import Path
from unittest import mock

from runtime.relevance import UNSUPPORTED_ANSWER, evidence_is_supported


class EvidenceIsSupportedTests(unittest.TestCase):
    def test_empty_passages_are_unsupported(self) -> None:
        self.assertFalse(evidence_is_supported([], 0.3))

    def test_low_top_score_is_unsupported(self) -> None:
        passages = [{"reranker_score": 0.0006, "text": "Article 32"}]
        self.assertFalse(evidence_is_supported(passages, 0.3))

    def test_high_top_score_is_supported(self) -> None:
        passages = [{"reranker_score": 0.6192, "text": "Article 22"}]
        self.assertTrue(evidence_is_supported(passages, 0.3))


@unittest.skipUnless(
    (Path(os.getenv("RAG_ARTIFACTS_DIR", "runtime_artifacts")) / "legal.faiss").is_file(),
    "retrieval artifacts are required for integration checks",
)
class RetrievalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from runtime.service import LegalRAGService
        from runtime.settings import Settings

        cls.service = LegalRAGService(Settings())

    def test_supported_article_question_passes_gate(self) -> None:
        passages = self.service.retrieve(
            self.service.transform_query(
                "What protection does Article 22 provide to an arrested person?"
            )
        )
        self.assertTrue(evidence_is_supported(passages, self.service.settings.reranker_score_threshold))
        self.assertGreaterEqual(passages[0]["reranker_score"], 0.3)
        self.assertEqual(passages[0]["article_reference"], "Article 22")

    def test_off_corpus_question_fails_gate(self) -> None:
        passages = self.service.retrieve(
            self.service.transform_query(
                "If my mobile phone is stolen, what legal remedies are available to me in India?"
            )
        )
        self.assertFalse(evidence_is_supported(passages, self.service.settings.reranker_score_threshold))
        self.assertLess(passages[0]["reranker_score"], 0.3)


class AskPipelineTests(unittest.TestCase):
    def test_unsupported_question_skips_generation(self) -> None:
        from runtime.service import LegalRAGService
        from runtime.settings import Settings

        service = LegalRAGService.__new__(LegalRAGService)
        service.settings = Settings()
        service.generator = mock.Mock()
        service.transform_query = lambda question: LegalRAGService.transform_query(question)
        service.classify_question = lambda question: LegalRAGService.classify_question(question)
        service._format_sources = lambda passages: LegalRAGService._format_sources(service, passages)
        service.retrieve = mock.Mock(
            return_value=[{
                "text": "writ jurisdiction",
                "title": "Constitution",
                "article_reference": "Article 32",
                "reranker_score": 0.0006,
            }]
        )

        result = service.ask(
            "If my mobile phone is stolen, what legal remedies are available to me in India?"
        )

        service.generator.answer.assert_not_called()
        self.assertFalse(result["supported"])
        self.assertEqual(result["answer"], UNSUPPORTED_ANSWER)
        self.assertEqual(result["sources"], [])

    def test_supported_question_generates_answer(self) -> None:
        from runtime.service import LegalRAGService
        from runtime.settings import Settings

        service = LegalRAGService.__new__(LegalRAGService)
        service.settings = Settings()
        service.generator = mock.Mock()
        service.generator.answer = mock.Mock(return_value="Article 22 protects arrested persons.")
        service.transform_query = lambda question: LegalRAGService.transform_query(question)
        service.classify_question = lambda question: LegalRAGService.classify_question(question)
        service._format_sources = lambda passages: LegalRAGService._format_sources(service, passages)
        service.retrieve = mock.Mock(
            return_value=[{
                "text": "Protection against arrest and detention in certain cases.",
                "title": "Constitution",
                "article_reference": "Article 22",
                "reranker_score": 0.6192,
                "rank": 1,
            }]
        )

        result = service.ask("What protection does Article 22 provide to an arrested person?")

        service.generator.answer.assert_called_once()
        self.assertTrue(result["supported"])
        self.assertEqual(result["answer"], "Article 22 protects arrested persons.")
        self.assertEqual(len(result["sources"]), 1)


if __name__ == "__main__":
    unittest.main()
