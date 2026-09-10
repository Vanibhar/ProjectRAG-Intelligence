#/legal_dc_baseline/metrics.py
"""Legal-DC-compatible and article-identifier retrieval/generation metrics."""

import re

from rouge_score import rouge_scorer


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def retrieval_metrics(records: list[dict], k: int) -> dict:
    """Evaluate article relevance and the repository's exact-gold-passage convention."""
    article_hits = 0
    reciprocal_ranks: list[float] = []
    legal_dc_gold_hits = 0

    for record in records:
        retrieved = record["retrieval"][:k]
        target_article = record["article_reference"]
        ranks = [item["rank"] for item in retrieved if item["article_reference"] == target_article]
        if ranks:
            article_hits += 1
            reciprocal_ranks.append(1.0 / min(ranks))
        else:
            reciprocal_ranks.append(0.0)

        merged = _compact("".join(item["text"] for item in retrieved))
        gold_passages = record["document"]
        if all(_compact(passage) in merged for passage in gold_passages if passage.strip()):
            legal_dc_gold_hits += 1

    total = len(records)
    return {
        f"article_recall_at_{k}": article_hits / total if total else 0.0,
        f"article_mrr_at_{k}": sum(reciprocal_ranks) / total if total else 0.0,
        f"legal_dc_exact_gold_recall_at_{k}": legal_dc_gold_hits / total if total else 0.0,
        "queries": total,
    }


def generation_metrics(records: list[dict]) -> dict:
    """English replacement for jieba tokenization: regex tokens, BLEU-2 and ROUGE-L F1."""
    try:
        from nltk.translate.bleu_score import sentence_bleu
    except ImportError as error:  # covered by requirements, but make the failure actionable
        raise RuntimeError("Install nltk to calculate generation metrics.") from error

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu_scores: list[float] = []
    rouge_scores: list[float] = []
    bleu_accuracy: list[int] = []
    rouge_accuracy: list[int] = []

    for record in records:
        reference_tokens = re.findall(r"\w+", record["answer"].lower())
        hypothesis_tokens = re.findall(r"\w+", record["rag_answer"].lower())
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(0.6, 0.4))
        rouge_l = scorer.score(record["answer"], record["rag_answer"])["rougeL"].fmeasure
        record["bleu_2"] = bleu
        record["rouge_l_f1"] = rouge_l
        record["is_accuracy_bleu"] = int(bleu > 0.4)
        record["is_accuracy_rouge_l"] = int(rouge_l > 0.6)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)
        bleu_accuracy.append(record["is_accuracy_bleu"])
        rouge_accuracy.append(record["is_accuracy_rouge_l"])

    total = len(records)
    return {
        "bleu_2_average": sum(bleu_scores) / total if total else 0.0,
        "rouge_l_average": sum(rouge_scores) / total if total else 0.0,
        "bleu_2_accuracy_threshold_0.4": sum(bleu_accuracy) / total if total else 0.0,
        "rouge_l_accuracy_threshold_0.6": sum(rouge_accuracy) / total if total else 0.0,
        "queries": total,
    }
