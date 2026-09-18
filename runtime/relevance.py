"""Retrieval relevance gating for the production runtime pipeline."""

UNSUPPORTED_ANSWER = (
    "I'm sorry, but I don't have sufficient relevant information in my current legal "
    "knowledge base to answer this question reliably."
)


def evidence_is_supported(passages: list[dict], threshold: float) -> bool:
    """Return True when the strongest reranked passage meets the relevance threshold.

    Baseline and production measurements with ``BAAI/bge-reranker-base`` show a clear
    separation: in-corpus constitutional questions typically score above ~0.5, while
    off-corpus questions (for example mobile-phone theft or tax filing) score below
    ~0.002. A threshold of 0.3 blocks irrelevant retrievals while keeping common
    supported queries such as Article 22 (~0.62) safely above the cutoff.
    """
    if not passages:
        return False
    return float(passages[0]["reranker_score"]) >= threshold
