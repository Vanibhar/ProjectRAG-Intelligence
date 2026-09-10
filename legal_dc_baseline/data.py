#/legal_dc_baseline/data.py

"""Read-only loading and Legal-DC-style article chunk construction."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    article_reference: str
    title: str
    text: str


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def load_corpus(path: Path) -> list[Chunk]:
    """Convert each ``body`` mapping entry into one retrievable article chunk.

    This deliberately performs no cleaning, splitting, or mutation of source text.
    """
    chunks: list[Chunk] = []
    for document_number, document in enumerate(load_json(path)):
        title = document["title"]
        for article_reference, text in document["body"].items():
            chunks.append(
                Chunk(
                    chunk_id=f"{document_number}:{article_reference}",
                    article_reference=article_reference,
                    title=title,
                    text=text,
                )
            )
    return chunks


def load_qa(path: Path) -> list[dict]:
    required = {"query", "answer", "document", "article_reference", "class"}
    records = load_json(path)
    for position, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"QA record {position} is missing fields: {sorted(missing)}")
    return records
