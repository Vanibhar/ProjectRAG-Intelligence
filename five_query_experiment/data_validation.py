"""Validation for comparable four-query-set experiment inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QUERY_SET_FILENAMES: dict[str, str] = {
    "set_1_original": "set_1_original.json",
    "set_2_legal_terminology": "set_2_legal_terminology.json",
    "set_3_semantic_paraphrase": "set_3_semantic_paraphrase.json",
    "set_4_context_enriched": "set_4_context_enriched.json",
}

IMMUTABLE_FIELDS = (
    "answer",
    "article_reference",
    "document",
    "class",
    "original_query_id",
    "original_query",
)

REQUIRED_FIELDS = (
    "query_id",
    "original_query_id",
    "query_set",
    "query",
    "original_query",
    "answer",
    "article_reference",
    "document",
    "class",
    "transformation_type",
    "transformation_rationale",
)


class QuerySetValidationError(ValueError):
    """Raised when a query set is unsuitable for paired comparison."""


def load_query_set(path: Path) -> list[dict[str, Any]]:
    """Read one JSON query-set file and require a top-level array."""
    with path.open("r", encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise QuerySetValidationError(f"{path} must contain a top-level JSON array.")
    if not all(isinstance(record, dict) for record in records):
        raise QuerySetValidationError(f"{path} must contain JSON objects only.")
    return records


def _require_nonempty_metadata(record: dict[str, Any], set_name: str, position: int) -> None:
    for field in ("query_set", "transformation_type", "transformation_rationale"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise QuerySetValidationError(
                f"{set_name} record {position} requires non-empty {field!r}."
            )


def validate_query_set(
    records: list[dict[str, Any]], set_name: str, expected_count: int = 1071
) -> dict[str, dict[str, Any]]:
    """Validate one set and return an index keyed by ``query_id``."""
    if len(records) != expected_count:
        raise QuerySetValidationError(
            f"{set_name} has {len(records)} records; expected exactly {expected_count}."
        )

    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise QuerySetValidationError(
                f"{set_name} record {position} is missing required fields: {missing}."
            )
        query_id = record["query_id"]
        if not isinstance(query_id, str) or not query_id.strip():
            raise QuerySetValidationError(f"{set_name} record {position} has an invalid query_id.")
        if query_id in indexed:
            raise QuerySetValidationError(f"{set_name} has duplicate query_id {query_id!r}.")
        if record["query_set"] != set_name:
            raise QuerySetValidationError(
                f"{set_name} record {position} has query_set={record['query_set']!r}."
            )
        _require_nonempty_metadata(record, set_name, position)
        if not isinstance(record["document"], list) or not all(
            isinstance(item, str) for item in record["document"]
        ):
            raise QuerySetValidationError(
                f"{set_name} record {position} must contain document as a list of strings."
            )
        indexed[query_id] = record

    if set_name == "set_1_original":
        for query_id, record in indexed.items():
            if record["query"] != record["original_query"]:
                raise QuerySetValidationError(
                    f"Set 1 query {query_id!r} must exactly equal original_query."
                )
    return indexed


def validate_query_sets(
    query_sets: dict[str, list[dict[str, Any]]], expected_count: int = 1071
) -> dict[str, list[dict[str, Any]]]:
    """Validate all four sets and their invariant fields for paired evaluation."""
    expected_names = set(QUERY_SET_FILENAMES)
    actual_names = set(query_sets)
    if actual_names != expected_names:
        raise QuerySetValidationError(
            f"Expected query sets {sorted(expected_names)}, got {sorted(actual_names)}."
        )

    indexed_sets = {
        name: validate_query_set(records, name, expected_count)
        for name, records in query_sets.items()
    }
    baseline = indexed_sets["set_1_original"]
    baseline_ids = set(baseline)

    for set_name, indexed in indexed_sets.items():
        if set(indexed) != baseline_ids:
            missing = sorted(baseline_ids.difference(indexed))[:5]
            extra = sorted(set(indexed).difference(baseline_ids))[:5]
            raise QuerySetValidationError(
                f"{set_name} query_id set differs from Set 1; missing={missing}, extra={extra}."
            )
        for query_id, baseline_record in baseline.items():
            record = indexed[query_id]
            for field in IMMUTABLE_FIELDS:
                if record[field] != baseline_record[field]:
                    raise QuerySetValidationError(
                        f"{set_name} query {query_id!r} changes immutable field {field!r}."
                    )
    return query_sets


def load_and_validate_query_sets(
    query_set_dir: Path, expected_count: int = 1071
) -> dict[str, list[dict[str, Any]]]:
    """Load the four required files from a directory, then validate them together."""
    missing_paths = [
        query_set_dir / filename
        for filename in QUERY_SET_FILENAMES.values()
        if not (query_set_dir / filename).is_file()
    ]
    if missing_paths:
        formatted = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required query-set files: {formatted}")
    query_sets = {
        name: load_query_set(query_set_dir / filename)
        for name, filename in QUERY_SET_FILENAMES.items()
    }
    return load_and_validate_records(query_sets, expected_count)


def load_and_validate_records(
    query_sets: dict[str, list[dict[str, Any]]], expected_count: int = 1071
) -> dict[str, list[dict[str, Any]]]:
    """Named helper for callers that already hold records in memory."""
    return validate_query_sets(query_sets, expected_count)
