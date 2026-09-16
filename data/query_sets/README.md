# Five-query-set experiment input schema

This directory will contain exactly these five UTF-8 JSON files:

- `set_1_original.json`
- `set_2_legal_terminology.json`
- `set_3_semantic_paraphrase.json`
- `set_4_context_enriched.json`
- `set_5_evidence_optimized.json`

Each file must be a JSON array with exactly 1,071 records. The five files must
have the identical set of unique `query_id` values. A record uses this schema:

```json
{
  "query_id": "q_0001",
  "original_query_id": "q_0001",
  "query_set": "set_1_original",
  "query": "Can Parliament establish a new state without any conditions?",
  "original_query": "Can Parliament establish a new state without any conditions?",
  "answer": "No, Parliament may by law establish a new State on such terms and conditions as it thinks fit.",
  "article_reference": "Article 2",
  "document": ["Admission or establishment of new States..."],
  "class": "logic",
  "transformation_type": "original",
  "transformation_rationale": "Verbatim baseline query retained for paired comparison.",
  "transformation_notes": {
    "optional": "Additional auditable transformation details may be stored here."
  }
}
```

Required fields are `query_id`, `original_query_id`, `query_set`, `query`,
`original_query`, `answer`, `article_reference`, `document`, `class`,
`transformation_type`, and `transformation_rationale`.

`document` must be an array of strings. The validator requires non-empty string
values for the transformation metadata fields.

Across a given `query_id`, these fields are immutable and must exactly match
Set 1: `original_query_id`, `original_query`, `answer`, `article_reference`,
`document`, and `class`. Only the query formulation and transformation metadata
may change. In `set_1_original.json`, `query` must exactly equal
`original_query`.

Use the exact `query_set` value matching each filename stem. Recommended
transformation types are `original`, `legal_terminology_enrichment`,
`semantic_paraphrase`, `context_enrichment`, and `evidence_optimized`.

Set 5 must document the empirical basis for its final wording in
`transformation_rationale` and/or `transformation_notes`; it should be created
only after analyzing Sets 1--4.
