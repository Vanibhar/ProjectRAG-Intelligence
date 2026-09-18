# Legal-DC-style baseline for the Indian Constitution

This project runs a modular reproduction of the public Legal-DC retrieval-augmented-generation baseline against the two original files in `data/`. It never modifies either input file.

## Install

Use Python 3.10 or later, then create and activate a virtual environment if desired.

```powershell
python -m pip install -r requirements.txt
```

## Run

Run the full baseline, including local answer generation:

```powershell
python run_baseline.py
```

The first run downloads the public embedding model, reranker, and `google/flan-t5-base` generator. For retrieval evaluation only (and no generator download), run:

```powershell
python run_baseline.py --generator-model none
```

The baseline supports current Transformers releases, including version 5. It loads
the FLAN-T5 tokenizer and encoder-decoder model directly rather than depending on
the removed `text2text-generation` pipeline alias.

Use `python run_baseline.py --help` to set models, device, or retrieval cut-offs. Each run creates a timestamped directory under `results/` containing:

- `predictions.json`: query, gold fields, five reranked passages, and generated answer/metrics when generation is enabled.
- `metrics.json`: configuration plus aggregate retrieval and generation scores.

## Method

The implementation follows the repository's `hybrid_rag.py` pattern:

1. Every source body entry is a retrieval unit. Here, that means one Constitution article per chunk; no additional splitting or cleaning occurs.
2. Dense retrieval returns 10 candidates.
3. Lexical BM25 retrieval returns 10 candidates.
4. Both lists are concatenated and duplicate chunk text is removed.
5. A cross-encoder reranks the merged candidates, returning five passages.
6. Those five passages are passed to the generator and evaluated against the reference answer.

Retrieval uses `article_reference` as the primary relevance label: it reports article Recall@5 and MRR@5. It also reports the Legal-DC-style exact-gold-passage recall, which requires every `document` gold snippet to be present in the concatenated five retrieved passages.

Generation reports Legal-DC-style BLEU-2 (weights 0.6, 0.4), ROUGE-L F1, and the original threshold accuracies (BLEU > 0.4, ROUGE-L > 0.6).

## Deliberate changes from Legal-DC

These are the only intentional departures, made to run the methodology locally on an English Constitution corpus:

| Legal-DC component | This implementation | Reason |
| --- | --- | --- |
| `maidalun1020/bce-embedding-base_v1` | `BAAI/bge-base-en-v1.5` | Legal-DC's selected embedding model is Chinese-focused; the corpus and questions here are English. |
| Elasticsearch `match` retrieval | In-process `rank-bm25` | It retains lexical BM25 while removing an undeclared external Elasticsearch server/index prerequisite. |
| BCE reranker | `BAAI/bge-reranker-base` | A public cross-encoder that supports English, used in the same merge-then-rerank role. |
| Jieba segmentation + `rouge_chinese` | Regex English tokens + `rouge-score` | Jieba is Chinese-specific; the score definitions and thresholds are preserved. |
| Wenxin/Qwen/Baichuan remote generators | Configurable local Hugging Face text-to-text generator, default `google/flan-t5-base` | Legal-DC requires unavailable service credentials and uses Chinese models; this keeps the RAG generation step executable locally. |
| Only title metadata | Title plus `article_reference` retained | The requested article reference is the stable retrieval relevance label needed for Indian-dataset evaluation. |

The original datasets contain apparent mojibake punctuation sequences. This implementation preserves them exactly and performs no automatic text repair, so source data remain untouched and results are traceable to the supplied files.

## Production Legal AI Assistant

The research runners above are unchanged. The production assistant is a separate
runtime: it never runs query-set experiments or calculates retrieval metrics for a
user question.

### One-time index build

Install dependencies, then build the retrieval artifacts once. This writes FAISS,
BM25, and source metadata to the ignored `runtime_artifacts/` directory.

```powershell
python -m pip install -r requirements.txt
python build_index.py
```

### Run the API and frontend

Start the API in one terminal. Startup loads saved artifacts and the embedding,
reranker, and generator models once.

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Start the React/Vite frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (normally `http://localhost:5173`). The frontend
sends `POST /api/ask` requests to `http://localhost:8000` and shows the generated
answer plus reranked constitutional passages. `GET /api/health` reports whether
startup completed.

Copy `.env.example` to `.env` to configure models, device, CORS, and artifact
location. The included generator is local and needs no API key. If a remote LLM
provider is introduced later, keep its key only in the backend `.env` (for example
`OPENAI_API_KEY`) and never use a `VITE_` prefix for it.

The API defaults to `RAG_LOCAL_MODELS_ONLY=true`, so the online service never waits
on Hugging Face after initial setup. If startup says that a model is absent, set it
to `false` temporarily, start once to download the model, then return it to `true`.
