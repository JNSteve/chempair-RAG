# PRP_002 — Unified Idempotent Ingestion Pipeline

> Implements `docs/prd/PRD_002_Unified_Idempotent_Ingestion_Pipeline.md`.

## Implementation steps

1. **`ingest_pipeline.py`** (importable, dependency-light): source-marker
   formatting, pdfplumber table formatting with table-number detection,
   page-ordered merge of text+table items, ingest planning
   (skip/ingest/replace decisions from manifest state), and LightRAG
   doc-id lookup from `kv_store_doc_status.json`. PDF libraries are
   imported lazily inside the extraction functions so planning/formatting
   stays testable without them.
2. **`ingest_corpus.py`** (CLI): wires `ingest_pipeline` to RAGAnything
   with the existing LLM/embedding setup (unchanged this phase), the
   rate-limit stop guard from the legacy scripts, manifest read/update via
   `scripts/corpus_manifest.py`, `--dry-run`/`--replace`/`--rebuild`
   flags, and JSON run reports under `reports/ingest/`.
3. **Delete** `ingest.py` and `ingest_tables.py`.
4. **Docs**: update `corpus/README.md` (ingestion section) and the
   roadmap status table.
5. **Tests**: `tests/test_ingest_pipeline.py` — markers, table formatting,
   merge ordering, plan decisions (fresh/unchanged/hash-drift/replace/
   rebuild/superseded), doc-id lookup, and manifest state updates.

## Verification

- `python -m pytest tests/test_ingest_pipeline.py tests/test_corpus_manifest.py tests/test_eval_scoring.py tests/test_golden_questions_schema.py`
- `python -m ruff check` / `ruff format --check` on all new/changed files.
- Owner (with corpus + OpenAI key): `python ingest_corpus.py --dry-run`
  shows the plan; a real run twice in a row reports zero inserts on the
  second pass.

## Rollback

Revert the PR. The legacy scripts return with it; `rag_storage/` is not
migrated by this change (the new pipeline only writes when run), so no KB
state needs restoring.
