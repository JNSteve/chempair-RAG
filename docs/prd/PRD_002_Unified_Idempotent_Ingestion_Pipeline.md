# PRD_002 — Unified Idempotent Ingestion Pipeline (Roadmap Phase 2)

> Roadmap: `docs/RAG_PERFECTION_ROADMAP.md`. Depends on PRD_001 (corpus
> manifest). No changes to `server.py`, routing, or the frontend contract.

## Problem

The knowledge base is built by two ad-hoc scripts (`ingest.py`,
`ingest_tables.py`) that:

- glob a local folder instead of reading the corpus manifest, so unfiled
  PDFs are silently ingested and superseded documents can't be excluded;
- ingest text and tables as **separate passes with separate synthetic
  filenames** (`tables_<name>.pdf`), which the citation code then has to
  un-mangle;
- carry no chunk-level metadata — page numbers are captured at extraction
  but lost by retrieval time, forcing `server.py` to regex-guess locators
  from prose;
- are only partially idempotent (tables pass only) and have no way to
  update or replace a single document.

## What

One manifest-driven CLI, `ingest_corpus.py`, replacing both legacy scripts:

1. **Manifest-driven scope.** Ingests exactly the `status: current`
   documents in `corpus/manifest.yaml`; superseded documents are skipped.
   Unregistered PDFs are never ingested.
2. **Text + tables in one pass per document.** pypdfium2 text and
   pdfplumber tables merged into a single page-ordered content list under
   the document's real filename. The `tables_` hack is gone.
3. **Source markers on every chunk.** Each content item is prefixed with a
   structured marker: `[source: <filename> | doc: <doc_id> | page <n>]`,
   plus `| Table <id>` where a table number is detected. Markers make
   locators machine-readable at retrieval time (Phase 4 parses them
   directly), and — because today's `_citation_locator` already regexes
   chunk content for `page N` / `Table X` — they improve citation locators
   immediately with zero server changes.
4. **Idempotent and incremental.**
   - Default run ingests only documents whose manifest `sha256` differs
     from their recorded `ingested_sha256` (or that were never ingested).
   - `--replace <doc_id>` forces one document; `--rebuild` forces all.
   - Replacement deletes the document's previous LightRAG entries (found
     via `kv_store_doc_status.json` by file path) before re-inserting.
   - On success the manifest records `ingested_at` + `ingested_sha256`,
     so the manifest is the single source of truth for KB state.
5. **Ingest report.** Every run writes a JSON report (per-document action,
   pages, tables, chunk counts, failures) and a console summary.
6. **Dry-run.** `--dry-run` prints the plan (ingest/replace/skip and why)
   without touching the KB, the manifest, or the LLM.

The LLM/embedding configuration is unchanged in this phase (gpt-5.4-mini +
MiniLM) — model choices are Phase 3's measured decision.

## Acceptance criteria

1. Running the pipeline twice over an unchanged corpus performs zero
   inserts on the second run.
2. Changing one PDF (hash change) or passing `--replace <doc_id>`
   re-ingests only that document; other documents are untouched.
3. Superseded and unregistered documents are never ingested; the run
   report says so explicitly.
4. Every inserted content item begins with a source marker containing the
   real filename, doc_id, and page; table items also carry a table locator
   when one is detectable.
5. `--dry-run` makes no API calls and no writes.
6. Extraction/planning/marker logic is unit-tested without the heavy RAG
   dependencies or an OpenAI key.
7. `ingest.py` and `ingest_tables.py` are removed; `corpus/README.md` and
   the roadmap point at the new pipeline.

## Out of scope

- Embedding/model changes and the LightRAG-vs-vector decision (Phase 3).
- Server-side citation changes (Phase 4) — though locators improve as a
  side effect of markers.
- Running the actual re-ingest (owner runs it locally against `my_pdfs/`
  with an OpenAI key; roadmap Phase 3 pairs it with the model bake-off).

## Risks

- LightRAG document deletion is best-effort (its doc-status store maps
  file paths to internal ids). If deletion fails, the pipeline aborts the
  replacement rather than double-ingesting.
- Source markers add ~80 characters per chunk; negligible against the
  embedding token budget, and they carry retrieval-relevant tokens
  (document names, table ids).
