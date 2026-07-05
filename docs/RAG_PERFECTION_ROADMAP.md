# RAG Perfection Roadmap

Phased plan to take the Chempair workspace RAG ("Alfie") from working to
trustworthy and maintainable. Each phase is one PRD/PRP + PR-sized chunk with
its own exit criteria. The serving layer (`server.py` routing/contract) is
kept; the ingestion/filing layer is rebuilt.

Frontend counterpart: `JNSteve/enviro-sage` (`workspace-rag-query` edge
function proxy + `workspaceChatContext.ts` schema v4 payload). The contract is
documented in `frontend-rag-handoff.md`.

## Phase 0 — Eval harness (measure before touching anything)

- Expand the 5 Package-4 golden questions to a ~30-question eval set:
  threshold lookups, table-specific, jurisdiction-specific (NSW/VIC/QLD),
  source/pathway, follow-ups, injection attempts.
- Runnable eval script scoring route accuracy, answer content, citation
  presence/exactness, and grounding against the **live** KB (not mocks).
- Record the baseline scorecard in the repo.

**Exit:** baseline scorecard committed.

## Phase 1 — Corpus filing (the manifest)

- `corpus/manifest.yaml`: every KB document registered with `doc_id`, title,
  guideline family, jurisdiction, version/year, status (current/superseded),
  SHA-256 hash, source URL.
- Naming conventions + validation tooling (`scripts/corpus_manifest.py`).
- Inventory of the current `my_pdfs/` corpus and gap analysis against the
  criteria families in enviro-sage's `criteriaData.ts`.

**Exit:** every KB document registered, versioned, and hash-verified.

## Phase 2 — One idempotent ingestion pipeline

- Single CLI replacing `ingest.py` + `ingest_tables.py`: text **and** tables
  in one pass per document; no `tables_` filename hack.
- Every chunk carries `doc_id`, page, section heading, table locator.
- Idempotent and incremental: hash-based skip, `--replace` for a single
  superseded document, `--rebuild` for the full corpus. Ingest report per run.

**Exit:** re-ingesting the same corpus twice is a no-op; replacing one
document doesn't disturb the rest.

## Phase 3 — Retrieval upgrade + full re-ingest

- Embedding bake-off using the Phase 0 eval set: current MiniLM vs a
  BGE-class local model vs API embeddings. Pick on measured recall.
- Decide whether LightRAG's LLM-built knowledge graph earns its ingestion
  cost vs plain vector retrieval + reranking with the new metadata.
- Full re-ingest through the new pipeline with the winning setup.

**Exit:** eval scorecard beats the Phase 0 baseline.

## Phase 4 — Exact citations

- Citation extraction reads document title, page, and table locator from
  chunk **metadata** instead of regex-guessing from prose.
- Citations become verifiable: "NEPM 2013 Schedule B1, Table 1A(1), p. 14".
- Frontend already renders `source/title/locator/snippet`; verify end-to-end.

**Exit:** citation-exactness score ~100% on covered documents.

## Phase 5 — Answer quality & routing polish

- Re-run golden + injection evals live; tune routing guardrails and the
  context-bot spec against real failures found in Phases 3–4.
- Revisit the answer LLM choice with measurements, not assumptions.
- Clear the pre-existing ruff debt (mostly dissolved by deleting the old
  ingest scripts).

**Exit:** full eval suite green; injection suite green; routing regressions
covered by tests.

## Phase 6 — Ops & repeatability

- Long-term home for the corpus and `rag_storage` (object storage / release
  artifacts), with a documented rebuild runbook.
- KB versioning: `/health` reports corpus version + manifest hash.
- CI for this repo (ruff + pytest + eval smoke), mirroring enviro-sage's
  quality gates.

**Exit:** anyone can rebuild or update the KB from the runbook.

## Status

| Phase | PRD/PRP | Status |
|-------|---------|--------|
| 0 — Eval harness | PRD_001 / PRP_001 | In progress |
| 1 — Corpus manifest | PRD_001 / PRP_001 | In progress |
| 2 — Ingestion pipeline | — | Not started |
| 3 — Retrieval upgrade | — | Not started |
| 4 — Exact citations | — | Not started |
| 5 — Answer quality | — | Not started |
| 6 — Ops | — | Not started |
