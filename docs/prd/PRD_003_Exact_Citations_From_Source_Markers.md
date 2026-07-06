# PRD_003 — Exact Citations from Source Markers (Roadmap Phase 4)

> Roadmap: `docs/RAG_PERFECTION_ROADMAP.md`. Depends on PRD_002 (chunks now
> carry `[source: <file> | doc: <id> | page <n> | Table <x>]` markers).

## Problem

Citation locators are regex-guessed from chunk prose and internal chunk
ids (`_citation_locator` in `server.py`), falling back to "source passage"
when nothing matches. For a regulatory tool, citations must be exact and
human-checkable.

## What

1. Extract the citation logic out of `server.py` into a dependency-light
   `citation_extraction.py` module.
2. When a retrieved chunk starts with a Phase 2 source marker, build the
   citation from it verbatim: source = marker filename, locator =
   `"Table <x>, p. <n>"` or `"p. <n>"`. No guessing.
3. Strip markers from citation snippets so users see document text.
4. Keep the legacy heuristics as the fallback for chunks ingested before
   Phase 2, byte-for-byte compatible (the existing response-contract tests
   pass unchanged), so the server works against old and new KBs alike.

## Acceptance criteria

1. Marker chunks yield exact locators and marker-derived sources; marker
   filename wins over the (possibly stale) reference file path.
2. Legacy chunks produce identical citations to today — verified by the
   untouched `TestResponseContract` tests.
3. Snippets never contain marker plumbing.
4. The new module is unit-tested without the RAG stack; the full server
   suite passes.
5. `/query` response schema is unchanged (frontend needs no changes).

## Out of scope

- Re-ingesting the corpus so markers exist in production (Phase 3 run).
- Golden-eval `exact_locator` checks flipping from `severity: info` to
  blocking — do that after the re-ingested KB ships.
