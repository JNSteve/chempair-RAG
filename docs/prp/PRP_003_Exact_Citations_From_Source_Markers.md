# PRP_003 — Exact Citations from Source Markers

> Implements `docs/prd/PRD_003_Exact_Citations_From_Source_Markers.md`.

## Implementation steps

1. Add `citation_extraction.py`: marker parsing (`parse_source_marker`,
   `strip_source_marker`), marker-based citation building, and the legacy
   helpers moved verbatim from `server.py` (`_file_source_name`,
   `_citation_title`, `_extract_table_locator`, `_citation_locator`,
   `_bounded_snippet`, `extract_citations_from_rag_payload`,
   `MAX_CITATIONS`, `MAX_SNIPPET_LENGTH`).
2. `server.py`: delete the moved code; import
   `extract_citations_from_rag_payload`; `_fetch_rag_citations` calls it.
   No other server changes.
3. Tests: `tests/test_citation_extraction.py` — marker parsing, exact
   marker citations, legacy behavior preservation (mirroring the
   `TestResponseContract` fixtures), snippet stripping/bounding, citation
   cap.
4. Housekeeping in the same change: repo-wide ruff green (one legacy
   `start.py` fix) and `.github/workflows/ci.yml` (lint + light tests +
   full suite), starting roadmap Phases 5–6.

## Verification

- `python -m pytest tests -q` (full suite, including the existing
  `test_query_context.py` response-contract tests, unchanged).
- `ruff check .` and `ruff format --check .` — repo-wide clean.

## Rollback

Revert the PR: the helpers return to `server.py` and behavior is identical
(legacy path is byte-compatible). No data or schema impact.
