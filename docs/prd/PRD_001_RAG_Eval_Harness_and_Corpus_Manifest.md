# PRD_001 — RAG Eval Harness and Corpus Manifest (Roadmap Phases 0–1)

> Roadmap: `docs/RAG_PERFECTION_ROADMAP.md`. This PRD covers Phase 0 (eval
> harness) and Phase 1 (corpus filing). No production behavior changes —
> `server.py`, routing, and the frontend contract are untouched.

## Problem

1. **Live answer quality has never been measured.** The Package 4 golden QA
   (2026-05-12) verified routing and contract handling with mocked KB
   retrieval. Corpus recall, citation quality, and grounded-answer quality
   against the real knowledge base are unverified. Later roadmap phases
   (embedding bake-off, LightRAG vs vector retrieval, exact citations) need a
   measured baseline to prove improvement against.
2. **The corpus is unfiled.** The knowledge base is whatever PDFs sit in a
   local `my_pdfs/` folder. There is no record of which documents are in the
   KB, their versions, jurisdictions, or hashes. Superseded guidelines cannot
   be identified or replaced; the corpus cannot be audited or rebuilt
   reproducibly.

## What

### Phase 0 — Golden eval harness (`evals/`)

- A versioned golden question set (`evals/golden_questions.yaml`) of ~30
  questions across categories: project-evidence answers (contaminants,
  exceedances), threshold lookups, criteria explanations, source/pathway,
  jurisdiction-specific KB questions, follow-up questions, prompt-injection
  attempts, and off-topic guardrails.
- Reusable workspace-context fixtures (`evals/context_profiles.yaml`)
  matching the enviro-sage schema v4 payload (Ducat soil profile from the
  Package 4 QA, plus a no-project profile).
- A runner (`evals/run_eval.py`) that POSTs each question to a live backend
  `/query`, scores the response against per-question expectations, and writes
  a JSON report + markdown scorecard.
- Scoring dimensions: route accuracy, required/forbidden answer content,
  citation presence, citation exactness (real locator vs "source passage"
  fallback), grounding flags, and injection safety.

### Phase 1 — Corpus manifest (`corpus/`)

- `corpus/manifest.yaml`: registry of every KB document — `doc_id`, title,
  guideline family, jurisdiction, version, status (current/superseded-by),
  filename, SHA-256, source URL, ingestion timestamp.
- `scripts/corpus_manifest.py` CLI:
  - `add` — register a PDF from the local corpus folder (computes hash).
  - `validate` — schema check, unique IDs/filenames, supersession references
    resolve; with `--corpus-dir`, verify files exist, hashes match, and flag
    unregistered PDFs.
  - `list` — human-readable inventory.
- `corpus/README.md`: filing conventions (naming, families, versioning,
  supersession).

## Acceptance criteria

1. `python evals/run_eval.py --base-url <url>` runs the full golden set
   against a live backend and produces a JSON report and markdown scorecard
   with per-category and overall scores. Exit code reflects pass/fail against
   a configurable threshold.
2. The question set covers all seven categories above, includes the five
   original Package 4 golden questions unchanged, and every question declares
   machine-checkable expectations.
3. `python scripts/corpus_manifest.py validate` fails on: duplicate
   `doc_id`/filename, missing required fields, invalid status, dangling
   `superseded_by`; and with `--corpus-dir` on missing files, hash
   mismatches, and unregistered PDFs.
4. `add` registers a real PDF with a correct SHA-256 without hand-editing.
5. Unit tests cover the eval scoring logic and manifest validation without
   requiring the heavy RAG dependencies (lightrag/sentence-transformers) or
   network access.
6. No changes to `server.py`, routing modules, or the frontend contract.

## Out of scope

- Running the baseline eval (requires the live corpus, which exists only on
  the owner's machine — the owner runs the harness and commits the
  scorecard).
- Populating the manifest with the real corpus (owner runs `add` against
  `my_pdfs/`; placeholder-free manifest ships empty).
- Any ingestion pipeline changes (Phase 2), retrieval changes (Phase 3), or
  citation changes (Phase 4).

## Risks

- Live-KB answer text is nondeterministic; expectations are therefore
  keyword/route/citation-based, not exact-match. Route and citation
  assertions carry the signal.
- The eval calls a paid LLM per question (~30 calls/run); the runner supports
  `--filter` to run subsets during iteration.
