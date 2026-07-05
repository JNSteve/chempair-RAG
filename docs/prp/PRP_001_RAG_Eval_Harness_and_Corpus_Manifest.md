# PRP_001 — RAG Eval Harness and Corpus Manifest

> Implements `docs/prd/PRD_001_RAG_Eval_Harness_and_Corpus_Manifest.md`.

## Implementation steps

1. **Docs**: add `docs/RAG_PERFECTION_ROADMAP.md` (phase plan + status
   table), this PRP, and the PRD.
2. **Eval harness** (`evals/`):
   - `evals/context_profiles.yaml` — schema v4 workspace-context fixtures
     (`ducat_soil` from Package 4 QA; `none` for no-project questions).
   - `evals/golden_questions.yaml` — ~30 questions with per-question
     expectations (`route_used`, `must_include`, `must_not_include`,
     `min_citations`, `citation_source_pattern`, `grounded`,
     `exact_locator`). Follow-ups reference `follow_up_of` and reuse the
     parent's session.
   - `evals/scoring.py` — pure scoring functions (no I/O): evaluate one
     response against expectations, aggregate per-category and overall.
   - `evals/run_eval.py` — stdlib-only HTTP runner (urllib): loads YAML,
     resolves context profiles, POSTs to `/query` with optional bearer key,
     threads `session_id` for follow-ups, writes `report.json` +
     `scorecard.md`, exits nonzero below `--min-score`.
3. **Corpus manifest** (`corpus/`, `scripts/`):
   - `corpus/manifest.yaml` — empty registry with documented schema header.
   - `corpus/README.md` — filing conventions.
   - `scripts/corpus_manifest.py` — `add` / `validate` / `list` CLI,
     stdlib + PyYAML only.
4. **Tests** (`tests/`): `test_eval_scoring.py`,
   `test_golden_questions_schema.py`, `test_corpus_manifest.py` — no heavy
   deps, no network.
5. **Deps**: add `pyyaml` to `requirements.txt`.

## Verification

- `python -m pytest tests/test_eval_scoring.py tests/test_golden_questions_schema.py tests/test_corpus_manifest.py`
- `python -m ruff check evals scripts tests/test_eval_scoring.py tests/test_golden_questions_schema.py tests/test_corpus_manifest.py`
- Existing test suites unchanged and unaffected (no imports from new code in
  `server.py` or routing modules).

## Owner follow-up (post-merge)

1. Register the corpus:
   `python scripts/corpus_manifest.py add my_pdfs/*.pdf` then
   `python scripts/corpus_manifest.py validate --corpus-dir my_pdfs`.
2. Run the baseline:
   `python evals/run_eval.py --base-url <backend-url> --api-key $RAG_API_KEY --out evals/baseline/`
   and commit the scorecard to `evals/baseline/`.

## Rollback

Additive-only change (new directories + one dev dependency). Rollback =
revert the PR; no data, schema, or runtime impact.
