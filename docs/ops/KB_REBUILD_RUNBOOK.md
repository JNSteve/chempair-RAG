# KB rebuild & publish runbook

The full cycle for rebuilding the knowledge base and shipping it to the
deployed backend. Total hands-on time is ~15 minutes; the ingest run itself
is unattended (resume-safe if rate-limited: just re-run the same command).

Prereqs: repo checkout, `pip install -r requirements.txt`, corpus PDFs in
`my_pdfs/`, `OPENAI_API_KEY` in `.env`.

## 1. Register the corpus (first time only)

```bash
python scripts/corpus_manifest.py seed --corpus-dir my_pdfs
python scripts/corpus_manifest.py validate --corpus-dir my_pdfs
```

Fix anything `validate` flags (unregistered PDFs, hash mismatches) before
continuing. New/updated documents later: `corpus_manifest.py add` (new doc)
or edit the manifest to mark the old edition `superseded`.

## 2. Baseline before touching anything (first rebuild only)

With the backend running against the OLD KB:

```bash
python evals/run_eval.py --base-url <backend-url> --api-key $RAG_API_KEY \
    --out evals/baseline/full-pre-rebuild/
```

Commit the scorecard — this is the "before" for every later comparison.

## 3. Rebuild

```bash
python ingest_corpus.py --dry-run    # sanity-check the plan
python ingest_corpus.py --rebuild    # unattended; re-run to resume after rate limits
```

Check the run report under `reports/ingest/` — every document should be
`success` before publishing. Re-run until the plan shows all `SKIP: up to
date`.

## 4. Verify the new KB locally

```bash
RAG_STORAGE=./rag_storage python start.py   # or run server.py directly
python evals/run_eval.py --base-url http://localhost:8000 --api-key $RAG_API_KEY \
    --out evals/baseline/full-post-rebuild/
python evals/retrieval_eval.py --rag-storage ./rag_storage   # retrieval-only metrics
```

The post-rebuild scorecard should beat the pre-rebuild one; citation
locators should show `Table X, p. N` (markers) instead of `source passage`.

## 5. Package & publish

```bash
python scripts/package_kb.py --rag-storage ./rag_storage --label v2.0
```

1. Create a GitHub release tagged (e.g.) `v2.0` and upload
   `dist/rag_storage.tar.gz` as the asset.
2. On the deployment, set `KB_RELEASE_URL` to the new asset URL and clear
   the data volume (or delete `vdb_entities.json`) so `start.py` re-downloads
   on next boot.
3. Confirm `GET /health` — the `kb.storage.packaged.label` field should show
   the new label, and `kb.manifest.manifest_sha256` should match the repo's
   manifest.

## Rollback

Point `KB_RELEASE_URL` back at the previous release asset and clear the data
volume again. Manifests are versioned in git; the old KB artifact stays on
its release.
