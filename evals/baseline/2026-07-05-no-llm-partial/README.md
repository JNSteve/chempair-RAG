# Partial baseline — 2026-07-05

Run from a cloud session against the **production KB snapshot** (the
`rag_storage.tar.gz` release artifact, v1.0) with **no OpenAI key** and no
egress to OpenAI/HuggingFace. That means:

- Deterministic project-evidence routes ran fully: **4/4 passed**
  (`contaminants_project`, `exceedances_project`, `worst_arsenic_sample`,
  `injection_original`).
- Every question needing the LLM or KB retrieval errored (`HTTP 500`,
  missing key / blocked tokenizer download) — not scored.

This is a harness/contract validation and a KB audit, **not** the full
Phase 0 baseline. The full baseline still needs one run from a machine
with `OPENAI_API_KEY` and the backend running:

```bash
python evals/run_eval.py --base-url <url> --api-key $RAG_API_KEY --out evals/baseline/<date>/
```

## KB audit findings (from the v1.0 release artifact itself)

Inventory of `kv_store_doc_status.json` in the deployed KB:

- **57 LightRAG documents** from ~29 source PDFs (text + legacy `tables_`
  passes), 1,347 text chunks total.
- **All 22 NEPM 2013 volumes (`F2013C00288VOL01–22.pdf`) have text status
  `failed`** — partial chunks exist (rate-limit aborts mid-document), so
  coverage of the single most important guideline is incomplete and
  unquantified. `NEPM_2013 (1).pdf` and `pfas-nemp-3.pdf` likewise.
- Only 4 documents fully processed for text: `140796-classify-waste`,
  `20p2233-consultants-reporting…`, `QLD_Waste_Framework`,
  `QLD_Waste_Operating_ERA`.
- Table passes: mixed — `tables_F2013C00288VOL02–11` failed,
  VOL12–20 processed.

**Implication:** the Phase 2 pipeline's full re-ingest (`ingest_corpus.py
--rebuild`) is not just hygiene — it repairs a partially-failed KB build.
The manifest for the corpus should be seeded from the 29 source PDFs
listed by `kv_store_doc_status.json`.
