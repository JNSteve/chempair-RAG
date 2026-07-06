# Corpus filing conventions

The knowledge base is built from the PDFs registered in `manifest.yaml`.
PDFs themselves live in the local corpus folder (`my_pdfs/`, gitignored);
the manifest is the versioned source of truth for what the KB contains.

## Rules

1. **Every PDF in the corpus folder must be registered** in `manifest.yaml`,
   and every manifest entry must point at an existing file with a matching
   SHA-256. `scripts/corpus_manifest.py validate --corpus-dir my_pdfs`
   enforces both directions.
2. **Register via the CLI**, which computes the hash for you:

   ```bash
   python scripts/corpus_manifest.py add my_pdfs/NEPM_2013_Schedule_B1.pdf \
       --doc-id nepm-2013-schedule-b1 \
       --title "NEPM 2013 Schedule B1 — Investigation Levels for Soil and Groundwater" \
       --family NEPM --jurisdiction AU --version 2013
   ```

3. **Filenames**: `<Family>_<Year>_<Descriptor>.pdf`, no spaces
   (e.g. `ANZECC_2000_Water_Quality_Guidelines.pdf`).
4. **Never delete a superseded document's entry.** Mark it
   `status: superseded` with `superseded_by: <new-doc-id>` so answers can be
   traced to the KB version that produced them. The Phase 2 ingestion
   pipeline skips superseded documents.
5. **A new edition is a new document** with a new `doc_id` (version suffix),
   not an in-place edit.

## Commands

```bash
python scripts/corpus_manifest.py seed --corpus-dir my_pdfs    # one-command bulk registration (corpus/seed_metadata.yaml)
python scripts/corpus_manifest.py list                          # inventory
python scripts/corpus_manifest.py validate                      # schema-only
python scripts/corpus_manifest.py validate --corpus-dir my_pdfs # + files/hashes
```

## Ingestion

The knowledge base is built from the manifest by `ingest_corpus.py`
(requires `OPENAI_API_KEY`):

```bash
python ingest_corpus.py --dry-run           # show what would happen
python ingest_corpus.py                     # ingest new/changed documents
python ingest_corpus.py --replace <doc_id>  # force one document
python ingest_corpus.py --rebuild           # force everything
```

Runs are idempotent: a document is only re-ingested when its file hash
differs from the manifest's `ingested_sha256` (or when forced). Superseded
and unregistered documents are never ingested. Every run writes a JSON
report under `reports/ingest/`.
