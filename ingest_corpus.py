"""Manifest-driven corpus ingestion for the Chempair RAG knowledge base.

Replaces the legacy ingest.py / ingest_tables.py two-pass scripts.

    python ingest_corpus.py --dry-run              # show the plan only
    python ingest_corpus.py                        # ingest new/changed docs
    python ingest_corpus.py --replace <doc_id>     # force one document
    python ingest_corpus.py --rebuild              # force everything

Scope comes from corpus/manifest.yaml (status: current documents only);
register PDFs with scripts/corpus_manifest.py first. Each run writes a JSON
report under reports/ingest/ and records ingested_at / ingested_sha256 back
into the manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_pipeline as pipeline  # noqa: E402
from corpus_manifest import load_manifest, save_manifest, validate_documents  # noqa: E402

RAG_STORAGE = os.environ.get("RAG_STORAGE", "./rag_storage")
LLM_MODEL = "gpt-5.4-mini"  # model changes are Phase 3's measured decision
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DELAY_BETWEEN_DOCS = 10  # seconds, matches legacy rate-limit spacing

_rate_limited = False


def build_rag():
    """RAGAnything wired exactly like the legacy scripts (heavy imports
    kept local so --dry-run and tests never need them)."""
    import numpy as np
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    async def local_embed(texts: list[str]):
        return np.array(
            embed_model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        global _rate_limited
        if _rate_limited:
            raise RuntimeError("RATE_LIMIT_STOP")
        try:
            return await openai_complete_if_cache(
                LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("OPENAI_API_KEY"),
                **kwargs,
            )
        except Exception as exc:
            message = str(exc).lower()
            if any(
                word in message for word in ("rate", "quota", "429", "insufficient")
            ):
                _rate_limited = True
                print("\n!!! RATE LIMIT / QUOTA HIT — STOPPING !!!")
                raise RuntimeError("RATE_LIMIT_STOP") from exc
            raise

    rag = RAGAnything(
        config=RAGAnythingConfig(
            working_dir=RAG_STORAGE,
            enable_image_processing=False,
            enable_table_processing=True,
            enable_equation_processing=False,
        ),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM, max_token_size=8192, func=local_embed
        ),
    )
    # Content is pre-extracted here; skip RAGAnything's parser check.
    rag._parser_installation_checked = True
    return rag


async def delete_previous(rag, filename: str) -> None:
    """Remove a document's existing KB entries before re-inserting. Raises
    on failure so a replacement never double-ingests."""
    for lightrag_id in pipeline.lightrag_doc_ids_for_file(RAG_STORAGE, filename):
        await rag.lightrag.adelete_by_doc_id(lightrag_id)


async def run(args: argparse.Namespace) -> int:
    global _rate_limited

    manifest = load_manifest(args.manifest)
    corpus_dir = Path(args.corpus_dir)
    problems = validate_documents(manifest, corpus_dir if corpus_dir.is_dir() else None)
    if problems:
        for problem in problems:
            print(f"manifest error: {problem}", file=sys.stderr)
        return 1

    documents = manifest.get("documents") or []
    if not documents:
        print("manifest is empty — register PDFs with scripts/corpus_manifest.py add")
        return 1

    plan = pipeline.plan_actions(
        documents, rebuild=args.rebuild, replace_ids=tuple(args.replace)
    )
    pending = [p for p in plan if p.action != pipeline.SKIP]
    for planned in plan:
        print(f"  {planned.action.upper():7s} {planned.doc_id}: {planned.reason}")
    print(
        f"\n{len(pending)} document(s) to process, {len(plan) - len(pending)} skipped."
    )
    if args.dry_run or not pending:
        return 0

    rag = build_rag()
    report_entries = []
    processed = 0
    for planned in pending:
        if _rate_limited:
            report_entries.append(
                {
                    "doc_id": planned.doc_id,
                    "action": planned.action,
                    "status": "not_run",
                }
            )
            continue

        doc = planned.doc
        filename = doc["filename"]
        pdf_path = corpus_dir / filename
        print(f"\n[{planned.action}] {planned.doc_id} ({filename})")
        entry = {
            "doc_id": planned.doc_id,
            "action": planned.action,
            "reason": planned.reason,
        }
        try:
            text_items = pipeline.extract_text_items(
                str(pdf_path), filename, doc["doc_id"]
            )
            table_items = pipeline.extract_table_items(
                str(pdf_path), filename, doc["doc_id"]
            )
            content_list = pipeline.build_content_list(text_items, table_items)
            entry.update(
                {
                    "pages": len(text_items),
                    "tables": len(table_items),
                    "items": len(content_list),
                }
            )
            if not content_list:
                raise RuntimeError("no text or tables extracted")

            if planned.action == pipeline.REPLACE:
                await delete_previous(rag, filename)
            await rag.insert_content_list(content_list=content_list, file_path=filename)

            pipeline.mark_ingested(
                doc, datetime.now(timezone.utc).isoformat(timespec="seconds")
            )
            save_manifest(args.manifest, manifest)  # persist per doc: crash-safe
            entry["status"] = "success"
            processed += 1
            print(f"  OK: {len(text_items)} pages, {len(table_items)} tables")
        except Exception as exc:  # noqa: BLE001 — per-document isolation
            entry["status"] = "failed"
            entry["error"] = str(exc)[:300]
            print(f"  ERROR: {entry['error']}")
        report_entries.append(entry)

        if planned is not pending[-1] and not _rate_limited:
            await asyncio.sleep(DELAY_BETWEEN_DOCS)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"ingest-{stamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "run_at": stamp,
                "corpus_dir": str(corpus_dir),
                "rag_storage": RAG_STORAGE,
                "llm_model": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "rate_limited": _rate_limited,
                "documents": report_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    failed = sum(1 for entry in report_entries if entry.get("status") != "success")
    print(
        f"\nDone: {processed} succeeded, {failed} failed/not run. Report: {report_path}"
    )
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "corpus" / "manifest.yaml"
    )
    parser.add_argument(
        "--corpus-dir", default="my_pdfs", help="folder containing the PDFs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; touch nothing"
    )
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="DOC_ID",
        help="force re-ingestion of one document (repeatable)",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="force re-ingestion of everything"
    )
    parser.add_argument(
        "--report-dir", default="reports/ingest", help="run report output folder"
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
