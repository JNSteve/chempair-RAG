"""Knowledge-base identity for /health and packaged KB artifacts.

Answers "which knowledge base is this server actually running?" — the
corpus manifest fingerprint plus the deployed storage's own document
counts. Everything here is best-effort and never raises: /health must not
fail because a file is missing or malformed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

KB_META_FILENAME = "kb_meta.json"


def manifest_fingerprint(manifest_path: str | Path) -> dict | None:
    """Identity of the corpus manifest checked into the repo: file hash +
    registered/current/ingested document counts."""
    path = Path(manifest_path)
    try:
        raw = path.read_bytes()
        import yaml

        documents = (yaml.safe_load(raw) or {}).get("documents") or []
        return {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "documents_registered": len(documents),
            "documents_current": sum(
                1
                for doc in documents
                if isinstance(doc, dict) and doc.get("status") == "current"
            ),
            "documents_ingested": sum(
                1
                for doc in documents
                if isinstance(doc, dict) and doc.get("ingested_sha256")
            ),
        }
    except Exception:
        return None


def storage_snapshot(rag_storage: str | Path) -> dict | None:
    """What the deployed LightRAG storage says about itself: doc counts by
    status, plus the kb_meta.json stamp if the KB was packaged by
    scripts/package_kb.py."""
    storage = Path(rag_storage)
    snapshot: dict = {}

    status_file = storage / "kv_store_doc_status.json"
    try:
        with open(status_file, encoding="utf-8") as handle:
            doc_status = json.load(handle)
        counts: dict[str, int] = {}
        for value in doc_status.values():
            if isinstance(value, dict):
                status = str(value.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
        snapshot["lightrag_documents"] = counts
    except Exception:
        pass

    meta_file = storage / KB_META_FILENAME
    try:
        with open(meta_file, encoding="utf-8") as handle:
            snapshot["packaged"] = json.load(handle)
    except Exception:
        pass

    return snapshot or None


def health_kb_info(manifest_path: str | Path, rag_storage: str | Path) -> dict:
    """The `kb` block for /health. Always returns a dict."""
    return {
        "manifest": manifest_fingerprint(manifest_path),
        "storage": storage_snapshot(rag_storage),
    }
