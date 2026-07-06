"""Package a rebuilt knowledge base for publishing as a GitHub release asset.

    python scripts/package_kb.py --rag-storage ./rag_storage --label v2.0

Stamps kb_meta.json into the storage dir (manifest fingerprint, doc counts,
label), then tars the storage contents into dist/rag_storage.tar.gz with
files at the archive root — the layout start.py expects when it downloads
and extracts the artifact on deploy.

Publish flow after this script (see docs/ops/KB_REBUILD_RUNBOOK.md):
  1. Create a GitHub release (e.g. tag v2.0) and upload dist/rag_storage.tar.gz
  2. Point the deployment at it via the KB_RELEASE_URL env var (start.py)
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb_info import KB_META_FILENAME, manifest_fingerprint, storage_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rag-storage", type=Path, default=Path("rag_storage"))
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "corpus" / "manifest.yaml"
    )
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--label", required=True, help="release label, e.g. v2.0")
    args = parser.parse_args()

    storage = args.rag_storage
    if not (storage / "vdb_entities.json").is_file():
        print(
            f"error: {storage} does not look like a LightRAG storage dir",
            file=sys.stderr,
        )
        return 1

    meta = {
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": manifest_fingerprint(args.manifest),
    }
    (storage / KB_META_FILENAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    tarball = args.out / "rag_storage.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for path in sorted(storage.iterdir()):
            if path.is_file():
                tar.add(path, arcname=path.name)

    snapshot = storage_snapshot(storage) or {}
    doc_counts = snapshot.get("lightrag_documents", {})
    size_mb = tarball.stat().st_size / (1024 * 1024)
    print(f"packaged {tarball} ({size_mb:.0f} MB)")
    print(f"  label: {args.label}")
    print(f"  lightrag documents by status: {doc_counts}")
    print("\nnext steps:")
    print(f"  1. create a GitHub release tagged {args.label} and upload {tarball}")
    print(
        "  2. set KB_RELEASE_URL on the deployment to the new asset URL "
        "(start.py downloads it when the data volume is empty)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
