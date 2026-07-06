import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import kb_info  # noqa: E402

MANIFEST_YAML = """\
schema_version: 1
documents:
  - doc_id: doc-a
    status: current
    ingested_sha256: aaa
  - doc_id: doc-b
    status: superseded
    ingested_sha256: null
"""


def test_manifest_fingerprint_counts_and_hash(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(MANIFEST_YAML, encoding="utf-8")

    info = kb_info.manifest_fingerprint(manifest)
    assert info == {
        "manifest_sha256": hashlib.sha256(MANIFEST_YAML.encode()).hexdigest(),
        "documents_registered": 2,
        "documents_current": 1,
        "documents_ingested": 1,
    }


def test_manifest_fingerprint_missing_file_is_none(tmp_path):
    assert kb_info.manifest_fingerprint(tmp_path / "nope.yaml") is None


def test_storage_snapshot_counts_and_packaged_meta(tmp_path):
    (tmp_path / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                "d1": {"status": "processed"},
                "d2": {"status": "processed"},
                "d3": {"status": "failed"},
                "junk": "ignored",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / kb_info.KB_META_FILENAME).write_text(
        json.dumps({"label": "v2.0"}), encoding="utf-8"
    )

    snapshot = kb_info.storage_snapshot(tmp_path)
    assert snapshot["lightrag_documents"] == {"processed": 2, "failed": 1}
    assert snapshot["packaged"]["label"] == "v2.0"


def test_storage_snapshot_empty_dir_is_none(tmp_path):
    assert kb_info.storage_snapshot(tmp_path) is None


def test_health_kb_info_never_raises(tmp_path):
    info = kb_info.health_kb_info(tmp_path / "missing.yaml", tmp_path / "missing-dir")
    assert info == {"manifest": None, "storage": None}


def test_package_kb_stamps_meta_and_builds_flat_tarball(tmp_path, monkeypatch, capsys):
    import package_kb

    storage = tmp_path / "rag_storage"
    storage.mkdir()
    (storage / "vdb_entities.json").write_text("{}", encoding="utf-8")
    (storage / "kv_store_doc_status.json").write_text(
        json.dumps({"d1": {"status": "processed"}}), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(MANIFEST_YAML, encoding="utf-8")
    out = tmp_path / "dist"

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            rag_storage=storage, manifest=manifest, out=out, label="v2.0"
        ),
    )
    assert package_kb.main() == 0

    meta = json.loads((storage / kb_info.KB_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["label"] == "v2.0"
    assert meta["manifest"]["documents_registered"] == 2

    with tarfile.open(out / "rag_storage.tar.gz") as tar:
        names = sorted(tar.getnames())
    # Flat layout, exactly what start.py's extraction expects.
    assert names == ["kb_meta.json", "kv_store_doc_status.json", "vdb_entities.json"]
