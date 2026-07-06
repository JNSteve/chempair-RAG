import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import corpus_manifest  # noqa: E402


def _entry(**overrides) -> dict:
    entry = {
        "doc_id": "nepm-2013-schedule-b1",
        "title": "NEPM 2013 Schedule B1",
        "family": "NEPM",
        "jurisdiction": "AU",
        "version": "2013",
        "status": "current",
        "superseded_by": None,
        "filename": "NEPM_2013_Schedule_B1.pdf",
        "sha256": "a" * 64,
        "source_url": None,
        "ingested_at": None,
    }
    entry.update(overrides)
    return entry


def test_valid_manifest_has_no_problems():
    manifest = {"schema_version": 1, "documents": [_entry()]}
    assert corpus_manifest.validate_documents(manifest, corpus_dir=None) == []


def test_empty_manifest_is_valid():
    assert corpus_manifest.validate_documents({"documents": []}, corpus_dir=None) == []


def test_missing_required_field_is_reported():
    manifest = {"documents": [_entry(title=None)]}
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=None)
    assert any("missing required field 'title'" in p for p in problems)


def test_duplicate_doc_id_and_filename_are_reported():
    manifest = {"documents": [_entry(), _entry()]}
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=None)
    assert any("duplicate doc_id" in p for p in problems)
    assert any("duplicate filename" in p for p in problems)


def test_invalid_status_and_family_are_reported():
    manifest = {"documents": [_entry(status="retired", family="ISO")]}
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=None)
    assert any("invalid status" in p for p in problems)
    assert any("unknown family" in p for p in problems)


def test_superseded_requires_resolving_reference():
    manifest = {"documents": [_entry(status="superseded", superseded_by="ghost-doc")]}
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=None)
    assert any("not a known doc_id" in p for p in problems)

    manifest = {"documents": [_entry(status="superseded", superseded_by=None)]}
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=None)
    assert any("must set superseded_by" in p for p in problems)


def test_superseded_by_without_superseded_status_is_reported():
    replacement = _entry(doc_id="nepm-2025-schedule-b1", filename="NEPM_2025.pdf")
    manifest = {
        "documents": [_entry(superseded_by="nepm-2025-schedule-b1"), replacement]
    }
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=None)
    assert any("status is not 'superseded'" in p for p in problems)


def test_malformed_sha256_is_reported():
    manifest = {"documents": [_entry(sha256="not-a-hash")]}
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=None)
    assert any("sha256" in p for p in problems)


def test_corpus_dir_checks_files_hashes_and_strays(tmp_path):
    pdf = tmp_path / "NEPM_2013_Schedule_B1.pdf"
    pdf.write_bytes(b"pdf-bytes")
    stray = tmp_path / "Unfiled_Guideline.pdf"
    stray.write_bytes(b"stray")

    good = _entry(sha256=corpus_manifest.sha256_of(pdf))
    missing = _entry(
        doc_id="anzecc-2000-guidelines",
        filename="ANZECC_2000.pdf",
        family="ANZECC",
    )
    manifest = {"documents": [good, missing]}
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=tmp_path)
    assert any("file not found" in p for p in problems)
    assert any(
        "unregistered PDF" in p and "Unfiled_Guideline.pdf" in p for p in problems
    )
    assert not any("sha256 mismatch" in p for p in problems)

    manifest = {"documents": [_entry(sha256="b" * 64)]}
    problems = corpus_manifest.validate_documents(manifest, corpus_dir=tmp_path)
    assert any("sha256 mismatch" in p for p in problems)


@pytest.fixture()
def manifest_file(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        "# Corpus manifest test fixture\n\nschema_version: 1\ndocuments: []\n",
        encoding="utf-8",
    )
    return path


def test_cmd_add_registers_pdf_with_real_hash(manifest_file, tmp_path, capsys):
    pdf = tmp_path / "NEPM_2013_Schedule_B1.pdf"
    pdf.write_bytes(b"pdf-bytes")

    args = _add_args(manifest_file, pdf)
    assert corpus_manifest.cmd_add(args) == 0

    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    (entry,) = manifest["documents"]
    assert entry["doc_id"] == "nepm-2013-schedule-b1"
    assert entry["sha256"] == corpus_manifest.sha256_of(pdf)
    assert entry["status"] == "current"
    # Header comment survives the rewrite.
    assert manifest_file.read_text(encoding="utf-8").startswith(
        "# Corpus manifest test fixture"
    )
    # And the result validates cleanly against the corpus dir.
    assert corpus_manifest.validate_documents(manifest, corpus_dir=tmp_path) == []


def test_cmd_add_rejects_duplicates(manifest_file, tmp_path):
    pdf = tmp_path / "NEPM_2013_Schedule_B1.pdf"
    pdf.write_bytes(b"pdf-bytes")
    assert corpus_manifest.cmd_add(_add_args(manifest_file, pdf)) == 0
    assert corpus_manifest.cmd_add(_add_args(manifest_file, pdf)) == 1


def _add_args(manifest_file, pdf):
    import argparse

    return argparse.Namespace(
        manifest=manifest_file,
        pdf=str(pdf),
        doc_id=None,
        title="NEPM 2013 Schedule B1",
        family="NEPM",
        jurisdiction="AU",
        version="2013",
        source_url=None,
    )


def test_cmd_seed_bulk_registers_and_is_idempotent(manifest_file, tmp_path, capsys):
    import argparse

    (tmp_path / "NEPM_2013_Schedule_B1.pdf").write_bytes(b"pdf-one")
    metadata = tmp_path / "seed_metadata.yaml"
    metadata.write_text(
        """
documents:
  - filename: NEPM_2013_Schedule_B1.pdf
    doc_id: nepm-2013-schedule-b1
    title: NEPM 2013 Schedule B1
    family: NEPM
    jurisdiction: AU
    version: "2013"
  - filename: Missing_Doc.pdf
    doc_id: missing-doc
    title: Missing Document
    family: OTHER
    jurisdiction: AU
    version: unknown
""",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest=manifest_file, metadata=metadata, corpus_dir=str(tmp_path)
    )

    # First run: one added, one missing -> exit 1 flags the gap
    assert corpus_manifest.cmd_seed(args) == 1
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    (entry,) = manifest["documents"]
    assert entry["doc_id"] == "nepm-2013-schedule-b1"
    assert entry["sha256"] == corpus_manifest.sha256_of(
        tmp_path / "NEPM_2013_Schedule_B1.pdf"
    )

    # Second run: already registered -> skipped, still flags the missing file
    assert corpus_manifest.cmd_seed(args) == 1
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    assert len(manifest["documents"]) == 1


def test_repo_seed_metadata_is_well_formed():
    metadata_path = ROOT / "corpus" / "seed_metadata.yaml"
    entries = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))["documents"]
    assert len(entries) == 30
    ids = [e["doc_id"] for e in entries]
    filenames = [e["filename"] for e in entries]
    assert len(set(ids)) == 30 and len(set(filenames)) == 30
    for entry in entries:
        assert corpus_manifest.DOC_ID_PATTERN.match(entry["doc_id"])
        assert entry["family"] in corpus_manifest.VALID_FAMILIES
        assert entry["title"] and entry["jurisdiction"]
