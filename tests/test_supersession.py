"""Superseded corpus documents must never read as current sources: the
citation title carries a warning and the rendered KB evidence tags the
passage so the answer model treats it as historical (KB v2.1 guard-rail)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import citation_extraction as ce  # noqa: E402
import server  # noqa: E402


MANIFEST = """
documents:
  - doc_id: anzecc-2000-vol1
    title: "ANZECC 2000 Water Quality Guidelines Volume 1"
    filename: anzecc-vol1.pdf
    status: superseded
    superseded_by: anzg-2026-toxicant-dgv-master-table
  - doc_id: anzg-2026-toxicant-dgv-master-table
    title: "ANZG Toxicant default guideline values - master table (2026)"
    filename: toxicants-dgvs-mastertable.pdf
    status: current
"""


def _use_manifest(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(MANIFEST, encoding="utf-8")
    monkeypatch.setattr(ce, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(ce, "_manifest_titles", None)
    monkeypatch.setattr(ce, "_manifest_titles_by_filename", None)
    monkeypatch.setattr(ce, "_manifest_meta", None)
    monkeypatch.setattr(ce, "_manifest_meta_by_filename", None)


def _superseded_chunk() -> dict:
    return {
        "reference_id": "ref-1",
        "file_path": "/kb/anzecc-vol1.pdf",
        "chunk_id": "chunk-aa",
        "content": (
            "[source: anzecc-vol1.pdf | doc: anzecc-2000-vol1 | page 12]\n"
            "Trigger value for zinc in freshwater ecosystems."
        ),
    }


def test_supersession_note_resolves_replacement_title(tmp_path, monkeypatch):
    _use_manifest(tmp_path, monkeypatch)
    note = ce.manifest_supersession_note("anzecc-2000-vol1", "anzecc-vol1.pdf")
    assert note == (
        "superseded by ANZG Toxicant default guideline values - master table (2026)"
    )
    assert (
        ce.manifest_supersession_note("anzg-2026-toxicant-dgv-master-table", None)
        is None
    )
    assert ce.manifest_supersession_note("unknown-doc", "unknown.pdf") is None


def test_superseded_citation_title_carries_warning(tmp_path, monkeypatch):
    _use_manifest(tmp_path, monkeypatch)
    payload = {
        "status": "success",
        "data": {
            "references": [
                {"reference_id": "ref-1", "file_path": "/kb/anzecc-vol1.pdf"}
            ],
            "chunks": [_superseded_chunk()],
        },
    }
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["title"] == (
        "ANZECC 2000 Water Quality Guidelines Volume 1 (superseded by "
        "ANZG Toxicant default guideline values - master table (2026))"
    )


def test_rendered_evidence_tags_superseded_passages(tmp_path, monkeypatch):
    _use_manifest(tmp_path, monkeypatch)
    payload = {
        "status": "success",
        "data": {
            "references": [],
            "chunks": [
                _superseded_chunk(),
                {
                    "reference_id": "ref-2",
                    "file_path": "/kb/toxicants-dgvs-mastertable.pdf",
                    "chunk_id": "chunk-bb",
                    "content": (
                        "[source: toxicants-dgvs-mastertable.pdf | "
                        "doc: anzg-2026-toxicant-dgv-master-table | page 3]\n"
                        "Zinc freshwater DGV at 95% species protection."
                    ),
                },
            ],
        },
    }
    evidence = server._render_kb_evidence(payload)
    assert "[SUPERSEDED — superseded by ANZG Toxicant" in evidence
    assert "historical context only" in evidence
    # The current document renders without any warning tag.
    assert "--- Passage 2 ---" in evidence


def test_unified_prompt_carries_supersession_rule():
    assert "marked SUPERSEDED are historical context" in server.UNIFIED_ANSWER_SYSTEM
