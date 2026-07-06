import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ingest_pipeline as pipeline  # noqa: E402


def _doc(**overrides) -> dict:
    doc = {
        "doc_id": "nepm-2013-schedule-b1",
        "status": "current",
        "filename": "NEPM_2013_Schedule_B1.pdf",
        "sha256": "a" * 64,
        "ingested_at": None,
        "ingested_sha256": None,
    }
    doc.update(overrides)
    return doc


# --- source markers ---


def test_source_marker_contains_filename_doc_and_page():
    marker = pipeline.source_marker("NEPM_2013.pdf", "nepm-2013", 14)
    assert marker == "[source: NEPM_2013.pdf | doc: nepm-2013 | page 14]"


def test_source_marker_with_table_locator():
    marker = pipeline.source_marker("NEPM_2013.pdf", "nepm-2013", 14, "Table 1A(1)")
    assert marker.endswith("| page 14 | Table 1A(1)]")


# --- table detection and formatting ---


def test_detect_table_number_from_cells():
    rows = [["Table 1A(1) Health investigation levels", ""], ["Analyte", "HIL A"]]
    assert pipeline.detect_table_number(rows) == "Table 1A(1)"


def test_detect_table_number_from_nearby_text():
    rows = [["Analyte", "HIL A"], ["Arsenic", "100"]]
    assert pipeline.detect_table_number(rows, "See Table 7 below.") == "Table 7"


def test_detect_table_number_absent():
    rows = [["Analyte", "HIL A"], ["Arsenic", "100"]]
    assert pipeline.detect_table_number(rows) is None


def test_format_table_produces_marked_markdown():
    rows = [
        ["Analyte", "HIL A (mg/kg)", "HIL B (mg/kg)"],
        ["Arsenic", "100", "500"],
        [None, "", ""],  # dropped: empty row
        ["Lead", "300", None],  # padded
    ]
    formatted = pipeline.format_table(
        rows, "NEPM_2013.pdf", "nepm-2013", 14, 1, "Table 1A(1) HILs"
    )
    lines = formatted.splitlines()
    assert (
        lines[0] == "[source: NEPM_2013.pdf | doc: nepm-2013 | page 14 | Table 1A(1)]"
    )
    assert lines[1] == "Analyte | HIL A (mg/kg) | HIL B (mg/kg)"
    assert lines[3] == "Arsenic | 100 | 500"
    assert lines[4] == "Lead | 300 | "


def test_format_table_falls_back_to_positional_locator():
    rows = [
        ["Analyte name column", "Result value column"],
        ["Arsenic measured in soil", "100 milligrams per kilogram"],
        ["Lead measured in soil", "300 milligrams per kilogram"],
    ]
    formatted = pipeline.format_table(rows, "Doc.pdf", "doc", 9, 2)
    assert "| Table p9.2]" in formatted.splitlines()[0]


def test_format_table_rejects_tiny_tables():
    assert pipeline.format_table([["only header"]], "Doc.pdf", "doc", 1, 1) is None
    assert pipeline.format_table([], "Doc.pdf", "doc", 1, 1) is None


# --- content merging ---


def test_build_content_list_orders_by_page_with_text_first():
    text_items = [
        {"type": "text", "text": "page2 text", "page_idx": 1},
        {"type": "text", "text": "page1 text", "page_idx": 0},
    ]
    table_items = [
        {"type": "text", "text": "page1 table", "page_idx": 0},
        {"type": "text", "text": "page3 table", "page_idx": 2},
    ]
    merged = pipeline.build_content_list(text_items, table_items)
    assert [item["text"] for item in merged] == [
        "page1 text",
        "page1 table",
        "page2 text",
        "page3 table",
    ]


# --- planning ---


def test_plan_fresh_document_is_ingested():
    (planned,) = pipeline.plan_actions([_doc()])
    assert planned.action == pipeline.INGEST
    assert planned.reason == "never ingested"


def test_plan_up_to_date_document_is_skipped():
    (planned,) = pipeline.plan_actions([_doc(ingested_sha256="a" * 64)])
    assert planned.action == pipeline.SKIP
    assert planned.reason == "up to date"


def test_plan_hash_drift_triggers_replace():
    (planned,) = pipeline.plan_actions([_doc(ingested_sha256="b" * 64)])
    assert planned.action == pipeline.REPLACE
    assert "changed" in planned.reason


def test_plan_superseded_document_is_always_skipped():
    (planned,) = pipeline.plan_actions(
        [_doc(status="superseded", ingested_sha256="b" * 64)], rebuild=True
    )
    assert planned.action == pipeline.SKIP


def test_plan_replace_flag_forces_single_document():
    docs = [
        _doc(ingested_sha256="a" * 64),
        _doc(
            doc_id="anzecc-2000", filename="ANZECC_2000.pdf", ingested_sha256="a" * 64
        ),
    ]
    plan = pipeline.plan_actions(docs, replace_ids=("anzecc-2000",))
    by_id = {planned.doc_id: planned for planned in plan}
    assert by_id["nepm-2013-schedule-b1"].action == pipeline.SKIP
    assert by_id["anzecc-2000"].action == pipeline.REPLACE


def test_plan_replace_of_never_ingested_document_is_plain_ingest():
    (planned,) = pipeline.plan_actions([_doc()], replace_ids=("nepm-2013-schedule-b1",))
    assert planned.action == pipeline.INGEST


def test_plan_rebuild_replaces_previously_ingested_documents():
    (planned,) = pipeline.plan_actions([_doc(ingested_sha256="a" * 64)], rebuild=True)
    assert planned.action == pipeline.REPLACE


def test_plan_unknown_replace_id_raises():
    with pytest.raises(KeyError, match="ghost-doc"):
        pipeline.plan_actions([_doc()], replace_ids=("ghost-doc",))


# --- LightRAG doc-id lookup ---


def test_lightrag_doc_ids_match_filename_and_legacy_tables_prefix(tmp_path):
    status = {
        "doc-1": {
            "file_path": "my_pdfs/NEPM_2013_Schedule_B1.pdf",
            "status": "processed",
        },
        "doc-2": {
            "file_path": "tables_NEPM_2013_Schedule_B1.pdf",
            "status": "processed",
        },
        "doc-3": {
            "file_path": "C:\\corpus\\NEPM_2013_Schedule_B1.pdf",
            "status": "processed",
        },
        "doc-4": {"file_path": "my_pdfs/ANZECC_2000.pdf", "status": "processed"},
        "not-a-dict": "ignored",
    }
    (tmp_path / "kv_store_doc_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )

    matches = pipeline.lightrag_doc_ids_for_file(tmp_path, "NEPM_2013_Schedule_B1.pdf")
    assert sorted(matches) == ["doc-1", "doc-2", "doc-3"]


def test_lightrag_doc_ids_without_state_file(tmp_path):
    assert pipeline.lightrag_doc_ids_for_file(tmp_path, "NEPM_2013.pdf") == []


# --- manifest state updates ---


def test_mark_ingested_records_timestamp_and_hash():
    doc = _doc()
    pipeline.mark_ingested(doc, "2026-07-05T00:00:00+00:00")
    assert doc["ingested_at"] == "2026-07-05T00:00:00+00:00"
    assert doc["ingested_sha256"] == doc["sha256"]
