import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import citation_extraction as ce  # noqa: E402


def _payload(chunks: list[dict], references: list[dict]) -> dict:
    return {"status": "success", "data": {"references": references, "chunks": chunks}}


def _marker_chunk(
    reference_id: str = "ref-1",
    filename: str = "NEPM_2013_Schedule_B1.pdf",
    doc_id: str = "nepm-2013-schedule-b1",
    page: int = 14,
    table: str | None = "Table 1A(1)",
    body: str = "Arsenic 100 mg/kg HIL A residential garden settings.",
) -> dict:
    marker = f"[source: {filename} | doc: {doc_id} | page {page}"
    if table:
        marker += f" | {table}"
    return {
        "reference_id": reference_id,
        "file_path": f"/kb/{filename}",
        "chunk_id": "chunk-xyz",
        "content": marker + "]\n" + body,
    }


# --- marker parsing ---


def test_parse_source_marker_with_table():
    marker = ce.parse_source_marker(
        "[source: NEPM_2013.pdf | doc: nepm-2013 | page 14 | Table 1A(1)]\nbody"
    )
    assert marker == {
        "filename": "NEPM_2013.pdf",
        "doc_id": "nepm-2013",
        "page": 14,
        "table": "Table 1A(1)",
    }


def test_parse_source_marker_without_table():
    marker = ce.parse_source_marker(
        "[source: ANZECC_2000.pdf | doc: anzecc-2000 | page 7]\nbody"
    )
    assert marker["table"] is None
    assert marker["page"] == 7


def test_parse_source_marker_rejects_prose_and_legacy_prefixes():
    assert ce.parse_source_marker("Table 1B(7) shows TRH C6-C10 fractions") is None
    assert (
        ce.parse_source_marker("[Table from NEPM_2013.pdf, page 45]\nAnalyte | HSL-A")
        is None
    )
    assert ce.parse_source_marker(None) is None


def test_strip_source_marker_removes_only_the_marker():
    content = "[source: A.pdf | doc: a | page 3]\nActual document text."
    assert ce.strip_source_marker(content) == "Actual document text."
    assert ce.strip_source_marker("No marker here.") == "No marker here."


# --- marker-based citations (Phase 2 chunks) ---


def test_marker_chunk_produces_exact_citation():
    payload = _payload(
        [_marker_chunk()],
        [{"reference_id": "ref-1", "file_path": "/kb/NEPM_2013_Schedule_B1.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["source"] == "NEPM_2013_Schedule_B1.pdf"
    assert citation["title"] == "NEPM 2013 Schedule B1"
    assert citation["locator"] == "Table 1A(1), p. 14"
    assert citation["snippet"] == "Arsenic 100 mg/kg HIL A residential garden settings."
    assert "[source:" not in citation["snippet"]


def test_marker_chunk_without_table_uses_page_locator():
    payload = _payload(
        [_marker_chunk(table=None, page=9)],
        [{"reference_id": "ref-1", "file_path": "/kb/NEPM_2013_Schedule_B1.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["locator"] == "p. 9"


def test_marker_filename_wins_over_reference_path():
    payload = _payload(
        [_marker_chunk(filename="ANZECC_2000.pdf", doc_id="anzecc-2000")],
        [{"reference_id": "ref-1", "file_path": "/kb/stale_path.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["source"] == "ANZECC_2000.pdf"


# --- legacy citations (pre-marker chunks) keep their exact behavior ---


def test_legacy_chunk_regex_locator_preserved():
    payload = _payload(
        [
            {
                "reference_id": "ref-1",
                "file_path": "/kb/NEPM_2013.pdf",
                "chunk_id": "page_123_chunk_1",
                "content": (
                    "Table 1B(7) shows TRH C6-C10 and related hydrocarbon fractions "
                    "should be assessed against the selected land use criteria in the NEPM."
                ),
            }
        ],
        [{"reference_id": "ref-1", "file_path": "/kb/NEPM_2013.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["source"] == "NEPM_2013.pdf"
    assert citation["title"] == "NEPM 2013"
    assert citation["locator"] == "Table 1B(7), p. 123"


def test_legacy_tables_prefix_still_normalised():
    payload = _payload(
        [
            {
                "reference_id": "ref-table",
                "file_path": "/kb/tables_NEPM_2013.pdf",
                "chunk_id": "table_page_45_chunk_2",
                "content": (
                    "[Table from NEPM_2013.pdf, page 45]\nAnalyte | HSL-A\nBenzene | 0.5"
                ),
            }
        ],
        [{"reference_id": "ref-table", "file_path": "/kb/tables_NEPM_2013.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["source"] == "NEPM_2013.pdf"
    assert citation["locator"] == "p. 45"


def test_legacy_chunk_without_signals_falls_back_to_source_passage():
    payload = _payload(
        [
            {
                "reference_id": None,
                "file_path": None,
                "chunk_id": None,
                "content": "General guidance text with no locator signals at all.",
            }
        ],
        [{"reference_id": None, "file_path": None}],
    )
    # reference without reference_id gets no chunk mapping; snippet empty -> skipped
    assert ce.extract_citations_from_rag_payload(payload) == []


def test_hex_chunk_hash_never_becomes_a_page_number():
    # Regression: "p" + digits inside LightRAG's hex chunk hashes used to be
    # read as a page ("chunk-9d4ap3915bc" -> "p. 3915") and composed into a
    # confident-looking wrong locator.
    payload = _payload(
        [
            {
                "reference_id": "ref-1",
                "file_path": "/kb/nsw-2022-sampling-design-part1.pdf",
                "chunk_id": "chunk-9d4ap3915bc",
                "content": (
                    "Table 9 sets out sampling density considerations for "
                    "systematic designs across assessment areas."
                ),
            }
        ],
        [
            {
                "reference_id": "ref-1",
                "file_path": "/kb/nsw-2022-sampling-design-part1.pdf",
            }
        ],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["locator"] == "Table 9"
    assert "3915" not in citation["locator"]


def test_bare_p_digits_in_content_is_not_a_page():
    payload = _payload(
        [
            {
                "reference_id": "ref-1",
                "file_path": "/kb/guide.pdf",
                "chunk_id": "chunk-aa11",
                "content": "Refer to appendix item p3915 for laboratory codes.",
            }
        ],
        [{"reference_id": "ref-1", "file_path": "/kb/guide.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["locator"] == "source passage"


def test_written_page_reference_in_content_still_counts():
    payload = _payload(
        [
            {
                "reference_id": "ref-1",
                "file_path": "/kb/guide.pdf",
                "chunk_id": "chunk-bb22",
                "content": "Minimum sampling densities are given in Table 2 (p. 94).",
            }
        ],
        [{"reference_id": "ref-1", "file_path": "/kb/guide.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["locator"] == "Table 2, p. 94"


def test_table_mentioned_deep_in_prose_is_not_this_passages_locator():
    payload = _payload(
        [
            {
                "reference_id": "ref-1",
                "file_path": "/kb/guide.pdf",
                "chunk_id": "chunk-cc33",
                "content": ("General discussion of site history requirements. " * 12)
                + "See also Table 7 for a related summary.",
            }
        ],
        [{"reference_id": "ref-1", "file_path": "/kb/guide.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["locator"] == "source passage"


# --- shared behavior ---


def test_citation_cap_and_empty_snippet_skip():
    chunks = [
        _marker_chunk(reference_id=f"ref-{i}", page=10 + i, body=f"Body text {i}.")
        for i in range(6)
    ]
    chunks.append(
        {"reference_id": "ref-empty", "file_path": "/kb/x.pdf", "content": "   "}
    )
    references = [
        {"reference_id": f"ref-{i}", "file_path": "/kb/a.pdf"} for i in range(6)
    ]
    references.insert(0, {"reference_id": "ref-empty", "file_path": "/kb/x.pdf"})

    citations = ce.extract_citations_from_rag_payload(_payload(chunks, references))
    assert len(citations) == ce.MAX_CITATIONS


def test_snippet_is_bounded():
    long_body = "word " * 100
    payload = _payload(
        [_marker_chunk(body=long_body)],
        [{"reference_id": "ref-1", "file_path": "/kb/a.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert len(citation["snippet"]) <= ce.MAX_SNIPPET_LENGTH
    assert citation["snippet"].endswith("...")


def test_non_dict_payload_returns_empty():
    assert ce.extract_citations_from_rag_payload(None) == []
    assert ce.extract_citations_from_rag_payload({"data": "nope"}) == []


def test_duplicate_source_locator_citations_are_deduped():
    chunks = [
        _marker_chunk(reference_id="ref-1", body="First excerpt from the table."),
        _marker_chunk(reference_id="ref-2", body="Second excerpt, same table."),
        _marker_chunk(reference_id="ref-3", page=99, body="Different page."),
    ]
    references = [
        {"reference_id": "ref-1", "file_path": "/kb/a.pdf"},
        {"reference_id": "ref-2", "file_path": "/kb/a.pdf"},
        {"reference_id": "ref-3", "file_path": "/kb/a.pdf"},
    ]
    citations = ce.extract_citations_from_rag_payload(_payload(chunks, references))
    assert len(citations) == 2
    assert citations[0]["locator"] == "Table 1A(1), p. 14"
    assert citations[1]["locator"] == "Table 1A(1), p. 99"


def test_marker_citation_title_resolves_from_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
documents:
  - doc_id: nepm-asc-2013-vol02
    title: "NEPM (Assessment of Site Contamination) 2013 compilation - Volume 2 of 22"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(ce, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(ce, "_manifest_titles", None)

    payload = _payload(
        [
            _marker_chunk(
                filename="F2013C00288VOL02.pdf",
                doc_id="nepm-asc-2013-vol02",
                page=31,
                table="Table 1A(1)",
            )
        ],
        [{"reference_id": "ref-1", "file_path": "/kb/F2013C00288VOL02.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["title"] == (
        "NEPM (Assessment of Site Contamination) 2013 compilation - Volume 2 of 22"
    )
    assert citation["source"] == "F2013C00288VOL02.pdf"
    assert citation["locator"] == "Table 1A(1), p. 31"


def test_marker_citation_title_falls_back_without_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "MANIFEST_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(ce, "_manifest_titles", None)

    payload = _payload(
        [_marker_chunk(filename="F2013C00288VOL02.pdf", doc_id="nepm-asc-2013-vol02")],
        [{"reference_id": "ref-1", "file_path": "/kb/F2013C00288VOL02.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["title"] == "F2013C00288VOL02"


def test_markerless_chunk_resolves_title_from_manifest_by_filename(
    tmp_path, monkeypatch
):
    # A page-continuation chunk (no marker) must still show the real document
    # title, resolved from the manifest by filename.
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
documents:
  - doc_id: nepm-asc-2013-vol08
    title: "NEPM (Assessment of Site Contamination) 2013 compilation - Volume 8 of 22"
    filename: F2013C00288VOL08.pdf
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(ce, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(ce, "_manifest_titles", None)
    monkeypatch.setattr(ce, "_manifest_titles_by_filename", None)

    payload = _payload(
        [
            {
                "reference_id": "ref-1",
                "file_path": "/kb/F2013C00288VOL08.pdf",
                "chunk_id": "page_45_chunk_2",
                "content": "Continuation text with no source marker prefix, HIL values for lead.",
            }
        ],
        [{"reference_id": "ref-1", "file_path": "/kb/F2013C00288VOL08.pdf"}],
    )
    (citation,) = ce.extract_citations_from_rag_payload(payload)
    assert citation["source"] == "F2013C00288VOL08.pdf"
    assert citation["title"] == (
        "NEPM (Assessment of Site Contamination) 2013 compilation - Volume 8 of 22"
    )
