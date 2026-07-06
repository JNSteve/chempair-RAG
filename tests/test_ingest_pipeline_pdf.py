"""End-to-end extraction test against a real generated PDF.

Exercises the actual pypdfium2/pdfplumber path in ingest_pipeline (the other
ingest tests use synthetic rows). Skipped automatically when the PDF stack
or fpdf2 (test-only dependency) is unavailable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fpdf = pytest.importorskip("fpdf")
pytest.importorskip("pypdfium2")
pytest.importorskip("pdfplumber")

import ingest_pipeline as pipeline  # noqa: E402


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    pdf = fpdf.FPDF()
    pdf.set_font("Helvetica", size=11)

    pdf.add_page()
    pdf.multi_cell(
        0,
        6,
        "Health investigation levels (HILs) are scientifically based, generic "
        "assessment criteria designed to be used in the first stage of an "
        "assessment of potential risks to human health from chronic exposure "
        "to contaminants.",
    )
    pdf.ln(4)
    pdf.multi_cell(0, 6, "Table 1A(1) Health investigation levels for metals")
    with pdf.table() as table:
        for row_values in (
            ("Analyte", "HIL A (mg/kg)", "HIL B (mg/kg)"),
            ("Arsenic", "100", "500"),
            ("Lead", "300", "1200"),
        ):
            row = table.row()
            for value in row_values:
                row.cell(value)

    pdf.add_page()
    pdf.multi_cell(
        0,
        6,
        "Soil vapour assessment should be considered where volatile "
        "contaminants are present beneath or adjacent to buildings.",
    )

    path = tmp_path_factory.mktemp("pdfs") / "NEPM_Test_Fixture.pdf"
    pdf.output(str(path))
    return str(path)


def test_text_extraction_markers_and_pages(sample_pdf):
    items = pipeline.extract_text_items(
        sample_pdf, "NEPM_Test_Fixture.pdf", "nepm-test"
    )

    assert len(items) == 2
    assert items[0]["page_idx"] == 0
    assert items[0]["text"].startswith(
        "[source: NEPM_Test_Fixture.pdf | doc: nepm-test | page 1]\n"
    )
    assert "Health investigation levels" in items[0]["text"]
    assert items[1]["text"].startswith(
        "[source: NEPM_Test_Fixture.pdf | doc: nepm-test | page 2]\n"
    )
    assert "Soil vapour" in items[1]["text"]


def test_table_extraction_detects_printed_table_number(sample_pdf):
    items = pipeline.extract_table_items(
        sample_pdf, "NEPM_Test_Fixture.pdf", "nepm-test"
    )

    assert len(items) == 1
    text = items[0]["text"]
    assert items[0]["page_idx"] == 0
    # Locator comes from the printed caption near the table, not a positional fallback.
    assert text.startswith(
        "[source: NEPM_Test_Fixture.pdf | doc: nepm-test | page 1 | Table 1A(1)]"
    )
    assert "Arsenic | 100 | 500" in text
    assert "Lead | 300 | 1200" in text


def test_full_document_content_list_round_trip(sample_pdf):
    text_items = pipeline.extract_text_items(
        sample_pdf, "NEPM_Test_Fixture.pdf", "nepm-test"
    )
    table_items = pipeline.extract_table_items(
        sample_pdf, "NEPM_Test_Fixture.pdf", "nepm-test"
    )
    merged = pipeline.build_content_list(text_items, table_items)

    assert len(merged) == 3
    # Page 1 text, page 1 table, page 2 text — and every chunk parseable by
    # the citation extractor.
    from citation_extraction import parse_source_marker

    markers = [parse_source_marker(item["text"]) for item in merged]
    assert all(markers)
    assert [(m["page"], bool(m["table"])) for m in markers] == [
        (1, False),
        (1, True),
        (2, False),
    ]
