"""Extraction, source-marker, and planning logic for corpus ingestion.

Pure logic lives here so it is unit-testable (tests/test_ingest_pipeline.py)
without the PDF/RAG dependencies; pypdfium2 and pdfplumber are imported
lazily inside the extraction functions. The CLI wrapper is ingest_corpus.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Matches NEPM-style table ids: "Table 1", "Table 1A(1)", "Table 5B".
TABLE_NUMBER_PATTERN = re.compile(
    r"(?i)\btable\s+([A-Za-z]?\d+[A-Za-z]?(?:\([^)]+\))*)"
)

INGEST = "ingest"
REPLACE = "replace"
SKIP = "skip"


def source_marker(
    filename: str, doc_id: str, page_number: int, table_locator: str | None = None
) -> str:
    """Structured chunk prefix. `page N` / `Table X` are also what the
    current server-side locator regex looks for, so markers improve
    citations immediately; Phase 4 parses them directly."""
    marker = f"[source: {filename} | doc: {doc_id} | page {page_number}"
    if table_locator:
        marker += f" | {table_locator}"
    return marker + "]"


def detect_table_number(rows: list[list], nearby_text: str = "") -> str | None:
    """Find a printed table number in the table's own cells or the
    surrounding page text."""
    cell_text = " ".join(str(cell) for row in rows[:3] for cell in row if cell)
    for candidate in (cell_text, nearby_text):
        match = TABLE_NUMBER_PATTERN.search(candidate)
        if match:
            return f"Table {match.group(1)}"
    return None


def format_table(
    rows: list[list],
    filename: str,
    doc_id: str,
    page_number: int,
    table_index: int,
    nearby_text: str = "",
) -> str | None:
    """Render one pdfplumber table as a marker-prefixed markdown table.
    Returns None for tables too small to be meaningful."""
    cleaned = []
    for row in rows or []:
        cells = [str(cell).strip() if cell else "" for cell in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < 2:
        return None

    locator = (
        detect_table_number(cleaned, nearby_text)
        or f"Table p{page_number}.{table_index}"
    )
    header = cleaned[0]
    lines = [
        source_marker(filename, doc_id, page_number, locator),
        " | ".join(header),
        " | ".join(["---"] * len(header)),
    ]
    for row in cleaned[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append(" | ".join(padded[: len(header)]))
    body = "\n".join(lines)
    return body if len(body) > 120 else None


def extract_text_items(pdf_path: str, filename: str, doc_id: str) -> list[dict]:
    """Per-page text items, marker-prefixed. Lazy pypdfium2 import."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        items = []
        for page_idx in range(len(doc)):
            text = doc[page_idx].get_textpage().get_text_bounded()
            if text and text.strip():
                items.append(
                    {
                        "type": "text",
                        "text": source_marker(filename, doc_id, page_idx + 1)
                        + "\n"
                        + text.strip(),
                        "page_idx": page_idx,
                    }
                )
        return items
    finally:
        doc.close()


def extract_table_items(pdf_path: str, filename: str, doc_id: str) -> list[dict]:
    """Per-table items, marker-prefixed with a table locator. Lazy
    pdfplumber import."""
    import pdfplumber

    items = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue
            nearby_text = page.extract_text() or ""
            for table_index, rows in enumerate(tables, 1):
                formatted = format_table(
                    rows,
                    filename,
                    doc_id,
                    page.page_number,
                    table_index,
                    nearby_text,
                )
                if formatted:
                    items.append(
                        {
                            "type": "text",
                            "text": formatted,
                            "page_idx": page.page_number - 1,
                        }
                    )
    return items


def build_content_list(text_items: list[dict], table_items: list[dict]) -> list[dict]:
    """Merge text and table items into one page-ordered content list
    (text before tables on the same page)."""
    keyed = [
        (item["page_idx"], 0, index, item) for index, item in enumerate(text_items)
    ]
    keyed += [
        (item["page_idx"], 1, index, item) for index, item in enumerate(table_items)
    ]
    return [item for *_ignored, item in sorted(keyed, key=lambda entry: entry[:3])]


@dataclass
class PlannedAction:
    doc: dict
    action: str  # ingest | replace | skip
    reason: str

    @property
    def doc_id(self) -> str:
        return self.doc.get("doc_id", "?")


def plan_actions(
    documents: list[dict],
    rebuild: bool = False,
    replace_ids: tuple[str, ...] = (),
) -> list[PlannedAction]:
    """Decide per-document work from manifest state. The manifest's
    `ingested_sha256` is the record of what the KB currently holds."""
    known_ids = {doc.get("doc_id") for doc in documents}
    unknown = set(replace_ids) - known_ids
    if unknown:
        raise KeyError(f"--replace references unknown doc_id(s): {sorted(unknown)}")

    plan = []
    for doc in documents:
        previous = doc.get("ingested_sha256")
        redo_action = REPLACE if previous else INGEST
        if doc.get("status") != "current":
            plan.append(PlannedAction(doc, SKIP, f"status is {doc.get('status')!r}"))
        elif doc.get("doc_id") in replace_ids:
            plan.append(PlannedAction(doc, redo_action, "requested via --replace"))
        elif rebuild:
            plan.append(PlannedAction(doc, redo_action, "requested via --rebuild"))
        elif not previous:
            plan.append(PlannedAction(doc, INGEST, "never ingested"))
        elif previous != doc.get("sha256"):
            plan.append(
                PlannedAction(doc, REPLACE, "content changed since last ingest")
            )
        else:
            plan.append(PlannedAction(doc, SKIP, "up to date"))
    return plan


def lightrag_doc_ids_for_file(rag_storage: str | Path, filename: str) -> list[str]:
    """Find LightRAG internal doc ids whose recorded file path matches this
    corpus filename, so a replacement can delete the old entries first.
    Also matches legacy `tables_<filename>` entries from the retired
    two-pass scripts."""
    status_file = Path(rag_storage) / "kv_store_doc_status.json"
    if not status_file.is_file():
        return []
    with open(status_file, encoding="utf-8") as handle:
        doc_status = json.load(handle)

    matches = []
    for lightrag_id, info in doc_status.items():
        if not isinstance(info, dict):
            continue
        recorded = str(info.get("file_path", "")).replace("\\", "/").split("/")[-1]
        if recorded in (filename, f"tables_{filename}"):
            matches.append(lightrag_id)
    return matches


def mark_ingested(doc: dict, ingested_at_iso: str) -> None:
    """Record a successful ingest on the manifest entry."""
    doc["ingested_at"] = ingested_at_iso
    doc["ingested_sha256"] = doc.get("sha256")
