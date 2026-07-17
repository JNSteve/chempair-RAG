"""Citation extraction from LightRAG retrieval payloads.

Chunks ingested by ingest_corpus.py (roadmap Phase 2) begin with a
structured source marker:

    [source: <filename> | doc: <doc_id> | page <n>]
    [source: <filename> | doc: <doc_id> | page <n> | Table <id>]

When a chunk carries a marker, the citation's source and locator are read
from it verbatim — exact, no guessing. Chunks ingested by the retired
two-pass scripts have no marker, so the legacy heuristics (filename from
the reference path with the `tables_` prefix stripped; locator regexed out
of chunk content/ids) are kept as the fallback until the corpus is fully
re-ingested.

Dependency-light on purpose: unit-tested in tests/test_citation_extraction.py
without the RAG stack.
"""

from __future__ import annotations

import re
from pathlib import Path

MAX_CITATIONS = 4
MAX_SNIPPET_LENGTH = 220

# Citation titles resolve doc_id -> the manifest's human title ("NEPM
# (Assessment of Site Contamination) 2013 compilation — Volume 2 of 22")
# instead of the raw filename stem ("F2013C00288VOL02"). Improving a title
# is a one-line manifest edit — no re-ingest needed.
MANIFEST_PATH = Path(__file__).resolve().parent / "corpus" / "manifest.yaml"
# Two indexes into the same manifest: doc_id -> title (from marker chunks)
# and filename -> title (for chunks that lost the marker — LightRAG splits a
# page into several chunks and only the first keeps the prefix, so a citation
# should still show the real document name via its filename).
_manifest_titles: dict[str, str] | None = None
_manifest_titles_by_filename: dict[str, str] | None = None


def _load_manifest_indexes() -> tuple[dict[str, str], dict[str, str]]:
    """(doc_id -> title, filename -> title) from the corpus manifest; empty
    (and harmless) when the manifest is absent, empty, or unreadable."""
    global _manifest_titles, _manifest_titles_by_filename
    if _manifest_titles is None or _manifest_titles_by_filename is None:
        by_doc_id: dict[str, str] = {}
        by_filename: dict[str, str] = {}
        try:
            import yaml

            manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
            for doc in manifest.get("documents") or []:
                if not isinstance(doc, dict) or not doc.get("title"):
                    continue
                title = str(doc["title"])
                if doc.get("doc_id"):
                    by_doc_id[str(doc["doc_id"])] = title
                if doc.get("filename"):
                    by_filename[str(doc["filename"])] = title
        except Exception:
            pass
        _manifest_titles = by_doc_id
        _manifest_titles_by_filename = by_filename
    return _manifest_titles, _manifest_titles_by_filename


def _load_manifest_titles() -> dict[str, str]:
    """doc_id -> title (back-compat accessor)."""
    return _load_manifest_indexes()[0]


def _manifest_title(doc_id: str | None, filename: str | None) -> str | None:
    """Resolve a document's human title from the manifest by doc_id first,
    then by filename. Returns None when neither is registered."""
    by_doc_id, by_filename = _load_manifest_indexes()
    if doc_id and doc_id in by_doc_id:
        return by_doc_id[doc_id]
    if filename and filename in by_filename:
        return by_filename[filename]
    return None


SOURCE_MARKER_PATTERN = re.compile(
    r"^\[source:\s*(?P<filename>[^|\]]+?)\s*"
    r"\|\s*doc:\s*(?P<doc_id>[^|\]]+?)\s*"
    r"\|\s*page\s+(?P<page>\d+)\s*"
    r"(?:\|\s*(?P<table>[^\]]+?)\s*)?\]"
)


def parse_source_marker(content: str | None) -> dict | None:
    """Read the Phase 2 source marker off the start of a chunk, if present."""
    match = SOURCE_MARKER_PATTERN.match((content or "").lstrip())
    if not match:
        return None
    return {
        "filename": match.group("filename"),
        "doc_id": match.group("doc_id"),
        "page": int(match.group("page")),
        "table": match.group("table"),
    }


def strip_source_marker(content: str) -> str:
    """Remove the marker prefix so snippets show document text, not plumbing."""
    return SOURCE_MARKER_PATTERN.sub("", content.lstrip(), count=1).lstrip("\n ")


def _marker_locator(marker: dict) -> str:
    if marker.get("table"):
        return f"{marker['table']}, p. {marker['page']}"
    return f"p. {marker['page']}"


# ---- legacy fallback heuristics (pre-marker chunks) ----


def _file_source_name(file_path: str | None, reference_id: str | None) -> str:
    if file_path:
        cleaned = str(file_path).replace("\\", "/").rstrip("/")
        if cleaned:
            source = cleaned.split("/")[-1]
            return source.removeprefix("tables_")
    if reference_id:
        return f"reference-{reference_id}"
    return "reference"


def _citation_title(source: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", source).replace("_", " ").strip()
    return stem or source


# Table labels announce themselves at the start of a table chunk (or in a
# structural chunk id) — a "Table 9" mentioned deep in prose is a cross-
# reference, not this passage's location.
MAX_TABLE_SCAN_CHARS = 300

# Only explicit page signals count: the word "page" ("page 45", "page_45",
# "table_page_45_chunk_2") or a written "p. 45". A bare "p" followed by
# digits is NOT a page — it matches inside LightRAG's hex chunk hashes
# (chunk-…4ap3915b… -> "p. 3915") and produces confident-looking wrong
# locators. "(?:^|[^a-z0-9])" instead of \b because underscore-delimited
# chunk ids have no word boundaries.
PAGE_SIGNAL_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])page[\s._-]?(\d{1,4})|(?:^|[^a-z0-9])p\.\s*(\d{1,4})",
    re.IGNORECASE,
)


def _extract_table_locator(text: str | None, chunk_id: str | None = None) -> str | None:
    combined = " ".join(
        part for part in ((text or "")[:MAX_TABLE_SCAN_CHARS], chunk_id) if part
    )
    table_match = re.search(
        r"\bTable\s+([A-Za-z]?\d+[A-Za-z]?(?:\([^)]+\))*)",
        combined,
        re.IGNORECASE,
    )
    if not table_match:
        return None
    return f"Table {table_match.group(1)}"


def _extract_page_locator(*parts: str | None) -> str | None:
    combined = " ".join(part for part in parts if part)
    page_match = PAGE_SIGNAL_PATTERN.search(combined)
    if not page_match:
        return None
    return page_match.group(1) or page_match.group(2)


def _citation_locator(
    reference_id: str | None,
    file_path: str | None,
    chunk_id: str | None,
    content: str | None = None,
) -> str:
    table_locator = _extract_table_locator(content, chunk_id)
    page = _extract_page_locator(file_path, chunk_id, content)
    if table_locator and page:
        return f"{table_locator}, p. {page}"
    if table_locator:
        return table_locator
    if page:
        return f"p. {page}"
    # No reliable signal: say so plainly rather than surfacing a raw chunk
    # hash or a guessed number.
    return "source passage"


def _bounded_snippet(text: str | None) -> str:
    snippet = re.sub(r"\s+", " ", strip_source_marker(text or "")).strip()
    if len(snippet) <= MAX_SNIPPET_LENGTH:
        return snippet
    return snippet[: MAX_SNIPPET_LENGTH - 3].rstrip() + "..."


def extract_citations_from_rag_payload(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []

    data = payload.get("data", {})
    references = data.get("references", []) if isinstance(data, dict) else []
    chunks = data.get("chunks", []) if isinstance(data, dict) else []

    chunks_by_reference: dict[str, list[dict]] = {}
    for chunk in chunks:
        reference_id = chunk.get("reference_id")
        if reference_id:
            chunks_by_reference.setdefault(reference_id, []).append(chunk)

    citations: list[dict] = []
    seen_locations: set[tuple[str, str]] = set()
    for reference in references:
        reference_id = reference.get("reference_id")
        file_path = reference.get("file_path")
        ref_chunks = chunks_by_reference.get(reference_id, [])
        primary_chunk = ref_chunks[0] if ref_chunks else {}
        content = primary_chunk.get("content")
        snippet = _bounded_snippet(content)
        if not snippet:
            continue

        marker = parse_source_marker(content)
        if marker:
            source = marker["filename"]
            locator = _marker_locator(marker)
            title = _manifest_title(marker["doc_id"], source) or _citation_title(source)
        else:
            source = _file_source_name(file_path, reference_id)
            locator = _citation_locator(
                reference_id,
                file_path,
                primary_chunk.get("chunk_id"),
                content,
            )
            # Marker-less chunks (page continuations) still resolve their
            # real title from the manifest via filename.
            title = _manifest_title(None, source) or _citation_title(source)

        location = (source, locator)
        if location in seen_locations:
            continue
        seen_locations.add(location)

        citations.append(
            {
                "source": source,
                "title": title,
                "locator": locator,
                "snippet": snippet,
            }
        )
        if len(citations) >= MAX_CITATIONS:
            break

    return citations
