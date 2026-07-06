"""Corpus manifest CLI: register, validate, and list KB documents.

See corpus/README.md for filing conventions.

    python scripts/corpus_manifest.py add my_pdfs/NEPM_2013_Schedule_B1.pdf \
        --doc-id nepm-2013-schedule-b1 --title "..." --family NEPM \
        --jurisdiction AU --version 2013
    python scripts/corpus_manifest.py validate [--corpus-dir my_pdfs]
    python scripts/corpus_manifest.py list
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "corpus" / "manifest.yaml"

REQUIRED_FIELDS = (
    "doc_id",
    "title",
    "family",
    "jurisdiction",
    "version",
    "status",
    "filename",
    "sha256",
)
VALID_STATUSES = ("current", "superseded")
VALID_FAMILIES = ("NEPM", "ANZECC", "ASC", "NSW", "VIC", "QLD", "OTHER")
DOC_ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or "documents" not in manifest:
        raise SystemExit(f"{path}: not a manifest (missing 'documents')")
    return manifest


def save_manifest(path: Path, manifest: dict) -> None:
    header_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            header_lines.append(line)
        else:
            break
    body = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    path.write_text("\n".join([*header_lines, body]).lstrip("\n"), encoding="utf-8")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_documents(manifest: dict, corpus_dir: Path | None) -> list[str]:
    """Return a list of problems; empty means valid."""
    problems: list[str] = []
    documents = manifest.get("documents") or []
    if not isinstance(documents, list):
        return ["'documents' must be a list"]

    seen_ids: set[str] = set()
    seen_filenames: set[str] = set()
    all_ids = {doc.get("doc_id") for doc in documents if isinstance(doc, dict)}

    for index, doc in enumerate(documents):
        label = doc.get("doc_id") if isinstance(doc, dict) else f"documents[{index}]"
        if not isinstance(doc, dict):
            problems.append(f"{label}: entry is not a mapping")
            continue
        for fieldname in REQUIRED_FIELDS:
            if not doc.get(fieldname):
                problems.append(f"{label}: missing required field {fieldname!r}")

        doc_id = doc.get("doc_id")
        if doc_id:
            if not DOC_ID_PATTERN.match(str(doc_id)):
                problems.append(f"{label}: doc_id must be kebab-case")
            if doc_id in seen_ids:
                problems.append(f"{label}: duplicate doc_id")
            seen_ids.add(doc_id)

        filename = doc.get("filename")
        if filename:
            if filename in seen_filenames:
                problems.append(f"{label}: duplicate filename {filename!r}")
            seen_filenames.add(filename)

        status = doc.get("status")
        if status and status not in VALID_STATUSES:
            problems.append(
                f"{label}: invalid status {status!r} (expected {VALID_STATUSES})"
            )
        if status == "superseded":
            replacement = doc.get("superseded_by")
            if not replacement:
                problems.append(f"{label}: superseded document must set superseded_by")
            elif replacement not in all_ids:
                problems.append(
                    f"{label}: superseded_by {replacement!r} is not a known doc_id"
                )
        elif doc.get("superseded_by"):
            problems.append(
                f"{label}: superseded_by set but status is not 'superseded'"
            )

        family = doc.get("family")
        if family and family not in VALID_FAMILIES:
            problems.append(
                f"{label}: unknown family {family!r} (expected {VALID_FAMILIES})"
            )

        sha256 = doc.get("sha256")
        if sha256 and not SHA256_PATTERN.match(str(sha256)):
            problems.append(f"{label}: sha256 is not a 64-char lowercase hex digest")

        if corpus_dir is not None and filename:
            pdf_path = corpus_dir / filename
            if not pdf_path.is_file():
                problems.append(f"{label}: file not found: {pdf_path}")
            elif (
                sha256
                and SHA256_PATTERN.match(str(sha256))
                and sha256_of(pdf_path) != sha256
            ):
                problems.append(f"{label}: sha256 mismatch for {pdf_path}")

    if corpus_dir is not None:
        unregistered = sorted(
            pdf.name
            for pdf in corpus_dir.glob("*.pdf")
            if pdf.name not in seen_filenames
        )
        for name in unregistered:
            problems.append(f"unregistered PDF in corpus dir: {name}")

    return problems


def _default_doc_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", stem)).strip("-")


def cmd_add(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"error: {pdf_path} not found", file=sys.stderr)
        return 1

    manifest = load_manifest(args.manifest)
    documents = manifest.setdefault("documents", []) or []
    manifest["documents"] = documents

    doc_id = args.doc_id or _default_doc_id(pdf_path.name)
    if any(doc.get("doc_id") == doc_id for doc in documents):
        print(f"error: doc_id {doc_id!r} already registered", file=sys.stderr)
        return 1
    if any(doc.get("filename") == pdf_path.name for doc in documents):
        print(f"error: filename {pdf_path.name!r} already registered", file=sys.stderr)
        return 1

    entry = {
        "doc_id": doc_id,
        "title": args.title or pdf_path.stem.replace("_", " "),
        "family": args.family,
        "jurisdiction": args.jurisdiction,
        "version": str(args.version),
        "status": "current",
        "superseded_by": None,
        "filename": pdf_path.name,
        "sha256": sha256_of(pdf_path),
        "source_url": args.source_url,
        "ingested_at": None,
    }
    documents.append(entry)

    problems = validate_documents(manifest, corpus_dir=None)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    save_manifest(args.manifest, manifest)
    print(f"registered {doc_id} ({pdf_path.name})")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Bulk-register PDFs from a pre-classified metadata file (skips
    already-registered documents and reports missing files)."""
    manifest = load_manifest(args.manifest)
    documents = manifest.setdefault("documents", []) or []
    manifest["documents"] = documents

    with open(args.metadata, encoding="utf-8") as handle:
        entries = (yaml.safe_load(handle) or {}).get("documents") or []
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_dir():
        print(f"error: corpus dir not found: {corpus_dir}", file=sys.stderr)
        return 1

    existing_ids = {doc.get("doc_id") for doc in documents}
    existing_files = {doc.get("filename") for doc in documents}
    added, skipped, missing = 0, 0, 0

    for entry in entries:
        filename = entry["filename"]
        doc_id = entry["doc_id"]
        if doc_id in existing_ids or filename in existing_files:
            skipped += 1
            continue
        pdf_path = corpus_dir / filename
        if not pdf_path.is_file():
            print(f"missing: {filename} (not in {corpus_dir})", file=sys.stderr)
            missing += 1
            continue
        documents.append(
            {
                "doc_id": doc_id,
                "title": entry["title"],
                "family": entry["family"],
                "jurisdiction": entry["jurisdiction"],
                "version": str(entry.get("version", "unknown")),
                "status": "current",
                "superseded_by": None,
                "filename": filename,
                "sha256": sha256_of(pdf_path),
                "source_url": entry.get("source_url"),
                "ingested_at": None,
            }
        )
        existing_ids.add(doc_id)
        existing_files.add(filename)
        added += 1
        print(f"registered {doc_id} ({filename})")

    problems = validate_documents(manifest, corpus_dir=None)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    if added:
        save_manifest(args.manifest, manifest)
    print(
        f"\nseed complete: {added} added, {skipped} already registered, {missing} missing"
    )
    return 0 if missing == 0 else 1


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else None
    if corpus_dir is not None and not corpus_dir.is_dir():
        print(f"error: corpus dir not found: {corpus_dir}", file=sys.stderr)
        return 1
    problems = validate_documents(manifest, corpus_dir)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(f"\n{len(problems)} problem(s) found", file=sys.stderr)
        return 1
    count = len(manifest.get("documents") or [])
    print(f"manifest OK ({count} document(s))")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    documents = manifest.get("documents") or []
    if not documents:
        print("manifest is empty — register documents with `corpus_manifest.py add`")
        return 0
    for doc in documents:
        status = doc.get("status", "?")
        suffix = f" -> {doc.get('superseded_by')}" if status == "superseded" else ""
        print(
            f"{doc.get('doc_id')}: {doc.get('title')} "
            f"[{doc.get('family')}/{doc.get('jurisdiction')} {doc.get('version')}] "
            f"({status}{suffix}) {doc.get('filename')}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="register a PDF in the manifest")
    add.add_argument("pdf", help="path to the PDF file")
    add.add_argument("--doc-id", help="kebab-case id (default: derived from filename)")
    add.add_argument("--title", help="document title (default: derived from filename)")
    add.add_argument("--family", required=True, choices=VALID_FAMILIES)
    add.add_argument("--jurisdiction", required=True, help="AU, NSW, VIC, QLD, ...")
    add.add_argument("--version", required=True, help="publication version/year")
    add.add_argument("--source-url", help="where the document was obtained")
    add.set_defaults(func=cmd_add)

    seed = subparsers.add_parser(
        "seed", help="bulk-register PDFs from a pre-classified metadata file"
    )
    seed.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_MANIFEST.parent / "seed_metadata.yaml",
        help="metadata file (default: corpus/seed_metadata.yaml)",
    )
    seed.add_argument(
        "--corpus-dir", default="my_pdfs", help="folder containing the PDFs"
    )
    seed.set_defaults(func=cmd_seed)

    validate = subparsers.add_parser("validate", help="validate the manifest")
    validate.add_argument(
        "--corpus-dir", help="also verify files and hashes in this folder"
    )
    validate.set_defaults(func=cmd_validate)

    list_cmd = subparsers.add_parser("list", help="print the corpus inventory")
    list_cmd.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
