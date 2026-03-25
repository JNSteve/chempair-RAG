"""
Extract tables from PDFs using pdfplumber and insert them into the existing
knowledge graph. This fills in structured table data that pypdfium2 missed.
"""

import asyncio
import os
import logging
import numpy as np
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger

from raganything import RAGAnything, RAGAnythingConfig

load_dotenv()

# ---- Configuration ----
PDF_FOLDER = "./my_pdfs"
RAG_STORAGE = "./rag_storage"
LLM_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DELAY_BETWEEN_FILES = 10

# ---- Rate limit protection ----
_rate_limited = False

# ---- Embedding setup ----
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


async def local_embed(texts: list[str]) -> np.ndarray:
    model = get_embed_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


# ---- LLM setup ----
async def safe_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    global _rate_limited
    if _rate_limited:
        raise Exception("RATE_LIMIT_STOP")
    try:
        result = await openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs,
        )
        return result
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "quota" in err or "429" in err or "insufficient" in err:
            _rate_limited = True
            print("\n!!! RATE LIMIT / QUOTA HIT — STOPPING !!!")
            raise Exception("RATE_LIMIT_STOP")
        raise


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return safe_llm_func(prompt, system_prompt, history_messages, **kwargs)


# ---- Table extraction ----
def format_table(table: list[list], page_num: int, pdf_name: str) -> str | None:
    """Format a raw pdfplumber table into readable structured text."""
    if not table or len(table) < 2:
        return None

    # Clean up None values
    cleaned = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
        # Skip rows that are entirely empty
        if any(c for c in cleaned_row):
            cleaned.append(cleaned_row)

    if len(cleaned) < 2:
        return None

    # Build a markdown-style table
    lines = [f"[Table from {pdf_name}, page {page_num}]"]

    # Use first non-empty row as header
    header = cleaned[0]
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * len(header)))

    for row in cleaned[1:]:
        # Pad or trim row to match header length
        padded = row + [""] * (len(header) - len(row))
        lines.append(" | ".join(padded[:len(header)]))

    return "\n".join(lines)


def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """Extract all tables from a PDF, return as content_list items."""
    content_list = []
    pdf_name = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                formatted = format_table(table, page.page_number, pdf_name)
                if formatted and len(formatted) > 50:  # skip tiny/empty tables
                    content_list.append({
                        "type": "text",
                        "text": formatted,
                        "page_idx": page.page_number - 1,
                    })

    return content_list


# ---- Main ----
async def main():
    global _rate_limited

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set your OPENAI_API_KEY in the .env file")
        return

    config = RAGAnythingConfig(
        working_dir=RAG_STORAGE,
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=False,
    )

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=local_embed,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # Skip parser check since we're not parsing — just inserting
    rag._parser_installation_checked = True

    pdf_folder = Path(PDF_FOLDER)
    pdfs = sorted(pdf_folder.glob("*.pdf"))

    # Check which files already have tables ingested
    import json
    already_done = set()
    status_file = Path(RAG_STORAGE) / "kv_store_doc_status.json"
    if status_file.exists():
        with open(status_file, encoding="utf-8") as f:
            doc_status = json.load(f)
        for doc_id, info in doc_status.items():
            if isinstance(info, dict):
                fp = info.get("file_path", "")
                status = info.get("status", "")
                if fp.startswith("tables_") and status == "processed":
                    # Extract original filename from "tables_XXXXX.pdf"
                    original = fp.replace("tables_", "")
                    already_done.add(original)

    # First pass: extract all tables (free, no API calls)
    print(f"Scanning {len(pdfs)} PDFs for tables...")
    all_tables = {}
    total_tables = 0
    skipped = 0
    for pdf_path in pdfs:
        if pdf_path.name in already_done:
            print(f"  {pdf_path.name}: already ingested, skipping")
            skipped += 1
            continue
        tables = extract_tables_from_pdf(str(pdf_path))
        if tables:
            all_tables[pdf_path.name] = tables
            total_tables += len(tables)
            print(f"  {pdf_path.name}: {len(tables)} tables found")
        else:
            print(f"  {pdf_path.name}: no tables")

    print(f"\nSkipped: {skipped} (already done)")
    print(f"New: {total_tables} tables from {len(all_tables)} files")

    print(f"\nTotal: {total_tables} tables from {len(all_tables)} files")
    print(f"Now inserting into knowledge graph (this costs API tokens)...\n")

    # Second pass: insert into knowledge graph (costs tokens)
    success = 0
    failed = 0
    for i, (pdf_name, tables) in enumerate(all_tables.items(), 1):
        if _rate_limited:
            print(f"\nStopping — rate limited. {len(all_tables) - i + 1} files remaining.")
            break

        print(f"[{i}/{len(all_tables)}] Inserting {len(tables)} tables from {pdf_name}...")
        try:
            await rag.insert_content_list(
                content_list=tables,
                file_path=f"tables_{pdf_name}",
            )
            print(f"  SUCCESS!")
            success += 1
        except Exception as e:
            err_str = str(e)
            if "RATE_LIMIT_STOP" in err_str:
                print(f"  STOPPED — rate limit hit.")
                break
            print(f"  ERROR: {err_str[:200]}")
            failed += 1

        if i < len(all_tables) and not _rate_limited:
            print(f"  (pausing {DELAY_BETWEEN_FILES}s...)")
            await asyncio.sleep(DELAY_BETWEEN_FILES)

    print(f"\n{'='*50}")
    print(f"Table ingestion complete! Success: {success}, Failed: {failed}")
    if _rate_limited:
        print("NOTE: Stopped early due to rate limit.")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
