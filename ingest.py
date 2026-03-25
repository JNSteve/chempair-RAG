"""
Ingest PDFs from my_pdfs/ folder into RAG-Anything using OpenAI
and sentence-transformers for local embeddings (no extra API key needed).

Bypasses MinerU/docling/paddleocr parsers — extracts text directly with pypdfium2
since these PDFs are text-based (not scanned images).
"""

import asyncio
import os
import sys
import logging
import numpy as np
from pathlib import Path

import pypdfium2 as pdfium
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
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, free, local
EMBEDDING_DIM = 384
DELAY_BETWEEN_FILES = 10  # seconds pause between files to avoid rate limits


# ---- PDF text extraction (no heavy parser needed) ----
def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text from a text-based PDF using pypdfium2, returns content_list format."""
    doc = pdfium.PdfDocument(pdf_path)
    content_list = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        if text and text.strip():
            content_list.append({
                "type": "text",
                "text": text.strip(),
                "page_idx": page_idx,
            })
    doc.close()
    return content_list


# ---- Embedding setup (local, no API key needed) ----
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


# ---- LLM setup (OpenAI) with rate limit protection ----
_rate_limited = False


async def safe_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    global _rate_limited
    if _rate_limited:
        raise Exception("RATE_LIMIT_STOP: Already hit rate limit, stopping to save money.")
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
            print("\n!!! RATE LIMIT / QUOTA HIT — STOPPING TO SAVE MONEY !!!")
            print("Add credits or wait, then re-run the script.")
            raise Exception("RATE_LIMIT_STOP")
        raise


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return safe_llm_func(prompt, system_prompt, history_messages, **kwargs)


# ---- Main ingestion ----
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

    pdf_folder = Path(PDF_FOLDER)
    pdfs = sorted(pdf_folder.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {PDF_FOLDER}")
    print(f"Using model: {LLM_MODEL}")
    print(f"Embeddings: {EMBEDDING_MODEL_NAME} (local)")
    print(f"Storage: {RAG_STORAGE}")
    print()

    success = 0
    failed = 0
    for i, pdf_path in enumerate(pdfs, 1):
        if _rate_limited:
            print(f"\nStopping early — rate limited. {len(pdfs) - i + 1} files remaining.")
            break

        print(f"[{i}/{len(pdfs)}] Processing {pdf_path.name}...")
        try:
            content_list = extract_pdf_text(str(pdf_path))
            if not content_list:
                print(f"  WARNING: No text extracted from {pdf_path.name}, skipping")
                failed += 1
                continue

            print(f"  Extracted {len(content_list)} pages of text")
            await rag.insert_content_list(
                content_list=content_list,
                file_path=str(pdf_path),
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

        # Pause between files to avoid rate limits
        if i < len(pdfs) and not _rate_limited:
            print(f"  (pausing {DELAY_BETWEEN_FILES}s before next file...)")
            await asyncio.sleep(DELAY_BETWEEN_FILES)

    print(f"\n{'='*50}")
    print(f"Ingestion complete! Success: {success}, Failed: {failed}")
    if _rate_limited:
        print("NOTE: Stopped early due to rate limit. Re-run after adding credits.")
    print(f"{'='*50}")

    # Quick test query
    if success > 0 and not _rate_limited:
        print("\n--- Test Query ---")
        try:
            result = await rag.aquery(
                "What are the main topics covered in these documents?",
                mode="hybrid",
            )
            print(f"Answer: {result}")
        except Exception as e:
            print(f"Query error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
