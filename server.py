"""
FastAPI server for querying the RAG knowledge graph.
Your Chempair app calls this API to get answers from the ingested regulatory docs.
Supports conversational sessions so users can refine their questions.
"""

import json
import logging
import os
import uuid
import time
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer

from context_models import (
    WorkspaceContext,
    build_grounding_prompt,
    MAX_CONTEXT_PAYLOAD_BYTES,
)

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig

load_dotenv()

# ---- Configuration ----
RAG_STORAGE = os.environ.get("RAG_STORAGE", "./rag_storage")
LLM_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SESSION_TTL = 3600  # sessions expire after 1 hour of inactivity
MAX_EXCHANGES = 3  # auto-reset session after 3 exchanges to avoid context pollution

logger = logging.getLogger("chempair.query")

# ---- Session storage ----
# Each session stores chat history so users can refine questions
sessions: dict[str, dict] = {}


def get_or_create_session(session_id: str | None) -> tuple[str, list[dict]]:
    """Get existing session or create a new one. Returns (session_id, history)."""
    if session_id and session_id in sessions:
        sessions[session_id]["last_used"] = time.time()
        return session_id, sessions[session_id]["history"]

    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = {"history": [], "last_used": time.time()}
    return new_id, sessions[new_id]["history"]


def cleanup_sessions():
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["last_used"] > SESSION_TTL]
    for sid in expired:
        del sessions[sid]


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
def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs,
    )


# ---- FastAPI app ----
app = FastAPI(title="Chempair RAG API", description="Environmental regulatory document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG on startup
rag = None


@app.on_event("startup")
async def startup():
    global rag
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
    # Skip parser check — we're only querying, not parsing
    rag._parser_installation_checked = True
    # Initialize the LightRAG storage so it loads existing data
    result = await rag._ensure_lightrag_initialized()
    if result.get("success"):
        print("RAG loaded and ready!")
    else:
        print(f"RAG init failed: {result.get('error')}")


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"  # local, global, hybrid, naive, mix
    session_id: str | None = None  # pass to continue a conversation
    context: WorkspaceContext | None = None  # optional structured workspace context


class QueryResponse(BaseModel):
    answer: str
    mode: str
    session_id: str  # return to client so they can continue the conversation
    session_reset: bool = False  # true when session hit max exchanges and was cleared
    context_used: bool = False  # true when structured workspace context was applied


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    cleanup_sessions()
    session_id, history = get_or_create_session(req.session_id)

    # ── Context validation & grounding ──────────────────────────
    system_prompt: str | None = None
    context_used = False

    if req.context is not None:
        # Size guardrail: reject oversized context payloads
        try:
            context_bytes = len(req.context.model_dump_json().encode("utf-8"))
        except Exception:
            context_bytes = 0

        if context_bytes > MAX_CONTEXT_PAYLOAD_BYTES:
            logger.warning(
                "context_rejected reason=oversize size_bytes=%d limit=%d session=%s",
                context_bytes,
                MAX_CONTEXT_PAYLOAD_BYTES,
                session_id,
            )
            raise HTTPException(
                status_code=422,
                detail=f"context payload too large ({context_bytes} bytes, limit {MAX_CONTEXT_PAYLOAD_BYTES})",
            )

        grounding = build_grounding_prompt(req.context)
        if grounding:
            system_prompt = grounding
            context_used = True
            logger.info(
                "context_accepted schema_version=%s project=%s session=%s",
                req.context.schemaVersion,
                req.context.project.projectName if req.context.project else None,
                session_id,
            )
        else:
            logger.info(
                "context_present_but_empty session=%s",
                session_id,
            )
    else:
        logger.info("context_absent session=%s", session_id)

    try:
        # Build a contextual query using chat history
        if history:
            # Summarize recent conversation for context
            context_parts = []
            for msg in history[-6:]:  # last 3 exchanges
                context_parts.append(f"{msg['role'].upper()}: {msg['content'][:300]}")
            conversation_context = "\n".join(context_parts)

            enhanced_query = (
                f"Previous conversation:\n{conversation_context}\n\n"
                f"Follow-up question: {req.question}"
            )
        else:
            enhanced_query = req.question

        result = await rag.aquery(
            enhanced_query,
            mode=req.mode,
            system_prompt=system_prompt,
        )

        # Store in session history
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": result})

        # Auto-reset if we've hit the max exchanges
        session_reset = False
        if len(history) >= MAX_EXCHANGES * 2:  # 2 messages per exchange (user + assistant)
            del sessions[session_id]
            session_reset = True

        return QueryResponse(
            answer=result,
            mode=req.mode,
            session_id=session_id,
            session_reset=session_reset,
            context_used=context_used,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/new")
async def new_session():
    """Start a fresh conversation session."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"history": [], "last_used": time.time()}
    return {"session_id": session_id}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": sessions[session_id]["history"]}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """End a conversation session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": LLM_MODEL,
        "storage": RAG_STORAGE,
        "active_sessions": len(sessions),
    }
