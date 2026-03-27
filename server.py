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

# ---- Alfie persona for the RAG answer step (user_prompt) ----
# This goes into LightRAG's "Additional Instructions" slot.
# Keep concise — shares token budget with KB context.
ALFIE_USER_PROMPT = (
    "You are Alfie, a senior Australian environmental scientist. "
    "Respond in Australian English. Never mention RAG, LLM, or AI.\n"
    "Style: concise, practical, technically sound. Short paragraphs, plain headings. "
    "No decorative markdown (no ***), no nested bullets, no chatbot phrasing.\n"
    "Give a direct answer first. Ground statements in retrieved data. "
    "If information is missing, say so briefly. Do not invent values or sample codes.\n"
    "Structure (skip sections if not needed): "
    "Answer (1 short paragraph) → Key findings (1 sentence per issue) → "
    "Implications (1 paragraph if useful) → Next steps (2-4 actions if useful)."
)

# ---- Prompts for the two-step context flow ----
CONTEXT_EXTRACTION_SYSTEM = (
    "You are a workspace-context extraction step for Alfie.\n\n"
    "Your job is to read the current workspace/project state and the user's question, "
    "then produce a compact structured context object containing ONLY the data relevant "
    "to the question. This will be used for downstream retrieval and answer generation.\n\n"
    "Requirements:\n"
    "- Extract only information that is actually present in the current workspace.\n"
    "- Filter to data relevant to the user's question.\n"
    "- Do not infer unsupported facts.\n"
    "- Keep the payload compact and useful for retrieval.\n"
    "- Prioritise analytes, samples, exceedances, criteria, and other current project "
    "signals that matter for environmental interpretation.\n"
    "- Preserve exact sample IDs, analyte names, units, criteria names, and project "
    "metadata where available.\n"
    "- Omit sections that are unavailable or irrelevant to the question.\n"
    "- Do not write a narrative answer.\n"
    "- Do not include markdown.\n"
    "- If the question is a standalone regulatory question that does not need project data "
    "(e.g. 'what is HIL-A for lead?'), respond with exactly: DIRECT_LOOKUP\n\n"
    "Output valid JSON only with these fields (omit if empty):\n"
    "- project: {projectName, siteName, projectType, landUse}\n"
    "- criteria: {applicableCriteria, landUse, state, relevantDetails: [{name, thresholds}]}\n"
    "- exceedances: [{analyte, sampleCode, value, unit, criterion, criterionValue}]\n"
    "- relevantSamples: [{sampleCode, depth, analyteValues: [{analyte, value, unit}]}]\n"
    "- summary: one sentence stating what the user wants to know about this data"
)

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

    # ── Context validation ──────────────────────────────────────
    grounding: str = ""
    context_used = False

    if req.context is not None:
        try:
            context_bytes = len(req.context.model_dump_json().encode("utf-8"))
        except Exception:
            context_bytes = 0

        if context_bytes > MAX_CONTEXT_PAYLOAD_BYTES:
            logger.warning(
                "context_rejected reason=oversize size_bytes=%d limit=%d session=%s",
                context_bytes, MAX_CONTEXT_PAYLOAD_BYTES, session_id,
            )
            raise HTTPException(
                status_code=422,
                detail=f"context payload too large ({context_bytes} bytes, limit {MAX_CONTEXT_PAYLOAD_BYTES})",
            )

        grounding = build_grounding_prompt(req.context)
        if grounding:
            context_used = True
            logger.info(
                "context_accepted schema_version=%s project=%s session=%s",
                req.context.schemaVersion,
                req.context.project.projectName if req.context.project else None,
                session_id,
            )
        else:
            logger.info("context_present_but_empty session=%s", session_id)
    else:
        logger.info("context_absent session=%s", session_id)

    try:
        # Build conversation context from history
        conversation_prefix = ""
        if history:
            context_parts = []
            for msg in history[-6:]:
                context_parts.append(f"{msg['role'].upper()}: {msg['content'][:300]}")
            conversation_prefix = (
                f"Previous conversation:\n" + "\n".join(context_parts) + "\n\n"
            )

        # ── Two-step flow when workspace context is available ─────
        if context_used and grounding:
            # Step 1: Extract question-relevant project data as structured JSON
            extraction_prompt = (
                f"User question: {req.question}\n\n"
                f"## Workspace Data\n{grounding}"
            )
            extraction_result = await openai_complete_if_cache(
                LLM_MODEL,
                extraction_prompt,
                system_prompt=CONTEXT_EXTRACTION_SYSTEM,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            if extraction_result.strip() == "DIRECT_LOOKUP":
                # Pure regulatory question — send straight to RAG
                logger.info("context_gate=direct_lookup session=%s", session_id)
                rag_query = conversation_prefix + req.question
            else:
                # Project-data question — build a focused RAG query from the JSON
                logger.info("context_gate=project_query session=%s", session_id)
                try:
                    # Parse JSON and build a focused query
                    filtered = json.loads(extraction_result)
                    query_parts = [req.question]

                    # Add project context
                    proj = filtered.get("project", {})
                    if proj.get("siteName") or proj.get("projectType"):
                        query_parts.append(
                            f"Site: {proj.get('siteName', 'unknown')}, "
                            f"land use: {proj.get('landUse', proj.get('projectType', 'unknown'))}"
                        )

                    # Add exceedances as specific lookup items
                    for ex in filtered.get("exceedances", []):
                        query_parts.append(
                            f"{ex.get('analyte', '?')} at {ex.get('value', '?')} {ex.get('unit', '')} "
                            f"in {ex.get('sampleCode', '?')} "
                            f"(criterion: {ex.get('criterion', '?')} = {ex.get('criterionValue', '?')})"
                        )

                    # Add relevant sample data
                    for row in filtered.get("relevantSamples", []):
                        vals = ", ".join(
                            f"{av.get('analyte', '?')}={av.get('value', '?')} {av.get('unit', '')}"
                            for av in row.get("analyteValues", [])
                        )
                        if vals:
                            query_parts.append(
                                f"Sample {row.get('sampleCode', '?')} ({row.get('depth', '?')}): {vals}"
                            )

                    # Add summary if present
                    summary = filtered.get("summary", "")
                    if summary:
                        query_parts.append(summary)

                    rag_query = conversation_prefix + "\n".join(query_parts)

                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    # JSON parse failed — fall back to using raw extraction as query context
                    logger.warning("context_json_parse_failed error=%s session=%s", str(e), session_id)
                    rag_query = (
                        f"{conversation_prefix}{req.question}\n\n"
                        f"Site data context:\n{extraction_result[:1000]}"
                    )
        else:
            # No context — straight to RAG
            rag_query = conversation_prefix + req.question if conversation_prefix else req.question

        # Step 2: Query the knowledge base with Alfie persona
        result = await rag.aquery(
            rag_query,
            mode=req.mode,
            user_prompt=ALFIE_USER_PROMPT,
        )

        # Store in session history
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": result})

        # Auto-reset if we've hit the max exchanges
        session_reset = False
        if len(history) >= MAX_EXCHANGES * 2:
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
