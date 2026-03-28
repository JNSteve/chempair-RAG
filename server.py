"""
FastAPI server for querying the RAG knowledge graph.
Your Chempair app calls this API to get answers from the ingested regulatory docs.
Supports conversational sessions so users can refine questions.
"""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from context_models import WorkspaceContext, MAX_CONTEXT_PAYLOAD_BYTES
from lightrag.base import QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

load_dotenv()

# ---- Configuration ----
RAG_STORAGE = os.environ.get("RAG_STORAGE", "./rag_storage")
LLM_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SESSION_TTL = 3600
MAX_EXCHANGES = 3

logger = logging.getLogger("chempair.query")

ALFIE_USER_PROMPT = (
    "You are Alfie, a senior Australian environmental scientist. "
    "Respond in Australian English. Never mention RAG, LLM, or AI.\n"
    "Give a direct answer first. Write like a concise consultant email. "
    "Do not use decorative markdown, bold text, or long bullet lists. "
    "Do not invent values, criteria, or sample codes. "
    "When project-applied criteria are supplied, treat them as authoritative over any general reference values. "
    "Keep soil criteria distinct from vapour criteria and other media or pathways. "
    "If notable analyte or sample issues exist, keep each issue to one sentence where practical. "
    "Keep responses to short structured prose."
)

PROJECT_ONLY_ANSWER_SYSTEM = (
    "You are Alfie, a senior Australian environmental scientist.\n\n"
    "Answer using only the supplied project context JSON.\n"
    "Do not mention the knowledge base, RAG, prompts, or internal routing.\n"
    "Be concise, practical, and human. Use Australian professional English.\n"
    "Do not invent values, criteria, or sample codes.\n"
    "Treat the supplied selected criteria, criteria details, and exceedance data as authoritative.\n"
    "For criterion or exceedance-value questions, prefer the project criterionValue first, then the matching threshold under the selected criterion.\n"
    "Do not substitute a threshold from a different medium, pathway, depth band, or land use.\n"
    "If the project context does not contain the answer, say so plainly.\n"
    "Prefer short paragraphs. Avoid decorative markdown and unnecessary bullet points."
)

CONTEXT_EXTRACTION_SYSTEM = (
    "You are a workspace-context extraction step for Alfie.\n\n"
    "Your job is to read the current workspace/project state JSON and the user's question, "
    "then classify the question and extract ONLY the project data relevant to answering it.\n\n"
    "Route the question as one of:\n"
    "- kb_only: the question is purely regulatory or knowledge-base driven and does not need project data.\n"
    "- project_only: the question can be answered directly from the project data without the knowledge base.\n"
    "- blended: the question needs both project data and knowledge-base interpretation.\n\n"
    "Requirements:\n"
    "- If the user is asking for an applied criterion, threshold, guideline value, or exceedance value that is already present in selected criteria, criteria details, exceedances, or retrieved project rows, route as project_only.\n"
    "- If the question asks about source, origin, likely cause, meaning, significance, indication, compliance, risk, health, vapour, remediation, disposal, management, or what something means under a regulation, do not use project_only.\n"
    "- If project analytes, exceedances, selected criteria, or regulations are relevant to an interpretive question, prefer blended.\n"
    "- Extract only information that is actually present in the current workspace.\n"
    "- Filter to data relevant to the user's question.\n"
    "- Do not infer unsupported facts.\n"
    "- Preserve exact sample IDs, analyte names, units, criteria names, and project metadata where available.\n"
    "- Keep soil, soil vapour, groundwater, and other media/pathways distinct. Never merge thresholds across different media/pathways.\n"
    "- Omit sections that are unavailable or irrelevant to the question.\n"
    "- Do not write a narrative answer.\n"
    "- Do not include markdown.\n"
    "Output valid JSON only with these fields (omit if empty):\n"
    "- route: kb_only | project_only | blended\n"
    "- project: {projectName, siteName, address, projectType, labReportNumber}\n"
    "- selectedCriteria: {applicableCriteria, regulations, landUse, state, criteriaNames, criteriaCount}\n"
    "- retrievalContext: {matchedAnalytes, matchedSampleCodes}\n"
    "- criteria: {criteriaDetails: [{name, thresholds}]}\n"
    "- exceedanceSummary: {totalExceedances, affectedSamples, affectedAnalytes, exceededCriteria, hotspotCount}\n"
    "- exceedances: [{analyte, sampleCode, value, unit, criterion, criterionValue}]\n"
    "- relevantSamples: [{sampleCode, depth, analyteValues: [{analyte, value, unit}]}]\n"
    "- summary: one sentence stating what the user wants to know about this data"
)

sessions: dict[str, dict] = {}
CRITERION_LOOKUP_TERMS = (
    "criterion",
    "criteria",
    "guideline",
    "guidelines",
    "threshold",
    "screening",
    "hsl",
    "hil",
)
CRITERION_VALUE_TERMS = (
    "value",
    "limit",
    "exceedance",
    "screening level",
    "criterion value",
    "guideline value",
    "threshold",
)
INTERPRETIVE_ROUTE_PATTERNS = (
    r"\bsource\b",
    r"\borigin\b",
    r"\blikely from\b",
    r"\bcaused by\b",
    r"\bdue to\b",
    r"\bsignificance\b",
    r"\bimplication(?:s)?\b",
    r"\bmeaning\b",
    r"\bmean(?:ing|s)?\b",
    r"\bindicat(?:e|es|ed|ing|ion)\b",
    r"\bcompliance\b",
    r"\baccording to\b",
    r"\bunder\b",
    r"\brisk\b",
    r"\bhealth\b",
    r"\bvapou?r\b",
    r"\bremediation\b",
    r"\bdisposal\b",
    r"\bmanagement\b",
)
PROJECT_FACT_PATTERNS = (
    "main exceedances",
    "what are the exceedances",
    "which sample had the highest",
    "which sample has the highest",
    "highest ",
    "lowest ",
    "selected criteria",
    "selected criterion",
    "criteria have i selected",
    "criterion have i selected",
    "how many exceedances",
    "total exceedances",
)
GENERIC_KB_PATTERNS = (
    "tell me about",
    "what does the nepm say",
    "what do the guidelines say",
    "what are the guidelines",
    "guidelines",
    "guidance",
    "standard",
    "standards",
)
PROJECT_REFERENCE_TERMS = (
    "this project",
    "this site",
    "these exceedances",
    "these results",
    "our project",
    "my project",
    "in this project",
)
PROJECT_INTERPRETIVE_REFERENTS = (
    "what does this mean",
    "what does this indicate",
    "what is this from",
    "what are these from",
    "this contamination",
    "this exceedance",
    "these exceedances",
    "this result",
    "these results",
    "this pattern",
    "these concentrations",
)
GENERIC_REGULATION_TOKENS = {
    "act",
    "commercial",
    "criteria",
    "criterion",
    "epm",
    "guideline",
    "guidelines",
    "health",
    "hil",
    "hsl",
    "industrial",
    "low",
    "medium",
    "nepm",
    "qld",
    "nsw",
    "nt",
    "residential",
    "sa",
    "sand",
    "screening",
    "soil",
    "tas",
    "vic",
    "vapour",
    "wa",
}
MAX_CITATIONS = 4
MAX_SNIPPET_LENGTH = 220


@dataclass
class RouteGuardrails:
    route_hint: str | None = None
    project_only_allowed: bool = True
    reason: str | None = None


def get_or_create_session(session_id: str | None) -> tuple[str, list[dict]]:
    if session_id and session_id in sessions:
        sessions[session_id]["last_used"] = time.time()
        return session_id, sessions[session_id]["history"]

    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = {"history": [], "last_used": time.time()}
    return new_id, sessions[new_id]["history"]


def cleanup_sessions():
    now = time.time()
    expired = [sid for sid, session in sessions.items() if now - session["last_used"] > SESSION_TTL]
    for session_id in expired:
        del sessions[session_id]


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


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs,
    )


app = FastAPI(title="Chempair RAG API", description="Environmental regulatory document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    rag._parser_installation_checked = True
    result = await rag._ensure_lightrag_initialized()
    if result.get("success"):
        print("RAG loaded and ready!")
    else:
        print(f"RAG init failed: {result.get('error')}")


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    session_id: str | None = None
    context: WorkspaceContext | None = None


class Citation(BaseModel):
    source: str
    title: str
    locator: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    mode: str
    session_id: str
    session_reset: bool = False
    context_used: bool = False
    route_used: str
    grounded: bool
    citations: list[Citation]


def _has_usable_context(ctx: WorkspaceContext) -> bool:
    payload = ctx.model_dump(exclude_none=True)
    return any(key not in {"schemaVersion", "generatedAtIso"} for key in payload.keys())


def _build_context_json(ctx: WorkspaceContext) -> str:
    return json.dumps(ctx.model_dump(exclude_none=True), ensure_ascii=False, indent=2)


def _normalise_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _format_scalar(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, "g")

    text = str(value).strip()
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return text

    if number.is_integer():
        return str(int(number))
    return format(number, "g")


def _format_value(value, unit: str | None = None) -> str:
    rendered = _format_scalar(value)
    return f"{rendered} {unit}".strip() if unit else rendered


def _coerce_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return None


def _question_targets_criterion_lookup(question: str) -> bool:
    normalised = _normalise_text(question)
    if normalised.startswith("how ") or normalised.startswith("why "):
        return False
    return any(term in normalised for term in CRITERION_LOOKUP_TERMS) and any(
        term in normalised for term in CRITERION_VALUE_TERMS
    )


def _iter_thresholds(ctx: WorkspaceContext):
    project_state = ctx.projectState
    if not project_state or not project_state.criteriaDetails:
        return

    for detail in project_state.criteriaDetails:
        if not detail.thresholds:
            continue
        for threshold in detail.thresholds:
            yield detail, threshold


def _find_question_analyte(question: str, ctx: WorkspaceContext) -> str | None:
    candidates: list[str] = []
    question_normalised = _normalise_text(question)

    if ctx.retrievalContext and ctx.retrievalContext.matchedAnalytes:
        candidates.extend(
            analyte for analyte in ctx.retrievalContext.matchedAnalytes if analyte
        )

    for _, threshold in _iter_thresholds(ctx):
        if threshold.analyte:
            candidates.append(threshold.analyte)

    project_state = ctx.projectState
    if project_state and project_state.exceedances:
        candidates.extend(ex.analyte for ex in project_state.exceedances if ex.analyte)
    if project_state and project_state.projectResults:
        for row in project_state.projectResults:
            if not row.analyteValues:
                continue
            candidates.extend(
                item.analyte for item in row.analyteValues if item.analyte
            )

    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        key = _normalise_text(candidate)
        if key and key not in seen:
            seen.add(key)
            deduped.append(candidate)

    for candidate in sorted(deduped, key=len, reverse=True):
        if _normalise_text(candidate) in question_normalised:
            return candidate

    return deduped[0] if deduped else None


def _selected_criterion_names(ctx: WorkspaceContext) -> list[str]:
    project_state = ctx.projectState
    if not project_state or not project_state.selectedCriteria:
        return []

    selected = project_state.selectedCriteria
    names: list[str] = []
    if selected.criteriaNames:
        names.extend(name for name in selected.criteriaNames if name)
    if selected.applicableCriteria:
        names.append(selected.applicableCriteria)
    return names


def _collect_context_analytes(ctx: WorkspaceContext) -> list[str]:
    analytes: list[str] = []
    if ctx.retrievalContext and ctx.retrievalContext.matchedAnalytes:
        analytes.extend(analyte for analyte in ctx.retrievalContext.matchedAnalytes if analyte)

    project_state = ctx.projectState
    if project_state and project_state.criteriaDetails:
        for detail in project_state.criteriaDetails:
            if not detail.thresholds:
                continue
            analytes.extend(
                threshold.analyte
                for threshold in detail.thresholds
                if threshold.analyte
            )
    if project_state and project_state.exceedances:
        analytes.extend(ex.analyte for ex in project_state.exceedances if ex.analyte)
    if project_state and project_state.projectResults:
        for row in project_state.projectResults:
            if not row.analyteValues:
                continue
            analytes.extend(
                item.analyte for item in row.analyteValues if item.analyte
            )
    if ctx.retrievalContext and ctx.retrievalContext.retrievedRows:
        for row in ctx.retrievalContext.retrievedRows:
            if not row.analyteValues:
                continue
            analytes.extend(
                item.analyte for item in row.analyteValues if item.analyte
            )

    seen: set[str] = set()
    deduped: list[str] = []
    for analyte in analytes:
        key = _normalise_text(analyte)
        if key and key not in seen:
            seen.add(key)
            deduped.append(analyte)
    return deduped


def _collect_sample_codes(ctx: WorkspaceContext) -> list[str]:
    codes: list[str] = []
    if ctx.retrievalContext and ctx.retrievalContext.matchedSampleCodes:
        codes.extend(code for code in ctx.retrievalContext.matchedSampleCodes if code)

    project_state = ctx.projectState
    if project_state and project_state.exceedances:
        codes.extend(ex.sampleCode for ex in project_state.exceedances if ex.sampleCode)
    if project_state and project_state.projectResults:
        codes.extend(row.sampleCode for row in project_state.projectResults if row.sampleCode)
    if ctx.retrievalContext and ctx.retrievalContext.retrievedRows:
        codes.extend(
            row.sampleCode for row in ctx.retrievalContext.retrievedRows if row.sampleCode
        )

    seen: set[str] = set()
    deduped: list[str] = []
    for code in codes:
        key = _normalise_text(code)
        if key and key not in seen:
            seen.add(key)
            deduped.append(code)
    return deduped


def _question_mentions_any(
    question: str,
    candidates: list[str],
    ignored_tokens: set[str] | None = None,
) -> bool:
    question_key = _normalise_text(question)
    for candidate in candidates:
        candidate_key = _normalise_text(candidate)
        if not candidate_key:
            continue
        if candidate_key in question_key:
            return True

        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", candidate_key)
        filtered_tokens = [
            token
            for token in tokens
            if token not in (ignored_tokens or set())
            and (len(token) >= 3 or any(char.isdigit() for char in token))
        ]
        if any(token in question_key for token in filtered_tokens):
            return True

    return False


def _has_regulatory_context(ctx: WorkspaceContext) -> bool:
    project_state = ctx.projectState
    selected = project_state.selectedCriteria if project_state else None
    return bool(
        selected
        and (
            selected.regulations
            or selected.criteriaNames
            or selected.applicableCriteria
            or selected.state
            or selected.landUse
        )
    )


def _question_mentions_project_context(question: str, ctx: WorkspaceContext) -> bool:
    question_key = _normalise_text(question)
    if any(term in question_key for term in PROJECT_REFERENCE_TERMS):
        return True
    if _question_mentions_any(question, _collect_sample_codes(ctx)):
        return True
    if _question_mentions_any(question, _collect_context_analytes(ctx)):
        return True
    if _question_mentions_any(
        question,
        _selected_criterion_names(ctx),
        ignored_tokens=GENERIC_REGULATION_TOKENS,
    ):
        return True
    return False


def _question_needs_project_grounding(question: str, ctx: WorkspaceContext) -> bool:
    question_key = _normalise_text(question)
    if any(term in question_key for term in PROJECT_INTERPRETIVE_REFERENTS):
        return True
    return _question_mentions_project_context(question, ctx)


def _is_interpretive_question(question: str, ctx: WorkspaceContext) -> bool:
    question_key = _normalise_text(question)
    if any(re.search(pattern, question_key) for pattern in INTERPRETIVE_ROUTE_PATTERNS):
        return True
    if re.search(r"\bwhat (?:is|are).+\bfrom\b", question_key):
        return True
    if "what does this indicate" in question_key or "what does this mean" in question_key:
        return True
    if (
        _question_mentions_project_context(question, ctx)
        and any(term in question_key for term in ("contamination", "exceedance", "criterion", "criteria"))
        and any(term in question_key for term in ("from", "mean", "means", "indicate", "indicates"))
    ):
        return True
    return False


def _is_deterministic_project_fact_question(question: str, ctx: WorkspaceContext) -> bool:
    question_key = _normalise_text(question)
    if _is_interpretive_question(question, ctx):
        return False
    if _question_targets_criterion_lookup(question):
        return True
    if any(pattern in question_key for pattern in PROJECT_FACT_PATTERNS):
        return True
    if _question_mentions_any(question, _collect_sample_codes(ctx)) and any(
        token in question_key for token in ("highest", "lowest", "value", "values", "exceed", "exceedance")
    ):
        return True
    if _question_mentions_any(question, _collect_context_analytes(ctx)) and any(
        token in question_key
        for token in ("highest", "lowest", "main exceedances", "criterion", "threshold", "value in this project")
    ):
        return True
    return False


def _is_generic_kb_question(question: str, ctx: WorkspaceContext) -> bool:
    question_key = _normalise_text(question)
    if _question_mentions_project_context(question, ctx):
        return False
    return any(pattern in question_key for pattern in GENERIC_KB_PATTERNS)


def _deterministic_route_guardrails(question: str, ctx: WorkspaceContext) -> RouteGuardrails:
    question_key = _normalise_text(question)
    needs_project_grounding = _question_needs_project_grounding(question, ctx)

    if _is_deterministic_project_fact_question(question, ctx):
        return RouteGuardrails(
            route_hint="project_only",
            project_only_allowed=True,
            reason="deterministic_project_fact",
        )

    if _is_interpretive_question(question, ctx):
        route_hint = "blended" if needs_project_grounding else "kb_only"
        return RouteGuardrails(
            route_hint=route_hint,
            project_only_allowed=False,
            reason="interpretive_or_causal_question",
        )

    if (
        needs_project_grounding
        and any(term in question_key for term in ("nepm", "guideline", "guidelines", "under", "according to"))
    ):
        return RouteGuardrails(
            route_hint="blended",
            project_only_allowed=False,
            reason="project_regulatory_question",
        )

    if _is_generic_kb_question(question, ctx):
        return RouteGuardrails(
            route_hint="kb_only",
            project_only_allowed=False,
            reason="generic_guidance_question",
        )

    return RouteGuardrails()


def _coerce_route(route: str | None, guardrails: RouteGuardrails, context_used: bool) -> str:
    if not context_used:
        return "kb_only"

    normalised_route = route if route in {"project_only", "kb_only", "blended"} else "blended"

    if guardrails.route_hint:
        return guardrails.route_hint

    if not guardrails.project_only_allowed and normalised_route == "project_only":
        return "blended"

    return normalised_route


def _canonical_filtered_context(ctx: WorkspaceContext) -> dict:
    snapshot: dict[str, object] = {}
    project_state = ctx.projectState
    retrieval_context = ctx.retrievalContext

    if project_state and project_state.project:
        snapshot["project"] = project_state.project.model_dump(exclude_none=True)
    if project_state and project_state.selectedCriteria:
        snapshot["selectedCriteria"] = project_state.selectedCriteria.model_dump(exclude_none=True)
    if retrieval_context:
        snapshot["retrievalContext"] = retrieval_context.model_dump(exclude_none=True)
    if project_state and project_state.criteriaDetails:
        snapshot["criteria"] = {
            "criteriaDetails": [
                detail.model_dump(exclude_none=True) for detail in project_state.criteriaDetails
            ]
        }
    if project_state and project_state.exceedanceSummary:
        snapshot["exceedanceSummary"] = project_state.exceedanceSummary.model_dump(exclude_none=True)
    if project_state and project_state.exceedances:
        snapshot["exceedances"] = [
            exceedance.model_dump(exclude_none=True) for exceedance in project_state.exceedances
        ]

    relevant_rows = []
    if retrieval_context and retrieval_context.retrievedRows:
        relevant_rows = [
            row.model_dump(exclude_none=True) for row in retrieval_context.retrievedRows
        ]
    elif project_state and project_state.projectResults:
        relevant_rows = [
            row.model_dump(exclude_none=True) for row in project_state.projectResults[:10]
        ]
    if relevant_rows:
        snapshot["relevantSamples"] = relevant_rows

    return snapshot


def _merge_filtered_with_context(filtered: dict, ctx: WorkspaceContext) -> dict:
    merged = _canonical_filtered_context(ctx)

    for key, value in filtered.items():
        if value in (None, "", [], {}):
            continue
        if key in {"project", "selectedCriteria", "criteria", "exceedanceSummary", "retrievalContext"}:
            base = merged.get(key, {})
            if isinstance(base, dict) and isinstance(value, dict):
                merged[key] = {**base, **value}
            else:
                merged[key] = value
        else:
            merged[key] = value

    return merged


def _find_matching_threshold(question: str, ctx: WorkspaceContext, analyte: str):
    analyte_key = _normalise_text(analyte)
    question_key = _normalise_text(question)
    selected_names = _selected_criterion_names(ctx)
    selected_keys = {_normalise_text(name) for name in selected_names if name}

    prioritised = []
    fallback = []
    for detail, threshold in _iter_thresholds(ctx):
        if _normalise_text(threshold.analyte) != analyte_key:
            continue

        detail_name = detail.name or ""
        detail_key = _normalise_text(detail_name)
        if detail_key and detail_key in question_key:
            prioritised.insert(0, (detail, threshold))
        elif detail_key and detail_key in selected_keys:
            prioritised.append((detail, threshold))
        else:
            fallback.append((detail, threshold))

    if prioritised:
        return prioritised[0]
    if len(fallback) == 1:
        return fallback[0]
    return None


def _try_answer_direct_criterion_lookup(question: str, ctx: WorkspaceContext) -> str | None:
    if not _question_targets_criterion_lookup(question):
        return None

    project_state = ctx.projectState
    if not project_state or not project_state.criteriaDetails:
        return None

    analyte = _find_question_analyte(question, ctx)
    if not analyte:
        return None
    if _normalise_text(analyte) not in _normalise_text(question):
        return None

    matched = _find_matching_threshold(question, ctx, analyte)
    if not matched:
        return None

    detail, threshold = matched
    criterion_name = detail.name or (_selected_criterion_names(ctx)[:1] or ["the selected criterion"])[0]
    criterion_value_text = _format_value(threshold.value, threshold.unit)
    return f"The applied {analyte} criterion for {criterion_name} is {criterion_value_text}."


def _build_blended_rag_query(question: str, filtered: dict) -> str:
    query_parts = [question]

    project = filtered.get("project", {})
    if project:
        project_bits = []
        if project.get("siteName"):
            project_bits.append(f"Site: {project['siteName']}")
        if project.get("address"):
            project_bits.append(f"Address: {project['address']}")
        if project.get("projectType"):
            project_bits.append(f"Project type: {project['projectType']}")
        if project.get("labReportNumber"):
            project_bits.append(f"Lab report: {project['labReportNumber']}")
        if project_bits:
            query_parts.append("; ".join(project_bits))

    selected_criteria = filtered.get("selectedCriteria", {})
    if selected_criteria:
        criteria_bits = []
        if selected_criteria.get("applicableCriteria"):
            criteria_bits.append(
                f"Applicable criteria: {selected_criteria['applicableCriteria']}"
            )
        if selected_criteria.get("regulations"):
            criteria_bits.append(
                f"Regulations: {', '.join(selected_criteria['regulations'])}"
            )
        if selected_criteria.get("landUse"):
            criteria_bits.append(f"Land use: {selected_criteria['landUse']}")
        if selected_criteria.get("state"):
            criteria_bits.append(f"State: {selected_criteria['state']}")
        if selected_criteria.get("criteriaNames"):
            criteria_bits.append(
                f"Selected criteria: {', '.join(selected_criteria['criteriaNames'])}"
            )
        if criteria_bits:
            query_parts.append("; ".join(criteria_bits))

    retrieval_context = filtered.get("retrievalContext", {})
    if retrieval_context:
        retrieval_bits = []
        if retrieval_context.get("matchedAnalytes"):
            retrieval_bits.append(
                f"Matched analytes: {', '.join(retrieval_context['matchedAnalytes'])}"
            )
        if retrieval_context.get("matchedSampleCodes"):
            retrieval_bits.append(
                f"Matched samples: {', '.join(retrieval_context['matchedSampleCodes'])}"
            )
        if retrieval_bits:
            query_parts.append("; ".join(retrieval_bits))

    exceedance_summary = filtered.get("exceedanceSummary", {})
    if exceedance_summary:
        summary_bits = []
        if exceedance_summary.get("totalExceedances") is not None:
            summary_bits.append(
                f"Total exceedances: {exceedance_summary['totalExceedances']}"
            )
        if exceedance_summary.get("exceededCriteria"):
            summary_bits.append(
                f"Exceeded criteria: {', '.join(exceedance_summary['exceededCriteria'])}"
            )
        if exceedance_summary.get("affectedAnalytes"):
            summary_bits.append(
                f"Affected analytes: {', '.join(exceedance_summary['affectedAnalytes'])}"
            )
        if summary_bits:
            query_parts.append("; ".join(summary_bits))

    criteria = filtered.get("criteria", {})
    for detail in criteria.get("criteriaDetails", []):
        threshold_bits = []
        for threshold in detail.get("thresholds", []):
            analyte = threshold.get("analyte", "?")
            value = threshold.get("value", "?")
            unit = threshold.get("unit", "")
            threshold_bits.append(f"{analyte}={value} {unit}".strip())
        if threshold_bits:
            query_parts.append(
                f"Criterion {detail.get('name', '?')}: {'; '.join(threshold_bits)}"
            )

    for exceedance in filtered.get("exceedances", []):
        query_parts.append(
            f"{exceedance.get('analyte', '?')} at {exceedance.get('value', '?')} "
            f"{exceedance.get('unit', '')} in {exceedance.get('sampleCode', '?')} "
            f"(criterion: {exceedance.get('criterion', '?')} = {exceedance.get('criterionValue', '?')})"
        )

    for row in filtered.get("relevantSamples", []):
        values = ", ".join(
            f"{item.get('analyte', '?')}={item.get('value', '?')} {item.get('unit', '')}".strip()
            for item in row.get("analyteValues", [])
        )
        if values:
            query_parts.append(
                f"Sample {row.get('sampleCode', '?')} ({row.get('depth', '?')}): {values}"
            )

    summary = filtered.get("summary")
    if summary:
        query_parts.append(summary)

    return "\n".join(query_parts)


def _file_source_name(file_path: str | None, reference_id: str | None) -> str:
    if file_path:
        cleaned = str(file_path).replace("\\", "/").rstrip("/")
        if cleaned:
            return cleaned.split("/")[-1]
    if reference_id:
        return f"reference-{reference_id}"
    return "reference"


def _citation_title(source: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", source).replace("_", " ").strip()
    return stem or source


def _citation_locator(reference_id: str | None, file_path: str | None, chunk_id: str | None) -> str:
    combined = " ".join(part for part in (file_path, chunk_id, reference_id) if part)
    page_match = re.search(r"(?:page|p)[\s._-]?(\d{1,4})", combined, re.IGNORECASE)
    if page_match:
        return f"p. {page_match.group(1)}"
    if chunk_id:
        return chunk_id
    if reference_id:
        return f"ref {reference_id}"
    return "source passage"


def _bounded_snippet(text: str | None) -> str:
    snippet = re.sub(r"\s+", " ", (text or "")).strip()
    if len(snippet) <= MAX_SNIPPET_LENGTH:
        return snippet
    return snippet[: MAX_SNIPPET_LENGTH - 3].rstrip() + "..."


def _extract_citations_from_rag_payload(payload: dict | None) -> list[dict]:
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
    for reference in references:
        reference_id = reference.get("reference_id")
        file_path = reference.get("file_path")
        ref_chunks = chunks_by_reference.get(reference_id, [])
        primary_chunk = ref_chunks[0] if ref_chunks else {}
        snippet = _bounded_snippet(primary_chunk.get("content"))
        if not snippet:
            continue

        source = _file_source_name(file_path, reference_id)
        citations.append(
            {
                "source": source,
                "title": _citation_title(source),
                "locator": _citation_locator(
                    reference_id,
                    file_path,
                    primary_chunk.get("chunk_id"),
                ),
                "snippet": snippet,
            }
        )
        if len(citations) >= MAX_CITATIONS:
            break

    return citations


async def _fetch_rag_citations(query: str, mode: str) -> list[dict]:
    lightrag = getattr(rag, "lightrag", None)
    if lightrag is None or not hasattr(lightrag, "aquery_data"):
        return []

    data = await lightrag.aquery_data(
        query,
        param=QueryParam(mode=mode),
    )
    return _extract_citations_from_rag_payload(data)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    cleanup_sessions()
    session_id, history = get_or_create_session(req.session_id)
    context_used = False
    route_used = "kb_only"
    grounded = False
    citations: list[dict] = []

    if req.context is not None:
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

        if _has_usable_context(req.context):
            context_used = True
            logger.info(
                "context_accepted schema_version=%s project=%s session=%s",
                req.context.schemaVersion,
                req.context.projectState.project.projectName
                if req.context.projectState and req.context.projectState.project
                else None,
                session_id,
            )
        else:
            logger.info("context_present_but_empty session=%s", session_id)
    else:
        logger.info("context_absent session=%s", session_id)

    try:
        conversation_prefix = ""
        if history:
            context_parts = []
            for message in history[-6:]:
                context_parts.append(f"{message['role'].upper()}: {message['content'][:300]}")
            conversation_prefix = "Previous conversation:\n" + "\n".join(context_parts) + "\n\n"

        result = None
        rag_query = conversation_prefix + req.question if conversation_prefix else req.question

        if context_used:
            direct_context_answer = _try_answer_direct_criterion_lookup(
                req.question, req.context
            )
            if direct_context_answer:
                result = direct_context_answer
                route_used = "project_only"

        if context_used and result is None:
            guardrails = _deterministic_route_guardrails(req.question, req.context)
            filtered = _canonical_filtered_context(req.context)

            if guardrails.route_hint == "kb_only":
                route_used = "kb_only"
            else:
                extraction_notes = []
                if guardrails.route_hint:
                    extraction_notes.append(
                        f"Deterministic route guardrail: prefer {guardrails.route_hint}."
                    )
                if not guardrails.project_only_allowed:
                    extraction_notes.append(
                        "Deterministic route guardrail: project_only is not allowed."
                    )

                extraction_prompt = (
                    f"User question: {req.question}\n\n"
                    f"{' '.join(extraction_notes)}\n\n"
                    f"Workspace context JSON:\n{_build_context_json(req.context)}"
                )
                extraction_result = await openai_complete_if_cache(
                    LLM_MODEL,
                    extraction_prompt,
                    system_prompt=CONTEXT_EXTRACTION_SYSTEM,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )

                try:
                    extracted = json.loads(extraction_result)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("context_json_parse_failed session=%s", session_id)
                    extracted = {"summary": str(extraction_result)[:1000]}

                filtered = _merge_filtered_with_context(extracted, req.context)
                route_used = _coerce_route(
                    extracted.get("route"),
                    guardrails,
                    context_used=True,
                )

            logger.info(
                "context_route=%s reason=%s session=%s",
                route_used,
                guardrails.reason if 'guardrails' in locals() else None,
                session_id,
            )

            if route_used == "project_only":
                answer_prompt = (
                    f"User question: {req.question}\n\n"
                    f"Relevant project context JSON:\n"
                    f"{json.dumps(filtered, ensure_ascii=False, indent=2)}"
                )
                result = await openai_complete_if_cache(
                    LLM_MODEL,
                    answer_prompt,
                    system_prompt=PROJECT_ONLY_ANSWER_SYSTEM,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            elif route_used == "kb_only":
                rag_query = conversation_prefix + req.question
            else:
                rag_query = conversation_prefix + _build_blended_rag_query(
                    req.question, filtered
                )
        if not context_used:
            route_used = "kb_only"

        if result is None:
            result = await rag.aquery(
                rag_query,
                mode=req.mode,
                user_prompt=ALFIE_USER_PROMPT,
            )
            try:
                citations = await _fetch_rag_citations(rag_query, req.mode)
            except Exception as citation_error:
                logger.warning(
                    "citation_collection_failed session=%s error=%s",
                    session_id,
                    citation_error,
                )
                citations = []
            grounded = bool(citations)

        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": result})

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
            route_used=route_used,
            grounded=grounded,
            citations=citations,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/session/new")
async def new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"history": [], "last_used": time.time()}
    return {"session_id": session_id}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": sessions[session_id]["history"]}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
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
