"""
Pydantic models for the optional structured workspace context
sent by the Chempair frontend on POST /query.

All fields are optional and defensively validated.
Unknown future-additive fields are allowed (extra="allow").
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Guardrail constants ─────────────────────────────────────────
MAX_EXCEEDANCES = 8
MAX_RETRIEVED_ROWS = 8
MAX_ANALYTE_VALUES_PER_ROW = 6
MAX_CONVERSATION_MESSAGES = 8
MAX_MATCHED_ANALYTES = 12
MAX_MATCHED_SAMPLE_CODES = 12
MAX_QUESTION_TOKENS = 24
MAX_REGULATIONS = 12
MAX_RELEVANT_DETAILS = 6
MAX_THRESHOLDS_PER_DETAIL = 6
MAX_SAMPLE_TYPES = 12
MAX_CONTEXT_PAYLOAD_BYTES = 16 * 1024  # 16 KB


# ── Nested models ───────────────────────────────────────────────

class ProjectInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    projectName: Optional[str] = None
    projectId: Optional[str] = None
    siteName: Optional[str] = None
    labReportNumber: Optional[str] = None
    projectType: Optional[str] = None
    sourceFile: Optional[str] = None
    totalSamples: Optional[int] = None
    totalAnalytes: Optional[int] = None


class RetrievalHints(BaseModel):
    model_config = ConfigDict(extra="allow")

    matchedAnalytes: Optional[List[str]] = Field(
        default=None, max_length=MAX_MATCHED_ANALYTES
    )
    matchedSampleCodes: Optional[List[str]] = Field(
        default=None, max_length=MAX_MATCHED_SAMPLE_CODES
    )
    questionTokens: Optional[List[str]] = Field(
        default=None, max_length=MAX_QUESTION_TOKENS
    )


class CriterionThreshold(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    value: Optional[float | int | str] = None
    unit: Optional[str] = None


class RelevantDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    thresholds: Optional[List[CriterionThreshold]] = Field(
        default=None, max_length=MAX_THRESHOLDS_PER_DETAIL
    )


class CriteriaInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    applicableCriteria: Optional[str] = None
    landUse: Optional[str] = None
    state: Optional[str] = None
    regulations: Optional[List[str]] = Field(
        default=None, max_length=MAX_REGULATIONS
    )
    relevantDetails: Optional[List[RelevantDetail]] = Field(
        default=None, max_length=MAX_RELEVANT_DETAILS
    )


class FieldSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    hasFieldData: Optional[bool] = None
    sessionCount: Optional[int] = None
    boreholeCount: Optional[int] = None
    fieldSampleCount: Optional[int] = None
    lithologyLogCount: Optional[int] = None
    latestSessionDate: Optional[str] = None
    sampleTypes: Optional[List[str]] = Field(
        default=None, max_length=MAX_SAMPLE_TYPES
    )
    depthRange: Optional[str] = None
    hasGpsData: Optional[bool] = None


class Exceedance(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    sampleCode: Optional[str] = None
    criterion: Optional[str] = None
    value: Optional[float | int | str] = None
    criterionValue: Optional[float | int] = None
    exceedanceFactor: Optional[float] = None
    unit: Optional[str] = None


class Coordinates(BaseModel):
    model_config = ConfigDict(extra="allow")

    lat: Optional[float] = None
    lng: Optional[float] = None


class AnalyteValue(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    value: Optional[float | int | str] = None
    unit: Optional[str] = None


class RetrievedRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    sampleCode: Optional[str] = None
    depth: Optional[str] = None
    collectionDate: Optional[str] = None
    sampleType: Optional[str] = None
    labName: Optional[str] = None
    coordinates: Optional[Coordinates] = None
    analyteValues: Optional[List[AnalyteValue]] = Field(
        default=None, max_length=MAX_ANALYTE_VALUES_PER_ROW
    )


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Optional[str] = None
    content: Optional[str] = None


# ── Top-level context model ─────────────────────────────────────

class WorkspaceContext(BaseModel):
    """
    Structured workspace context sent by the Chempair frontend.
    All sections are optional so legacy and partial payloads are accepted.
    """

    model_config = ConfigDict(extra="allow")

    schemaVersion: Optional[int] = None
    generatedAtIso: Optional[str] = None
    project: Optional[ProjectInfo] = None
    retrieval: Optional[RetrievalHints] = None
    criteria: Optional[CriteriaInfo] = None
    fieldSummary: Optional[FieldSummary] = None
    exceedances: Optional[List[Exceedance]] = Field(
        default=None, max_length=MAX_EXCEEDANCES
    )
    retrievedRows: Optional[List[RetrievedRow]] = Field(
        default=None, max_length=MAX_RETRIEVED_ROWS
    )
    conversation: Optional[List[ConversationMessage]] = Field(
        default=None, max_length=MAX_CONVERSATION_MESSAGES
    )


# ── Grounding prompt builder ────────────────────────────────────

def build_grounding_prompt(ctx: WorkspaceContext) -> str:
    """
    Convert structured workspace context into a grounding prompt
    that is passed as system_prompt to the RAG query.

    Returns an empty string when context has no usable data.
    """
    sections: list[str] = []

    # Project overview
    if ctx.project:
        p = ctx.project
        parts = []
        if p.projectName:
            parts.append(f"Project: {p.projectName}")
        if p.siteName:
            parts.append(f"Site: {p.siteName}")
        if p.projectType:
            parts.append(f"Type: {p.projectType}")
        if p.labReportNumber:
            parts.append(f"Lab report: {p.labReportNumber}")
        if p.sourceFile:
            parts.append(f"Source file: {p.sourceFile}")
        if p.totalSamples is not None:
            parts.append(f"Total samples: {p.totalSamples}")
        if p.totalAnalytes is not None:
            parts.append(f"Total analytes: {p.totalAnalytes}")
        if parts:
            sections.append("## Project\n" + "\n".join(parts))

    # Regulatory criteria
    if ctx.criteria:
        c = ctx.criteria
        parts = []
        if c.applicableCriteria:
            parts.append(f"Applicable criteria: {c.applicableCriteria}")
        if c.landUse:
            parts.append(f"Land use: {c.landUse}")
        if c.state:
            parts.append(f"State: {c.state}")
        if c.regulations:
            parts.append(f"Regulations: {', '.join(c.regulations)}")
        if c.relevantDetails:
            for detail in c.relevantDetails[:20]:  # cap display
                if detail.name and detail.thresholds:
                    thresh_strs = []
                    for t in detail.thresholds[:20]:
                        if t.analyte and t.value is not None:
                            thresh_strs.append(
                                f"{t.analyte}: {t.value} {t.unit or ''}"
                            )
                    if thresh_strs:
                        parts.append(
                            f"Criterion {detail.name}: {'; '.join(thresh_strs)}"
                        )
        if parts:
            sections.append("## Regulatory Criteria\n" + "\n".join(parts))

    # Field summary
    if ctx.fieldSummary:
        f = ctx.fieldSummary
        parts = []
        if f.hasFieldData is not None:
            parts.append(f"Has field data: {'Yes' if f.hasFieldData else 'No'}")
        if f.boreholeCount is not None:
            parts.append(f"Boreholes: {f.boreholeCount}")
        if f.fieldSampleCount is not None:
            parts.append(f"Field samples: {f.fieldSampleCount}")
        if f.depthRange:
            parts.append(f"Depth range: {f.depthRange}")
        if f.sampleTypes:
            parts.append(f"Sample types: {', '.join(f.sampleTypes)}")
        if f.latestSessionDate:
            parts.append(f"Latest session: {f.latestSessionDate}")
        if f.hasGpsData is not None:
            parts.append(f"GPS data: {'Yes' if f.hasGpsData else 'No'}")
        if parts:
            sections.append("## Field Summary\n" + "\n".join(parts))

    # Exceedances
    if ctx.exceedances:
        rows = []
        for ex in ctx.exceedances[:50]:  # cap display for prompt size
            if ex.analyte and ex.value is not None:
                row = f"- {ex.analyte}"
                if ex.sampleCode:
                    row += f" @ {ex.sampleCode}"
                row += f": {ex.value}"
                if ex.unit:
                    row += f" {ex.unit}"
                if ex.criterionValue is not None and ex.criterion:
                    row += f" (limit {ex.criterion}={ex.criterionValue}"
                    if ex.exceedanceFactor is not None:
                        row += f", {ex.exceedanceFactor}x"
                    row += ")"
                rows.append(row)
        if rows:
            sections.append("## Exceedances\n" + "\n".join(rows))

    # Retrieved sample rows
    if ctx.retrievedRows:
        rows = []
        for r in ctx.retrievedRows[:30]:  # cap display for prompt size
            if r.sampleCode:
                header = r.sampleCode
                if r.depth:
                    header += f" ({r.depth})"
                if r.collectionDate:
                    header += f" [{r.collectionDate}]"
                vals = []
                if r.analyteValues:
                    for av in r.analyteValues[:20]:
                        if av.analyte and av.value is not None:
                            vals.append(
                                f"{av.analyte}={av.value}{' ' + av.unit if av.unit else ''}"
                            )
                row = f"- {header}"
                if vals:
                    row += ": " + ", ".join(vals)
                rows.append(row)
        if rows:
            sections.append("## Retrieved Sample Data\n" + "\n".join(rows))

    # Retrieval hints (compact)
    if ctx.retrieval:
        parts = []
        if ctx.retrieval.matchedAnalytes:
            parts.append(
                f"Matched analytes: {', '.join(ctx.retrieval.matchedAnalytes[:30])}"
            )
        if ctx.retrieval.matchedSampleCodes:
            parts.append(
                f"Matched samples: {', '.join(ctx.retrieval.matchedSampleCodes[:30])}"
            )
        if parts:
            sections.append("## Retrieval Hints\n" + "\n".join(parts))

    if not sections:
        return ""

    header = (
        "You have access to the following workspace context from the user's "
        "environmental data project. Use it to ground your answer.\n\n"
    )
    return header + "\n\n".join(sections)
