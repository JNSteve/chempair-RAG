"""
Pydantic models for the structured workspace context sent by the frontend.

The contract separates full live project state from compact retrieval context so
the backend can route questions as project-only, KB-only, or blended.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_CONVERSATION_MESSAGES = 20
MAX_CONTEXT_PAYLOAD_BYTES = 512 * 1024  # 512 KB

PROJECT_STATE_KEYS = {
    "project",
    "selectedCriteria",
    "criteriaDetails",
    "exceedanceSummary",
    "exceedances",
    "projectResults",
    "fieldSummary",
}
RETRIEVAL_CONTEXT_KEYS = {
    "matchedAnalytes",
    "matchedSampleCodes",
    "questionTokens",
    "retrievedRows",
}
PROJECT_INFO_KEYS = {
    "projectName",
    "projectId",
    "siteName",
    "address",
    "labReportNumber",
    "projectType",
    "sourceFile",
    "totalSamples",
    "totalAnalytes",
}


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _merge_if_missing(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        target.setdefault(key, value)
    return target


class ProjectInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    projectName: Optional[str] = None
    projectId: Optional[str] = None
    siteName: Optional[str] = None
    address: Optional[str] = None
    labReportNumber: Optional[str] = None
    projectType: Optional[str] = None
    sourceFile: Optional[str] = None
    totalSamples: Optional[int] = None
    totalAnalytes: Optional[int] = None


class SelectedCriteria(BaseModel):
    model_config = ConfigDict(extra="allow")

    applicableCriteria: Optional[str] = None
    regulations: Optional[List[str]] = None
    landUse: Optional[str] = None
    state: Optional[str] = None
    criteriaNames: Optional[List[str]] = None
    criteriaCount: Optional[int] = None


class CriterionThreshold(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    value: Optional[float | int | str] = None
    unit: Optional[str] = None


class CriteriaDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    thresholds: Optional[List[CriterionThreshold]] = None


class ExceedanceSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    totalExceedances: Optional[int] = None
    affectedSamples: Optional[List[str]] = None
    affectedAnalytes: Optional[List[str]] = None
    exceededCriteria: Optional[List[str]] = None
    hotspotCount: Optional[int] = None


class Exceedance(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    sampleCode: Optional[str] = None
    criterion: Optional[str] = None
    value: Optional[float | int | str] = None
    criterionValue: Optional[float | int | str] = None
    exceedanceFactor: Optional[float] = None
    isHotspot: Optional[bool] = None
    unit: Optional[str] = None
    date: Optional[str] = None


class Coordinates(BaseModel):
    model_config = ConfigDict(extra="allow")

    lat: Optional[float] = None
    lng: Optional[float] = None


class AnalyteValue(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    value: Optional[float | int | str] = None
    unit: Optional[str] = None


class ProjectResultRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    sampleCode: Optional[str] = None
    depth: Optional[str] = None
    collectionDate: Optional[str] = None
    sampleType: Optional[str] = None
    sampleRound: Optional[str] = None
    labName: Optional[str] = None
    labReportNumber: Optional[str] = None
    coordinates: Optional[Coordinates] = None
    analyteValues: Optional[List[AnalyteValue]] = None


class FieldSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    hasFieldData: Optional[bool] = None
    sessionCount: Optional[int] = None
    boreholeCount: Optional[int] = None
    fieldSampleCount: Optional[int] = None
    lithologyLogCount: Optional[int] = None
    latestSessionDate: Optional[str] = None
    sampleTypes: Optional[List[str]] = None
    depthRange: Optional[str] = None
    hasGpsData: Optional[bool] = None


class ProjectState(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: Optional[ProjectInfo] = None
    selectedCriteria: Optional[SelectedCriteria] = None
    criteriaDetails: Optional[List[CriteriaDetail]] = None
    exceedanceSummary: Optional[ExceedanceSummary] = None
    exceedances: Optional[List[Exceedance]] = None
    projectResults: Optional[List[ProjectResultRow]] = None
    fieldSummary: Optional[FieldSummary] = None


class ProjectEvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    summary: Optional[str] = None
    totalExceedances: Optional[int] = None
    affectedSamples: Optional[List[str]] = None
    affectedAnalytes: Optional[List[str]] = None
    exceededCriteria: Optional[List[str]] = None
    contaminantsOfConcern: Optional[List[str]] = None
    topExceedances: Optional[List[Exceedance]] = None


class RetrievalContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    matchedAnalytes: Optional[List[str]] = None
    matchedSampleCodes: Optional[List[str]] = None
    questionTokens: Optional[List[str]] = None
    retrievedRows: Optional[List[ProjectResultRow]] = None


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Optional[str] = None
    content: Optional[str] = None


class WorkspaceContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    schemaVersion: Optional[int] = None
    generatedAtIso: Optional[str] = None
    questionIntent: Optional[str] = None
    requiresProjectContext: Optional[bool] = None
    targetAnalytes: Optional[List[str]] = None
    targetSampleCodes: Optional[List[str]] = None
    preferredAnswerShape: Optional[str] = None
    projectEvidenceSummary: Optional[ProjectEvidenceSummary] = None
    projectState: Optional[ProjectState] = None
    retrievalContext: Optional[RetrievalContext] = None
    conversation: Optional[List[ConversationMessage]] = Field(
        default=None, max_length=MAX_CONVERSATION_MESSAGES
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_legacy_context(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw = dict(data)
        project_state = _to_plain_dict(raw.pop("projectState", None))
        retrieval_context = _to_plain_dict(raw.pop("retrievalContext", None))

        canonical_project = _to_plain_dict(project_state.get("project"))
        flat_project = _to_plain_dict(raw.pop("project", None))
        if flat_project:
            _merge_if_missing(canonical_project, flat_project)

        flat_project_fields = {}
        for key in list(raw.keys()):
            if key in PROJECT_INFO_KEYS:
                flat_project_fields[key] = raw.pop(key)
        if flat_project_fields:
            _merge_if_missing(canonical_project, flat_project_fields)
        if canonical_project:
            project_state["project"] = canonical_project

        for key in PROJECT_STATE_KEYS - {"project"}:
            flat_value = raw.pop(key, None)
            if flat_value is not None and key not in project_state:
                project_state[key] = flat_value

        if "retrieval" in raw:
            retrieval_context = _merge_if_missing(
                retrieval_context, _to_plain_dict(raw.pop("retrieval"))
            )

        for key in RETRIEVAL_CONTEXT_KEYS:
            flat_value = raw.pop(key, None)
            if flat_value is not None and key not in retrieval_context:
                retrieval_context[key] = flat_value

        if "conversationHistory" in raw and "conversation" not in raw:
            raw["conversation"] = raw.pop("conversationHistory")
        if "messages" in raw and "conversation" not in raw:
            raw["conversation"] = raw.pop("messages")

        if project_state:
            raw["projectState"] = project_state
        if retrieval_context:
            raw["retrievalContext"] = retrieval_context

        return raw


def build_grounding_prompt(ctx: WorkspaceContext) -> str:
    """
    Convert structured workspace context into a readable grounding summary.

    This is primarily used for operator visibility and tests. The query route
    now prefers the raw context JSON for extraction so full project state is
    available to the classification step.
    """
    sections: list[str] = []

    project_state = ctx.projectState
    retrieval_context = ctx.retrievalContext

    v4_parts = []
    if ctx.questionIntent:
        v4_parts.append(f"Question intent: {ctx.questionIntent}")
    if ctx.requiresProjectContext is not None:
        v4_parts.append(
            f"Requires project context: {'Yes' if ctx.requiresProjectContext else 'No'}"
        )
    if ctx.targetAnalytes:
        v4_parts.append(f"Target analytes: {', '.join(ctx.targetAnalytes)}")
    if ctx.targetSampleCodes:
        v4_parts.append(f"Target samples: {', '.join(ctx.targetSampleCodes)}")
    if ctx.preferredAnswerShape:
        v4_parts.append(f"Preferred answer shape: {ctx.preferredAnswerShape}")
    if v4_parts:
        sections.append("## Request Context\n" + "\n".join(v4_parts))

    if ctx.projectEvidenceSummary:
        evidence = ctx.projectEvidenceSummary
        parts = []
        if evidence.summary:
            parts.append(evidence.summary)
        if evidence.totalExceedances is not None:
            parts.append(f"Total exceedances: {evidence.totalExceedances}")
        if evidence.affectedAnalytes:
            parts.append(f"Affected analytes: {', '.join(evidence.affectedAnalytes)}")
        if evidence.exceededCriteria:
            parts.append(f"Exceeded criteria: {', '.join(evidence.exceededCriteria)}")
        if evidence.topExceedances:
            rendered = []
            for exceedance in evidence.topExceedances:
                if exceedance.analyte and exceedance.value is not None:
                    row = f"{exceedance.analyte}"
                    if exceedance.sampleCode:
                        row += f" @ {exceedance.sampleCode}"
                    row += f": {exceedance.value}"
                    if exceedance.unit:
                        row += f" {exceedance.unit}"
                    rendered.append(row)
            if rendered:
                parts.append("Top exceedances: " + "; ".join(rendered))
        if parts:
            sections.append("## Project Evidence Summary\n" + "\n".join(parts))

    if project_state and project_state.project:
        p = project_state.project
        parts = []
        if p.projectName:
            parts.append(f"Project: {p.projectName}")
        if p.siteName:
            parts.append(f"Site: {p.siteName}")
        if p.address:
            parts.append(f"Address: {p.address}")
        if p.projectType:
            parts.append(f"Type: {p.projectType}")
        if p.labReportNumber:
            parts.append(f"Lab report: {p.labReportNumber}")
        if p.totalSamples is not None:
            parts.append(f"Total samples: {p.totalSamples}")
        if p.totalAnalytes is not None:
            parts.append(f"Total analytes: {p.totalAnalytes}")
        if parts:
            sections.append("## Project\n" + "\n".join(parts))

    if project_state and project_state.selectedCriteria:
        c = project_state.selectedCriteria
        parts = []
        if c.applicableCriteria:
            parts.append(f"Applicable criteria: {c.applicableCriteria}")
        if c.landUse:
            parts.append(f"Land use: {c.landUse}")
        if c.state:
            parts.append(f"State: {c.state}")
        if c.regulations:
            parts.append(f"Regulations: {', '.join(c.regulations)}")
        if c.criteriaNames:
            parts.append(f"Selected criteria: {', '.join(c.criteriaNames)}")
        if parts:
            sections.append("## Selected Criteria\n" + "\n".join(parts))

    if project_state and project_state.criteriaDetails:
        rows = []
        for detail in project_state.criteriaDetails:
            if not detail.name or not detail.thresholds:
                continue
            threshold_bits = []
            for threshold in detail.thresholds:
                if threshold.analyte and threshold.value is not None:
                    threshold_bits.append(
                        f"{threshold.analyte}={threshold.value}"
                        f"{' ' + threshold.unit if threshold.unit else ''}"
                    )
            if threshold_bits:
                rows.append(f"- {detail.name}: {', '.join(threshold_bits)}")
        if rows:
            sections.append("## Criteria Details\n" + "\n".join(rows))

    if project_state and project_state.exceedanceSummary:
        s = project_state.exceedanceSummary
        parts = []
        if s.totalExceedances is not None:
            parts.append(f"Total exceedances: {s.totalExceedances}")
        if s.exceededCriteria:
            parts.append(f"Exceeded criteria: {', '.join(s.exceededCriteria)}")
        if s.affectedAnalytes:
            parts.append(f"Affected analytes: {', '.join(s.affectedAnalytes)}")
        if s.affectedSamples:
            parts.append(f"Affected samples: {', '.join(s.affectedSamples)}")
        if s.hotspotCount is not None:
            parts.append(f"Hotspots: {s.hotspotCount}")
        if parts:
            sections.append("## Exceedance Summary\n" + "\n".join(parts))

    if project_state and project_state.exceedances:
        rows = []
        for ex in project_state.exceedances:
            if ex.analyte and ex.value is not None:
                row = f"- {ex.analyte}"
                if ex.sampleCode:
                    row += f" @ {ex.sampleCode}"
                row += f": {ex.value}"
                if ex.unit:
                    row += f" {ex.unit}"
                if ex.criterion:
                    row += f" against {ex.criterion}"
                rows.append(row)
        if rows:
            sections.append("## Exceedances\n" + "\n".join(rows))

    if project_state and project_state.projectResults:
        rows = []
        for result in project_state.projectResults[:20]:
            if result.sampleCode:
                header = result.sampleCode
                if result.depth:
                    header += f" ({result.depth})"
                vals = []
                if result.analyteValues:
                    for analyte_value in result.analyteValues[:20]:
                        if analyte_value.analyte and analyte_value.value is not None:
                            vals.append(
                                f"{analyte_value.analyte}={analyte_value.value}"
                                f"{' ' + analyte_value.unit if analyte_value.unit else ''}"
                            )
                row = f"- {header}"
                if vals:
                    row += ": " + ", ".join(vals)
                rows.append(row)
        if rows:
            sections.append("## Project Results\n" + "\n".join(rows))

    if project_state and project_state.fieldSummary:
        f = project_state.fieldSummary
        parts = []
        if f.hasFieldData is not None:
            parts.append(f"Has field data: {'Yes' if f.hasFieldData else 'No'}")
        if f.sessionCount is not None:
            parts.append(f"Field sessions: {f.sessionCount}")
        if f.boreholeCount is not None:
            parts.append(f"Boreholes: {f.boreholeCount}")
        if f.fieldSampleCount is not None:
            parts.append(f"Field samples: {f.fieldSampleCount}")
        if f.lithologyLogCount is not None:
            parts.append(f"Lithology logs: {f.lithologyLogCount}")
        if f.latestSessionDate:
            parts.append(f"Latest session: {f.latestSessionDate}")
        if f.sampleTypes:
            parts.append(f"Sample types: {', '.join(f.sampleTypes)}")
        if f.depthRange:
            parts.append(f"Depth range: {f.depthRange}")
        if f.hasGpsData is not None:
            parts.append(f"GPS data: {'Yes' if f.hasGpsData else 'No'}")
        if parts:
            sections.append("## Field Summary\n" + "\n".join(parts))

    if retrieval_context:
        parts = []
        if retrieval_context.matchedAnalytes:
            parts.append(
                f"Matched analytes: {', '.join(retrieval_context.matchedAnalytes)}"
            )
        if retrieval_context.matchedSampleCodes:
            parts.append(
                f"Matched samples: {', '.join(retrieval_context.matchedSampleCodes)}"
            )
        if retrieval_context.retrievedRows:
            parts.append(f"Retrieved rows: {len(retrieval_context.retrievedRows)}")
        if parts:
            sections.append("## Retrieval Context\n" + "\n".join(parts))

    return "\n\n".join(sections)
