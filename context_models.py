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


class MapContext(BaseModel):
    """Summary of the active/saved site-map view (schema v5, optional).

    Numbers only — the frontend sends app-computed figures (contour areas,
    zone counts), never polygon coordinate dumps or imagery. Alfie reports
    these verbatim; it must not derive spatial figures itself.
    """

    model_config = ConfigDict(extra="allow")

    mapViewName: Optional[str] = None
    selectedAnalyte: Optional[str] = None
    selectedCriteriaName: Optional[str] = None
    criteriaValue: Optional[float | int] = None
    criteriaUnit: Optional[str] = None
    depthFilter: Optional[str] = None
    concentrationPointCount: Optional[int] = None
    contourAreaM2: Optional[float] = None
    exceedanceZoneCount: Optional[int] = None
    criticalZoneCount: Optional[int] = None
    hotspotCount: Optional[int] = None
    hotspotDiameterM: Optional[float] = None
    # Voronoi contaminated-volume estimate (map engine, PRD_038/039)
    volumeM3: Optional[float] = None
    massTonnes: Optional[float] = None
    contaminatedAreaM2: Optional[float] = None
    averageDepthM: Optional[float] = None
    volumeConfidence: Optional[str] = None
    volumeDepthAssumed: Optional[bool] = None
    exceedingLocations: Optional[int] = None
    totalLocations: Optional[int] = None
    capturedAtIso: Optional[str] = None


class SaqpPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    depthFromM: Optional[float] = None
    depthToM: Optional[float] = None
    priority: Optional[str] = None
    matrix: Optional[str] = None
    executionStatus: Optional[str] = None
    notes: Optional[str] = None


class SaqpContext(BaseModel):
    """Summary of the project's sampling plan (SAQP) and its sufficiency
    advisory (schema v5, optional). All figures are computed by the app's
    sufficiency engine — Alfie reports them verbatim."""

    model_config = ConfigDict(extra="allow")

    points: Optional[List[SaqpPoint]] = None
    pointsTruncated: Optional[bool] = None
    planStatus: Optional[str] = None
    sufficiencyStatus: Optional[str] = None
    computedStatus: Optional[str] = None
    plannedPoints: Optional[int] = None
    requiredPoints: Optional[int] = None
    areaHa: Optional[float] = None
    gridEnabled: Optional[bool] = None
    gridSizeM: Optional[float] = None
    rulesetKey: Optional[str] = None
    rulesetVersion: Optional[str] = None
    advisoryMessage: Optional[str] = None
    overrideActive: Optional[bool] = None
    overrideJustification: Optional[str] = None
    completedPoints: Optional[int] = None
    skippedPoints: Optional[int] = None
    relocatedPoints: Optional[int] = None


class ProposalEntityRef(BaseModel):
    """A stable app id plus a short human label, so the proposal model can
    pick real targets. Ids are opaque; labels are display-only."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    label: Optional[str] = None


class ProposalLinkageRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    label: Optional[str] = None
    # "generated" | "consultant" — consultant-authored linkages are rejected
    # at apply time in the app, so the prompt steers away from them.
    origin: Optional[str] = None


class ProposalSaqpContext(BaseModel):
    """SAQP artifact identity + editable targets for edit proposals
    (contract: docs/ops/proposal-context-contract.md)."""

    model_config = ConfigDict(extra="allow")

    planId: Optional[str] = None
    planUpdatedAt: Optional[str] = None
    points: Optional[List[ProposalEntityRef]] = None
    samples: Optional[List[ProposalEntityRef]] = None


class ProposalCsmContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    csmId: Optional[str] = None
    csmUpdatedAt: Optional[str] = None
    sources: Optional[List[ProposalEntityRef]] = None
    pathways: Optional[List[ProposalEntityRef]] = None
    receptors: Optional[List[ProposalEntityRef]] = None
    linkages: Optional[List[ProposalLinkageRef]] = None
    media: Optional[List[str]] = None


class ProposalContext(BaseModel):
    """Ids + artifact versions the frontend sends so /query can emit
    apply-able edit proposals. Absent on old clients — proposals are then
    never generated."""

    model_config = ConfigDict(extra="allow")

    saqp: Optional[ProposalSaqpContext] = None
    csm: Optional[ProposalCsmContext] = None


class BoreholeLithologyInterval(BaseModel):
    model_config = ConfigDict(extra="allow")

    depthFromM: Optional[float] = None
    depthToM: Optional[float] = None
    soilType: Optional[str] = None
    colour: Optional[str] = None
    moisture: Optional[str] = None
    uscsCode: Optional[str] = None
    observations: Optional[str] = None


class BoreholeFieldSample(BaseModel):
    model_config = ConfigDict(extra="allow")

    sampleId: Optional[str] = None
    depthFromM: Optional[float] = None
    depthToM: Optional[float] = None
    pidReading: Optional[float] = None
    pidUnit: Optional[str] = None
    odour: Optional[str] = None
    observations: Optional[str] = None


class BoreholeLog(BaseModel):
    model_config = ConfigDict(extra="allow")

    boreholeId: Optional[str] = None
    totalDepthM: Optional[float] = None
    groundwaterDepthM: Optional[float] = None
    waterFirstStrikeM: Optional[float] = None
    waterStandingM: Optional[float] = None
    groundwaterNotes: Optional[str] = None
    drillingMethod: Optional[str] = None
    lithology: Optional[List[BoreholeLithologyInterval]] = None
    samples: Optional[List[BoreholeFieldSample]] = None


class FieldContext(BaseModel):
    """Field-collected evidence (schema v5, optional; PRD_101 Phase C):
    borehole logs with lithology intervals, groundwater, and field samples
    including PID readings. Captured from the app's field tables — Alfie
    reports the values verbatim and never invents intervals or readings."""

    model_config = ConfigDict(extra="allow")

    sessionCount: Optional[int] = None
    latestSessionDate: Optional[str] = None
    boreholeCount: Optional[int] = None
    fieldSampleCount: Optional[int] = None
    boreholes: Optional[List[BoreholeLog]] = None
    truncated: Optional[bool] = None


class CriteriaCoverage(BaseModel):
    model_config = ConfigDict(extra="allow")

    scopeLabel: Optional[str] = None
    coveredCount: Optional[int] = None
    notCoveredCount: Optional[int] = None
    excludedCount: Optional[int] = None
    matchRatePct: Optional[float] = None
    notCoveredAnalytes: Optional[List[str]] = None
    excludedAnalytes: Optional[List[str]] = None


class DetectionStat(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    detectionCount: Optional[int] = None
    totalSamples: Optional[int] = None
    detectionRatePct: Optional[float] = None


class QaQcSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    duplicatePairCount: Optional[int] = None
    rpdFailures: Optional[int] = None
    blankDetections: Optional[int] = None
    blankSampleCount: Optional[int] = None
    unpairedDuplicates: Optional[int] = None
    findings: Optional[List[str]] = None


class SamplingProgram(BaseModel):
    model_config = ConfigDict(extra="allow")

    rounds: Optional[List[str]] = None
    sampleTypes: Optional[List[str]] = None
    earliestDate: Optional[str] = None
    latestDate: Optional[str] = None


class CsmLinkage(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: Optional[str] = None
    pathway: Optional[str] = None
    receptor: Optional[str] = None
    status: Optional[str] = None


class CsmContext(BaseModel):
    """Trimmed summary of the consultant's draft conceptual site model
    (schema v6, optional). The consultant's own working model — site
    context, not regulatory evidence."""

    model_config = ConfigDict(extra="allow")

    summary: Optional[str] = None
    landUse: Optional[str] = None
    siteHistory: Optional[str] = None
    contaminantsOfConcern: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    pathways: Optional[List[str]] = None
    receptors: Optional[List[str]] = None
    linkages: Optional[List[CsmLinkage]] = None
    dataGaps: Optional[List[str]] = None


class UclEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    sampleCount: Optional[int] = None
    mean: Optional[float] = None
    ucl95: Optional[float] = None
    method: Optional[str] = None
    distribution: Optional[str] = None
    unit: Optional[str] = None
    criterionValue: Optional[float] = None
    exceedsCriteria: Optional[bool] = None
    maxDetected: Optional[float] = None
    isReliable: Optional[bool] = None
    reliabilityWarning: Optional[str] = None


class AnalyteColumnResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    sampleCode: Optional[str] = None
    depth: Optional[str] = None
    value: Optional[float | int | str] = None


class AnalyteColumn(BaseModel):
    """Every reported result for one analyte across the project. Unlike
    projectResults (a relevance-selected subset), this listing is complete
    for its analyte and may be enumerated as such."""

    model_config = ConfigDict(extra="allow")

    analyte: Optional[str] = None
    unit: Optional[str] = None
    criterionValue: Optional[float] = None
    results: Optional[List[AnalyteColumnResult]] = None
    totalResults: Optional[int] = None
    truncated: Optional[bool] = None


class ProjectState(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: Optional[ProjectInfo] = None
    selectedCriteria: Optional[SelectedCriteria] = None
    criteriaDetails: Optional[List[CriteriaDetail]] = None
    exceedanceSummary: Optional[ExceedanceSummary] = None
    exceedances: Optional[List[Exceedance]] = None
    projectResults: Optional[List[ProjectResultRow]] = None
    fieldSummary: Optional[FieldSummary] = None
    uclSummary: Optional[List[UclEntry]] = None
    criteriaCoverage: Optional[CriteriaCoverage] = None
    detectionStats: Optional[List[DetectionStat]] = None
    qaqc: Optional[QaQcSummary] = None
    samplingProgram: Optional[SamplingProgram] = None
    analyteColumns: Optional[List[AnalyteColumn]] = None


class ProjectEvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: Optional[ProjectInfo] = None
    selectedCriteria: Optional[SelectedCriteria] = None
    exceedanceSummary: Optional[ExceedanceSummary] = None
    summary: Optional[str] = None
    totalExceedances: Optional[int] = None
    affectedSamples: Optional[List[str]] = None
    affectedAnalytes: Optional[List[str]] = None
    exceededCriteria: Optional[List[str]] = None
    contaminantsOfConcern: Optional[List[str]] = None
    topExceedances: Optional[List[Exceedance]] = None
    topExceedancesByMagnitude: Optional[List[Exceedance]] = None
    matchedAnalytes: Optional[List[str]] = None
    matchedSampleLocations: Optional[List[str]] = None
    relevantResultRows: Optional[List[ProjectResultRow]] = None


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
    mapContext: Optional[MapContext] = None
    saqpContext: Optional[SaqpContext] = None
    fieldContext: Optional[FieldContext] = None
    csmContext: Optional[CsmContext] = None
    proposalContext: Optional[ProposalContext] = None
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
        total_exceedances = evidence.totalExceedances
        affected_analytes = evidence.affectedAnalytes
        exceeded_criteria = evidence.exceededCriteria
        if evidence.exceedanceSummary:
            total_exceedances = (
                total_exceedances
                if total_exceedances is not None
                else evidence.exceedanceSummary.totalExceedances
            )
            affected_analytes = (
                affected_analytes or evidence.exceedanceSummary.affectedAnalytes
            )
            exceeded_criteria = (
                exceeded_criteria or evidence.exceedanceSummary.exceededCriteria
            )
        if total_exceedances is not None:
            parts.append(f"Total exceedances: {total_exceedances}")
        if affected_analytes:
            parts.append(f"Affected analytes: {', '.join(affected_analytes)}")
        if exceeded_criteria:
            parts.append(f"Exceeded criteria: {', '.join(exceeded_criteria)}")
        top_exceedances = evidence.topExceedances or evidence.topExceedancesByMagnitude
        if top_exceedances:
            rendered = []
            for exceedance in top_exceedances:
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
            total = (
                project_state.exceedanceSummary.totalExceedances
                if project_state.exceedanceSummary
                else None
            )
            if total and total > len(rows):
                rows.append(
                    f"(showing the {len(rows)} largest of {total} exceedances "
                    "— not the full list)"
                )
            sections.append("## Exceedances\n" + "\n".join(rows))

    if project_state and project_state.uclSummary:
        rows = []
        for entry in project_state.uclSummary:
            if not entry.analyte or entry.ucl95 is None:
                continue
            unit = f" {entry.unit}" if entry.unit else ""
            line = f"- {entry.analyte}: 95% UCL = {entry.ucl95:g}{unit}"
            details = []
            if entry.method:
                details.append(entry.method)
            if entry.sampleCount is not None:
                details.append(f"n={entry.sampleCount}")
            if entry.mean is not None:
                details.append(f"mean {entry.mean:g}{unit}")
            if entry.maxDetected is not None:
                details.append(f"max {entry.maxDetected:g}{unit}")
            if details:
                line += f" ({', '.join(details)})"
            if entry.criterionValue is not None:
                verdict = (
                    "EXCEEDS"
                    if entry.exceedsCriteria
                    else "below"
                    if entry.exceedsCriteria is not None
                    else "vs"
                )
                line += f" — {verdict} criterion {entry.criterionValue:g}{unit}"
            if entry.isReliable is False and entry.reliabilityWarning:
                line += f" [caution: {entry.reliabilityWarning}]"
            rows.append(line)
        if rows:
            sections.append("## UCL95 Statistics\n" + "\n".join(rows))

    for column in (project_state.analyteColumns or []) if project_state else []:
        if not column.analyte or not column.results:
            continue
        unit = f" {column.unit}" if column.unit else ""
        rows = []
        for result in column.results:
            if result.sampleCode is None or result.value is None:
                continue
            label = result.sampleCode
            if result.depth:
                label += f" ({result.depth})"
            rows.append(f"- {label}: {result.value}{unit}")
        if not rows:
            continue
        total = column.totalResults if column.totalResults is not None else len(rows)
        header = f"## Complete {column.analyte} Results ({total} samples with results)"
        if column.criterionValue is not None:
            rows.insert(0, f"Criterion: {column.criterionValue:g}{unit}")
        if column.truncated:
            rows.append(
                f"(showing the first {len([r for r in rows if r.startswith('- ')])} "
                f"of {total} results — list truncated)"
            )
        else:
            rows.append("(complete listing — every reported result for this analyte)")
        sections.append(header + "\n" + "\n".join(rows))

    if project_state and project_state.criteriaCoverage:
        c = project_state.criteriaCoverage
        parts = []
        if c.scopeLabel:
            parts.append(f"Scope: {c.scopeLabel}")
        counts = []
        if c.coveredCount is not None:
            counts.append(f"{c.coveredCount} covered")
        if c.notCoveredCount is not None:
            counts.append(f"{c.notCoveredCount} with NO applicable criterion")
        if c.excludedCount is not None and c.excludedCount:
            counts.append(f"{c.excludedCount} excluded")
        if counts:
            parts.append("Analytes: " + ", ".join(counts))
        if c.notCoveredAnalytes:
            parts.append(
                "Not screened (no criterion applied): "
                + ", ".join(c.notCoveredAnalytes)
            )
        if c.excludedAnalytes:
            parts.append("Excluded from matching: " + ", ".join(c.excludedAnalytes))
        if parts:
            sections.append("## Criteria Coverage\n" + "\n".join(parts))

    if project_state and project_state.detectionStats:
        rows = []
        for stat in project_state.detectionStats:
            if not stat.analyte:
                continue
            line = f"- {stat.analyte}"
            if stat.detectionCount is not None and stat.totalSamples is not None:
                line += (
                    f": detected in {stat.detectionCount}/{stat.totalSamples} samples"
                )
            if stat.detectionRatePct is not None:
                line += f" ({stat.detectionRatePct:g}%)"
            rows.append(line)
        if rows:
            sections.append(
                "## Detection Frequency (partially detected analytes)\n"
                + "\n".join(rows)
            )

    if project_state and project_state.qaqc:
        q = project_state.qaqc
        parts = []
        if q.duplicatePairCount is not None:
            parts.append(f"Duplicate pairs assessed: {q.duplicatePairCount}")
        if q.rpdFailures is not None:
            parts.append(f"RPD failures: {q.rpdFailures}")
        if q.blankSampleCount is not None:
            blanks = f"Blank samples: {q.blankSampleCount}"
            if q.blankDetections is not None:
                blanks += f" ({q.blankDetections} with detections)"
            parts.append(blanks)
        if q.unpairedDuplicates:
            parts.append(f"Unpaired duplicates: {q.unpairedDuplicates}")
        for finding in q.findings or []:
            parts.append(f"- {finding}")
        if parts:
            sections.append("## QA/QC Summary\n" + "\n".join(parts))

    if project_state and project_state.samplingProgram:
        sp = project_state.samplingProgram
        parts = []
        if sp.rounds:
            parts.append(f"Sampling rounds: {', '.join(sp.rounds)}")
        if sp.sampleTypes:
            parts.append(f"Sample types: {', '.join(sp.sampleTypes)}")
        if sp.earliestDate and sp.latestDate:
            parts.append(f"Sampling dates: {sp.earliestDate} to {sp.latestDate}")
        elif sp.earliestDate or sp.latestDate:
            parts.append(f"Sampling date: {sp.earliestDate or sp.latestDate}")
        if parts:
            sections.append("## Sampling Program\n" + "\n".join(parts))

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
            total_samples = (
                project_state.project.totalSamples if project_state.project else None
            )
            if total_samples and total_samples > len(rows):
                rows.append(
                    f"(a relevance-selected {len(rows)} of {total_samples} total "
                    "samples — not the complete results table)"
                )
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

    if ctx.mapContext:
        m = ctx.mapContext
        parts = []
        if m.mapViewName:
            parts.append(f"Map view: {m.mapViewName}")
        if m.selectedAnalyte:
            parts.append(f"Mapped analyte: {m.selectedAnalyte}")
        if m.selectedCriteriaName:
            criteria = f"Map criteria: {m.selectedCriteriaName}"
            if m.criteriaValue is not None:
                criteria += f" ({m.criteriaValue}{' ' + m.criteriaUnit if m.criteriaUnit else ''})"
            parts.append(criteria)
        if m.depthFilter:
            parts.append(f"Depth filter: {m.depthFilter}")
        if m.contourAreaM2 is not None:
            parts.append(f"Contour area: {m.contourAreaM2:g} m2")
        if m.exceedanceZoneCount is not None:
            parts.append(f"Exceedance zones: {m.exceedanceZoneCount}")
        if m.criticalZoneCount is not None:
            parts.append(f"Critical zones: {m.criticalZoneCount}")
        if m.hotspotCount is not None:
            parts.append(f"Hotspots: {m.hotspotCount}")
        if m.hotspotDiameterM is not None:
            parts.append(f"Hotspot diameter: {m.hotspotDiameterM:g} m")
        if m.concentrationPointCount is not None:
            parts.append(f"Mapped sample points: {m.concentrationPointCount}")
        if m.volumeM3 is not None:
            volume = f"Estimated contaminated volume: {m.volumeM3:g} m3"
            if m.massTonnes is not None:
                volume += f" (~{m.massTonnes:g} t)"
            if m.volumeConfidence:
                volume += f", confidence {m.volumeConfidence}"
            parts.append(volume)
        if m.contaminatedAreaM2 is not None:
            parts.append(f"Contaminated area: {m.contaminatedAreaM2:g} m2")
        if m.averageDepthM is not None:
            parts.append(f"Average contaminated depth: {m.averageDepthM:g} m")
        if parts:
            sections.append("## Map Context\n" + "\n".join(parts))

    if ctx.saqpContext:
        s = ctx.saqpContext
        parts = []
        if s.planStatus:
            parts.append(f"Plan status: {s.planStatus}")
        if s.sufficiencyStatus:
            parts.append(f"Sufficiency: {s.sufficiencyStatus}")
        if s.plannedPoints is not None:
            planned = f"Planned points: {s.plannedPoints}"
            if s.requiredPoints is not None:
                planned += f" (guidance minimum {s.requiredPoints})"
            parts.append(planned)
        if s.areaHa is not None:
            parts.append(f"Assessment area: {s.areaHa:g} ha")
        if s.gridSizeM is not None:
            parts.append(f"Grid spacing: {s.gridSizeM:g} m")
        if s.rulesetKey:
            ruleset = f"Ruleset: {s.rulesetKey}"
            if s.rulesetVersion:
                ruleset += f" v{s.rulesetVersion}"
            parts.append(ruleset)
        if s.advisoryMessage:
            parts.append(f"Advisory: {s.advisoryMessage}")
        if s.overrideActive:
            override = "Manual override active"
            if s.overrideJustification:
                override += f": {s.overrideJustification}"
            parts.append(override)
        if s.completedPoints is not None:
            parts.append(f"Field progress: {s.completedPoints} completed")
        for point in s.points or []:
            if not point.name:
                continue
            line = f"- {point.name}"
            if point.depthFromM is not None and point.depthToM is not None:
                line += f" ({point.depthFromM:g}-{point.depthToM:g} m)"
            details = [
                d for d in (point.matrix, point.priority, point.executionStatus) if d
            ]
            if details:
                line += f" [{', '.join(details)}]"
            if point.notes:
                line += f" — {point.notes}"
            parts.append(line)
        if s.pointsTruncated:
            parts.append("(planned point list truncated — not all points shown)")
        if parts:
            sections.append("## Sampling Plan (SAQP)\n" + "\n".join(parts))

    if ctx.fieldContext:
        f = ctx.fieldContext
        parts = []
        if f.sessionCount is not None:
            parts.append(f"Field sessions: {f.sessionCount}")
        if f.latestSessionDate:
            parts.append(f"Latest session: {f.latestSessionDate}")
        if f.boreholeCount is not None:
            parts.append(f"Boreholes: {f.boreholeCount}")
        if f.fieldSampleCount is not None:
            parts.append(f"Field samples: {f.fieldSampleCount}")
        for hole in f.boreholes or []:
            if not hole.boreholeId:
                continue
            header_bits = []
            if hole.totalDepthM is not None:
                header_bits.append(f"total depth {hole.totalDepthM:g} m")
            if hole.groundwaterDepthM is not None:
                header_bits.append(f"groundwater {hole.groundwaterDepthM:g} m")
            if hole.waterFirstStrikeM is not None:
                header_bits.append(f"water first strike {hole.waterFirstStrikeM:g} m")
            if hole.waterStandingM is not None:
                header_bits.append(f"standing water {hole.waterStandingM:g} m")
            if hole.drillingMethod:
                header_bits.append(hole.drillingMethod)
            header = f"- {hole.boreholeId}"
            if header_bits:
                header += f" ({', '.join(header_bits)})"
            parts.append(header)
            if hole.groundwaterNotes:
                parts.append(f"  groundwater: {hole.groundwaterNotes}")
            for interval in hole.lithology or []:
                if interval.depthFromM is None or interval.depthToM is None:
                    continue
                bits = [
                    bit
                    for bit in (
                        interval.soilType,
                        interval.colour,
                        interval.moisture,
                        interval.uscsCode,
                    )
                    if bit
                ]
                line = f"  {interval.depthFromM:g}-{interval.depthToM:g} m: " + (
                    ", ".join(bits) if bits else "no description"
                )
                if interval.observations:
                    line += f" — {interval.observations}"
                parts.append(line)
            for sample in hole.samples or []:
                bits = []
                if sample.depthFromM is not None:
                    depth = f"{sample.depthFromM:g}"
                    if (
                        sample.depthToM is not None
                        and sample.depthToM != sample.depthFromM
                    ):
                        depth += f"-{sample.depthToM:g}"
                    bits.append(f"@ {depth} m")
                if sample.pidReading is not None:
                    bits.append(f"PID {sample.pidReading:g} {sample.pidUnit or 'ppm'}")
                if sample.odour:
                    bits.append(f"odour: {sample.odour}")
                if sample.observations:
                    bits.append(sample.observations)
                if not bits and not sample.sampleId:
                    continue
                parts.append(
                    f"  sample {sample.sampleId or '(unlabelled)'} " + " ".join(bits)
                )
        if f.truncated:
            parts.append("(borehole list truncated — not all field data is shown)")
        if parts:
            sections.append("## Borehole Logs & Field Data\n" + "\n".join(parts))

    if ctx.csmContext:
        c = ctx.csmContext
        parts = []
        if c.summary:
            parts.append(c.summary)
        if c.landUse:
            parts.append(f"Land use: {c.landUse}")
        if c.siteHistory:
            parts.append(f"Site history: {c.siteHistory}")
        if c.contaminantsOfConcern:
            parts.append(
                f"Contaminants of concern: {', '.join(c.contaminantsOfConcern)}"
            )
        if c.sources:
            parts.append(f"Potential sources: {', '.join(c.sources)}")
        if c.pathways:
            parts.append(f"Pathways: {', '.join(c.pathways)}")
        if c.receptors:
            parts.append(f"Receptors: {', '.join(c.receptors)}")
        for linkage in c.linkages or []:
            bits = [b for b in (linkage.source, linkage.pathway, linkage.receptor) if b]
            if not bits:
                continue
            line = "- " + " -> ".join(bits)
            if linkage.status:
                line += f" [{linkage.status}]"
            parts.append(line)
        if c.dataGaps:
            parts.append("Data gaps: " + "; ".join(c.dataGaps))
        if parts:
            sections.append(
                "## Conceptual Site Model (consultant's draft)\n" + "\n".join(parts)
            )

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
            row_count = len(retrieval_context.retrievedRows)
            total_samples = (
                project_state.project.totalSamples
                if project_state and project_state.project
                else None
            )
            if total_samples and total_samples > row_count:
                parts.append(
                    f"Retrieved rows: {row_count} of {total_samples} total "
                    "samples — a relevance-selected subset, NOT the complete "
                    "results table."
                )
            else:
                parts.append(f"Retrieved rows: {row_count}")
            # Question-matched sample rows are the most relevant lab evidence —
            # render their values, not just the count.
            for row in retrieval_context.retrievedRows[:30]:
                if not row.sampleCode:
                    continue
                header = row.sampleCode
                if row.depth:
                    header += f" ({row.depth})"
                vals = []
                for analyte_value in (row.analyteValues or [])[:20]:
                    if analyte_value.analyte and analyte_value.value is not None:
                        vals.append(
                            f"{analyte_value.analyte}={analyte_value.value}"
                            f"{' ' + analyte_value.unit if analyte_value.unit else ''}"
                        )
                line = f"- {header}"
                if vals:
                    line += ": " + ", ".join(vals)
                parts.append(line)
        if parts:
            sections.append("## Retrieval Context\n" + "\n".join(parts))

    return "\n\n".join(sections)
