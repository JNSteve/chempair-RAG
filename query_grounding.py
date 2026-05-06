from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from context_models import WorkspaceContext
from query_normalization import normalise_text, resolve_candidate_matches


PROJECT_REFERENCE_TERMS = (
    "this project",
    "this site",
    "these exceedances",
    "these results",
    "our project",
    "my project",
    "in this project",
)
CRITERION_LOOKUP_TERMS = (
    "criterion",
    "criteria",
    "guideline",
    "guidelines",
    "threshold",
    "screening",
    "screening level",
    "screening levels",
    "investigation level",
    "investigation levels",
    "ecological investigation level",
    "ecological investigation levels",
    "action level",
    "action levels",
    "hsl",
    "hil",
    "esl",
    "eil",
    "management level",
    "management levels",
    "management limit",
    "management limits",
    "waste classification",
    "waste class",
    "waste category",
    "disposal classification",
)
CRITERION_VALUE_TERMS = (
    "value",
    "limit",
    "exceedance",
    "screening level",
    "investigation level",
    "investigation levels",
    "criterion value",
    "guideline value",
    "threshold",
    "allowable",
    "maximum",
    "trigger",
    "trigger value",
    "trigger values",
    "management level",
    "management levels",
    "management limit",
    "management limits",
)


@dataclass
class GroundedQuestion:
    question: str
    normalised_question: str
    matched_analytes: list[str] = field(default_factory=list)
    matched_sample_codes: list[str] = field(default_factory=list)
    matched_criteria_names: list[str] = field(default_factory=list)
    project_referenced: bool = False
    criterion_lookup: bool = False

    @property
    def has_entity_matches(self) -> bool:
        return bool(
            self.matched_analytes
            or self.matched_sample_codes
            or self.matched_criteria_names
            or self.project_referenced
        )


def question_targets_criterion_lookup(question: str) -> bool:
    normalised = normalise_text(question)
    if normalised.startswith("how ") or normalised.startswith("why "):
        return False
    return any(term in normalised for term in CRITERION_LOOKUP_TERMS) and any(
        term in normalised for term in CRITERION_VALUE_TERMS
    )


def iter_thresholds(ctx: WorkspaceContext):
    project_state = ctx.projectState
    if not project_state or not project_state.criteriaDetails:
        return

    for detail in project_state.criteriaDetails:
        if not detail.thresholds:
            continue
        for threshold in detail.thresholds:
            yield detail, threshold


def selected_criterion_names(ctx: WorkspaceContext) -> list[str]:
    project_state = ctx.projectState
    if not project_state or not project_state.selectedCriteria:
        return []

    selected = project_state.selectedCriteria
    names: list[str] = []
    if selected.criteriaNames:
        names.extend(name for name in selected.criteriaNames if name)
    if selected.applicableCriteria:
        names.append(selected.applicableCriteria)
    return _dedupe_strings(names)


def collect_context_analytes(ctx: WorkspaceContext) -> list[str]:
    analytes: list[str] = []
    if ctx.targetAnalytes:
        analytes.extend(analyte for analyte in ctx.targetAnalytes if analyte)
    if ctx.retrievalContext and ctx.retrievalContext.matchedAnalytes:
        analytes.extend(
            analyte for analyte in ctx.retrievalContext.matchedAnalytes if analyte
        )

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
            analytes.extend(item.analyte for item in row.analyteValues if item.analyte)
    if ctx.retrievalContext and ctx.retrievalContext.retrievedRows:
        for row in ctx.retrievalContext.retrievedRows:
            if not row.analyteValues:
                continue
            analytes.extend(item.analyte for item in row.analyteValues if item.analyte)

    return _dedupe_strings(analytes)


def collect_sample_codes(ctx: WorkspaceContext) -> list[str]:
    codes: list[str] = []
    if ctx.targetSampleCodes:
        codes.extend(code for code in ctx.targetSampleCodes if code)
    if ctx.retrievalContext and ctx.retrievalContext.matchedSampleCodes:
        codes.extend(code for code in ctx.retrievalContext.matchedSampleCodes if code)

    project_state = ctx.projectState
    if project_state and project_state.exceedances:
        codes.extend(ex.sampleCode for ex in project_state.exceedances if ex.sampleCode)
    if project_state and project_state.projectResults:
        codes.extend(
            row.sampleCode for row in project_state.projectResults if row.sampleCode
        )
    if ctx.retrievalContext and ctx.retrievalContext.retrievedRows:
        codes.extend(
            row.sampleCode
            for row in ctx.retrievalContext.retrievedRows
            if row.sampleCode
        )

    return _dedupe_strings(codes)


def resolve_grounded_question(
    question: str,
    ctx: WorkspaceContext,
    ignored_tokens: set[str] | None = None,
) -> GroundedQuestion:
    question_key = normalise_text(question)
    return GroundedQuestion(
        question=question,
        normalised_question=question_key,
        matched_analytes=resolve_candidate_matches(
            question,
            collect_context_analytes(ctx),
        ),
        matched_sample_codes=resolve_candidate_matches(
            question,
            collect_sample_codes(ctx),
        ),
        matched_criteria_names=resolve_candidate_matches(
            question,
            selected_criterion_names(ctx),
            ignored_tokens=ignored_tokens,
        ),
        project_referenced=any(
            term in question_key for term in PROJECT_REFERENCE_TERMS
        ),
        criterion_lookup=question_targets_criterion_lookup(question),
    )


def build_grounded_context(
    ctx: WorkspaceContext,
    grounded: GroundedQuestion,
    include_regulatory_snapshot: bool = False,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    project_state = ctx.projectState
    retrieval_context = ctx.retrievalContext

    if project_state and project_state.project:
        snapshot["project"] = project_state.project.model_dump(exclude_none=True)
    if project_state and project_state.selectedCriteria:
        snapshot["selectedCriteria"] = project_state.selectedCriteria.model_dump(
            exclude_none=True
        )
    if ctx.projectEvidenceSummary:
        snapshot["projectEvidenceSummary"] = ctx.projectEvidenceSummary.model_dump(
            exclude_none=True
        )
    if ctx.questionIntent:
        snapshot["questionIntent"] = ctx.questionIntent
    if ctx.requiresProjectContext is not None:
        snapshot["requiresProjectContext"] = ctx.requiresProjectContext
    if ctx.targetAnalytes:
        snapshot["targetAnalytes"] = ctx.targetAnalytes
    if ctx.targetSampleCodes:
        snapshot["targetSampleCodes"] = ctx.targetSampleCodes

    matched_analyte_keys = {
        normalise_text(analyte) for analyte in grounded.matched_analytes
    }
    matched_sample_keys = {
        normalise_text(code) for code in grounded.matched_sample_codes
    }
    matched_criteria_keys = {
        normalise_text(name) for name in grounded.matched_criteria_names
    }

    if retrieval_context:
        filtered_retrieval: dict[str, Any] = {}
        if grounded.matched_analytes:
            filtered_retrieval["matchedAnalytes"] = grounded.matched_analytes
        if grounded.matched_sample_codes:
            filtered_retrieval["matchedSampleCodes"] = grounded.matched_sample_codes
        if filtered_retrieval:
            snapshot["retrievalContext"] = filtered_retrieval

    if project_state and project_state.criteriaDetails:
        criteria_details: list[dict[str, Any]] = []
        for detail in project_state.criteriaDetails:
            detail_key = normalise_text(detail.name)
            keep_detail = include_regulatory_snapshot or (
                detail_key and detail_key in matched_criteria_keys
            )
            thresholds = []
            for threshold in detail.thresholds or []:
                analyte_key = normalise_text(threshold.analyte)
                if matched_analyte_keys and analyte_key not in matched_analyte_keys:
                    continue
                thresholds.append(threshold.model_dump(exclude_none=True))

            if thresholds:
                criteria_details.append(
                    {
                        "name": detail.name,
                        "thresholds": thresholds,
                    }
                )
            elif keep_detail and detail.thresholds:
                criteria_details.append(detail.model_dump(exclude_none=True))

        if criteria_details:
            snapshot["criteria"] = {"criteriaDetails": criteria_details}

    filtered_exceedances: list[dict[str, Any]] = []
    if project_state and project_state.exceedances:
        for exceedance in project_state.exceedances:
            analyte_key = normalise_text(exceedance.analyte)
            sample_key = normalise_text(exceedance.sampleCode)
            if matched_analyte_keys and analyte_key not in matched_analyte_keys:
                continue
            if matched_sample_keys and sample_key not in matched_sample_keys:
                continue
            filtered_exceedances.append(exceedance.model_dump(exclude_none=True))

    if filtered_exceedances:
        snapshot["exceedances"] = filtered_exceedances

    if project_state and project_state.exceedanceSummary:
        summary = project_state.exceedanceSummary.model_dump(exclude_none=True)
        if filtered_exceedances:
            summary["totalExceedances"] = len(filtered_exceedances)
            if grounded.matched_analytes:
                summary["affectedAnalytes"] = grounded.matched_analytes
            if grounded.matched_sample_codes:
                summary["affectedSamples"] = grounded.matched_sample_codes
        snapshot["exceedanceSummary"] = summary

    relevant_rows = []
    row_source = []
    if retrieval_context and retrieval_context.retrievedRows:
        row_source = retrieval_context.retrievedRows
    elif project_state and project_state.projectResults:
        row_source = project_state.projectResults

    for row in row_source:
        sample_key = normalise_text(row.sampleCode)
        if matched_sample_keys and sample_key not in matched_sample_keys:
            continue

        analyte_values = []
        for analyte_value in row.analyteValues or []:
            analyte_key = normalise_text(analyte_value.analyte)
            if matched_analyte_keys and analyte_key not in matched_analyte_keys:
                continue
            analyte_values.append(analyte_value.model_dump(exclude_none=True))

        if matched_analyte_keys and not analyte_values:
            continue

        row_payload = row.model_dump(exclude_none=True)
        if analyte_values:
            row_payload["analyteValues"] = analyte_values
        relevant_rows.append(row_payload)

    if relevant_rows:
        snapshot["relevantSamples"] = relevant_rows[:10]

    return snapshot


def merge_grounded_context(
    base: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)

    for key, value in overlay.items():
        if value in (None, "", [], {}):
            continue
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = {**existing, **value}
        else:
            merged[key] = value

    return merged


def find_question_analyte(question: str, ctx: WorkspaceContext) -> str | None:
    grounded = resolve_grounded_question(question, ctx)
    if grounded.matched_analytes:
        return grounded.matched_analytes[0]
    analytes = collect_context_analytes(ctx)
    return analytes[0] if analytes else None


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = normalise_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped
