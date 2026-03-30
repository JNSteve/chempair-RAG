from __future__ import annotations

import re
from dataclasses import dataclass

from context_models import WorkspaceContext
from query_grounding import GroundedQuestion
from query_normalization import normalise_text


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
PROJECT_CRITERIA_TERMS = (
    "hsl",
    "hil",
    "esl",
    "eil",
    "criterion",
    "criteria",
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
    "management level",
    "management levels",
    "management limit",
    "management limits",
    "waste classification",
    "waste class",
    "waste category",
    "disposal classification",
)
CRITERION_SCOPE_QUALIFIERS = (
    "clay",
    "sand",
    "silt",
    "fine soil",
    "coarse soil",
    "residential",
    "commercial",
    "industrial",
    "public open space",
    "parkland",
    "low density",
    "high density",
    "ecological",
    "freshwater",
    "marine",
)
DOCUMENT_SCOPE_PATTERNS = (
    "all soil types",
    "all land uses",
    "all values",
    "for each soil type",
    "for each land use",
    "across the nepm",
    "in the nepm",
    "across nepm",
    "compare",
    "comparison",
    "table",
    "full table",
    "all hsl",
    "all hil",
    "all esl",
    "all eil",
)


@dataclass
class RouteGuardrails:
    route_hint: str | None = None
    project_only_allowed: bool = True
    reason: str | None = None


def has_regulatory_context(ctx: WorkspaceContext) -> bool:
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


def question_requests_non_selected_scope(question: str, ctx: WorkspaceContext) -> bool:
    question_key = normalise_text(question)
    selected_names: list[str] = []

    project_state = ctx.projectState
    selected = project_state.selectedCriteria if project_state else None
    if selected:
        if selected.criteriaNames:
            selected_names.extend(name for name in selected.criteriaNames if name)
        if selected.applicableCriteria:
            selected_names.append(selected.applicableCriteria)

    selected_key = " ".join(normalise_text(name) for name in selected_names if name)
    if not selected_key:
        return False

    qualifier_mismatch = any(
        qualifier in question_key and qualifier not in selected_key
        for qualifier in CRITERION_SCOPE_QUALIFIERS
    )
    if qualifier_mismatch:
        return True

    depth_mentions = re.findall(r"\b\d+\s*(?:-\s*\d+)?\s*m\b|\b<\s*\d+\s*m\b", question_key)
    if depth_mentions and not any(depth in selected_key for depth in depth_mentions):
        return True

    return False


def question_needs_project_grounding(grounded: GroundedQuestion) -> bool:
    if any(term in grounded.normalised_question for term in PROJECT_INTERPRETIVE_REFERENTS):
        return True
    return grounded.has_entity_matches


def is_interpretive_question(grounded: GroundedQuestion) -> bool:
    question_key = grounded.normalised_question
    if any(re.search(pattern, question_key) for pattern in INTERPRETIVE_ROUTE_PATTERNS):
        return True
    if re.search(r"\bwhat (?:is|are).+\bfrom\b", question_key):
        return True
    if "what does this indicate" in question_key or "what does this mean" in question_key:
        return True
    if (
        grounded.has_entity_matches
        and any(term in question_key for term in ("contamination", "exceedance", "criterion", "criteria"))
        and any(term in question_key for term in ("from", "mean", "means", "indicate", "indicates"))
    ):
        return True
    return False


def is_deterministic_project_fact_question(grounded: GroundedQuestion) -> bool:
    question_key = grounded.normalised_question
    if is_interpretive_question(grounded):
        return False
    if any(pattern in question_key for pattern in DOCUMENT_SCOPE_PATTERNS):
        return False
    if grounded.criterion_lookup:
        return True
    if any(pattern in question_key for pattern in PROJECT_FACT_PATTERNS):
        return True
    if grounded.matched_sample_codes and any(
        token in question_key for token in ("highest", "lowest", "value", "values", "exceed", "exceedance")
    ):
        return True
    if grounded.matched_analytes and any(
        token in question_key
        for token in ("highest", "lowest", "main exceedances", "criterion", "threshold", "value in this project")
    ):
        return True
    return False


def is_generic_kb_question(grounded: GroundedQuestion) -> bool:
    if grounded.has_entity_matches:
        return False
    return any(pattern in grounded.normalised_question for pattern in GENERIC_KB_PATTERNS)


def deterministic_route_guardrails(
    question: str,
    ctx: WorkspaceContext,
    grounded: GroundedQuestion,
) -> RouteGuardrails:
    question_key = normalise_text(question)
    needs_project_grounding = question_needs_project_grounding(grounded)

    if question_requests_non_selected_scope(question, ctx):
        return RouteGuardrails(
            route_hint="blended" if has_regulatory_context(ctx) else "kb_only",
            project_only_allowed=False,
            reason="non_selected_criteria_scope",
        )

    if is_deterministic_project_fact_question(grounded):
        return RouteGuardrails(
            route_hint="project_only",
            project_only_allowed=True,
            reason="deterministic_project_fact",
        )

    if is_interpretive_question(grounded):
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

    if any(pattern in question_key for pattern in DOCUMENT_SCOPE_PATTERNS):
        return RouteGuardrails(
            route_hint="blended" if has_regulatory_context(ctx) else "kb_only",
            project_only_allowed=False,
            reason="document_scope_criteria_question",
        )

    if (
        has_regulatory_context(ctx)
        and any(term in question_key for term in PROJECT_CRITERIA_TERMS)
        and not is_generic_kb_question(grounded)
    ):
        return RouteGuardrails(
            route_hint="blended",
            project_only_allowed=False,
            reason="project_criteria_guidance",
        )

    if is_generic_kb_question(grounded):
        return RouteGuardrails(
            route_hint="kb_only",
            project_only_allowed=False,
            reason="generic_guidance_question",
        )

    return RouteGuardrails()


def coerce_route(route: str | None, guardrails: RouteGuardrails, context_used: bool) -> str:
    if not context_used:
        return "kb_only"

    normalised_route = route if route in {"project_only", "kb_only", "blended"} else "blended"

    if guardrails.route_hint:
        return guardrails.route_hint

    if not guardrails.project_only_allowed and normalised_route == "project_only":
        return "blended"

    return normalised_route
