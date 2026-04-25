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
REGULATORY_ONLY_OVERRIDES = (
    "i dont care about the site",
    "i don't care about the site",
    "ignore the selected criteria",
    "ignore selected criteria",
    "just tell me what the nepm says",
    "not for this project",
    "not for the site",
    "in general",
    "in the document",
)
PROJECT_ONLY_SCOPE_TERMS = (
    "this project",
    "this site",
    "our site",
    "our project",
    "my site",
    "my project",
    "selected criteria",
    "selected criterion",
    "criteria have i selected",
    "criterion have i selected",
    "applied criterion",
    "applied criteria",
    "for this project",
    "for this site",
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


def _has_explicit_regulatory_override(question_key: str) -> bool:
    return any(term in question_key for term in REGULATORY_ONLY_OVERRIDES)


def _has_explicit_project_scope(question_key: str) -> bool:
    return any(term in question_key for term in PROJECT_ONLY_SCOPE_TERMS)


def _has_regulatory_framing(question_key: str) -> bool:
    return (
        any(pattern in question_key for pattern in DOCUMENT_SCOPE_PATTERNS)
        or any(pattern in question_key for pattern in GENERIC_KB_PATTERNS)
        or any(term in question_key for term in PROJECT_CRITERIA_TERMS)
        or any(term in question_key for term in ("nepm", "guideline", "guidelines", "table", "document"))
    )


def _mentions_regulation_or_criteria(question_key: str) -> bool:
    return _has_regulatory_framing(question_key) or any(
        term in question_key for term in ("criterion", "criteria", "threshold", "regulation", "regulations")
    )


def _is_follow_up_fragment(question_key: str) -> bool:
    token_count = len(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", question_key))
    if token_count <= 4:
        return True
    return question_key.startswith(
        ("in the", "across", "for all", "all ", "under", "compare", "what about", "and ", "or ")
    )


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
    if grounded.criterion_lookup and not any(
        term in question_key
        for term in (
            "mean",
            "means",
            "meaning",
            "indicate",
            "indicates",
            "indication",
            "from",
            "source",
            "origin",
            "significance",
            "implication",
            "risk",
            "health",
            "remediation",
            "disposal",
        )
    ):
        return False
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
    if any(pattern in question_key for pattern in PROJECT_FACT_PATTERNS):
        return True
    if _mentions_regulation_or_criteria(question_key):
        return False
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
    previous_route: str | None = None,
) -> RouteGuardrails:
    question_key = normalise_text(question)
    needs_project_grounding = question_needs_project_grounding(grounded)
    has_regulatory_framing = _has_regulatory_framing(question_key)
    mentions_regulation_or_criteria = _mentions_regulation_or_criteria(question_key)
    explicit_regulatory_override = _has_explicit_regulatory_override(question_key)
    explicit_project_scope = _has_explicit_project_scope(question_key)

    if explicit_regulatory_override:
        return RouteGuardrails(
            route_hint="regulatory_only",
            project_only_allowed=False,
            reason="explicit_regulatory_override",
        )

    if previous_route and _is_follow_up_fragment(question_key):
        if previous_route == "regulatory_only" and not explicit_project_scope:
            return RouteGuardrails(
                route_hint="regulatory_only",
                project_only_allowed=False,
                reason="follow_up_inherits_regulatory_only",
            )
        if previous_route == "project_only" and not has_regulatory_framing and not explicit_regulatory_override:
            return RouteGuardrails(
                route_hint="project_only",
                project_only_allowed=True,
                reason="follow_up_inherits_project_only",
            )
        if previous_route == "hybrid" and not explicit_regulatory_override and not explicit_project_scope:
            return RouteGuardrails(
                route_hint="hybrid",
                project_only_allowed=False,
                reason="follow_up_inherits_hybrid",
            )

    if is_deterministic_project_fact_question(grounded):
        return RouteGuardrails(
            route_hint="project_only",
            project_only_allowed=True,
            reason="deterministic_project_fact",
        )

    if mentions_regulation_or_criteria:
        return RouteGuardrails(
            route_hint="hybrid" if (grounded.has_entity_matches or explicit_project_scope) else "regulatory_only",
            project_only_allowed=False,
            reason="regulation_or_criteria_question",
        )

    if question_requests_non_selected_scope(question, ctx):
        return RouteGuardrails(
            route_hint="hybrid" if has_regulatory_context(ctx) else "regulatory_only",
            project_only_allowed=False,
            reason="non_selected_criteria_scope",
        )

    if explicit_project_scope:
        return RouteGuardrails(
            route_hint="hybrid" if has_regulatory_framing else "project_only",
            project_only_allowed=not has_regulatory_framing,
            reason="explicit_project_scope",
        )

    if is_interpretive_question(grounded):
        route_hint = "hybrid" if needs_project_grounding else "regulatory_only"
        return RouteGuardrails(
            route_hint=route_hint,
            project_only_allowed=False,
            reason="interpretive_or_causal_question",
        )

    if needs_project_grounding and has_regulatory_framing:
        return RouteGuardrails(
            route_hint="hybrid",
            project_only_allowed=False,
            reason="project_regulatory_question",
        )

    if any(pattern in question_key for pattern in DOCUMENT_SCOPE_PATTERNS):
        return RouteGuardrails(
            route_hint="hybrid" if has_regulatory_context(ctx) else "regulatory_only",
            project_only_allowed=False,
            reason="document_scope_criteria_question",
        )

    if (
        has_regulatory_context(ctx)
        and any(term in question_key for term in PROJECT_CRITERIA_TERMS)
        and not is_generic_kb_question(grounded)
    ):
        return RouteGuardrails(
            route_hint="hybrid",
            project_only_allowed=False,
            reason="project_criteria_guidance",
        )

    if is_generic_kb_question(grounded):
        return RouteGuardrails(
            route_hint="regulatory_only",
            project_only_allowed=False,
            reason="generic_guidance_question",
        )

    if grounded.has_entity_matches:
        return RouteGuardrails(
            route_hint="hybrid",
            project_only_allowed=False,
            reason="entity_matched_hybrid_default",
        )

    return RouteGuardrails(
        route_hint="regulatory_only",
        project_only_allowed=False,
        reason="regulatory_only_default",
    )


def coerce_route(route: str | None, guardrails: RouteGuardrails, context_used: bool) -> str:
    if not context_used:
        return "regulatory_only"

    normalised_route = (
        route if route in {"project_only", "regulatory_only", "hybrid"} else "hybrid"
    )

    if guardrails.route_hint:
        return guardrails.route_hint

    if not guardrails.project_only_allowed and normalised_route == "project_only":
        return "hybrid"

    return normalised_route
