"""Schema v5 saqpContext block: model, grounded snapshot, prompt, routing,
deterministic sufficiency answers (PRD_101 Phase B)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402
from context_models import (  # noqa: E402
    SaqpContext,
    WorkspaceContext,
    build_grounding_prompt,
)
from query_grounding import build_grounded_context, resolve_grounded_question  # noqa: E402
from query_routing import (  # noqa: E402
    deterministic_route_guardrails,
    is_saqp_plan_question,
)


def _saqp_payload() -> dict:
    return {
        "planStatus": "approved",
        "sufficiencyStatus": "borderline",
        "computedStatus": "borderline",
        "plannedPoints": 8,
        "requiredPoints": 9,
        "areaHa": 0.5,
        "gridEnabled": True,
        "gridSizeM": 25,
        "rulesetKey": "nepm-b2-density",
        "rulesetVersion": "2026.1",
        "advisoryMessage": (
            "Sampling plan is near guidance thresholds; review before field execution."
        ),
        "overrideActive": False,
        "completedPoints": 3,
        "skippedPoints": 1,
        "relocatedPoints": 0,
    }


def _ctx(payload: dict | None = None) -> WorkspaceContext:
    return WorkspaceContext.model_validate(
        {
            "schemaVersion": 5,
            "projectState": {"project": {"projectName": "Ducat"}},
            "saqpContext": payload if payload is not None else _saqp_payload(),
        }
    )


def test_saqp_context_parses_into_typed_model():
    ctx = _ctx()
    assert isinstance(ctx.saqpContext, SaqpContext)
    assert ctx.saqpContext.plannedPoints == 8
    assert ctx.saqpContext.rulesetKey == "nepm-b2-density"


def test_grounding_prompt_renders_saqp_section():
    prompt = build_grounding_prompt(_ctx())
    assert "## Sampling Plan (SAQP)" in prompt
    assert "Planned points: 8 (guidance minimum 9)" in prompt
    assert "Grid spacing: 25 m" in prompt
    assert "nepm-b2-density v2026.1" in prompt


def test_grounded_context_passes_saqp_through():
    ctx = _ctx()
    grounded = resolve_grounded_question("Have I planned enough samples?", ctx)
    snapshot = build_grounded_context(ctx, grounded)
    assert snapshot["saqpContext"]["plannedPoints"] == 8
    assert snapshot["saqpContext"]["sufficiencyStatus"] == "borderline"


def test_saqp_question_detection():
    assert is_saqp_plan_question("have i planned enough samples")
    assert is_saqp_plan_question("is my sampling density ok")
    assert is_saqp_plan_question("do i need more samples")
    assert not is_saqp_plan_question("what is the hil a for arsenic")


def test_saqp_question_routes_project_only():
    ctx = _ctx()
    question = "Have I planned enough samples?"
    grounded = resolve_grounded_question(question, ctx)
    guardrails = deterministic_route_guardrails(question, ctx, grounded)
    assert guardrails.route_hint == "project_only"
    assert guardrails.reason == "saqp_plan_evidence"


def test_saqp_question_with_regulatory_framing_routes_hybrid():
    ctx = _ctx()
    question = "Have I planned enough samples under the NEPM sampling guidance?"
    grounded = resolve_grounded_question(question, ctx)
    guardrails = deterministic_route_guardrails(question, ctx, grounded)
    assert guardrails.route_hint == "hybrid"
    assert guardrails.reason == "saqp_plan_evidence_with_regulatory_support"


def test_saqp_question_without_context_keeps_existing_routing():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    question = "Have I planned enough samples?"
    grounded = resolve_grounded_question(question, ctx)
    guardrails = deterministic_route_guardrails(question, ctx, grounded)
    assert guardrails.reason not in {
        "saqp_plan_evidence",
        "saqp_plan_evidence_with_regulatory_support",
    }


def test_deterministic_saqp_answer_borderline():
    answer = server._try_answer_saqp("Have I planned enough samples?", _ctx())
    assert answer is not None
    assert "close to the line" in answer
    assert "8 planned sampling points" in answer
    assert "guidance minimum of 9" in answer
    assert "0.5 ha" in answer
    assert "25 m grid" in answer
    assert "nepm-b2-density v2026.1" in answer
    assert "3 points completed" in answer
    assert "1 skipped" in answer


def test_deterministic_saqp_answer_sufficient():
    payload = _saqp_payload()
    payload["sufficiencyStatus"] = "sufficient"
    payload["computedStatus"] = "sufficient"
    payload["plannedPoints"] = 12
    answer = server._try_answer_saqp("is my sampling plan sufficient?", _ctx(payload))
    assert answer is not None
    assert answer.startswith("Yes")


def test_deterministic_saqp_answer_override():
    payload = _saqp_payload()
    payload["sufficiencyStatus"] = "override"
    payload["computedStatus"] = "insufficient"
    payload["overrideActive"] = True
    payload["overrideJustification"] = "Historical data covers the north half."
    answer = server._try_answer_saqp("have i planned enough samples?", _ctx(payload))
    assert answer is not None
    assert "manual override is active" in answer
    assert "Historical data covers the north half." in answer
    assert "falls short" in answer


def test_deterministic_saqp_answer_not_assessable():
    payload = _saqp_payload()
    payload["sufficiencyStatus"] = "not_assessable"
    payload["computedStatus"] = "not_assessable"
    payload["advisoryMessage"] = "Site area is required to assess sufficiency."
    answer = server._try_answer_saqp("planned enough samples?", _ctx(payload))
    assert answer is not None
    assert "cannot be assessed" in answer
    assert "Site area is required" in answer


def test_deterministic_saqp_answer_skips_other_questions():
    assert server._try_answer_saqp("what is the hil a for arsenic?", _ctx()) is None


def test_deterministic_saqp_answer_skips_without_context():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    assert server._try_answer_saqp("have i planned enough samples?", ctx) is None


def test_advice_questions_fall_through_to_llm():
    # "where should I add more samples?" is advice, not a figure lookup —
    # the template must not re-fire with the same sufficiency answer.
    assert (
        server._try_answer_saqp(
            "OK where should I add more samples to? what part of the site", _ctx()
        )
        is None
    )
    assert (
        server._try_answer_saqp("how should i improve my sampling plan?", _ctx())
        is None
    )
    # Plain sufficiency asks still answer deterministically.
    assert server._try_answer_saqp("have i planned enough samples?", _ctx()) is not None


def test_map_advice_questions_fall_through_to_llm():
    from context_models import WorkspaceContext as WC

    ctx = WC.model_validate(
        {
            "schemaVersion": 5,
            "projectState": {"project": {"projectName": "Ducat"}},
            "mapContext": {"exceedanceZoneCount": 3, "criticalZoneCount": 1},
        }
    )
    assert (
        server._try_answer_map_spatial("where are the hotspot zones on the map?", ctx)
        is None
    )
