"""Schema v5 saqpContext block: typed model and grounding-prompt rendering.

SAQP figures reach the model through the unified grounded answer path
(see tests/test_query_context.py::TestUnifiedAnswering) — there is no
routing and no deterministic sufficiency answerer any more.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_models import (  # noqa: E402
    SaqpContext,
    WorkspaceContext,
    build_grounding_prompt,
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


def test_v4_payload_without_saqp_context_still_parses():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    assert ctx.saqpContext is None


def test_grounding_prompt_renders_saqp_section():
    prompt = build_grounding_prompt(_ctx())
    assert "## Sampling Plan (SAQP)" in prompt
    assert "Planned points: 8 (guidance minimum 9)" in prompt
    assert "Grid spacing: 25 m" in prompt
    assert "nepm-b2-density v2026.1" in prompt
    assert "Sufficiency: borderline" in prompt


def test_grounding_prompt_renders_override_and_advisory():
    payload = _saqp_payload()
    payload["overrideActive"] = True
    payload["overrideJustification"] = "Historical data covers the north half."
    prompt = build_grounding_prompt(_ctx(payload))
    assert "Manual override active: Historical data covers the north half." in prompt
    assert "Advisory: Sampling plan is near guidance thresholds" in prompt


def test_grounding_prompt_omits_saqp_section_when_absent():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    assert "## Sampling Plan (SAQP)" not in build_grounding_prompt(ctx)
