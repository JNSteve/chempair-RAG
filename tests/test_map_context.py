"""Schema v5 mapContext block: model, grounded snapshot, prompt, routing.

Phase A of PRD_101 (enviro-sage): Alfie answers spatial-extent questions
from app-computed map figures. v4 payloads (no mapContext) are untouched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_models import (  # noqa: E402
    MapContext,
    WorkspaceContext,
    build_grounding_prompt,
)
from query_grounding import build_grounded_context, resolve_grounded_question  # noqa: E402
from query_routing import (  # noqa: E402
    deterministic_route_guardrails,
    is_spatial_extent_question,
)


def _map_context_payload() -> dict:
    return {
        "mapViewName": "Ducat baseline",
        "selectedAnalyte": "Arsenic",
        "selectedCriteriaName": "NEPM 2013 HIL-A residential",
        "criteriaValue": 100,
        "criteriaUnit": "mg/kg",
        "contourAreaM2": 2296,
        "exceedanceZoneCount": 3,
        "criticalZoneCount": 1,
        "hotspotCount": 2,
        "hotspotDiameterM": 10,
        "concentrationPointCount": 9,
    }


def _ctx_with_map(schema_version: int = 5) -> WorkspaceContext:
    return WorkspaceContext.model_validate(
        {
            "schemaVersion": schema_version,
            "projectState": {"project": {"projectName": "Ducat"}},
            "mapContext": _map_context_payload(),
        }
    )


def test_map_context_parses_into_typed_model():
    ctx = _ctx_with_map()
    assert isinstance(ctx.mapContext, MapContext)
    assert ctx.mapContext.contourAreaM2 == 2296
    assert ctx.mapContext.selectedAnalyte == "Arsenic"


def test_v4_payload_without_map_context_still_parses():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    assert ctx.mapContext is None


def test_grounding_prompt_renders_map_section():
    prompt = build_grounding_prompt(_ctx_with_map())
    assert "## Map Context" in prompt
    assert "Contour area: 2296 m2" in prompt
    assert "Hotspots: 2" in prompt
    assert "NEPM 2013 HIL-A residential" in prompt


def test_grounded_context_passes_map_context_through():
    ctx = _ctx_with_map()
    grounded = resolve_grounded_question("How big is the arsenic contour?", ctx)
    snapshot = build_grounded_context(ctx, grounded)
    assert snapshot["mapContext"]["contourAreaM2"] == 2296
    assert snapshot["mapContext"]["exceedanceZoneCount"] == 3


def test_grounded_context_omits_map_context_when_absent():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    grounded = resolve_grounded_question("How big is the arsenic contour?", ctx)
    snapshot = build_grounded_context(ctx, grounded)
    assert "mapContext" not in snapshot


def test_spatial_extent_detection():
    assert is_spatial_extent_question("how big is the arsenic contour")
    assert is_spatial_extent_question("what is the contaminated area on site")
    assert is_spatial_extent_question("is contamination contained within the site boundary")
    assert not is_spatial_extent_question("what is the hil a for arsenic")


def test_spatial_question_with_map_context_routes_project_only():
    ctx = _ctx_with_map()
    question = "How big is the arsenic contour?"
    grounded = resolve_grounded_question(question, ctx)
    guardrails = deterministic_route_guardrails(question, ctx, grounded)
    assert guardrails.route_hint == "project_only"
    assert guardrails.reason == "map_spatial_evidence"


def test_spatial_question_with_regulatory_framing_routes_hybrid():
    ctx = _ctx_with_map()
    question = "How big is the arsenic contour compared to NEPM guidance?"
    grounded = resolve_grounded_question(question, ctx)
    guardrails = deterministic_route_guardrails(question, ctx, grounded)
    assert guardrails.route_hint == "hybrid"
    assert guardrails.reason == "map_spatial_evidence_with_regulatory_support"


def test_spatial_question_without_map_context_keeps_existing_routing():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    question = "How big is the arsenic contour?"
    grounded = resolve_grounded_question(question, ctx)
    guardrails = deterministic_route_guardrails(question, ctx, grounded)
    assert guardrails.reason not in {
        "map_spatial_evidence",
        "map_spatial_evidence_with_regulatory_support",
    }
