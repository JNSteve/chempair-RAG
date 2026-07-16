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
        "volumeM3": 320,
        "massTonnes": 512,
        "contaminatedAreaM2": 410,
        "averageDepthM": 0.78,
        "volumeConfidence": "moderate",
        "volumeDepthAssumed": False,
        "exceedingLocations": 5,
        "totalLocations": 12,
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
    assert is_spatial_extent_question(
        "is contamination contained within the site boundary"
    )
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


def test_deterministic_contour_answer_uses_map_figure():
    import server

    ctx = _ctx_with_map()
    answer = server._try_answer_map_spatial("How big is the arsenic contour area?", ctx)
    assert answer is not None
    assert "2296 m2" in answer
    assert "Arsenic" in answer
    assert "Ducat baseline" in answer


def test_deterministic_contour_answer_honest_when_absent():
    import server

    payload = _map_context_payload()
    payload.pop("contourAreaM2")
    ctx = WorkspaceContext.model_validate(
        {
            "schemaVersion": 5,
            "projectState": {"project": {"projectName": "Ducat"}},
            "mapContext": payload,
        }
    )
    answer = server._try_answer_map_spatial("How big is the contour area?", ctx)
    assert answer is not None
    assert "does not include a drawn contour area" in answer


def test_deterministic_zone_answer_counts_zones():
    import server

    ctx = _ctx_with_map()
    answer = server._try_answer_map_spatial(
        "How many exceedance zones are on the map, and how many are critical?", ctx
    )
    assert answer is not None
    assert "3 exceedance zones" in answer
    assert "1 critical" in answer
    assert "9 mapped sample points" in answer


def test_deterministic_map_answer_skips_non_spatial_questions():
    import server

    ctx = _ctx_with_map()
    assert server._try_answer_map_spatial("what is the hil a for arsenic?", ctx) is None


def test_deterministic_map_answer_skips_without_map_context():
    import server

    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    assert server._try_answer_map_spatial("how big is the contour area?", ctx) is None


def test_deterministic_volume_answer():
    import server

    ctx = _ctx_with_map()
    answer = server._try_answer_map_spatial(
        "What volume of contaminated soil is on the map?", ctx
    )
    assert answer is not None
    assert "320 m3" in answer
    assert "512 t" in answer
    assert "410 m2" in answer
    assert "0.78 m" in answer
    assert "5 of 12 mapped locations" in answer


def test_deterministic_volume_answer_honest_when_absent():
    import server

    payload = _map_context_payload()
    for field in (
        "volumeM3",
        "massTonnes",
        "contaminatedAreaM2",
        "averageDepthM",
        "volumeConfidence",
        "volumeDepthAssumed",
        "exceedingLocations",
        "totalLocations",
    ):
        payload.pop(field, None)
    ctx = WorkspaceContext.model_validate(
        {
            "schemaVersion": 5,
            "projectState": {"project": {"projectName": "Ducat"}},
            "mapContext": payload,
        }
    )
    answer = server._try_answer_map_spatial("how many tonnes of soil?", ctx)
    assert answer is not None
    assert "does not include a volume estimate" in answer
