"""Schema v5 mapContext block: typed model and grounding-prompt rendering.

Map figures reach the model through the unified grounded answer path
(see tests/test_query_context.py::TestUnifiedAnswering) — there is no
routing and no deterministic map answerer any more.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_models import (  # noqa: E402
    MapContext,
    WorkspaceContext,
    build_grounding_prompt,
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


def test_grounding_prompt_renders_volume_figures():
    prompt = build_grounding_prompt(_ctx_with_map())
    assert "Estimated contaminated volume: 320 m3 (~512 t)" in prompt
    assert "Contaminated area: 410 m2" in prompt
    assert "Average contaminated depth: 0.78 m" in prompt


def test_grounding_prompt_omits_map_section_when_absent():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    assert "## Map Context" not in build_grounding_prompt(ctx)
