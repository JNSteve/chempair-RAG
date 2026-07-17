"""Schema v5 fieldContext block: typed model and grounding-prompt rendering
(PRD_101 Phase C — borehole logs and field data reach Alfie)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_models import (  # noqa: E402
    FieldContext,
    WorkspaceContext,
    build_grounding_prompt,
)


def _field_payload() -> dict:
    return {
        "sessionCount": 2,
        "latestSessionDate": "2026-07-10",
        "boreholeCount": 2,
        "fieldSampleCount": 3,
        "boreholes": [
            {
                "boreholeId": "BH01",
                "totalDepthM": 3.5,
                "groundwaterDepthM": 2.1,
                "drillingMethod": "hand auger",
                "lithology": [
                    {
                        "depthFromM": 0,
                        "depthToM": 0.5,
                        "soilType": "FILL",
                        "colour": "brown",
                        "moisture": "moist",
                        "observations": "brick fragments",
                    },
                    {
                        "depthFromM": 0.5,
                        "depthToM": 3.5,
                        "soilType": "CLAY",
                        "uscsCode": "CH",
                    },
                ],
                "samples": [
                    {
                        "sampleId": "BH01-0.5",
                        "depthFromM": 0.5,
                        "pidReading": 12.4,
                        "pidUnit": "ppm",
                        "odour": "slight hydrocarbon",
                    }
                ],
            },
            {
                "boreholeId": "BH02",
                "totalDepthM": 2.0,
                "lithology": [],
                "samples": [],
            },
        ],
        "truncated": False,
    }


def _ctx(payload: dict | None = None) -> WorkspaceContext:
    return WorkspaceContext.model_validate(
        {
            "schemaVersion": 5,
            "projectState": {"project": {"projectName": "Ducat"}},
            "fieldContext": payload if payload is not None else _field_payload(),
        }
    )


def test_field_context_parses_into_typed_model():
    ctx = _ctx()
    assert isinstance(ctx.fieldContext, FieldContext)
    assert ctx.fieldContext.boreholes[0].boreholeId == "BH01"
    assert ctx.fieldContext.boreholes[0].lithology[0].soilType == "FILL"
    assert ctx.fieldContext.boreholes[0].samples[0].pidReading == 12.4


def test_v4_payload_without_field_context_still_parses():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    assert ctx.fieldContext is None


def test_grounding_prompt_renders_borehole_logs():
    prompt = build_grounding_prompt(_ctx())
    assert "## Borehole Logs & Field Data" in prompt
    assert "- BH01 (total depth 3.5 m, groundwater 2.1 m, hand auger)" in prompt
    assert "0-0.5 m: FILL, brown, moist — brick fragments" in prompt
    assert "0.5-3.5 m: CLAY, CH" in prompt
    assert "sample BH01-0.5 @ 0.5 m PID 12.4 ppm odour: slight hydrocarbon" in prompt
    assert "- BH02 (total depth 2 m)" in prompt


def test_grounding_prompt_notes_truncation():
    payload = _field_payload()
    payload["truncated"] = True
    prompt = build_grounding_prompt(_ctx(payload))
    assert "borehole list truncated" in prompt


def test_grounding_prompt_omits_section_when_absent():
    ctx = WorkspaceContext.model_validate(
        {"schemaVersion": 4, "projectState": {"project": {"projectName": "Ducat"}}}
    )
    assert "## Borehole Logs" not in build_grounding_prompt(ctx)
