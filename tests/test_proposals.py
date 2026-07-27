"""Unit tests for the /query proposals[] pipeline: context models,
fail-closed validation (mirror of enviro-sage validate.ts @ df5fe84f),
LLM output parsing, and generation orchestration."""

from context_models import ProposalContext, WorkspaceContext


def _proposal_context_dict() -> dict:
    return {
        "saqp": {
            "planId": "plan-1",
            "updatedAt": "2026-07-27T00:00:00.000Z",
            "points": [{"id": "pt-1", "label": "SP01"}],
            "samples": [{"id": "smp-1", "label": "BH01_0.5"}],
        },
        "csm": {
            "id": "csm-1",
            "updatedAt": "2026-07-27T01:00:00.000Z",
            "sources": [{"id": "s1", "label": "Former UST"}],
            "pathways": [{"id": "p1", "label": "Leaching"}],
            "receptors": [{"id": "r1", "label": "Groundwater users"}],
            "linkages": [{"id": "l1", "label": "UST to GW", "origin": "generated"}],
            "media": ["Soil", "Groundwater"],
        },
    }


class TestProposalContextModels:
    def test_workspace_context_parses_proposal_context(self):
        ctx = WorkspaceContext.model_validate(
            {"schemaVersion": 5, "proposalContext": _proposal_context_dict()}
        )
        assert ctx.proposalContext is not None
        assert ctx.proposalContext.saqp.planId == "plan-1"
        assert ctx.proposalContext.saqp.points[0].id == "pt-1"
        assert ctx.proposalContext.csm.linkages[0].origin == "generated"
        assert ctx.proposalContext.csm.media == ["Soil", "Groundwater"]

    def test_proposal_context_absent_is_none(self):
        ctx = WorkspaceContext.model_validate({"schemaVersion": 5})
        assert ctx.proposalContext is None

    def test_proposal_context_tolerates_unknown_fields(self):
        payload = _proposal_context_dict()
        payload["saqp"]["futureField"] = "x"
        parsed = ProposalContext.model_validate(payload)
        assert parsed.saqp.planId == "plan-1"


import pytest

from proposals import (
    ProposalRejected,
    extract_artifacts,
    _validate_add_targeted_point,
    _validate_set_grid_parameters,
    _validate_update_point_attributes,
)


def _artifacts():
    return extract_artifacts(ProposalContext.model_validate(_proposal_context_dict()))


class TestExtractArtifacts:
    def test_extracts_both_artifacts(self):
        saqp, csm = _artifacts()
        assert saqp.plan_id == "plan-1"
        assert saqp.updated_at == "2026-07-27T00:00:00.000Z"
        assert saqp.point_ids == frozenset({"pt-1"})
        assert saqp.sample_ids == frozenset({"smp-1"})
        assert csm.csm_id == "csm-1"
        assert csm.linkage_ids == frozenset({"l1"})
        assert csm.media == frozenset({"Soil", "Groundwater"})

    def test_artifact_without_updated_at_is_dropped(self):
        payload = _proposal_context_dict()
        del payload["saqp"]["updatedAt"]
        saqp, csm = extract_artifacts(ProposalContext.model_validate(payload))
        assert saqp is None
        assert csm is not None

    def test_none_context_yields_no_artifacts(self):
        assert extract_artifacts(None) == (None, None)


class TestSetGridParameters:
    def test_valid_payload(self):
        result = _validate_set_grid_parameters({"gridEnabled": True, "gridSizeM": 40})
        assert result == {"gridEnabled": True, "gridSizeM": 40}

    @pytest.mark.parametrize("bad", [0, 501, 40.5, "40", True, None])
    def test_bad_grid_size_rejected(self, bad):
        with pytest.raises(ProposalRejected):
            _validate_set_grid_parameters({"gridEnabled": True, "gridSizeM": bad})

    def test_extra_key_rejected(self):
        with pytest.raises(ProposalRejected):
            _validate_set_grid_parameters(
                {"gridEnabled": True, "gridSizeM": 40, "latitude": -27.5}
            )


def _point_payload(**overrides) -> dict:
    payload = {
        "anchor": {"type": "sample", "id": "smp-1"},
        "offsetM": 30,
        "bearingDeg": 90,
        "sampleName": "TP01",
        "depthFromM": 0,
        "depthToM": 1,
        "matrix": "soil",
        "priority": "medium",
    }
    payload.update(overrides)
    return payload


class TestAddTargetedPoint:
    def test_valid_payload(self):
        saqp, _ = _artifacts()
        result = _validate_add_targeted_point(_point_payload(), saqp)
        assert result["anchor"] == {"type": "sample", "id": "smp-1"}
        assert result["sampleName"] == "TP01"

    def test_saqp_point_anchor_checks_point_ids(self):
        saqp, _ = _artifacts()
        result = _validate_add_targeted_point(
            _point_payload(anchor={"type": "saqp_point", "id": "pt-1"}), saqp
        )
        assert result["anchor"]["id"] == "pt-1"

    def test_unknown_anchor_id_rejected(self):
        saqp, _ = _artifacts()
        with pytest.raises(ProposalRejected):
            _validate_add_targeted_point(
                _point_payload(anchor={"type": "sample", "id": "nope"}), saqp
            )

    def test_coordinate_smuggling_rejected(self):
        saqp, _ = _artifacts()
        with pytest.raises(ProposalRejected):
            _validate_add_targeted_point(_point_payload(latitude=-27.5), saqp)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"offsetM": 501},
            {"offsetM": -1},
            {"bearingDeg": 361},
            {"depthFromM": 2, "depthToM": 1},
            {"depthToM": 101},
            {"sampleName": "x" * 121},
            {"priority": "urgent"},
            {"anchor": {"type": "sample", "id": "smp-1", "lat": 1}},
        ],
    )
    def test_out_of_contract_rejected(self, overrides):
        saqp, _ = _artifacts()
        with pytest.raises(ProposalRejected):
            _validate_add_targeted_point(_point_payload(**overrides), saqp)


class TestUpdatePointAttributes:
    def test_single_field_update_ok(self):
        saqp, _ = _artifacts()
        result = _validate_update_point_attributes(
            {"pointId": "pt-1", "priority": "high"}, saqp
        )
        assert result == {"pointId": "pt-1", "priority": "high"}

    def test_bare_point_id_rejected(self):
        saqp, _ = _artifacts()
        with pytest.raises(ProposalRejected, match="No fields to update"):
            _validate_update_point_attributes({"pointId": "pt-1"}, saqp)

    def test_unknown_point_id_rejected(self):
        saqp, _ = _artifacts()
        with pytest.raises(ProposalRejected):
            _validate_update_point_attributes(
                {"pointId": "ghost", "priority": "high"}, saqp
            )

    def test_inverted_depths_rejected(self):
        saqp, _ = _artifacts()
        with pytest.raises(ProposalRejected):
            _validate_update_point_attributes(
                {"pointId": "pt-1", "depthFromM": 3, "depthToM": 1}, saqp
            )

    def test_notes_cap_600(self):
        saqp, _ = _artifacts()
        with pytest.raises(ProposalRejected):
            _validate_update_point_attributes(
                {"pointId": "pt-1", "notes": "x" * 601}, saqp
            )
