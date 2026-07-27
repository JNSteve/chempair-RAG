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
