"""Unit tests for the /query proposals[] pipeline: context models,
fail-closed validation (mirror of enviro-sage validate.ts @ df5fe84f),
LLM output parsing, and generation orchestration."""

import asyncio
import json

import pytest

from context_models import ProposalContext, WorkspaceContext
from proposals import (
    MAX_PROPOSALS_PER_ANSWER,
    ProposalRejected,
    build_proposals_prompt,
    extract_artifacts,
    generate_proposals,
    parse_llm_proposals,
    validate_candidate,
    _validate_add_linkage,
    _validate_add_targeted_point,
    _validate_set_grid_parameters,
    _validate_update_linkage,
    _validate_update_narrative,
    _validate_update_point_attributes,
)


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


class TestAddLinkage:
    def test_valid_payload(self):
        _, csm = _artifacts()
        result = _validate_add_linkage(
            {
                "sourceId": "s1",
                "pathwayId": "p1",
                "receptorId": "r1",
                "riskLevel": "moderate",
                "isComplete": False,
                "reasoning": "Leaching pathway plausible given TRH exceedances.",
            },
            csm,
        )
        assert result["riskLevel"] == "moderate"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"sourceId": "ghost"},
            {"pathwayId": "ghost"},
            {"receptorId": "ghost"},
            {"riskLevel": "severe"},
            {"isComplete": "false"},
            {"reasoning": "x" * 601},
            {"extra": 1},
        ],
    )
    def test_out_of_contract_rejected(self, overrides):
        _, csm = _artifacts()
        payload = {
            "sourceId": "s1",
            "pathwayId": "p1",
            "receptorId": "r1",
            "riskLevel": "moderate",
            "isComplete": False,
            "reasoning": "ok",
        }
        payload.update(overrides)
        with pytest.raises(ProposalRejected):
            _validate_add_linkage(payload, csm)


class TestUpdateLinkage:
    def test_single_field_ok(self):
        _, csm = _artifacts()
        result = _validate_update_linkage({"linkageId": "l1", "isComplete": True}, csm)
        assert result == {"linkageId": "l1", "isComplete": True}

    def test_bare_linkage_id_rejected(self):
        # Mirrors enviro-sage PR #614.
        _, csm = _artifacts()
        with pytest.raises(ProposalRejected, match="No fields to update"):
            _validate_update_linkage({"linkageId": "l1"}, csm)

    def test_unknown_linkage_rejected(self):
        _, csm = _artifacts()
        with pytest.raises(ProposalRejected):
            _validate_update_linkage({"linkageId": "ghost", "isComplete": True}, csm)


class TestUpdateNarrative:
    def test_summary_text(self):
        _, csm = _artifacts()
        result = _validate_update_narrative(
            {"section": "csmSummary", "text": "Summary."}, csm
        )
        assert result == {"section": "csmSummary", "text": "Summary."}

    def test_key_findings_items(self):
        _, csm = _artifacts()
        result = _validate_update_narrative(
            {"section": "keyFindings", "items": ["Finding one", "Finding two"]}, csm
        )
        assert result["items"] == ["Finding one", "Finding two"]

    def test_exposure_justification_checks_media(self):
        _, csm = _artifacts()
        result = _validate_update_narrative(
            {"section": "exposureJustification", "medium": "Soil", "text": "Why."},
            csm,
        )
        assert result["medium"] == "Soil"
        with pytest.raises(ProposalRejected, match="not an affected medium"):
            _validate_update_narrative(
                {"section": "exposureJustification", "medium": "Air", "text": "Why."},
                csm,
            )

    @pytest.mark.parametrize(
        "payload",
        [
            {"section": "csmSummary", "text": "x" * 4001},
            {"section": "keyFindings", "items": []},
            {"section": "keyFindings", "items": ["x"] * 11},
            {"section": "keyFindings", "items": ["x" * 301]},
            {"section": "background", "text": "x"},
            {"section": "csmSummary", "text": "x", "bogus": 1},
        ],
    )
    def test_out_of_contract_rejected(self, payload):
        _, csm = _artifacts()
        with pytest.raises(ProposalRejected):
            _validate_update_narrative(payload, csm)


def _grid_candidate(**overrides) -> dict:
    candidate = {
        "operation": "saqp.set_grid_parameters",
        "payload": {"gridEnabled": True, "gridSizeM": 40},
        "rationale": "Systematic 40 m grid coverage per NEPM Sch B2 guidance.",
        "citations": [{"source": "NEPM 2013 Sch B2", "locator": "s4.2"}],
    }
    candidate.update(overrides)
    return candidate


class TestValidateCandidate:
    def test_valid_candidate_gets_full_envelope(self):
        saqp, csm = _artifacts()
        envelope = validate_candidate(_grid_candidate(), saqp, csm)
        assert envelope["kind"] == "saqp"
        assert envelope["operation"] == "saqp.set_grid_parameters"
        assert envelope["id"].startswith("prop-")
        assert len(envelope["id"]) <= 128
        assert envelope["baseline"] == {
            "artifactId": "plan-1",
            "updatedAt": "2026-07-27T00:00:00.000Z",
        }
        assert envelope["citations"] == [
            {"source": "NEPM 2013 Sch B2", "locator": "s4.2"}
        ]

    def test_csm_candidate_gets_csm_baseline(self):
        saqp, csm = _artifacts()
        envelope = validate_candidate(
            {
                "operation": "csm.update_linkage",
                "payload": {"linkageId": "l1", "isComplete": True},
                "rationale": "Linkage supported by groundwater exceedances.",
            },
            saqp,
            csm,
        )
        assert envelope["kind"] == "csm"
        assert envelope["baseline"]["artifactId"] == "csm-1"
        assert envelope["citations"] == []

    def test_model_supplied_envelope_fields_ignored(self):
        saqp, csm = _artifacts()
        envelope = validate_candidate(
            _grid_candidate(
                id="model-id",
                kind="csm",
                baseline={"artifactId": "fake", "updatedAt": "fake"},
            ),
            saqp,
            csm,
        )
        assert envelope["id"] != "model-id"
        assert envelope["kind"] == "saqp"
        assert envelope["baseline"]["artifactId"] == "plan-1"

    def test_unknown_operation_rejected(self):
        saqp, csm = _artifacts()
        with pytest.raises(ProposalRejected):
            validate_candidate(
                _grid_candidate(operation="saqp.delete_point"), saqp, csm
            )

    def test_saqp_operation_without_saqp_artifact_rejected(self):
        _, csm = _artifacts()
        with pytest.raises(ProposalRejected):
            validate_candidate(_grid_candidate(), None, csm)

    def test_missing_rationale_rejected(self):
        saqp, csm = _artifacts()
        candidate = _grid_candidate()
        del candidate["rationale"]
        with pytest.raises(ProposalRejected):
            validate_candidate(candidate, saqp, csm)

    def test_rationale_cap_600(self):
        saqp, csm = _artifacts()
        with pytest.raises(ProposalRejected):
            validate_candidate(_grid_candidate(rationale="x" * 601), saqp, csm)

    def test_citation_without_source_dropped(self):
        saqp, csm = _artifacts()
        envelope = validate_candidate(
            _grid_candidate(citations=[{"locator": "s1"}, {"source": "NEPM 2013"}]),
            saqp,
            csm,
        )
        assert envelope["citations"] == [{"source": "NEPM 2013"}]

    def test_seven_citations_rejected(self):
        saqp, csm = _artifacts()
        with pytest.raises(ProposalRejected):
            validate_candidate(
                _grid_candidate(citations=[{"source": f"S{i}"} for i in range(7)]),
                saqp,
                csm,
            )

    def test_overlong_citation_source_rejected(self):
        saqp, csm = _artifacts()
        with pytest.raises(ProposalRejected):
            validate_candidate(
                _grid_candidate(citations=[{"source": "x" * 201}]), saqp, csm
            )


class TestParseLlmProposals:
    def test_object_with_proposals_key(self):
        assert parse_llm_proposals(json.dumps({"proposals": [{"a": 1}]})) == [{"a": 1}]

    def test_top_level_list(self):
        assert parse_llm_proposals(json.dumps([{"a": 1}])) == [{"a": 1}]

    @pytest.mark.parametrize("raw", ["not json", "{}", '{"proposals": "x"}', "", None])
    def test_garbage_yields_empty(self, raw):
        assert parse_llm_proposals(raw) == []


class TestBuildProposalsPrompt:
    def test_prompt_lists_targets_and_blocks(self):
        ctx = ProposalContext.model_validate(_proposal_context_dict())
        saqp, csm = extract_artifacts(ctx)
        prompt = build_proposals_prompt(
            "Is coverage sufficient?", "The answer.", "SITE", "KB", ctx, saqp, csm
        )
        assert 'id="pt-1"' in prompt
        assert 'id="smp-1"' in prompt
        assert 'id="l1"' in prompt
        assert "origin=generated" in prompt
        assert "Groundwater, Soil" in prompt
        assert "Is coverage sufficient?" in prompt
        assert "The answer." in prompt

    def test_prompt_omits_unqualified_artifact(self):
        payload = _proposal_context_dict()
        del payload["csm"]["updatedAt"]
        ctx = ProposalContext.model_validate(payload)
        saqp, csm = extract_artifacts(ctx)
        prompt = build_proposals_prompt("q", "a", "s", "k", ctx, saqp, csm)
        assert csm is None
        assert 'id="l1"' not in prompt


def _run(coro):
    return asyncio.run(coro)


class TestGenerateProposals:
    def test_happy_path(self):
        ctx = ProposalContext.model_validate(_proposal_context_dict())

        async def complete(system_prompt, prompt):
            return json.dumps({"proposals": [_grid_candidate()]})

        result = _run(generate_proposals("q", "a", "site", "kb", ctx, complete))
        assert len(result) == 1
        assert result[0]["operation"] == "saqp.set_grid_parameters"

    def test_invalid_candidates_dropped_valid_kept(self):
        ctx = ProposalContext.model_validate(_proposal_context_dict())

        async def complete(system_prompt, prompt):
            return json.dumps(
                {
                    "proposals": [
                        _grid_candidate(operation="saqp.delete_point"),
                        _grid_candidate(),
                        "not even an object",
                    ]
                }
            )

        result = _run(generate_proposals("q", "a", "site", "kb", ctx, complete))
        assert len(result) == 1

    def test_cap_at_three(self):
        ctx = ProposalContext.model_validate(_proposal_context_dict())

        async def complete(system_prompt, prompt):
            return json.dumps({"proposals": [_grid_candidate() for _ in range(6)]})

        result = _run(generate_proposals("q", "a", "site", "kb", ctx, complete))
        assert len(result) == MAX_PROPOSALS_PER_ANSWER

    def test_completion_failure_yields_empty(self):
        ctx = ProposalContext.model_validate(_proposal_context_dict())

        async def complete(system_prompt, prompt):
            raise RuntimeError("upstream down")

        result = _run(generate_proposals("q", "a", "site", "kb", ctx, complete))
        assert result == []

    def test_no_qualified_artifacts_skips_completion(self):
        ctx = ProposalContext.model_validate({"saqp": {"planId": "plan-1"}})
        calls = []

        async def complete(system_prompt, prompt):
            calls.append(1)
            return "{}"

        result = _run(generate_proposals("q", "a", "site", "kb", ctx, complete))
        assert result == []
        assert calls == []
