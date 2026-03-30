import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from context_models import (
    MAX_CONTEXT_PAYLOAD_BYTES,
    AnalyteValue,
    ConversationMessage,
    Exceedance,
    ExceedanceSummary,
    FieldSummary,
    ProjectInfo,
    ProjectResultRow,
    ProjectState,
    RetrievalContext,
    SelectedCriteria,
    WorkspaceContext,
    build_grounding_prompt,
)


def _sample_grounding_payload() -> dict:
    return {
        "status": "success",
        "message": "Query executed successfully",
        "data": {
            "entities": [],
            "relationships": [],
            "references": [
                {
                    "reference_id": "ref-1",
                    "file_path": "/kb/NEPM_2013.pdf",
                }
            ],
            "chunks": [
                {
                    "reference_id": "ref-1",
                    "file_path": "/kb/NEPM_2013.pdf",
                    "chunk_id": "page_123_chunk_1",
                    "content": (
                        "Table 1B(7) shows TRH C6-C10 and related hydrocarbon fractions "
                        "should be assessed against the selected land use criteria in the NEPM."
                    ),
                }
            ],
        },
        "metadata": {"mode": "hybrid"},
    }


def _minimal_context() -> dict:
    return {
        "schemaVersion": 3,
        "generatedAtIso": "2026-03-27T10:00:00.000Z",
        "projectState": {
            "project": {
                "projectName": "Project One",
                "projectId": "project-1",
                "projectType": "soil",
                "totalSamples": 2,
                "totalAnalytes": 2,
            }
        },
    }


def _full_context() -> dict:
    return {
        "schemaVersion": 3,
        "generatedAtIso": "2026-03-27T10:00:00.000Z",
        "projectState": {
            "project": {
                "projectName": "Project One",
                "projectId": "project-1",
                "siteName": "Ducat",
                "address": "1 Test Street, Brisbane QLD",
                "labReportNumber": "LAB-001",
                "projectType": "soil",
                "sourceFile": "report.xlsx",
                "totalSamples": 2,
                "totalAnalytes": 2,
            },
            "selectedCriteria": {
                "applicableCriteria": "NEPM 2013 HIL-A",
                "regulations": ["NEPM 2013"],
                "landUse": "Residential",
                "state": "QLD",
                "criteriaNames": ["NEPM 2013 HIL-A"],
                "criteriaCount": 1,
            },
            "criteriaDetails": [
                {
                    "name": "NEPM 2013 HIL-A",
                    "thresholds": [
                        {"analyte": "Lead", "value": 300, "unit": "mg/kg"},
                        {"analyte": "PFOS", "value": 0.07, "unit": "mg/kg"},
                    ],
                }
            ],
            "exceedanceSummary": {
                "totalExceedances": 1,
                "affectedSamples": ["BH-01"],
                "affectedAnalytes": ["Lead"],
                "exceededCriteria": ["NEPM 2013 HIL-A"],
                "hotspotCount": 0,
            },
            "exceedances": [
                {
                    "analyte": "Lead",
                    "sampleCode": "BH-01",
                    "date": "2026-03-26",
                    "criterion": "NEPM 2013 HIL-A",
                    "value": 720,
                    "criterionValue": 300,
                    "exceedanceFactor": 2.4,
                    "isHotspot": False,
                    "unit": "mg/kg",
                }
            ],
            "projectResults": [
                {
                    "sampleCode": "BH-01",
                    "depth": "0-0.5m",
                    "collectionDate": "2026-03-26",
                    "sampleType": "soil",
                    "labName": "ALS",
                    "labReportNumber": "LAB-001",
                    "coordinates": {"lat": -27.5, "lng": 153.0},
                    "analyteValues": [
                        {"analyte": "Lead", "value": 720, "unit": "mg/kg"},
                        {"analyte": "PFOS", "value": 0.02, "unit": "mg/kg"},
                    ],
                },
                {
                    "sampleCode": "BH-02",
                    "depth": "0.5-1.0m",
                    "collectionDate": "2026-03-26",
                    "sampleType": "soil",
                    "labName": "ALS",
                    "labReportNumber": "LAB-001",
                    "analyteValues": [
                        {"analyte": "Lead", "value": 120, "unit": "mg/kg"}
                    ],
                },
            ],
            "fieldSummary": {
                "hasFieldData": True,
                "sessionCount": 2,
                "boreholeCount": 3,
                "fieldSampleCount": 4,
                "lithologyLogCount": 1,
                "latestSessionDate": "2026-03-25",
                "sampleTypes": ["soil"],
                "depthRange": "0-3 m",
                "hasGpsData": True,
            },
        },
        "retrievalContext": {
            "matchedAnalytes": ["Lead"],
            "matchedSampleCodes": ["BH-01"],
            "questionTokens": ["lead", "bh", "01"],
            "retrievedRows": [
                {
                    "sampleCode": "BH-01",
                    "depth": "0-0.5m",
                    "analyteValues": [
                        {"analyte": "Lead", "value": 720, "unit": "mg/kg"}
                    ],
                }
            ],
        },
        "conversation": [
            {"role": "user", "content": "Summarise this project"},
            {"role": "assistant", "content": "This project covers two samples."},
        ],
    }


def _flat_context() -> dict:
    context = _full_context()
    return {
        "schemaVersion": context["schemaVersion"],
        "generatedAtIso": context["generatedAtIso"],
        "project": context["projectState"]["project"],
        "selectedCriteria": context["projectState"]["selectedCriteria"],
        "criteriaDetails": context["projectState"]["criteriaDetails"],
        "exceedanceSummary": context["projectState"]["exceedanceSummary"],
        "exceedances": context["projectState"]["exceedances"],
        "projectResults": context["projectState"]["projectResults"],
        "fieldSummary": context["projectState"]["fieldSummary"],
        "matchedAnalytes": context["retrievalContext"]["matchedAnalytes"],
        "matchedSampleCodes": context["retrievalContext"]["matchedSampleCodes"],
        "questionTokens": context["retrievalContext"]["questionTokens"],
        "retrievedRows": context["retrievalContext"]["retrievedRows"],
        "conversation": context["conversation"],
    }


def _benzene_hsl_context() -> dict:
    return {
        "schemaVersion": 3,
        "generatedAtIso": "2026-03-27T10:00:00.000Z",
        "projectState": {
            "project": {
                "projectName": "Project Benzene",
                "projectId": "project-benzene",
                "projectType": "soil",
            },
            "selectedCriteria": {
                "applicableCriteria": "EPM 2013 HSL-A Low Density Residential Sand (0m to <1m)",
                "regulations": ["NEPM 2013"],
                "landUse": "Low Density Residential",
                "state": "QLD",
                "criteriaNames": [
                    "EPM 2013 HSL-A Low Density Residential Sand (0m to <1m)"
                ],
                "criteriaCount": 1,
            },
            "criteriaDetails": [
                {
                    "name": "EPM 2013 HSL-A Low Density Residential Sand (0m to <1m)",
                    "thresholds": [
                        {"analyte": "Benzene", "value": 0.5, "unit": "mg/kg"}
                    ],
                }
            ],
        },
        "retrievalContext": {
            "matchedAnalytes": ["Benzene"],
            "questionTokens": ["benzene", "hsl-a"],
        },
    }


def _benzene_eil_context() -> dict:
    return {
        "schemaVersion": 3,
        "generatedAtIso": "2026-03-27T10:00:00.000Z",
        "projectState": {
            "project": {
                "projectName": "Project EIL",
                "projectId": "project-eil",
                "projectType": "soil",
            },
            "selectedCriteria": {
                "applicableCriteria": "EIL Freshwater Investigation Level",
                "regulations": ["NEPM 2013"],
                "landUse": "Ecological",
                "state": "QLD",
                "criteriaNames": ["EIL Freshwater Investigation Level"],
                "criteriaCount": 1,
            },
            "criteriaDetails": [
                {
                    "name": "EIL Freshwater Investigation Level",
                    "thresholds": [
                        {"analyte": "Benzene", "value": 0.08, "unit": "mg/L"}
                    ],
                }
            ],
        },
        "retrievalContext": {
            "matchedAnalytes": ["Benzene"],
            "questionTokens": ["benzene", "eil"],
        },
    }


def _trh_context() -> dict:
    return {
        "schemaVersion": 3,
        "generatedAtIso": "2026-03-27T10:00:00.000Z",
        "projectState": {
            "project": {
                "projectName": "Hydrocarbon Project",
                "projectId": "project-trh",
                "siteName": "Depot Site",
                "projectType": "soil",
                "labReportNumber": "LAB-TRH",
            },
            "selectedCriteria": {
                "applicableCriteria": "NEPM 2013 HSL-A",
                "regulations": ["NEPM 2013"],
                "landUse": "Commercial",
                "state": "QLD",
                "criteriaNames": ["TRH C6-C10 less BTEX", "F1"],
                "criteriaCount": 2,
            },
            "criteriaDetails": [
                {
                    "name": "NEPM 2013 HSL-A",
                    "thresholds": [
                        {"analyte": "TRH C6-C10", "value": 100, "unit": "mg/kg"},
                        {
                            "analyte": "TRH C6-C10 less BTEX",
                            "value": 90,
                            "unit": "mg/kg",
                        },
                        {"analyte": "BTEX", "value": 15, "unit": "mg/kg"},
                        {"analyte": "F1", "value": 75, "unit": "mg/kg"},
                    ],
                }
            ],
            "exceedanceSummary": {
                "totalExceedances": 2,
                "affectedSamples": ["BH-TRH-01"],
                "affectedAnalytes": ["TRH C6-C10", "TRH C6-C10 less BTEX"],
                "exceededCriteria": ["NEPM 2013 HSL-A"],
                "hotspotCount": 1,
            },
            "exceedances": [
                {
                    "analyte": "TRH C6-C10",
                    "sampleCode": "BH-TRH-01",
                    "criterion": "NEPM 2013 HSL-A",
                    "value": 380,
                    "criterionValue": 100,
                    "unit": "mg/kg",
                }
            ],
            "projectResults": [
                {
                    "sampleCode": "BH-TRH-01",
                    "depth": "0-0.5m",
                    "analyteValues": [
                        {"analyte": "TRH C6-C10", "value": 380, "unit": "mg/kg"},
                        {
                            "analyte": "TRH C6-C10 less BTEX",
                            "value": 320,
                            "unit": "mg/kg",
                        },
                        {"analyte": "BTEX", "value": 10, "unit": "mg/kg"},
                        {"analyte": "F1", "value": 90, "unit": "mg/kg"},
                    ],
                }
            ],
        },
        "retrievalContext": {
            "matchedAnalytes": [
                "TRH C6-C10",
                "TRH C6-C10 less BTEX",
                "BTEX",
                "F1",
            ],
            "matchedSampleCodes": ["BH-TRH-01"],
            "questionTokens": ["trh", "contamination"],
            "retrievedRows": [
                {
                    "sampleCode": "BH-TRH-01",
                    "depth": "0-0.5m",
                    "analyteValues": [
                        {"analyte": "TRH C6-C10", "value": 380, "unit": "mg/kg"},
                        {
                            "analyte": "TRH C6-C10 less BTEX",
                            "value": 320,
                            "unit": "mg/kg",
                        },
                    ],
                }
            ],
        },
    }


@pytest.fixture()
def client():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        import server

        server.app.router.on_startup.clear()
        server.app.router.on_shutdown.clear()
        server.sessions.clear()

        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="mocked rag answer")
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value=_sample_grounding_payload()
        )
        mock_openai = AsyncMock()

        server.rag = mock_rag
        with patch.object(server, "openai_complete_if_cache", mock_openai):
            with TestClient(server.app, raise_server_exceptions=False) as test_client:
                yield test_client, server, mock_rag, mock_openai

        server.sessions.clear()


class TestLegacyRequest:
    def test_context_bot_spec_is_loaded_from_repo_file(self, client):
        _, server, _, _ = client

        assert server.CONTEXT_BOT_SPEC_PATH.name == "context-bot-spec.md"
        assert server.CONTEXT_BOT_SPEC_PATH.exists()
        assert "# Context Bot" in server.CONTEXT_BOT_SPEC

    def test_legacy_no_context(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={"question": "What are the soil guidelines?", "mode": "hybrid"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "mocked rag answer"
        assert body["context_used"] is False
        assert body["route_used"] == "regulatory_only"
        assert body["grounded"] is True
        assert body["citations"]
        assert body["citations"][0]["source"] == "NEPM_2013.pdf"
        assert body["citations"][0]["locator"] == "Table 1B(7), p. 123"
        assert "session_id" in body
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        assert mock_rag.aquery.await_args.args[0] == "What are the soil guidelines?"
        assert "name the supporting table inline" in mock_rag.aquery.await_args.kwargs["user_prompt"]

    def test_legacy_with_session(self, client):
        test_client, _, mock_rag, mock_openai = client

        first = test_client.post("/query", json={"question": "First question"})
        session_id = first.json()["session_id"]

        second = test_client.post(
            "/query",
            json={"question": "Follow-up", "session_id": session_id},
        )

        assert second.status_code == 200
        assert second.json()["session_id"] == session_id
        assert second.json()["context_used"] is False
        assert second.json()["route_used"] == "regulatory_only"
        mock_openai.assert_not_awaited()
        assert mock_rag.aquery.await_count == 2


class TestContextCompatibility:
    def test_canonical_nested_context_request_works(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.return_value = "The selected criteria are NEPM 2013 HIL-A."

        response = test_client.post(
            "/query",
            json={
                "question": "What criteria have I selected?",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["context_used"] is True
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        assert body["citations"]
        assert body["answer"] == "For this site, the selected criterion is NEPM 2013 HIL-A.\n\nmocked rag answer"
        mock_rag.aquery.assert_awaited_once()
        assert mock_rag.aquery.await_args.args[0] == "What criteria have I selected?"
        mock_openai.assert_not_awaited()

    def test_flat_schema_context_normalises_for_endpoint(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.return_value = "The main exceedance is Lead in BH-01."

        response = test_client.post(
            "/query",
            json={
                "question": "What are the main exceedances?",
                "context": _flat_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["context_used"] is True
        assert body["route_used"] == "project_only"
        assert body["grounded"] is False
        assert body["citations"] == []
        mock_rag.aquery.assert_not_awaited()
        assert mock_openai.await_count == 1

    def test_workspace_context_model_normalises_flat_schema(self):
        context = WorkspaceContext(**_flat_context())

        assert context.projectState is not None
        assert context.projectState.project.projectName == "Project One"
        assert context.projectState.selectedCriteria.applicableCriteria == "NEPM 2013 HIL-A"
        assert context.retrievalContext.matchedAnalytes == ["Lead"]
        assert context.conversation[0].role == "user"

    def test_mixed_flat_and_nested_payload_dumps_canonically(self):
        mixed = _full_context()
        mixed["project"] = {"siteName": "Flat Override Site"}
        mixed["selectedCriteria"] = {"applicableCriteria": "Flat Duplicate Criteria"}
        mixed["matchedAnalytes"] = ["Flat Duplicate Lead"]

        context = WorkspaceContext(**mixed)
        dumped = context.model_dump(exclude_none=True)

        assert "project" not in dumped
        assert "selectedCriteria" not in dumped
        assert "matchedAnalytes" not in dumped
        assert dumped["projectState"]["project"]["siteName"] == "Ducat"
        assert dumped["projectState"]["selectedCriteria"]["applicableCriteria"] == "NEPM 2013 HIL-A"
        assert dumped["retrievalContext"]["matchedAnalytes"] == ["Lead"]

    def test_empty_context_object_is_accepted_but_not_used(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": {}},
        )

        assert response.status_code == 200
        assert response.json()["context_used"] is False
        assert response.json()["route_used"] == "regulatory_only"
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()

    def test_context_null_is_legacy(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": None},
        )

        assert response.status_code == 200
        assert response.json()["context_used"] is False
        assert response.json()["route_used"] == "regulatory_only"
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()


class TestRoutingBehaviour:
    def test_main_exceedances_forces_project_only(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.return_value = "The main exceedance is Lead in BH-01."

        response = test_client.post(
            "/query",
            json={
                "question": "What are the main exceedances?",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "project_only"
        assert body["grounded"] is False
        assert body["citations"] == []
        mock_rag.aquery.assert_not_awaited()
        assert mock_openai.await_count == 1

    def test_generic_guidance_question_routes_kb_only(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "Tell me about the sample density guidelines",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "regulatory_only"
        assert body["grounded"] is True
        assert body["citations"]
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        assert (
            mock_rag.aquery.await_args.args[0]
            == "Tell me about the sample density guidelines"
        )

    def test_unrelated_interpretive_question_stays_kb_only(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What are the disposal implications of asbestos?",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "regulatory_only"
        assert body["grounded"] is True
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        assert (
            mock_rag.aquery.await_args.args[0]
            == "What are the disposal implications of asbestos?"
        )

    def test_shared_regulation_query_stays_kb_only(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What does the NEPM say about asbestos?",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "regulatory_only"
        assert body["grounded"] is True
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "What does the NEPM say about asbestos?"

    def test_explicit_site_rejection_forces_regulatory_only(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "I dont care about the site, whats in the NEPM",
                "context": _benzene_hsl_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "regulatory_only"
        assert body["answer"] == "mocked rag answer"
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        assert mock_rag.aquery.await_args.args[0] == "I dont care about the site, whats in the NEPM"

    def test_generic_nepm_exceedance_criteria_question_stays_kb_only(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What NEPM exceedance criterias can you tell me the values for",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "regulatory_only"
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "What NEPM exceedance criterias can you tell me the values for"

    def test_trh_source_question_forces_blended(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What is the TRH contamination from?",
                "context": _trh_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        assert body["citations"]
        mock_rag.aquery.assert_awaited_once()
        rag_query = mock_rag.aquery.await_args.args[0]
        mock_openai.assert_not_awaited()
        assert rag_query == "What is the TRH contamination from?"
        assert body["answer"] == "For this site, the selected criterion is NEPM 2013 HSL-A.\n\nmocked rag answer"

    def test_nepm_analyte_question_hits_rag(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What does the NEPM say about TRH C6-C10?",
                "context": _trh_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        mock_rag.aquery.assert_awaited_once()
        rag_query = mock_rag.aquery.await_args.args[0]
        mock_openai.assert_not_awaited()
        assert rag_query == "What does the NEPM say about TRH C6-C10?"
        assert body["answer"] == "For this site, the selected criterion is NEPM 2013 HSL-A, with TRH C6-C10 at 100 mg/kg.\n\nmocked rag answer"

    def test_direct_criterion_lookup_bypasses_rag(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": (
                    "EPM 2013 HSL-A Low Density Residential Sand (0m to <1m) "
                    "- whats the benzene exceedance value?"
                ),
                "context": _benzene_hsl_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert "0.5 mg/kg" in body["answer"]
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        assert body["citations"]
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        assert mock_rag.aquery.await_args.args[0] == (
            "EPM 2013 HSL-A Low Density Residential Sand (0m to <1m) - whats the benzene exceedance value?"
        )

    def test_broad_hsl_question_uses_blended_not_single_applied_value(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What are the criteria values in the HSL for benzene in the NEPM all soil types?",
                "context": _benzene_hsl_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "What are the criteria values in the HSL for benzene in the NEPM all soil types?"
        assert body["answer"] == (
            "For this site, the selected criterion is EPM 2013 HSL-A Low Density Residential Sand (0m to <1m), with Benzene at 0.5 mg/kg.\n\n"
            "mocked rag answer"
        )

    def test_hybrid_query_sent_to_rag_is_clean_user_question_only(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What is the HSL for benzene?",
                "context": _benzene_hsl_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "hybrid"
        assert mock_rag.aquery.await_args.args[0] == "What is the HSL for benzene?"
        assert body["answer"] == (
            "For this site, the selected criterion is EPM 2013 HSL-A Low Density Residential Sand (0m to <1m), with Benzene at 0.5 mg/kg.\n\n"
            "mocked rag answer"
        )
        mock_openai.assert_not_awaited()

    def test_non_selected_soil_type_does_not_collapse_to_selected_hsl(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What is the exceedance threshold for benzene in clay as per the HSL?",
                "context": _benzene_hsl_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "What is the exceedance threshold for benzene in clay as per the HSL?"
        assert body["answer"] == (
            "For this site, the selected criterion is EPM 2013 HSL-A Low Density Residential Sand (0m to <1m), with Benzene at 0.5 mg/kg.\n\n"
            "mocked rag answer"
        )

    def test_eil_criterion_lookup_bypasses_rag(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What's the exceedance value for benzene in the EIL?",
                "context": _benzene_eil_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert "0.08 mg/L" in body["answer"]
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        assert body["citations"]
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        assert mock_rag.aquery.await_args.args[0] == "What's the exceedance value for benzene in the EIL?"

    def test_management_limit_lookup_bypasses_rag(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What's the management limit for benzene in the EIL?",
                "context": _benzene_eil_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert "0.08 mg/L" in body["answer"]
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        assert body["citations"]
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        assert mock_rag.aquery.await_args.args[0] == "What's the management limit for benzene in the EIL?"

    def test_investigation_level_lookup_bypasses_rag(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "What's the investigation level for benzene in the EIL?",
                "context": _benzene_eil_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert "0.08 mg/L" in body["answer"]
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_awaited_once()
        assert mock_rag.aquery.await_args.args[0] == "What's the investigation level for benzene in the EIL?"

    def test_invalid_extraction_json_falls_back_to_safe_blended_route(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.return_value = "not valid json"

        response = test_client.post(
            "/query",
            json={
                "question": "What is the TRH contamination from?",
                "context": _trh_context(),
            },
        )

        assert response.status_code == 200
        assert response.json()["route_used"] == "hybrid"
        rag_query = mock_rag.aquery.await_args.args[0]
        assert mock_openai.await_count == 0
        assert rag_query == "What is the TRH contamination from?"

    def test_project_hsl_guidance_routes_blended_with_selected_criteria(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "Tell me about the HSL criteria for this project",
                "context": _benzene_hsl_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        mock_openai.assert_not_awaited()
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "Tell me about the HSL criteria for this project"
        assert body["answer"] == (
            "For this site, the selected criterion is EPM 2013 HSL-A Low Density Residential Sand (0m to <1m).\n\n"
            "mocked rag answer"
        )

    def test_waste_classification_for_project_routes_blended(self, client):
        test_client, _, mock_rag, mock_openai = client

        response = test_client.post(
            "/query",
            json={
                "question": "Tell me about waste classification for this project",
                "context": _benzene_eil_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        mock_openai.assert_not_awaited()
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "Tell me about waste classification for this project"
        assert body["answer"] == "For this site, the selected criterion is EIL Freshwater Investigation Level.\n\nmocked rag answer"

    def test_scope_follow_up_keeps_previous_analyte_and_routes_blended(self, client):
        test_client, _, mock_rag, mock_openai = client

        first = test_client.post(
            "/query",
            json={
                "question": "What are the criteria values in the HSL for benzene?",
                "context": _benzene_hsl_context(),
            },
        )
        session_id = first.json()["session_id"]

        second = test_client.post(
            "/query",
            json={
                "question": "in the NEPM all soil types",
                "session_id": session_id,
                "context": _benzene_hsl_context(),
            },
        )

        assert second.status_code == 200
        body = second.json()
        assert body["route_used"] == "hybrid"
        assert body["grounded"] is True
        mock_openai.assert_not_awaited()
        assert mock_rag.aquery.await_count == 2
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "What are the criteria values in the HSL for benzene? in the NEPM all soil types"
        assert body["answer"] == (
            "For this site, the selected criterion is EPM 2013 HSL-A Low Density Residential Sand (0m to <1m), with Benzene at 0.5 mg/kg.\n\n"
            "mocked rag answer"
        )


class TestResponseContract:
    def test_grounded_metadata_and_citations_returned_for_kb_route(self, client):
        test_client, _, _, _ = client

        response = test_client.post(
            "/query",
            json={"question": "What are the soil guidelines?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "regulatory_only"
        assert body["grounded"] is True
        assert body["citations"]
        assert body["citations"][0] == {
            "source": "NEPM_2013.pdf",
            "title": "NEPM 2013",
            "locator": "Table 1B(7), p. 123",
            "snippet": (
                "Table 1B(7) shows TRH C6-C10 and related hydrocarbon fractions "
                "should be assessed against the selected land use criteria in the NEPM."
            ),
        }

    def test_project_only_metadata_returns_empty_citations(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.side_effect = [
            json.dumps({"route": "project_only"}),
            "The main exceedance is Lead in BH-01.",
        ]

        response = test_client.post(
            "/query",
            json={
                "question": "What are the main exceedances?",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["route_used"] == "project_only"
        assert body["grounded"] is False
        assert body["citations"] == []


class TestInvalidContext:
    def test_context_string_rejected(self, client):
        test_client, _, _, _ = client

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": "not an object"},
        )

        assert response.status_code == 422

    def test_context_array_rejected(self, client):
        test_client, _, _, _ = client

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": [1, 2, 3]},
        )

        assert response.status_code == 422

    def test_context_number_rejected(self, client):
        test_client, _, _, _ = client

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": 42},
        )

        assert response.status_code == 422


class TestOversizeContext:
    def test_oversize_context_is_rejected(self, client):
        test_client, _, _, _ = client

        huge_context = _full_context()
        huge_context["projectState"]["projectResults"] = [
            {
                "sampleCode": f"BH-{index:04d}",
                "analyteValues": [
                    {
                        "analyte": "Lead",
                        "value": "x" * 5000,
                        "unit": "mg/kg",
                    }
                ],
            }
            for index in range(150)
        ]

        raw_bytes = len(json.dumps(huge_context).encode("utf-8"))
        assert raw_bytes > MAX_CONTEXT_PAYLOAD_BYTES

        response = test_client.post(
            "/query",
            json={"question": "Check exceedances", "context": huge_context},
        )

        assert response.status_code == 422
        assert "context payload too large" in response.json()["detail"]


class TestSessionBehaviour:
    def test_session_persists_with_context(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.side_effect = [
            json.dumps({"route": "project_only"}),
            "The main exceedance is Lead in BH-01.",
        ]

        first = test_client.post(
            "/query",
            json={"question": "What are the main exceedances?", "context": _full_context()},
        )
        session_id = first.json()["session_id"]

        second = test_client.post(
            "/query",
            json={"question": "Second", "session_id": session_id},
        )

        assert second.json()["session_id"] == session_id

    def test_session_reset_after_max_exchanges(self, client):
        test_client, server, _, mock_openai = client
        mock_openai.side_effect = [
            json.dumps({"route": "project_only"}),
            "The main exceedance is Lead in BH-01.",
        ]

        session_id = None
        for index in range(server.MAX_EXCHANGES):
            payload = {"question": f"Q{index}"}
            if session_id:
                payload["session_id"] = session_id
            if index == 0:
                payload["context"] = _full_context()
                payload["question"] = "What are the main exceedances?"

            response = test_client.post("/query", json=payload)
            assert response.status_code == 200
            session_id = response.json()["session_id"]

        assert response.json()["session_reset"] is True


class TestForwardCompatibility:
    def test_unknown_top_level_field_is_accepted(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.side_effect = [
            json.dumps({"route": "project_only"}),
            "The main exceedance is Lead in BH-01.",
        ]

        context = _minimal_context()
        context["futureField"] = {"data": "something new"}
        context["projectState"]["exceedances"] = [
            {"analyte": "Lead", "sampleCode": "BH-01", "value": 720}
        ]

        response = test_client.post(
            "/query",
            json={"question": "What are the main exceedances?", "context": context},
        )

        assert response.status_code == 200

    def test_unknown_nested_field_is_accepted(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.side_effect = [
            json.dumps({"route": "project_only"}),
            "The main exceedance is Lead in BH-01.",
        ]

        context = _full_context()
        context["projectState"]["project"]["newMetric"] = 42

        response = test_client.post(
            "/query",
            json={"question": "What are the main exceedances?", "context": context},
        )

        assert response.status_code == 200

    def test_unknown_exceedance_field_is_accepted(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.side_effect = [
            json.dumps({"route": "project_only"}),
            "The main exceedance is Lead in BH-01.",
        ]

        context = _full_context()
        context["projectState"]["exceedances"][0]["newFlag"] = True

        response = test_client.post(
            "/query",
            json={"question": "What are the main exceedances?", "context": context},
        )

        assert response.status_code == 200


class TestBuildGroundingPrompt:
    def test_empty_context(self):
        assert build_grounding_prompt(WorkspaceContext()) == ""

    def test_project_section_renders(self):
        prompt = build_grounding_prompt(
            WorkspaceContext(
                projectState=ProjectState(
                    project=ProjectInfo(projectName="Test", siteName="Site A")
                )
            )
        )

        assert "## Project" in prompt
        assert "Project: Test" in prompt
        assert "Site: Site A" in prompt

    def test_selected_criteria_section_renders(self):
        prompt = build_grounding_prompt(
            WorkspaceContext(
                projectState=ProjectState(
                    selectedCriteria=SelectedCriteria(
                        applicableCriteria="NEPM 2013",
                        landUse="Residential",
                        state="QLD",
                        regulations=["NEPM 2013"],
                        criteriaNames=["HIL-A"],
                    )
                )
            )
        )

        assert "## Selected Criteria" in prompt
        assert "NEPM 2013" in prompt
        assert "Residential" in prompt
        assert "HIL-A" in prompt

    def test_exceedance_summary_and_rows_render(self):
        prompt = build_grounding_prompt(
            WorkspaceContext(
                projectState=ProjectState(
                    exceedanceSummary=ExceedanceSummary(
                        totalExceedances=1,
                        exceededCriteria=["HIL-A"],
                        affectedAnalytes=["Lead"],
                    ),
                    exceedances=[
                        Exceedance(
                            analyte="Lead",
                            sampleCode="BH-01",
                            value=720,
                            criterion="HIL-A",
                            unit="mg/kg",
                        )
                    ],
                )
            )
        )

        assert "## Exceedance Summary" in prompt
        assert "Total exceedances: 1" in prompt
        assert "## Exceedances" in prompt
        assert "Lead @ BH-01: 720 mg/kg against HIL-A" in prompt

    def test_project_results_and_retrieval_context_render(self):
        prompt = build_grounding_prompt(
            WorkspaceContext(
                projectState=ProjectState(
                    projectResults=[
                        ProjectResultRow(
                            sampleCode="BH-01",
                            depth="0-0.5m",
                            analyteValues=[
                                AnalyteValue(analyte="Lead", value=720, unit="mg/kg")
                            ],
                        )
                    ],
                    fieldSummary=FieldSummary(boreholeCount=3),
                ),
                retrievalContext=RetrievalContext(
                    matchedAnalytes=["Lead"],
                    matchedSampleCodes=["BH-01"],
                    retrievedRows=[ProjectResultRow(sampleCode="BH-01")],
                ),
            )
        )

        assert "## Project Results" in prompt
        assert "Lead=720 mg/kg" in prompt
        assert "## Retrieval Context" in prompt
        assert "Matched analytes: Lead" in prompt
        assert "Retrieved rows: 1" in prompt


class TestContextModelValidation:
    def test_valid_full_context(self):
        context = WorkspaceContext(**_full_context())

        assert context.schemaVersion == 3
        assert context.projectState.project.projectName == "Project One"
        assert len(context.projectState.exceedances) == 1
        assert context.projectState.exceedances[0].analyte == "Lead"
        assert context.retrievalContext.matchedAnalytes == ["Lead"]

    def test_valid_empty_context(self):
        context = WorkspaceContext()

        assert context.projectState is None
        assert context.retrievalContext is None

    def test_extra_fields_allowed(self):
        context = WorkspaceContext(schemaVersion=3, unknownField="ok")
        assert context.schemaVersion == 3

    def test_conversation_max_length(self):
        with pytest.raises(Exception):
            WorkspaceContext(
                conversation=[
                    ConversationMessage(role="user", content=f"msg{index}")
                    for index in range(21)
                ]
            )
