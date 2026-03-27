import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from context_models import (
    MAX_CONTEXT_PAYLOAD_BYTES,
    AnalyteValue,
    ConversationMessage,
    CriteriaDetail,
    CriterionThreshold,
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


@pytest.fixture()
def client():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        import server

        server.app.router.on_startup.clear()
        server.app.router.on_shutdown.clear()
        server.sessions.clear()

        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="mocked rag answer")
        mock_openai = AsyncMock()

        server.rag = mock_rag
        with patch.object(server, "openai_complete_if_cache", mock_openai):
            with TestClient(server.app, raise_server_exceptions=False) as test_client:
                yield test_client, server, mock_rag, mock_openai

        server.sessions.clear()


class TestLegacyRequest:
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
        assert "session_id" in body
        mock_openai.assert_not_awaited()
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "What are the soil guidelines?"

    def test_legacy_with_session(self, client):
        test_client, _, _, mock_openai = client

        first = test_client.post("/query", json={"question": "First question"})
        session_id = first.json()["session_id"]

        second = test_client.post(
            "/query",
            json={"question": "Follow-up", "session_id": session_id},
        )

        assert second.status_code == 200
        assert second.json()["session_id"] == session_id
        assert second.json()["context_used"] is False
        mock_openai.assert_not_awaited()


class TestContextRequest:
    def test_minimal_context_routes_successfully(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.return_value = json.dumps({"route": "kb_only"})

        response = test_client.post(
            "/query",
            json={"question": "What are the exceedances?", "context": _minimal_context()},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["context_used"] is True
        assert body["answer"] == "mocked rag answer"
        assert mock_openai.await_count == 1

    def test_full_context_routes_successfully(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.return_value = json.dumps(
            {
                "route": "blended",
                "project": {
                    "siteName": "Ducat",
                    "address": "1 Test Street, Brisbane QLD",
                    "projectType": "soil",
                    "labReportNumber": "LAB-001",
                },
                "selectedCriteria": {
                    "applicableCriteria": "NEPM 2013 HIL-A",
                    "landUse": "Residential",
                    "state": "QLD",
                    "criteriaNames": ["NEPM 2013 HIL-A"],
                },
                "criteria": {
                    "criteriaDetails": [
                        {
                            "name": "NEPM 2013 HIL-A",
                            "thresholds": [
                                {"analyte": "Lead", "value": 300, "unit": "mg/kg"}
                            ],
                        }
                    ]
                },
                "exceedanceSummary": {
                    "totalExceedances": 1,
                    "exceededCriteria": ["NEPM 2013 HIL-A"],
                    "affectedAnalytes": ["Lead"],
                },
                "exceedances": [
                    {
                        "analyte": "Lead",
                        "sampleCode": "BH-01",
                        "value": 720,
                        "unit": "mg/kg",
                        "criterion": "NEPM 2013 HIL-A",
                        "criterionValue": 300,
                    }
                ],
                "relevantSamples": [
                    {
                        "sampleCode": "BH-01",
                        "depth": "0-0.5m",
                        "analyteValues": [
                            {"analyte": "Lead", "value": 720, "unit": "mg/kg"}
                        ],
                    }
                ],
                "summary": "The user wants to understand the project's lead exceedance.",
            }
        )

        response = test_client.post(
            "/query",
            json={
                "question": "Summarise exceedances for Lead",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        assert response.json()["context_used"] is True
        rag_query = mock_rag.aquery.await_args.args[0]
        assert "Applicable criteria: NEPM 2013 HIL-A" in rag_query
        assert "Sample BH-01 (0-0.5m): Lead=720 mg/kg" in rag_query

    def test_empty_context_object_is_accepted_but_not_used(self, client):
        test_client, _, _, mock_openai = client

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": {}},
        )

        assert response.status_code == 200
        assert response.json()["context_used"] is False
        mock_openai.assert_not_awaited()

    def test_context_null_is_legacy(self, client):
        test_client, _, _, mock_openai = client

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": None},
        )

        assert response.status_code == 200
        assert response.json()["context_used"] is False
        mock_openai.assert_not_awaited()


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
        mock_openai.return_value = json.dumps({"route": "kb_only"})

        first = test_client.post(
            "/query",
            json={"question": "First", "context": _full_context()},
        )
        session_id = first.json()["session_id"]

        second = test_client.post(
            "/query",
            json={"question": "Second", "session_id": session_id},
        )

        assert second.json()["session_id"] == session_id

    def test_session_reset_after_max_exchanges(self, client):
        test_client, server, _, mock_openai = client
        mock_openai.return_value = json.dumps({"route": "kb_only"})

        session_id = None
        for index in range(server.MAX_EXCHANGES):
            payload = {"question": f"Q{index}"}
            if session_id:
                payload["session_id"] = session_id
            if index == 0:
                payload["context"] = _minimal_context()

            response = test_client.post("/query", json=payload)
            assert response.status_code == 200
            session_id = response.json()["session_id"]

        assert response.json()["session_reset"] is True


class TestRoutingBehaviour:
    def test_direct_criterion_lookup_uses_project_context_threshold(self, client):
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
        answer = response.json()["answer"]
        assert "0.5 mg/kg" in answer
        assert "0.6 mg/kg" not in answer
        assert "0.1 mg/kg" not in answer
        mock_openai.assert_not_awaited()
        mock_rag.aquery.assert_not_awaited()

    def test_project_only_bypasses_rag(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.side_effect = [
            json.dumps(
                {
                    "route": "project_only",
                    "project": {"projectName": "Project One"},
                    "selectedCriteria": {"applicableCriteria": "NEPM 2013 HIL-A"},
                }
            ),
            "The selected criteria are NEPM 2013 HIL-A.",
        ]

        response = test_client.post(
            "/query",
            json={"question": "What criteria have I selected?", "context": _full_context()},
        )

        assert response.status_code == 200
        assert response.json()["answer"] == "The selected criteria are NEPM 2013 HIL-A."
        mock_rag.aquery.assert_not_awaited()
        assert mock_openai.await_count == 2

    def test_kb_only_uses_plain_question(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.return_value = json.dumps({"route": "kb_only"})

        response = test_client.post(
            "/query",
            json={
                "question": "How did they evaluate the HIL values in the NEPM?",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        rag_query = mock_rag.aquery.await_args.args[0]
        assert rag_query == "How did they evaluate the HIL values in the NEPM?"
        assert '"projectState"' not in rag_query

    def test_blended_query_carries_project_and_criteria_context(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.return_value = json.dumps(
            {
                "route": "blended",
                "project": {
                    "siteName": "Ducat",
                    "projectType": "soil",
                    "labReportNumber": "LAB-001",
                },
                "selectedCriteria": {
                    "applicableCriteria": "NEPM 2013 HIL-A",
                    "landUse": "Residential",
                    "state": "QLD",
                    "criteriaNames": ["NEPM 2013 HIL-A"],
                },
                "criteria": {
                    "criteriaDetails": [
                        {
                            "name": "NEPM 2013 HIL-A",
                            "thresholds": [
                                {"analyte": "Lead", "value": 300, "unit": "mg/kg"}
                            ],
                        }
                    ]
                },
                "exceedanceSummary": {
                    "totalExceedances": 1,
                    "exceededCriteria": ["NEPM 2013 HIL-A"],
                    "affectedAnalytes": ["Lead"],
                },
                "exceedances": [
                    {
                        "analyte": "Lead",
                        "sampleCode": "BH-01",
                        "value": 720,
                        "unit": "mg/kg",
                        "criterion": "NEPM 2013 HIL-A",
                        "criterionValue": 300,
                    }
                ],
                "relevantSamples": [
                    {
                        "sampleCode": "BH-01",
                        "depth": "0-0.5m",
                        "analyteValues": [
                            {"analyte": "Lead", "value": 720, "unit": "mg/kg"}
                        ],
                    }
                ],
                "summary": "The user wants to know which applied guideline is exceeded.",
            }
        )

        response = test_client.post(
            "/query",
            json={
                "question": "Which applied guideline have I exceeded?",
                "context": _full_context(),
            },
        )

        assert response.status_code == 200
        rag_query = mock_rag.aquery.await_args.args[0]
        assert "Site: Ducat" in rag_query
        assert "Applicable criteria: NEPM 2013 HIL-A" in rag_query
        assert "Criterion NEPM 2013 HIL-A: Lead=300 mg/kg" in rag_query
        assert "Lead at 720 mg/kg in BH-01" in rag_query

    def test_invalid_extraction_json_falls_back_to_blended_summary(self, client):
        test_client, _, mock_rag, mock_openai = client
        mock_openai.return_value = "not valid json"

        response = test_client.post(
            "/query",
            json={"question": "Which applied guideline have I exceeded?", "context": _full_context()},
        )

        assert response.status_code == 200
        rag_query = mock_rag.aquery.await_args.args[0]
        assert "Which applied guideline have I exceeded?" in rag_query
        assert "not valid json" in rag_query


class TestForwardCompatibility:
    def test_unknown_top_level_field_is_accepted(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.return_value = json.dumps({"route": "kb_only"})

        context = _minimal_context()
        context["futureField"] = {"data": "something new"}

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": context},
        )

        assert response.status_code == 200

    def test_unknown_nested_field_is_accepted(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.return_value = json.dumps({"route": "kb_only"})

        context = _minimal_context()
        context["projectState"]["project"]["newMetric"] = 42

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": context},
        )

        assert response.status_code == 200

    def test_unknown_exceedance_field_is_accepted(self, client):
        test_client, _, _, mock_openai = client
        mock_openai.return_value = json.dumps({"route": "kb_only"})

        context = _full_context()
        context["projectState"]["exceedances"][0]["newFlag"] = True

        response = test_client.post(
            "/query",
            json={"question": "Hello", "context": context},
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

    def test_full_context_grounding(self):
        prompt = build_grounding_prompt(WorkspaceContext(**_full_context()))

        assert "## Project" in prompt
        assert "## Selected Criteria" in prompt
        assert "## Exceedance Summary" in prompt
        assert "## Exceedances" in prompt
        assert "## Project Results" in prompt
        assert "## Retrieval Context" in prompt


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
