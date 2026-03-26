"""
Tests for the optional structured workspace context on POST /query.

Covers: contract validation, guardrails, grounding prompt assembly,
session behaviour, and the separation between context and question.
"""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from context_models import (
    WorkspaceContext,
    ProjectInfo,
    CriteriaInfo,
    RetrievalHints,
    FieldSummary,
    Exceedance,
    RetrievedRow,
    AnalyteValue,
    Coordinates,
    ConversationMessage,
    RelevantDetail,
    CriterionThreshold,
    build_grounding_prompt,
    MAX_CONTEXT_PAYLOAD_BYTES,
)


# ── Helpers ─────────────────────────────────────────────────────

def _minimal_context() -> dict:
    """Minimal valid context payload."""
    return {
        "schemaVersion": 1,
        "generatedAtIso": "2026-03-26T10:00:00.000Z",
        "project": {
            "projectName": "Project One",
            "projectId": "project-1",
        },
    }


def _full_context() -> dict:
    """Full context payload matching the frontend spec."""
    return {
        "schemaVersion": 1,
        "generatedAtIso": "2026-03-26T10:00:00.000Z",
        "project": {
            "projectName": "Project One",
            "projectId": "project-1",
            "siteName": "Site One",
            "labReportNumber": "LAB-001",
            "projectType": "soil",
            "sourceFile": "report.xlsx",
            "totalSamples": 24,
            "totalAnalytes": 18,
        },
        "retrieval": {
            "matchedAnalytes": ["Lead"],
            "matchedSampleCodes": ["BH-01"],
            "questionTokens": ["lead", "bh", "01"],
        },
        "criteria": {
            "applicableCriteria": "NEPM 2013",
            "landUse": "Residential",
            "state": "QLD",
            "regulations": ["NEPM 2013"],
            "relevantDetails": [
                {
                    "name": "HIL-A",
                    "thresholds": [
                        {"analyte": "Lead", "value": 300, "unit": "mg/kg"}
                    ],
                }
            ],
        },
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
        "exceedances": [
            {
                "analyte": "Lead",
                "sampleCode": "BH-01",
                "criterion": "HIL-A",
                "value": 720,
                "criterionValue": 300,
                "exceedanceFactor": 2.4,
                "unit": "mg/kg",
            }
        ],
        "retrievedRows": [
            {
                "sampleCode": "BH-01",
                "depth": "0-0.5m",
                "collectionDate": "2026-03-24",
                "sampleType": "soil",
                "labName": "ALS",
                "coordinates": {"lat": -27.5, "lng": 153.0},
                "analyteValues": [
                    {"analyte": "Lead", "value": 720, "unit": "mg/kg"}
                ],
            }
        ],
        "conversation": [
            {"role": "user", "content": "Summarise this project"},
            {"role": "assistant", "content": "This project covers..."},
        ],
    }


# ── Fixture: patched FastAPI TestClient ─────────────────────────

@pytest.fixture()
def client():
    """
    Create a TestClient with the RAG dependency mocked out so we
    can test the HTTP layer without needing LightRAG / embeddings.
    """
    # We need to mock the heavy imports before importing server
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        # Mock the heavy dependencies that server.py imports at module level
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="mocked answer")

        import server
        server.app.router.on_startup.clear()
        server.app.router.on_shutdown.clear()
        server.rag = mock_rag
        # Clear sessions between tests
        server.sessions.clear()

        with TestClient(server.app, raise_server_exceptions=False) as c:
            yield c

        server.sessions.clear()


# ── 1. Legacy request without context returns success ───────────

class TestLegacyRequest:
    def test_legacy_no_context(self, client):
        resp = client.post("/query", json={
            "question": "What are the soil guidelines?",
            "mode": "hybrid",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "mocked answer"
        assert body["context_used"] is False
        assert "session_id" in body

    def test_legacy_with_session(self, client):
        resp1 = client.post("/query", json={
            "question": "First question",
        })
        sid = resp1.json()["session_id"]

        resp2 = client.post("/query", json={
            "question": "Follow-up",
            "session_id": sid,
        })
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == sid
        assert resp2.json()["context_used"] is False


# ── 2. Request with valid context returns success ───────────────

class TestContextRequest:
    def test_minimal_context(self, client):
        resp = client.post("/query", json={
            "question": "What are the exceedances?",
            "context": _minimal_context(),
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "mocked answer"
        assert body["context_used"] is True

    def test_full_context(self, client):
        resp = client.post("/query", json={
            "question": "Summarise exceedances for Lead",
            "context": _full_context(),
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["context_used"] is True

    def test_empty_context_object(self, client):
        """An empty context {} is valid but produces no grounding."""
        resp = client.post("/query", json={
            "question": "Hello",
            "context": {},
        })
        assert resp.status_code == 200
        assert resp.json()["context_used"] is False

    def test_context_null_is_legacy(self, client):
        resp = client.post("/query", json={
            "question": "Hello",
            "context": None,
        })
        assert resp.status_code == 200
        assert resp.json()["context_used"] is False


# ── 3. Invalid context shape is rejected ────────────────────────

class TestInvalidContext:
    def test_context_string_rejected(self, client):
        """context must be an object, not a string."""
        resp = client.post("/query", json={
            "question": "Hello",
            "context": "not an object",
        })
        assert resp.status_code == 422

    def test_context_array_rejected(self, client):
        """context must be an object, not an array."""
        resp = client.post("/query", json={
            "question": "Hello",
            "context": [1, 2, 3],
        })
        assert resp.status_code == 422

    def test_context_number_rejected(self, client):
        resp = client.post("/query", json={
            "question": "Hello",
            "context": 42,
        })
        assert resp.status_code == 422


# ── 4. Oversize context is rejected ─────────────────────────────

class TestOversizeContext:
    def test_oversize_exceedances(self, client):
        """Context with huge exceedances list should be rejected."""
        ctx = _minimal_context()
        ctx["exceedances"] = [
            {
                "analyte": f"Analyte-{i}",
                "sampleCode": f"BH-{i:04d}",
                "criterion": "HIL-A",
                "value": 999.9,
                "criterionValue": 300,
                "exceedanceFactor": 3.3,
                "unit": "mg/kg",
                "extraData": "x" * 500,
            }
            for i in range(500)
        ]
        resp = client.post("/query", json={
            "question": "Check exceedances",
            "context": ctx,
        })
        # Should be 422 because the request exceeds server-side context bounds.
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "at most 8 items" in json.dumps(detail)

    def test_exceedances_list_max_length(self, client):
        """Pydantic rejects lists exceeding max_length."""
        ctx = _minimal_context()
        ctx["exceedances"] = [
            {"analyte": f"A-{i}", "value": i}
            for i in range(501)  # exceeds MAX_EXCEEDANCES=500
        ]
        resp = client.post("/query", json={
            "question": "Check",
            "context": ctx,
        })
        assert resp.status_code == 422


# ── 5. Session behaviour unchanged with/without context ────────

class TestSessionBehaviour:
    def test_session_persists_with_context(self, client):
        resp1 = client.post("/query", json={
            "question": "First",
            "context": _full_context(),
        })
        sid = resp1.json()["session_id"]

        resp2 = client.post("/query", json={
            "question": "Second",
            "session_id": sid,
        })
        assert resp2.json()["session_id"] == sid

    def test_session_reset_after_max_exchanges(self, client):
        """Session auto-resets after MAX_EXCHANGES regardless of context."""
        import server
        sid = None
        for i in range(server.MAX_EXCHANGES):
            payload = {"question": f"Q{i}"}
            if sid:
                payload["session_id"] = sid
            if i == 0:
                payload["context"] = _minimal_context()
            resp = client.post("/query", json=payload)
            assert resp.status_code == 200
            sid = resp.json()["session_id"]

        # The last response should have session_reset=True
        assert resp.json()["session_reset"] is True

    def test_session_id_returned_without_context(self, client):
        resp = client.post("/query", json={"question": "Hi"})
        assert "session_id" in resp.json()
        assert isinstance(resp.json()["session_id"], str)

    def test_session_id_returned_with_context(self, client):
        resp = client.post("/query", json={
            "question": "Hi",
            "context": _minimal_context(),
        })
        assert "session_id" in resp.json()
        assert isinstance(resp.json()["session_id"], str)


# ── 6. Context passed as structured grounding, not in question ──

class TestContextSeparation:
    def test_context_passed_as_system_prompt(self, client):
        """
        When context is provided, it must be passed via system_prompt
        to rag.aquery, NOT concatenated into the question string.
        """
        import server
        mock_rag = server.rag

        client.post("/query", json={
            "question": "What are the exceedances?",
            "context": _full_context(),
        })

        # Verify aquery was called
        mock_rag.aquery.assert_called_once()
        call_kwargs = mock_rag.aquery.call_args

        # The first positional arg is the query text
        query_text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("enhanced_query", "")

        # system_prompt should be set (context grounding)
        system_prompt = call_kwargs.kwargs.get("system_prompt")
        assert system_prompt is not None
        assert "Project One" in system_prompt
        assert "Lead" in system_prompt

        # The question text should NOT contain raw JSON context
        assert '"schemaVersion"' not in query_text
        assert '"projectName"' not in query_text

    def test_no_system_prompt_without_context(self, client):
        """Without context, system_prompt should be None."""
        import server
        mock_rag = server.rag
        mock_rag.aquery.reset_mock()

        client.post("/query", json={
            "question": "General question",
        })

        call_kwargs = mock_rag.aquery.call_args
        system_prompt = call_kwargs.kwargs.get("system_prompt")
        assert system_prompt is None


# ── 7. Unknown additive fields don't break the request ──────────

class TestForwardCompatibility:
    def test_unknown_top_level_field(self, client):
        ctx = _minimal_context()
        ctx["futureField"] = {"data": "something new"}
        resp = client.post("/query", json={
            "question": "Hello",
            "context": ctx,
        })
        assert resp.status_code == 200

    def test_unknown_nested_field(self, client):
        ctx = _minimal_context()
        ctx["project"]["newMetric"] = 42
        resp = client.post("/query", json={
            "question": "Hello",
            "context": ctx,
        })
        assert resp.status_code == 200

    def test_unknown_exceedance_field(self, client):
        ctx = _full_context()
        ctx["exceedances"][0]["newFlag"] = True
        resp = client.post("/query", json={
            "question": "Hello",
            "context": ctx,
        })
        assert resp.status_code == 200


# ── 8. Grounding prompt unit tests (no server needed) ───────────

class TestBuildGroundingPrompt:
    def test_empty_context(self):
        ctx = WorkspaceContext()
        assert build_grounding_prompt(ctx) == ""

    def test_project_only(self):
        ctx = WorkspaceContext(
            project=ProjectInfo(projectName="Test", siteName="Site A")
        )
        prompt = build_grounding_prompt(ctx)
        assert "Project: Test" in prompt
        assert "Site: Site A" in prompt

    def test_exceedances_rendered(self):
        ctx = WorkspaceContext(
            exceedances=[
                Exceedance(
                    analyte="Lead",
                    sampleCode="BH-01",
                    value=720,
                    criterion="HIL-A",
                    criterionValue=300,
                    exceedanceFactor=2.4,
                    unit="mg/kg",
                )
            ]
        )
        prompt = build_grounding_prompt(ctx)
        assert "Lead" in prompt
        assert "BH-01" in prompt
        assert "720" in prompt
        assert "HIL-A" in prompt

    def test_criteria_rendered(self):
        ctx = WorkspaceContext(
            criteria=CriteriaInfo(
                applicableCriteria="NEPM 2013",
                landUse="Residential",
                state="QLD",
                regulations=["NEPM 2013"],
                relevantDetails=[
                    RelevantDetail(
                        name="HIL-A",
                        thresholds=[
                            CriterionThreshold(analyte="Lead", value=300, unit="mg/kg")
                        ],
                    )
                ],
            )
        )
        prompt = build_grounding_prompt(ctx)
        assert "NEPM 2013" in prompt
        assert "Residential" in prompt
        assert "QLD" in prompt
        assert "HIL-A" in prompt

    def test_field_summary_rendered(self):
        ctx = WorkspaceContext(
            fieldSummary=FieldSummary(
                hasFieldData=True,
                boreholeCount=3,
                depthRange="0-3 m",
            )
        )
        prompt = build_grounding_prompt(ctx)
        assert "Boreholes: 3" in prompt
        assert "0-3 m" in prompt

    def test_retrieved_rows_rendered(self):
        ctx = WorkspaceContext(
            retrievedRows=[
                RetrievedRow(
                    sampleCode="BH-01",
                    depth="0-0.5m",
                    analyteValues=[
                        AnalyteValue(analyte="Lead", value=720, unit="mg/kg")
                    ],
                )
            ]
        )
        prompt = build_grounding_prompt(ctx)
        assert "BH-01" in prompt
        assert "Lead=720" in prompt

    def test_retrieval_hints_rendered(self):
        ctx = WorkspaceContext(
            retrieval=RetrievalHints(
                matchedAnalytes=["Lead", "Arsenic"],
                matchedSampleCodes=["BH-01"],
            )
        )
        prompt = build_grounding_prompt(ctx)
        assert "Lead" in prompt
        assert "Arsenic" in prompt
        assert "BH-01" in prompt

    def test_full_context_grounding(self):
        """All sections should appear in the grounding prompt."""
        ctx = WorkspaceContext(**_full_context())
        prompt = build_grounding_prompt(ctx)
        assert "## Project" in prompt
        assert "## Regulatory Criteria" in prompt
        assert "## Field Summary" in prompt
        assert "## Exceedances" in prompt
        assert "## Retrieved Sample Data" in prompt
        assert "## Retrieval Hints" in prompt


# ── 9. Pydantic model validation unit tests ─────────────────────

class TestContextModelValidation:
    def test_valid_full_context(self):
        ctx = WorkspaceContext(**_full_context())
        assert ctx.schemaVersion == 1
        assert ctx.project.projectName == "Project One"
        assert len(ctx.exceedances) == 1
        assert ctx.exceedances[0].analyte == "Lead"

    def test_valid_empty_context(self):
        ctx = WorkspaceContext()
        assert ctx.project is None
        assert ctx.exceedances is None

    def test_extra_fields_allowed(self):
        ctx = WorkspaceContext(schemaVersion=1, unknownField="ok")
        assert ctx.schemaVersion == 1

    def test_exceedances_max_length(self):
        with pytest.raises(Exception):
            WorkspaceContext(
                exceedances=[Exceedance(analyte=f"A{i}") for i in range(501)]
            )

    def test_retrieved_rows_max_length(self):
        with pytest.raises(Exception):
            WorkspaceContext(
                retrievedRows=[RetrievedRow(sampleCode=f"S{i}") for i in range(501)]
            )

    def test_conversation_max_length(self):
        with pytest.raises(Exception):
            WorkspaceContext(
                conversation=[
                    ConversationMessage(role="user", content=f"msg{i}")
                    for i in range(101)
                ]
            )
