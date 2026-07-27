"""Endpoint tests for /query proposals[]: flag gating, happy path,
failure isolation. Uses the same mocked-rag TestClient pattern as
test_query_resilience.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _context_with_proposals() -> dict:
    return {
        "schemaVersion": 5,
        "projectState": {
            "project": {"projectName": "Proj", "projectId": "p1", "totalSamples": 1}
        },
        "proposalContext": {
            "saqp": {
                "planId": "plan-1",
                "planUpdatedAt": "2026-07-27T00:00:00.000Z",
                "points": [{"id": "pt-1", "label": "SP01"}],
                "samples": [{"id": "smp-1", "label": "BH01_0.5"}],
            }
        },
    }


def _valid_llm_output() -> str:
    return json.dumps(
        {
            "proposals": [
                {
                    "operation": "saqp.set_grid_parameters",
                    "payload": {"gridEnabled": True, "gridSizeM": 40},
                    "rationale": "Grid coverage per NEPM Sch B2.",
                    "citations": [{"source": "NEPM 2013 Sch B2"}],
                }
            ]
        }
    )


@pytest.fixture()
def client(tmp_path):
    made = []

    def factory(extra_env=None):
        base_env = {"OPENAI_API_KEY": "test-key", "RAG_AUTH_REQUIRED": "false"}
        base_env.update(extra_env or {})
        env_patch = patch.dict("os.environ", base_env)
        env_patch.start()
        import server

        server.app.router.on_startup.clear()
        server.app.router.on_shutdown.clear()
        server.sessions.clear()

        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="kb only answer")
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={"status": "success", "data": {"references": [], "chunks": []}}
        )
        answer_mock = AsyncMock(return_value="the grounded answer")
        proposals_mock = AsyncMock(return_value=_valid_llm_output())

        storage_patch = patch.object(server, "RAG_STORAGE", str(tmp_path))
        storage_patch.start()
        server.rag = mock_rag
        answer_patch = patch.object(server, "openai_complete_if_cache", answer_mock)
        answer_patch.start()
        proposals_patch = patch.object(
            server, "_complete_proposals_json", proposals_mock
        )
        proposals_patch.start()
        test_client = TestClient(server.app, raise_server_exceptions=False)
        made.append((server, [env_patch, storage_patch, answer_patch, proposals_patch]))
        return test_client, server, proposals_mock

    yield factory

    for server_mod, patches in made:
        server_mod.sessions.clear()
        for patcher in reversed(patches):
            patcher.stop()


def test_flag_off_returns_empty_and_skips_call(client):
    test_client, _, proposals_mock = client({"RAG_ENABLE_PROPOSALS": "false"})
    response = test_client.post(
        "/query",
        json={
            "question": "Is coverage sufficient?",
            "context": _context_with_proposals(),
        },
    )
    assert response.status_code == 200
    assert response.json()["proposals"] == []
    proposals_mock.assert_not_awaited()


def test_flag_on_emits_validated_proposals(client):
    test_client, _, proposals_mock = client({"RAG_ENABLE_PROPOSALS": "true"})
    response = test_client.post(
        "/query",
        json={
            "question": "Is coverage sufficient?",
            "context": _context_with_proposals(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "the grounded answer"
    assert len(body["proposals"]) == 1
    proposal = body["proposals"][0]
    assert proposal["operation"] == "saqp.set_grid_parameters"
    assert proposal["kind"] == "saqp"
    assert proposal["baseline"] == {
        "artifactId": "plan-1",
        "updatedAt": "2026-07-27T00:00:00.000Z",
    }
    proposals_mock.assert_awaited_once()


def test_flag_on_without_proposal_context_skips_call(client):
    test_client, _, proposals_mock = client({"RAG_ENABLE_PROPOSALS": "true"})
    context = _context_with_proposals()
    del context["proposalContext"]
    response = test_client.post(
        "/query", json={"question": "Is coverage sufficient?", "context": context}
    )
    assert response.status_code == 200
    assert response.json()["proposals"] == []
    proposals_mock.assert_not_awaited()


def test_proposal_llm_failure_never_breaks_answer(client):
    test_client, _, proposals_mock = client({"RAG_ENABLE_PROPOSALS": "true"})
    proposals_mock.side_effect = RuntimeError("proposal model down")
    response = test_client.post(
        "/query",
        json={
            "question": "Is coverage sufficient?",
            "context": _context_with_proposals(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "the grounded answer"
    assert body["proposals"] == []


def test_kb_only_route_has_empty_proposals(client):
    test_client, _, proposals_mock = client({"RAG_ENABLE_PROPOSALS": "true"})
    response = test_client.post("/query", json={"question": "What is HIL-A?"})
    assert response.status_code == 200
    assert response.json()["proposals"] == []
    proposals_mock.assert_not_awaited()
