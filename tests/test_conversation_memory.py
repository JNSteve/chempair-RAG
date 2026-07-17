"""Conversational memory: the client-sent transcript renders into the
unified prompt so multi-turn references actually work, framed as
continuity — never as evidence."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402
from context_models import WorkspaceContext  # noqa: E402


def _ctx(conversation: list[dict] | None = None) -> dict:
    return {
        "schemaVersion": 4,
        "projectState": {"project": {"projectName": "Ducat"}},
        **({"conversation": conversation} if conversation is not None else {}),
    }


def test_render_conversation_labels_and_trims():
    long_answer = "x" * 700
    ctx = WorkspaceContext.model_validate(
        _ctx(
            [
                {"role": "user", "content": "is the arsenic a problem?"},
                {"role": "assistant", "content": long_answer},
                {"role": "user", "content": "   "},
            ]
        )
    )
    rendered = server._render_conversation(ctx)
    lines = rendered.splitlines()
    assert lines[0] == "Consultant: is the arsenic a problem?"
    assert lines[1].startswith("Alfie: xxx")
    assert lines[1].endswith("…")
    assert len(lines) == 2  # blank message dropped


def test_render_conversation_caps_message_count():
    conversation = [{"role": "user", "content": f"question {i}"} for i in range(12)]
    ctx = WorkspaceContext.model_validate(_ctx(conversation))
    rendered = server._render_conversation(ctx)
    assert len(rendered.splitlines()) == server.MAX_PROMPT_CONVERSATION_MESSAGES
    assert "question 11" in rendered
    assert "question 3" not in rendered


def test_render_conversation_empty_without_history():
    ctx = WorkspaceContext.model_validate(_ctx())
    assert server._render_conversation(ctx) == ""


@pytest.fixture()
def client():
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test-key", "RAG_AUTH_REQUIRED": "false"},
    ):
        server.app.router.on_startup.clear()
        server.app.router.on_shutdown.clear()
        server.sessions.clear()

        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="kb answer")
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={"status": "success", "data": {"references": [], "chunks": []}}
        )
        mock_openai = AsyncMock(return_value="unified answer")

        server.rag = mock_rag
        with patch.object(server, "openai_complete_if_cache", mock_openai):
            with TestClient(server.app, raise_server_exceptions=False) as test_client:
                yield test_client, mock_openai

        server.sessions.clear()


def test_conversation_reaches_the_model(client):
    test_client, mock_openai = client

    response = test_client.post(
        "/query",
        json={
            "question": "draft that paragraph for the report",
            "context": _ctx(
                [
                    {"role": "user", "content": "is the fill layer the concern?"},
                    {
                        "role": "assistant",
                        "content": "Yes - the fill is where the BaP exceedances sit.",
                    },
                ]
            ),
        },
    )

    assert response.status_code == 200
    prompt = mock_openai.await_args.args[1]
    assert "=== CONVERSATION SO FAR (continuity only — not evidence) ===" in prompt
    assert "Consultant: is the fill layer the concern?" in prompt
    assert "Alfie: Yes - the fill is where the BaP exceedances sit." in prompt
    # The conversation renders before the question, after the evidence.
    assert (
        prompt.index("KNOWLEDGE BASE EVIDENCE")
        < prompt.index("CONVERSATION SO FAR")
        < prompt.index("=== QUESTION ===")
    )


def test_no_conversation_block_without_history(client):
    test_client, mock_openai = client

    response = test_client.post(
        "/query",
        json={"question": "what criteria apply?", "context": _ctx()},
    )

    assert response.status_code == 200
    assert "CONVERSATION SO FAR" not in mock_openai.await_args.args[1]


def test_continuity_rule_in_system_prompt():
    assert "CONVERSATION SO FAR" in server.UNIFIED_ANSWER_SYSTEM
    assert "It is not evidence" in server.UNIFIED_ANSWER_SYSTEM
