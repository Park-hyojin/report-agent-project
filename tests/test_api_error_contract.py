""" 실행 코드 : python -m pytest -q -p no:cacheprovider tests """

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from db.session import get_session
from integrations.embedding_client import EmbeddingUnavailable
from integrations.llm_client import LLMTimeout, LLMUnavailable
from main import app
from services import report_service, search_service


@pytest.fixture
def client():
    app.dependency_overrides[get_session] = lambda: object()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_session, None)


def assert_error(response, status, code):
    assert response.status_code == status
    assert set(response.json()) == {"error"}
    error = response.json()["error"]
    assert set(error) == {"code", "message", "details"}
    assert error["code"] == code
    assert error["message"] and isinstance(error["details"], dict)


def test_request_validation_omits_input(client):
    response = client.post("/api/v1/agent/run", json={"action": "private input"})
    assert_error(response, 422, "VALIDATION_ERROR")
    assert "private input" not in response.text


def test_report_not_found(client, monkeypatch):
    monkeypatch.setattr(report_service, "get_report", Mock(return_value=None))
    assert_error(client.get("/api/v1/reports/1"), 404, "REPORT_NOT_FOUND")


def test_embedding_error(client, monkeypatch):
    monkeypatch.setattr(
        search_service, "search_similar", Mock(side_effect=EmbeddingUnavailable("private input"))
    )
    response = client.post("/api/v1/search/similar", json={"query": "example"})
    assert_error(response, 503, "EMBEDDING_UNAVAILABLE")
    assert "private input" not in response.text


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (LLMUnavailable("private input"), 503, "LLM_UNAVAILABLE"),
        (LLMTimeout("private input"), 504, "LLM_TIMEOUT"),
        (SQLAlchemyError("private input"), 500, "INTERNAL_ERROR"),
        (OperationalError(None, None, RuntimeError("private input")), 503, "DB_UNAVAILABLE"),
    ],
)
def test_summary_dependency_errors(client, monkeypatch, failure, status, code):
    monkeypatch.setattr(report_service, "summarize_report", Mock(side_effect=failure))
    response = client.post("/api/v1/reports/1/summarize", json={"style": "brief"})
    assert_error(response, status, code)
    assert "private input" not in response.text
