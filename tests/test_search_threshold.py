import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy.dialects import postgresql

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from repositories import report_repository
from schemas.search import SimilarSearchRequest
from services import search_service


@pytest.mark.parametrize(
    ("keyword_match", "score", "expected_reason"),
    [
        (False, 0.79, None),
        (False, 0.80, "semantic_threshold"),
        (True, 0.79, "keyword_match"),
    ],
)
def test_threshold_applies_to_semantic_fallback(monkeypatch, keyword_match, score, expected_reason):
    monkeypatch.setattr(search_service.settings, "search_min_similarity", 0.80)
    monkeypatch.setattr(search_service.embedding_client, "encode", Mock())
    candidate = {
        "report_id": 1,
        "title": "Example",
        "category_code": "EXPERIMENT",
        "overview": "Example overview",
        "keyword_match": keyword_match,
        "cosine_similarity": score,
    }
    monkeypatch.setattr(report_repository, "search_candidates", Mock(return_value=[candidate]))

    result = search_service.search_similar(Mock(), SimilarSearchRequest(query="example"))

    if expected_reason is None:
        assert result.status == "not_found"
        assert result.best_match is None
    else:
        assert result.status == "found"
        assert result.best_match.selection_reason == expected_reason


def test_category_filter_applies_to_both_candidate_groups():
    session = Mock()
    session.execute.return_value.mappings.return_value = []

    report_repository.search_candidates(session, "example", [1.0] * 384, "EXPERIMENT", 20, 20)

    statement = session.execute.call_args.args[0]
    for branch in statement.selects:
        compiled = branch.compile(dialect=postgresql.dialect())
        assert "report_read_view.category_code =" in str(compiled)
        assert "EXPERIMENT" in compiled.params.values()
