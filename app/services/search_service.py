from sqlalchemy.orm import Session

from core.config import settings
from integrations import embedding_client
from repositories import report_repository
from schemas.search import SearchMatch, SimilarSearchRequest, SimilarSearchResponse


def search_similar(session: Session, request: SimilarSearchRequest) -> SimilarSearchResponse:
    vector = embedding_client.encode(f"query: {request.query}")
    candidates = report_repository.search_candidates(
        session, request.query, vector, request.category_code,
        request.keyword_limit, request.semantic_limit,
    )
    eligible = (
        candidate for candidate in candidates
        if candidate["keyword_match"] or candidate["cosine_similarity"] >= settings.search_min_similarity
    )
    best = min(
        eligible,
        key=lambda candidate: (
            not candidate["keyword_match"], -candidate["cosine_similarity"], candidate["report_id"]
        ),
        default=None,
    )
    match = None
    if best is not None:
        keyword_match = best["keyword_match"]
        match = SearchMatch.model_validate({
            **best,
            "keyword_score": int(keyword_match),
            "semantic_score": best["cosine_similarity"],
            "selection_reason": "keyword_match" if keyword_match else "semantic_threshold",
        })

    return SimilarSearchResponse(
        status="found" if best is not None else "not_found",
        query=request.query,
        best_match=match,
    )
