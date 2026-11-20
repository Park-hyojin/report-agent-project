from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db.session import get_session
from integrations.embedding_client import EmbeddingUnavailable
from schemas.search import SimilarSearchRequest, SimilarSearchResponse
from services import search_service


router = APIRouter()


@router.post("/search/similar", responses={503: {"description": "Embedding unavailable"}})
def search_similar(
    request: SimilarSearchRequest,
    session: Annotated[Session, Depends(get_session)],
) -> SimilarSearchResponse:
    try:
        return search_service.search_similar(session, request)
    except EmbeddingUnavailable as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding unavailable"
        ) from error
