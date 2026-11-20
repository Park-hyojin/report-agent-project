from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.session import get_session
from schemas.search import SimilarSearchRequest, SimilarSearchResponse
from services import search_service


router = APIRouter()


@router.post("/search/similar", responses={503: {"description": "Embedding unavailable"}})
def search_similar(
    request: SimilarSearchRequest,
    session: Annotated[Session, Depends(get_session)],
) -> SimilarSearchResponse:
    return search_service.search_similar(session, request)
