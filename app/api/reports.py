from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.session import get_session
from schemas.report import CategoryListResponse
from services import report_service


router = APIRouter()


@router.get("/categories")
def list_categories(
    session: Annotated[Session, Depends(get_session)],
) -> CategoryListResponse:
    return CategoryListResponse.model_validate(
        {"items": report_service.list_categories(session)}
    )
