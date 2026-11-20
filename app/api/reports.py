from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db.session import get_session
from integrations.embedding_client import EmbeddingUnavailable
from schemas.report import CategoryListResponse, ReportCreate, ReportCreateResponse
from services import report_service


router = APIRouter()


@router.get("/categories")
def list_categories(
    session: Annotated[Session, Depends(get_session)],
) -> CategoryListResponse:
    return CategoryListResponse.model_validate(
        {"items": report_service.list_categories(session)}
    )


@router.post(
    "/reports",
    status_code=status.HTTP_201_CREATED,
    responses={
        500: {"description": "Database write failed"},
        503: {"description": "Embedding unavailable"},
    },
)
def create_report(
    request: ReportCreate,
    session: Annotated[Session, Depends(get_session)],
) -> ReportCreateResponse:
    try:
        return report_service.create_report(session, request)
    except report_service.CategoryNotFound as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown category code") from error
    except EmbeddingUnavailable as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding unavailable"
        ) from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Database write failed"
        ) from error
