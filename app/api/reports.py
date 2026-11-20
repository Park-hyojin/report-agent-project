from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db.session import get_session
from integrations.embedding_client import EmbeddingUnavailable
from integrations.llm_client import LLMTimeout, LLMUnavailable
from schemas.report import (
    CategoryListResponse,
    ReportCreate,
    ReportCreateResponse,
    ReportDetailResponse,
    ReportListResponse,
    ReportSummaryRequest,
    ReportSummaryResponse,
    ReportUpdate,
    ReportUpdateResponse,
)
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


@router.get("/reports")
def list_reports(
    session: Annotated[Session, Depends(get_session)],
    title: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
) -> ReportListResponse:
    return report_service.list_reports(session, title, limit, offset)


@router.get(
    "/reports/{report_id}",
    response_model=ReportDetailResponse,
    responses={404: {"description": "Report not found"}},
)
def get_report(
    report_id: Annotated[int, Path(gt=0, le=2**63 - 1)],
    session: Annotated[Session, Depends(get_session)],
) -> ReportDetailResponse | JSONResponse:
    report = report_service.get_report(session, report_id)
    if report is None:
        return report_not_found()
    return report


@router.put(
    "/reports/{report_id}",
    response_model=ReportUpdateResponse,
    responses={
        404: {"description": "Report not found"},
        500: {"description": "Database write failed"},
        503: {"description": "Embedding unavailable"},
    },
)
def update_report(
    report_id: Annotated[int, Path(gt=0, le=2**63 - 1)],
    request: ReportUpdate,
    session: Annotated[Session, Depends(get_session)],
) -> ReportUpdateResponse | JSONResponse:
    try:
        report = report_service.update_report(session, report_id, request)
    except ValidationError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid details for the report category"
        ) from error
    except report_service.PromptContentChange as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    except EmbeddingUnavailable as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding unavailable"
        ) from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Database write failed"
        ) from error
    if report is None:
        return report_not_found()
    return report


@router.delete(
    "/reports/{report_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={404: {"description": "Report not found"}},
)
def delete_report(
    report_id: Annotated[int, Path(gt=0, le=2**63 - 1)],
    session: Annotated[Session, Depends(get_session)],
) -> Response:
    if not report_service.delete_report(session, report_id):
        return report_not_found()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/reports/{report_id}/summarize",
    response_model=ReportSummaryResponse,
    responses={
        404: {"description": "Report not found"},
        503: {"description": "LLM unavailable"},
        504: {"description": "LLM request timed out"},
    },
)
def summarize_report(
    report_id: Annotated[int, Path(gt=0, le=2**63 - 1)],
    request: ReportSummaryRequest,
    session: Annotated[Session, Depends(get_session)],
) -> ReportSummaryResponse | JSONResponse:
    try:
        summary = report_service.summarize_report(session, report_id, request)
    except LLMTimeout as error:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "LLM request timed out") from error
    except LLMUnavailable as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM unavailable") from error
    if summary is None:
        return report_not_found()
    return summary


def report_not_found() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": {
                "code": "REPORT_NOT_FOUND",
                "message": "요청한 보고서를 찾을 수 없습니다.",
                "details": {},
            }
        },
    )
