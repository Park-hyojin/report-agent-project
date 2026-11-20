from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.exceptions import HTTPException

from integrations.embedding_client import EmbeddingUnavailable
from integrations.llm_client import LLMTimeout, LLMUnavailable


def error_response(
    status_code: int, code: str, message: str, *, details: dict | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or {}}},
        headers=headers,
    )


async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
    fields = [
        {"location": list(issue["loc"]), "type": issue["type"]}
        for issue in error.errors()
    ]
    return error_response(
        422, "VALIDATION_ERROR", "요청 값을 확인해 주세요.", details={"fields": fields}
    )


async def http_error(_: Request, error: HTTPException) -> JSONResponse:
    code = {400: "INVALID_ACTION", 404: "NOT_FOUND", 422: "VALIDATION_ERROR"}.get(
        error.status_code, "HTTP_ERROR"
    )
    return error_response(error.status_code, code, str(error.detail), headers=error.headers)


async def dependency_error(_: Request, error: Exception) -> JSONResponse:
    if isinstance(error, EmbeddingUnavailable):
        return error_response(503, "EMBEDDING_UNAVAILABLE", "임베딩 모델을 사용할 수 없습니다.")
    if isinstance(error, LLMTimeout):
        return error_response(504, "LLM_TIMEOUT", "요약 요청 시간이 초과됐습니다.")
    if isinstance(error, LLMUnavailable):
        return error_response(503, "LLM_UNAVAILABLE", "요약 모델을 사용할 수 없습니다.")
    if isinstance(error, OperationalError):
        sqlstate = getattr(error.orig, "sqlstate", None) or ""
        if error.statement is None or error.connection_invalidated or sqlstate.startswith("08"):
            return error_response(503, "DB_UNAVAILABLE", "데이터베이스에 연결할 수 없습니다.")
    return error_response(500, "INTERNAL_ERROR", "요청 처리 중 오류가 발생했습니다.")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(HTTPException, http_error)
    for error_type in (EmbeddingUnavailable, LLMUnavailable, LLMTimeout, SQLAlchemyError):
        app.add_exception_handler(error_type, dependency_error)
