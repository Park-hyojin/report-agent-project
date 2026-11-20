from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.reports import report_not_found
from db.session import get_session
from integrations.embedding_client import EmbeddingUnavailable
from integrations.llm_client import LLMTimeout, LLMUnavailable
from schemas.agent import AgentRequest, AgentResponse
from services import agent_service


router = APIRouter()


@router.post(
    "/agent/run",
    response_model=AgentResponse,
    responses={
        404: {"description": "Report not found"},
        503: {"description": "Embedding or LLM unavailable"},
        504: {"description": "LLM request timed out"},
    },
)
def run_agent(
    request: AgentRequest,
    session: Annotated[Session, Depends(get_session)],
) -> AgentResponse | JSONResponse:
    try:
        result = agent_service.run_agent(session, request)
    except EmbeddingUnavailable as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Embedding unavailable"
        ) from error
    except LLMUnavailable as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM unavailable") from error
    except LLMTimeout as error:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "LLM request timed out") from error
    if result is None:
        return report_not_found()
    return result
