from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.reports import report_not_found
from db.session import get_session
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
    result = agent_service.run_agent(session, request)
    if result is None:
        return report_not_found()
    return result
