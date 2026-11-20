from sqlalchemy.orm import Session

from schemas.agent import AgentRequest, AgentResponse, FindReportAction, GetReportAction
from schemas.report import ReportSummaryRequest
from schemas.search import SimilarSearchRequest
from services import report_service, search_service


def run_agent(session: Session, request: AgentRequest) -> AgentResponse | None:
    if isinstance(request, FindReportAction):
        data = search_service.search_similar(session, SimilarSearchRequest(query=request.query))
        return AgentResponse(action=request.action, tool_calls=["find_similar_report"], data=data)

    if isinstance(request, GetReportAction):
        data = report_service.get_report(session, request.report_id)
    else:
        data = report_service.summarize_report(
            session, request.report_id, ReportSummaryRequest(style="brief")
        )

    if data is None:
        return None
    return AgentResponse(action=request.action, tool_calls=[request.action], data=data)
