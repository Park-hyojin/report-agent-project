from sqlalchemy.orm import Session

from schemas.agent import (
    AgentRequest,
    AgentResponse,
    FindReportAction,
    GetReportAction,
    SearchAndSummarizeAction,
    SearchAndSummarizeResult,
)
from schemas.report import ReportSummaryRequest
from schemas.search import SimilarSearchRequest
from services import report_service, search_service


def run_agent(session: Session, request: AgentRequest) -> AgentResponse | None:
    if isinstance(request, SearchAndSummarizeAction):
        return search_and_summarize(session, request)

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


def search_and_summarize(
    session: Session, request: SearchAndSummarizeAction
) -> AgentResponse | None:
    with session.begin():
        search = search_service.search_similar(session, SimilarSearchRequest(query=request.query))

    if search.best_match is None:
        return AgentResponse(action=request.action, tool_calls=["find_similar_report"], data=search)

    summary = report_service.summarize_report(
        session, search.best_match.report_id, ReportSummaryRequest(style="brief")
    )
    if summary is None:
        return None

    return AgentResponse(
        action=request.action,
        tool_calls=["find_similar_report", "summarize_report"],
        data=SearchAndSummarizeResult(
            status=search.status, query=search.query, best_match=search.best_match, summary=summary
        ),
    )
