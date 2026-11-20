from typing import Annotated, Literal

from pydantic import BaseModel, Field

from schemas.report import ReportDetailResponse, ReportSummaryResponse
from schemas.search import SearchQuery, SimilarSearchResponse


ReportId = Annotated[int, Field(gt=0, le=2**63 - 1)]


class GetReportAction(BaseModel):
    action: Literal["get_report"]
    report_id: ReportId


class FindReportAction(BaseModel):
    action: Literal["find_report"]
    query: SearchQuery


class SummarizeReportAction(BaseModel):
    action: Literal["summarize_report"]
    report_id: ReportId


AgentRequest = Annotated[
    GetReportAction | FindReportAction | SummarizeReportAction,
    Field(discriminator="action"),
]


class AgentResponse(BaseModel):
    action: Literal["get_report", "find_report", "summarize_report"]
    status: Literal["completed"] = "completed"
    tool_calls: list[Literal["get_report", "find_similar_report", "summarize_report"]]
    data: ReportDetailResponse | SimilarSearchResponse | ReportSummaryResponse
