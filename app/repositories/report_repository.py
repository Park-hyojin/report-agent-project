from sqlalchemy import select
from sqlalchemy.orm import Session

from models.report import (
    DailyWorkReport,
    DataProcessingReport,
    ExperimentReport,
    MeetingMinutes,
    PromptReport,
    PromptVersion,
    Report,
    ReportCategory,
    ReportDetail,
    ReportEmbedding,
    ResearchPlan,
)
from schemas.report import CategoryCode, PromptDetails, ReportCreate


DETAIL_MODELS: dict[CategoryCode, type[ReportDetail]] = {
    "EXPERIMENT": ExperimentReport,
    "DAILY_WORK": DailyWorkReport,
    "RESEARCH_PLAN": ResearchPlan,
    "DATA_PROCESSING": DataProcessingReport,
    "PROMPT": PromptReport,
    "MEETING": MeetingMinutes,
}


def list_categories(session: Session) -> list[ReportCategory]:
    statement = select(ReportCategory).order_by(ReportCategory.category_id)
    return list(session.scalars(statement))


def find_category(session: Session, code: CategoryCode) -> ReportCategory | None:
    statement = select(ReportCategory).where(ReportCategory.code == code)
    return session.scalars(statement).one_or_none()


def find_report(
    session: Session, report_id: int
) -> tuple[Report, str, ReportDetail] | None:
    statement = (
        select(Report, ReportCategory.code)
        .join(ReportCategory, Report.category_id == ReportCategory.category_id)
        .where(Report.report_id == report_id)
    )
    row = session.execute(statement).one_or_none()
    if row is None:
        return None

    report, category_code = row
    detail = session.get(DETAIL_MODELS[category_code], report_id)
    if detail is None:
        raise RuntimeError(f"Report {report_id} has no detail row")
    return report, category_code, detail


def insert_report(
    session: Session,
    category_id: int,
    request: ReportCreate,
    embedding: list[float],
    model_name: str,
    overview_hash: str,
) -> Report:
    report = Report(
        category_id=category_id, overview=request.overview, revision=1, index_status="ready"
    )
    session.add(report)
    session.flush()

    detail_fields = request.details.model_dump(exclude={"prompt_text"})
    detail_model = DETAIL_MODELS[request.category_code]
    session.add(detail_model(
        report_id=report.report_id,
        title=request.title,
        author=request.author,
        content=request.content,
        **detail_fields,
    ))
    session.add(ReportEmbedding(
        report_id=report.report_id,
        embedding=embedding,
        model_name=model_name,
        overview_hash=overview_hash,
        indexed_revision=report.revision,
    ))

    if isinstance(request.details, PromptDetails):
        session.flush()
        session.add(PromptVersion(
            report_id=report.report_id, version_no=1, prompt_text=request.details.prompt_text
        ))

    return report
