from collections.abc import Mapping
from typing import Any

from sqlalchemy import column, func, select, table
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

REPORT_READ_VIEW = table(
    "report_read_view",
    column("report_id"),
    column("category_code"),
    column("title"),
    column("revision"),
    column("update_at"),
)


def list_categories(session: Session) -> list[ReportCategory]:
    statement = select(ReportCategory).order_by(ReportCategory.category_id)
    return list(session.scalars(statement))


def find_category(session: Session, code: CategoryCode) -> ReportCategory | None:
    statement = select(ReportCategory).where(ReportCategory.code == code)
    return session.scalars(statement).one_or_none()


def list_reports(
    session: Session, title: str | None, limit: int, offset: int
) -> tuple[list[Mapping[str, Any]], int]:
    view = REPORT_READ_VIEW
    matches = select(
        view.c.report_id, view.c.category_code, view.c.title, view.c.revision, view.c.update_at
    )
    if title is not None:
        matches = matches.where(view.c.title.icontains(title, autoescape=True))

    total = session.execute(select(func.count()).select_from(matches.subquery())).scalar_one()
    page = (
        matches.with_only_columns(view.c.report_id, view.c.category_code, view.c.title, view.c.revision)
        .order_by(view.c.update_at.desc(), view.c.report_id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.execute(page).mappings()), total


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
