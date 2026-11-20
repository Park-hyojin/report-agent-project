from collections.abc import Mapping
from typing import Any

from sqlalchemy import column, delete, func, or_, select, table
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
    column("content"),
    column("overview"),
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
    session: Session, report_id: int, *, for_update: bool = False
) -> tuple[Report, str, ReportDetail] | None:
    statement = (
        select(Report, ReportCategory.code)
        .join(ReportCategory, Report.category_id == ReportCategory.category_id)
        .where(Report.report_id == report_id)
    )
    if for_update:
        statement = statement.with_for_update(of=Report)
    row = session.execute(statement).one_or_none()
    if row is None:
        return None

    report, category_code = row
    detail = session.get(DETAIL_MODELS[category_code], report_id)
    if detail is None:
        raise RuntimeError(f"Report {report_id} has no detail row")
    return report, category_code, detail


def search_candidates(
    session: Session,
    query: str,
    vector: list[float],
    category_code: CategoryCode | None,
    keyword_limit: int,
    semantic_limit: int,
) -> list[Mapping[str, Any]]:
    view = REPORT_READ_VIEW
    keyword_match = or_(
        view.c.title.icontains(query, autoescape=True),
        view.c.overview.icontains(query, autoescape=True),
        view.c.content.icontains(query, autoescape=True),
    )
    distance = ReportEmbedding.embedding.cosine_distance(vector)
    candidates = (
        select(
            view.c.report_id,
            view.c.title,
            view.c.category_code,
            view.c.overview,
            keyword_match.label("keyword_match"),
            (1 - distance).label("cosine_similarity"),
        )
        .select_from(view.join(Report, view.c.report_id == Report.report_id))
        .join(ReportEmbedding, Report.report_id == ReportEmbedding.report_id)
        .where(Report.index_status == "ready", ReportEmbedding.indexed_revision == Report.revision)
    )
    if category_code is not None:
        candidates = candidates.where(view.c.category_code == category_code)

    ranked = candidates.order_by(distance, view.c.report_id)
    keywords = ranked.where(keyword_match).limit(keyword_limit).subquery()
    semantic = ranked.limit(semantic_limit).subquery()
    statement = select(keywords).union(select(semantic))
    return list(session.execute(statement).mappings())


def list_report_ids(session: Session) -> list[int]:
    return list(session.scalars(select(Report.report_id).order_by(Report.report_id)))


def lock_report(session: Session, report_id: int) -> Report | None:
    return session.get(Report, report_id, with_for_update=True)


def insert_prompt_version(session: Session, report_id: int, prompt_text: str) -> int:
    statement = select(func.max(PromptVersion.version_no)).where(
        PromptVersion.report_id == report_id
    )
    latest_version = session.execute(statement).scalar_one()
    version_no = (latest_version or 0) + 1
    session.add(PromptVersion(report_id=report_id, version_no=version_no, prompt_text=prompt_text))
    return version_no


def delete_report(session: Session, report_id: int) -> bool:
    statement = delete(Report).where(Report.report_id == report_id).returning(Report.report_id)
    return session.execute(statement).scalar_one_or_none() is not None


def find_embedding(session: Session, report_id: int) -> ReportEmbedding | None:
    return session.get(ReportEmbedding, report_id)


def add_embedding(session: Session, report_id: int) -> ReportEmbedding:
    embedding = ReportEmbedding(report_id=report_id)
    session.add(embedding)
    return embedding


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
