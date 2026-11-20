from datetime import UTC, datetime
from hashlib import sha256

from sqlalchemy.orm import Session

from core.config import settings
from integrations import embedding_client, llm_client
from models.report import PromptReport, ReportCategory
from repositories import report_repository
from schemas.report import (
    DETAIL_TYPES,
    CategoryCode,
    PromptVersionCreate,
    PromptVersionCreateResponse,
    PromptVersionItem,
    PromptVersionListResponse,
    ReportCreate,
    ReportCreateResponse,
    ReportDetailResponse,
    ReportListItem,
    ReportListResponse,
    ReportSummaryRequest,
    ReportSummaryResponse,
    ReportUpdate,
    ReportUpdateResponse,
)


class CategoryNotFound(ValueError):
    pass


class PromptContentChange(ValueError):
    pass


class NotPromptReport(ValueError):
    pass


def list_categories(session: Session) -> list[ReportCategory]:
    return report_repository.list_categories(session)


def list_reports(
    session: Session,
    title: str | None,
    limit: int,
    offset: int,
    *,
    category_code: CategoryCode | None = None,
    updated_after: datetime | None = None,
) -> ReportListResponse:
    rows, total = report_repository.list_reports(
        session, title, limit, offset, category_code=category_code, updated_after=updated_after
    )
    return ReportListResponse(
        items=[ReportListItem.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


def encode_overview(overview: str) -> list[float]:
    return embedding_client.encode(f"passage: {overview}")


def get_report(session: Session, report_id: int) -> ReportDetailResponse | None:
    result = report_repository.find_report(session, report_id)
    if result is None:
        return None

    report, category_code, detail = result
    details = {}
    for field_name in DETAIL_TYPES[category_code].model_fields:
        if field_name == "prompt_text" and isinstance(detail, PromptReport):
            details[field_name] = detail.content
            continue
        value = getattr(detail, field_name)
        details[field_name] = value.astimezone(UTC) if isinstance(value, datetime) else value

    return ReportDetailResponse(
        report_id=report.report_id,
        category_code=category_code,
        title=detail.title,
        content=detail.content,
        author=detail.author,
        overview=report.overview,
        details=details,
        revision=report.revision,
        index_status=report.index_status,
        created_at=detail.create_at,
        updated_at=detail.update_at,
    )


def summarize_report(
    session: Session, report_id: int, request: ReportSummaryRequest
) -> ReportSummaryResponse | None:
    with session.begin():
        result = report_repository.find_report(session, report_id, for_update=True)
        if result is None:
            return None
        report, _, detail = result
        content = detail.content
        source_revision = report.revision

    summary, model = llm_client.summarize(content, request.max_sentences)
    return ReportSummaryResponse(
        report_id=report_id,
        source_revision=source_revision,
        summary=summary,
        model=model,
    )


def create_report(session: Session, request: ReportCreate) -> ReportCreateResponse:
    with session.begin():
        category = report_repository.find_category(session, request.category_code)
        if category is None:
            raise CategoryNotFound(request.category_code)

        embedding = encode_overview(request.overview)
        model_name = settings.embedding_model
        report = report_repository.insert_report(
            session,
            category_id=category.category_id,
            request=request,
            embedding=embedding,
            model_name=model_name,
            overview_hash=sha256(request.overview.encode("utf-8")).hexdigest(),
        )
        result = ReportCreateResponse(
            report_id=report.report_id,
            category_code=request.category_code,
            title=request.title,
            author=request.author,
            revision=report.revision,
            index_status="ready",
            embedding_model=model_name,
        )
    return result


def find_prompt_report(
    session: Session, report_id: int, *, for_update: bool = False
) -> PromptReport | None:
    result = report_repository.find_report(session, report_id, for_update=for_update)
    if result is None:
        return None
    _, _, detail = result
    if not isinstance(detail, PromptReport):
        raise NotPromptReport
    return detail


def list_prompt_versions(session: Session, report_id: int) -> PromptVersionListResponse | None:
    prompt = find_prompt_report(session, report_id)
    if prompt is None:
        return None
    versions = report_repository.list_prompt_versions(session, report_id)
    return PromptVersionListResponse(
        items=[PromptVersionItem.model_validate(version) for version in versions],
        active_version_no=prompt.active_version_no,
    )


def create_prompt_version(
    session: Session, report_id: int, request: PromptVersionCreate
) -> PromptVersionCreateResponse | None:
    with session.begin():
        if find_prompt_report(session, report_id, for_update=True) is None:
            return None

        version_no = report_repository.insert_prompt_version(
            session, report_id, request.prompt_text
        )
    return PromptVersionCreateResponse(version_no=version_no)


def delete_report(session: Session, report_id: int) -> bool:
    with session.begin():
        return report_repository.delete_report(session, report_id)


def index_overview(session: Session, report_id: int) -> bool:
    with session.begin():
        report = report_repository.lock_report(session, report_id)
        if report is None:
            return False

        overview_hash = sha256(report.overview.encode("utf-8")).hexdigest()
        embedding = report_repository.find_embedding(session, report_id)
        if (
            report.index_status == "ready"
            and embedding is not None
            and embedding.indexed_revision == report.revision
            and embedding.overview_hash == overview_hash
            and embedding.model_name == settings.embedding_model
        ):
            return False

        vector = encode_overview(report.overview)
        if embedding is None:
            embedding = report_repository.add_embedding(session, report_id)
        embedding.embedding = vector
        embedding.overview_hash = overview_hash
        embedding.model_name = settings.embedding_model
        embedding.indexed_at = datetime.now(UTC)
        embedding.indexed_revision = report.revision
        report.index_status = "ready"
    return True


def update_report(
    session: Session, report_id: int, request: ReportUpdate
) -> ReportUpdateResponse | None:
    with session.begin():
        current = report_repository.find_report(session, report_id, for_update=True)
        if current is None:
            return None

        report, category_code, detail = current
        details = DETAIL_TYPES[category_code].model_validate(request.details)
        detail_fields = details.model_dump(exclude={"prompt_text"})
        if isinstance(detail, PromptReport) and (
            request.content != detail.content or request.details.get("prompt_text") != detail.content
        ):
            raise PromptContentChange("Prompt content must match the active version")

        embedding = report_repository.find_embedding(session, report_id)
        embedding_updated = embedding is None or report.overview != request.overview
        if embedding_updated:
            vector = encode_overview(request.overview)
            if embedding is None:
                embedding = report_repository.add_embedding(session, report_id)
            embedding.embedding = vector
            embedding.model_name = settings.embedding_model
            embedding.overview_hash = sha256(request.overview.encode("utf-8")).hexdigest()
            embedding.indexed_at = datetime.now(UTC)

        report.overview = request.overview
        report.revision += 1
        report.index_status = "ready"
        embedding.indexed_revision = report.revision

        detail.title = request.title
        detail.author = request.author
        detail.content = request.content
        detail.update_at = datetime.now(UTC)
        for field_name, value in detail_fields.items():
            setattr(detail, field_name, value)

        result = ReportUpdateResponse(
            report_id=report_id,
            revision=report.revision,
            index_status="ready",
            embedding_updated=embedding_updated,
        )
    return result
