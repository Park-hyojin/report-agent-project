from datetime import UTC, datetime
from hashlib import sha256

from sqlalchemy.orm import Session

from core.config import settings
from integrations import embedding_client
from models.report import PromptReport, ReportCategory
from repositories import report_repository
from schemas.report import DETAIL_TYPES, ReportCreate, ReportCreateResponse, ReportDetailResponse


class CategoryNotFound(ValueError):
    pass


def list_categories(session: Session) -> list[ReportCategory]:
    return report_repository.list_categories(session)


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
