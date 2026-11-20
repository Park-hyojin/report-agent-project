from hashlib import sha256

from sqlalchemy.orm import Session

from core.config import settings
from integrations import embedding_client
from models.report import ReportCategory
from repositories import report_repository
from schemas.report import ReportCreate, ReportCreateResponse


class CategoryNotFound(ValueError):
    pass


def list_categories(session: Session) -> list[ReportCategory]:
    return report_repository.list_categories(session)


def encode_overview(overview: str) -> list[float]:
    return embedding_client.encode(f"passage: {overview}")


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
