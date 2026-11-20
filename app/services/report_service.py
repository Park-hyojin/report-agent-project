from sqlalchemy.orm import Session

from integrations import embedding_client
from models.report import ReportCategory
from repositories import report_repository


def list_categories(session: Session) -> list[ReportCategory]:
    return report_repository.list_categories(session)


def encode_overview(overview: str) -> list[float]:
    return embedding_client.encode(f"passage: {overview}")
