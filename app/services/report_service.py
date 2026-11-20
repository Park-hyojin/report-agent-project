from sqlalchemy.orm import Session

from models.report import ReportCategory
from repositories import report_repository


def list_categories(session: Session) -> list[ReportCategory]:
    return report_repository.list_categories(session)
