from sqlalchemy import select
from sqlalchemy.orm import Session

from models.report import ReportCategory


def list_categories(session: Session) -> list[ReportCategory]:
    statement = select(ReportCategory).order_by(ReportCategory.category_id)
    return list(session.scalars(statement))
