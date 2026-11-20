from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, CHAR, DateTime, ForeignKey, Identity, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ReportCategory(Base):
    __tablename__ = "report_categories"

    category_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    code: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)


class Report(Base):
    __tablename__ = "reports"

    report_id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("report_categories.category_id", ondelete="RESTRICT"))
    overview: Mapped[str] = mapped_column(String(1000))
    index_status: Mapped[str] = mapped_column(String(20), server_default="pending")
    revision: Mapped[int] = mapped_column(server_default=text("1"))


class ReportEmbedding(Base):
    __tablename__ = "report_embeddings"

    report_id: Mapped[int] = mapped_column(ForeignKey("reports.report_id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    model_name: Mapped[str] = mapped_column(Text)
    overview_hash: Mapped[str] = mapped_column(CHAR(64))
    indexed_revision: Mapped[int]
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReportDetail:
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.report_id", ondelete="CASCADE"), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    author: Mapped[str] = mapped_column(String(100))
    create_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    update_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExperimentReport(ReportDetail, Base):
    __tablename__ = "experiment_reports"

    experiment_date: Mapped[date]
    hypothesis: Mapped[str] = mapped_column(Text)
    result_summary: Mapped[str] = mapped_column(Text)


class DailyWorkReport(ReportDetail, Base):
    __tablename__ = "daily_work_reports"

    work_date: Mapped[date]
    progress: Mapped[str] = mapped_column(Text)
    blockers: Mapped[str] = mapped_column(Text, server_default="")


class ResearchPlan(ReportDetail, Base):
    __tablename__ = "research_plans"

    research_goal: Mapped[str] = mapped_column(Text)
    methodology: Mapped[str] = mapped_column(Text)
    target_date: Mapped[date]


class DataProcessingReport(ReportDetail, Base):
    __tablename__ = "data_processing_reports"

    dataset_name: Mapped[str] = mapped_column(Text)
    processing_steps: Mapped[str] = mapped_column(Text)
    quality_notes: Mapped[str] = mapped_column(Text, server_default="")


class PromptReport(ReportDetail, Base):
    __tablename__ = "prompt_reports"

    purpose: Mapped[str] = mapped_column(Text)
    active_version_no: Mapped[int] = mapped_column(server_default=text("1"))


class MeetingMinutes(ReportDetail, Base):
    __tablename__ = "meeting_minutes"

    meeting_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    attendees: Mapped[str] = mapped_column(Text)
    decisions: Mapped[str] = mapped_column(Text)


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("report_id", "version_no", name="uq_prompt_versions_report_no"),)

    prompt_version_id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("prompt_reports.report_id", ondelete="CASCADE"))
    version_no: Mapped[int]
    prompt_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
