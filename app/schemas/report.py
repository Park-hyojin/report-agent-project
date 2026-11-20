from datetime import UTC, date, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)


class CategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    category_id: int
    code: str
    name: str
    description: str | None


class CategoryListResponse(BaseModel):
    items: list[CategoryResponse]


CategoryCode = Literal[
    "EXPERIMENT", "DAILY_WORK", "RESEARCH_PLAN", "DATA_PROCESSING", "PROMPT", "MEETING"
]
NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Title = Annotated[NonBlankText, StringConstraints(max_length=200)]
Author = Annotated[NonBlankText, StringConstraints(max_length=100)]
Content = Annotated[NonBlankText, StringConstraints(max_length=20_000)]
Overview = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1000)]


class ReportDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentDetails(ReportDetails):
    experiment_date: date
    hypothesis: NonBlankText
    result_summary: NonBlankText


class DailyWorkDetails(ReportDetails):
    work_date: date
    progress: NonBlankText
    blockers: str


class ResearchPlanDetails(ReportDetails):
    research_goal: NonBlankText
    methodology: NonBlankText
    target_date: date


class DataProcessingDetails(ReportDetails):
    dataset_name: NonBlankText
    processing_steps: NonBlankText
    quality_notes: str


class PromptDetails(ReportDetails):
    purpose: NonBlankText
    prompt_text: Content


class MeetingDetails(ReportDetails):
    meeting_at: AwareDatetime
    attendees: NonBlankText
    decisions: NonBlankText


DETAIL_TYPES: dict[CategoryCode, type[ReportDetails]] = {
    "EXPERIMENT": ExperimentDetails,
    "DAILY_WORK": DailyWorkDetails,
    "RESEARCH_PLAN": ResearchPlanDetails,
    "DATA_PROCESSING": DataProcessingDetails,
    "PROMPT": PromptDetails,
    "MEETING": MeetingDetails,
}


class ReportCreate(BaseModel):
    category_code: CategoryCode
    title: Title
    author: Author
    content: Content
    overview: Overview
    details: (
        ExperimentDetails
        | DailyWorkDetails
        | ResearchPlanDetails
        | DataProcessingDetails
        | PromptDetails
        | MeetingDetails
    )

    @field_validator("details", mode="before")
    @classmethod
    def validate_details(cls, value: object, info: ValidationInfo) -> object:
        category_code = info.data.get("category_code")
        if category_code is None:
            return value
        return DETAIL_TYPES[category_code].model_validate(value)

    @model_validator(mode="after")
    def validate_prompt_content(self) -> Self:
        if isinstance(self.details, PromptDetails) and self.content != self.details.prompt_text:
            raise ValueError("content must match details.prompt_text for PROMPT reports")
        return self


class ReportCreateResponse(BaseModel):
    report_id: int
    category_code: CategoryCode
    title: str
    author: str
    revision: int
    index_status: Literal["ready"]
    embedding_model: str


class ReportDetailResponse(BaseModel):
    report_id: int
    category_code: CategoryCode
    title: str
    content: str
    author: str
    overview: str
    details: dict[str, str | date | datetime]
    revision: int
    index_status: Literal["pending", "ready", "failed"]
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)
