from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from schemas.report import CategoryCode


SearchQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class SimilarSearchRequest(BaseModel):
    query: SearchQuery
    category_code: CategoryCode | None = None
    keyword_limit: int = Field(default=20, ge=1, le=100)
    semantic_limit: int = Field(default=20, ge=1, le=100)


class SearchMatch(BaseModel):
    report_id: int
    title: str
    category_code: CategoryCode
    overview: str
    keyword_match: bool
    cosine_similarity: float
    keyword_score: Literal[0, 1]
    semantic_score: float
    selection_reason: Literal["keyword_match", "semantic_threshold"]


class SimilarSearchResponse(BaseModel):
    status: Literal["found", "not_found"]
    query: str
    best_match: SearchMatch | None
