from pydantic import BaseModel, ConfigDict


class CategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    category_id: int
    code: str
    name: str
    description: str | None


class CategoryListResponse(BaseModel):
    items: list[CategoryResponse]
