import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field


class Settings(BaseModel):
    database_url: str = Field(default="", repr=False)
    embedding_model: str = "intfloat/multilingual-e5-small"
    search_min_similarity: float = Field(default=0.75, ge=-1, le=1)
    openai_api_key: str = Field(default="", repr=False)
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=30, gt=0)
    llm_max_output_tokens: int = Field(default=400, gt=0)
    project_level: Literal["basic", "intermediate", "advanced"] = "basic"


def load_settings() -> Settings:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    file_values = dotenv_values(env_path)
    values = {}
    for field_name in Settings.model_fields:
        env_name = field_name.upper()
        value = os.environ.get(env_name, file_values.get(env_name))
        if value is not None:
            values[field_name] = value
    return Settings.model_validate(values)


settings = load_settings()
