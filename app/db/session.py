from functools import lru_cache

from sqlalchemy import Engine, create_engine

from core.config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(settings.database_url)
