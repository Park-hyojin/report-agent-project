import uvicorn
from fastapi import FastAPI, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from api.agent import router as agent_router
from api.error_handlers import register_error_handlers
from api.reports import router as reports_router
from api.search import router as search_router
from db.session import get_engine


app = FastAPI(title="Report Agent", version="1.4")
register_error_handlers(app)
app.include_router(reports_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")


@app.get("/health", responses={503: {"description": "Database or pgvector unavailable"}})
def health(response: Response) -> dict[str, str]:
    try:
        with get_engine().connect() as connection:
            vector_ready = connection.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            ).scalar_one()
    except SQLAlchemyError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "db": "unavailable", "vector": "unknown"}

    if not vector_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "db": "ready", "vector": "unavailable"}

    return {"status": "ok", "db": "ready", "vector": "ready"}


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
