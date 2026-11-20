import sys
from pathlib import Path

from sqlalchemy.orm import Session

# Share the same module imports as python app/main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from db.session import get_engine
from integrations.embedding_client import EmbeddingUnavailable
from repositories.report_repository import list_report_ids
from services.report_service import index_overview


def main() -> int:
    with Session(get_engine()) as session:
        with session.begin():
            report_ids = list_report_ids(session)

        indexed = 0
        for report_id in report_ids:
            try:
                indexed += index_overview(session, report_id)
            except EmbeddingUnavailable:
                print(f"Embedding failed for report {report_id}. Indexed {indexed} reports.", file=sys.stderr)
                return 1

    print(f"Indexed {indexed} of {len(report_ids)} reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
