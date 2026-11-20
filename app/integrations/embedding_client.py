from functools import lru_cache
from math import isfinite
from typing import TYPE_CHECKING

from core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


EMBEDDING_DIMENSION = 384


class EmbeddingUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def encode(text: str) -> list[float]:
    try:
        model = _load_model(settings.embedding_model)
        vector = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()
    except Exception as error:
        raise EmbeddingUnavailable("Failed to encode text with the embedding model") from error

    if len(vector) != EMBEDDING_DIMENSION or not all(map(isfinite, vector)):
        raise EmbeddingUnavailable("The embedding model returned an invalid vector")
    return vector
