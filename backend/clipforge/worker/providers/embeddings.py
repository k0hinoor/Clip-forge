"""Optional sentence embeddings for semantic segmentation (TRD §12).

Embeddings are optional and local; the default is lexical TF-IDF (no model).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from clipforge.core.config import Settings
from clipforge.core.logging import get_logger

log = get_logger(__name__)
_lock = threading.Lock()
_model: Any = None


def build_embedder(settings: Settings) -> Callable[[list[str]], list[list[float]]] | None:
    """Return an embedding function, or None to use lexical TF-IDF."""
    if settings.EMBEDDINGS_PROVIDER != "sentence_transformers":
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError:
        log.warning("sentence-transformers not installed; using lexical segmentation")
        return None

    def embed(texts: list[str]) -> list[list[float]]:
        global _model
        with _lock:
            if _model is None:
                _model = SentenceTransformer(settings.EMBEDDINGS_MODEL)
        vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]

    return embed
