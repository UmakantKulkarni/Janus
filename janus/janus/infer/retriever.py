"""Real-time explainability data retrieval."""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import FeatureHasher, TfidfVectorizer

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class ExplainabilityRetriever:
    """Load indexes and fetch context for anomalous logs."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        path = self.index_dir
        with (path / "openapi_map.json").open() as f:
            self.openapi_map = json.load(f)
        with (path / "source_code_map.json").open() as f:
            self.code_map = json.load(f)
        self.faiss_index = faiss.read_index(str(path / "source_code.index"))
        with (path / "tfidf.pkl").open("rb") as f:
            self.vectorizer: TfidfVectorizer = pickle.load(f)
        with (path / "hasher.pkl").open("rb") as f:
            self.hasher: FeatureHasher = pickle.load(f)
        with (path / "3gpp_spec.index").open("rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        with (path / "3gpp_spec_content.json").open() as f:
            self.spec_docs: List[str] = json.load(f)
        logger.info("Explainability indexes loaded")

    def explain(self, log_window: str) -> Dict[str, Any]:
        """Return contextual explanations for ``log_window``."""
        return {
            "procedure": self._lookup_openapi(log_window),
            "spec": self._lookup_spec(log_window),
            "code": self._lookup_code(log_window),
        }

    def _lookup_openapi(self, text: str) -> Dict[str, Any] | None:
        for key, meta in self.openapi_map.items():
            if key.lower() in text.lower():
                return meta
        return None

    def _lookup_spec(self, text: str) -> Dict[str, Any] | None:
        tokens = text.split()
        scores = self.bm25.get_scores(tokens)
        if not scores.any():
            return None
        best = int(scores.argmax())
        return {"text": self.spec_docs[best], "score": float(scores[best])}

    def _lookup_code(self, text: str) -> List[Dict[str, Any]]:
        logs = re.findall(r"\"([^\"]+)\"", text)
        if not logs:
            logs = [text]
        vec = self.vectorizer.transform([" ".join(logs)])
        feats = {str(idx): float(vec[0, idx]) for idx in vec.nonzero()[1]}
        dense = self.hasher.transform([feats]).toarray()[0].astype("float32")
        distances, indices = self.faiss_index.search(dense.reshape(1, -1), 3)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "function": self.code_map[idx],
                "distance": float(dist),
            })
        return results
