#!/usr/bin/env python3
"""Offline builders for explainability indexes."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List

import faiss
import yaml
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import FeatureHasher, TfidfVectorizer

from janus.preprocess.control_flow_graph import LOG_FUNC_RE, extract_block

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def build_openapi_map(openapi_dir: Path) -> Dict[str, dict]:
    """Return mapping of simplified keywords to OpenAPI context."""
    mapping: Dict[str, dict] = {}
    for path in openapi_dir.rglob("*.yaml"):
        try:
            with path.open() as f:
                doc = yaml.safe_load(f)
        except Exception as exc:  # pragma: no cover - parsing edge cases
            logger.warning("Failed to parse %s: %s", path, exc)
            continue
        spec = doc.get("info", {}).get("title", path.stem)
        paths = doc.get("paths", {})
        for endpoint, ops in paths.items():
            if not isinstance(ops, dict):
                continue
            for details in ops.values():
                if not isinstance(details, dict):
                    continue
                summary = details.get("summary") or details.get("description", "")
                if not summary:
                    continue
                key = re.sub(r"\W+", "", summary)[:50]
                mapping[key] = {
                    "endpoint": endpoint,
                    "description": summary,
                    "spec": spec,
                }
    logger.info("Collected %d OpenAPI entries", len(mapping))
    return mapping


def collect_function_logs(source_dir: Path) -> Dict[str, List[str]]:
    """Map function identifier to list of log templates."""
    mapping: Dict[str, List[str]] = {}
    for path in source_dir.rglob("*"):
        if path.suffix not in {".c", ".h"}:
            continue
        text = path.read_text(errors="ignore")
        for m in re.finditer(r"\w[\w\s\*]+\s+(\w+)\s*\([^)]*\)\s*{", text):
            func = m.group(1)
            start = m.end() - 1
            try:
                block, _ = extract_block(text, start)
            except Exception:  # pragma: no cover - malformed code
                continue
            logs = LOG_FUNC_RE.findall(block)
            if logs:
                ident = f"{path.relative_to(source_dir)}::{func}"
                mapping[ident] = logs
    logger.info("Collected %d functions with logs", len(mapping))
    return mapping


def build_source_code_index(source_dir: Path, out_dir: Path) -> None:
    """Create FAISS index and related artifacts for source code logs."""
    func_map = collect_function_logs(source_dir)
    corpus = [" ".join(logs) for logs in func_map.values()]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    hasher = FeatureHasher(n_features=256, input_type="dict")

    index = faiss.IndexFlatL2(256)
    id_map: List[str] = []
    for i, (func, logs) in enumerate(func_map.items()):
        vec = vectorizer.transform([" ".join(logs)])
        feats = {str(idx): float(vec[0, idx]) for idx in vec.nonzero()[1]}
        dense = hasher.transform([feats]).toarray()[0].astype("float32")
        index.add(dense.reshape(1, -1))
        id_map.append(func)
    faiss.write_index(index, str(out_dir / "source_code.index"))
    with (out_dir / "source_code_map.json").open("w") as f:
        json.dump(id_map, f)
    with (out_dir / "tfidf.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)
    with (out_dir / "hasher.pkl").open("wb") as f:
        pickle.dump(hasher, f)
    logger.info("Saved source code index with %d entries", len(id_map))


def build_spec_index(spec_path: Path, out_dir: Path) -> None:
    """Create BM25 index from 3GPP specification JSON file."""
    with spec_path.open() as f:
        data = json.load(f)
    docs = [item["content"] for item in data]
    tokenized = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    with (out_dir / "3gpp_spec.index").open("wb") as f:
        pickle.dump(bm25, f)
    with (out_dir / "3gpp_spec_content.json").open("w") as f:
        json.dump(docs, f)
    logger.info("Saved spec index with %d documents", len(docs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explainability indexes")
    parser.add_argument("--openapi-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--spec-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    openapi_map = build_openapi_map(args.openapi_dir)
    with (args.output_dir / "openapi_map.json").open("w") as f:
        json.dump(openapi_map, f)

    build_source_code_index(args.source_dir, args.output_dir)
    build_spec_index(args.spec_path, args.output_dir)


if __name__ == "__main__":
    main()
