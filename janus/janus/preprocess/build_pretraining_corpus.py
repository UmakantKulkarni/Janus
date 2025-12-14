#!/usr/bin/env python3
"""Aggregate domain-specific text into a corpus for continual pre-training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Iterable, Any, Tuple

import ijson
import yaml
import re

from janus.dataset import chunk_content

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw_data")
OPENAPI_DIR = RAW_DATA_DIR / "3gpp_openapi_yaml_rel17"
SPEC_DIR = RAW_DATA_DIR / "spec_3gpp"
CODE_DIR = RAW_DATA_DIR / "open5gs_source_code"

OUTPUT_FILE = Path("data/preprocessed_data/pretraining_corpus.txt")

# Approximate maximum characters per output segment. This keeps each chunk
# within roughly 1k tokens so metadata is preserved.
MAX_SEGMENT_CHARS = 4000


def iter_strings(obj: Any) -> Iterable[str]:
    """Recursively yield all string values from ``obj``."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_strings(item)


def with_metadata(metadata: str, text: str) -> Iterator[str]:
    """Yield ``text`` split into chunks with ``metadata`` prepended."""
    for chunk in chunk_content(text, MAX_SEGMENT_CHARS - len(metadata)):
        yield metadata + chunk


def collect_openapi_texts(directory: Path) -> Iterator[str]:
    """Yield descriptive strings from YAML files."""
    for path in directory.rglob("*.yaml"):
        try:
            with path.open() as f:
                doc = yaml.safe_load(f)
        except Exception as exc:  # pragma: no cover - parse errors
            logger.warning("Failed to parse %s: %s", path, exc)
            continue
        for text in iter_strings(doc):
            if text.strip():
                yield text.strip()


def extract_named_functions(content: str, ext: str) -> Iterator[Tuple[str, str]]:
    """Return ``(name, body)`` tuples for functions in ``content``."""
    if ext in {".c", ".h"}:
        pattern = re.finditer(r"(\w[\w\s\*]+)\s+(\w+)\s*\([^;]*\)\s*\{", content)
        for match in pattern:
            name = match.group(2)
            start = match.start()
            brace = 1
            idx = match.end()
            while idx < len(content) and brace:
                brace += (content[idx] == "{") - (content[idx] == "}")
                idx += 1
            yield name, content[start:idx]
    elif ext == ".py":
        pattern = re.finditer(r"^def\s+(\w+)\s*\(.*?\):", content, re.M)
        for match in pattern:
            name = match.group(1)
            lines = content[match.start():].splitlines(True)
            indent = len(lines[0]) - len(lines[0].lstrip())
            body = lines[0]
            for ln in lines[1:]:
                if ln.startswith(" " * (indent + 1)):
                    body += ln
                else:
                    break
            yield name, body


def collect_spec_texts(directory: Path) -> Iterator[str]:
    """Yield paragraphs from spec JSON files."""
    for path in directory.glob("*.json"):
        try:
            with path.open("rb") as f:
                for item in ijson.items(f, "item"):
                    if isinstance(item, dict) and "content" in item:
                        spec = item.get("specnumber", "")
                        release = item.get("release", "")
                        title = item.get("title", "")
                        meta = f"[SPEC {spec} Release {release} Section {title}] "
                        text = item["content"]
                        if isinstance(text, str) and text.strip():
                            for chunk in with_metadata(meta, text.strip()):
                                yield chunk
        except Exception as exc:  # pragma: no cover - malformed JSON
            logger.warning("Failed to read %s: %s", path, exc)
            continue


def collect_code_texts(directory: Path) -> Iterator[str]:
    """Yield code chunks annotated with file and function name."""
    for path in directory.rglob("*"):
        if path.suffix.lower() not in {".c", ".h"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - read errors
            logger.warning("Failed to read %s: %s", path, exc)
            continue
        rel = path.relative_to(directory)
        funcs = list(extract_named_functions(text, path.suffix.lower()))
        if not funcs:
            meta = f"[CODE {rel}] "
            for chunk in with_metadata(meta, text):
                yield chunk
        else:
            for name, body in funcs:
                meta = f"[CODE {rel}::{name}] "
                for chunk in with_metadata(meta, body):
                    yield chunk


def main() -> None:
    """Build the aggregated pretraining corpus."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUTPUT_FILE.open("w") as out:
        # for text in collect_openapi_texts(OPENAPI_DIR):
        #     out.write(text + "\n")
        #     count += 1
        for text in collect_spec_texts(SPEC_DIR):
            out.write(text + "\n")
            count += 1
        for text in collect_code_texts(CODE_DIR):
            out.write(text + "\n")
            count += 1
    logger.info("Wrote %d segments to %s", count, OUTPUT_FILE)


if __name__ == "__main__":
    main()
