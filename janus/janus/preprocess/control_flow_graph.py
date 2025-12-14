#!/usr/bin/env python3
"""Static analysis for Open5GS control-flow guided log mapping."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from janus.utils.paths import resolve_path

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

LOG_FUNC_RE = re.compile(r'ogs_(?:info|error|warn|debug|notice|crit|emerg)\s*\(\s*"([^"]+)"')

def extract_block(text: str, start: int) -> Tuple[str, int]:
    """Return block substring and index after closing brace."""
    assert text[start] == "{"
    depth = 1
    i = start + 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start + 1 : i - 1], i


def find_if_else_pairs(text: str) -> List[Tuple[str, str]]:
    """Yield pairs of if/else block contents."""
    pairs: List[Tuple[str, str]] = []
    idx = 0
    while True:
        m = re.search(r"\bif\s*\([^)]*\)\s*{", text[idx:], re.S)
        if not m:
            break
        start = idx + m.end() - 1
        if_block, pos = extract_block(text, start)
        idx = pos
        m_else = re.match(r"\s*else\s*{", text[idx:], re.S)
        if not m_else:
            continue
        start_else = idx + m_else.end() - 1
        else_block, idx = extract_block(text, start_else)
        pairs.append((if_block, else_block))
    return pairs


def build_cfg_map(source_dir: Path) -> Dict[str, str]:
    """Build mapping of log templates from ``if`` to ``else`` blocks."""
    mapping: Dict[str, str] = {}
    for path in source_dir.rglob("*.c"):
        logger.info("Analyzing %s", path)
        text = path.read_text(errors="ignore")
        for if_block, else_block in find_if_else_pairs(text):
            if_msgs = LOG_FUNC_RE.findall(if_block)
            else_msgs = LOG_FUNC_RE.findall(else_block)
            if if_msgs and else_msgs:
                for msg in if_msgs:
                    mapping[msg] = else_msgs[0]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate control-flow map")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=resolve_path("data/raw_data/open5gs_source_code"),
        help="Source directory to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=resolve_path("data/preprocessed_data/log_code_cfg_map.json"),
        help="Output JSON file"
    )
    args = parser.parse_args()

    cfg_map = build_cfg_map(args.source_dir)
    with args.output.open("w+") as f:
        json.dump(cfg_map, f, indent=2)
    logger.info("Wrote %d pairs to %s", len(cfg_map), args.output)


if __name__ == "__main__":
    main()
