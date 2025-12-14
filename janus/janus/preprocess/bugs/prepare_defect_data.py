"""Build the defect fine-tuning corpus from curated bug reports.

This module materialises synthetic anomalous log files and associated
metadata. The generated corpus is persisted under
``data/preprocessed_data/defect_corpus`` so that the training pipeline can
consume it without creating temporary directories at runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from janus.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = resolve_path("data/preprocessed_data/defect_data")
DEFAULT_HIGH_CONF_FILE = resolve_path(
    "data/preprocessed_data/high_confidence_data.json"
)


def load_buggy_snippets(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load anomalous log snippets grouped by network function.

    The high-confidence bug file produced by :mod:`prepare_bug_data` contains a
    ``buggy_log_snippet`` mapping for each issue.  This helper normalises the
    structure to ``{nf: [(issue_id, snippet_text), ...]}`` so that downstream
    processing can iterate deterministically.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"High-confidence data not found: {path}")

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    nf_to_snippets: dict[str, list[tuple[str, str]]] = {}
    for entry in data:
        snippets = entry.get("buggy_log_snippet") or {}
        if not isinstance(snippets, dict):
            continue
        issue_id = entry.get("issue_id")
        issue_str = str(issue_id) if issue_id is not None else "unknown"
        for nf_name, text in snippets.items():
            if not isinstance(text, str):
                continue
            cleaned = text.strip("\n")
            if not cleaned:
                continue
            nf_key = str(nf_name).strip().lower()
            nf_to_snippets.setdefault(nf_key, []).append((issue_str, cleaned))

    return nf_to_snippets


def _write_anomalous_logs(
    anomaly_root: Path,
    snippets: dict[str, list[tuple[str, str]]],
    *,
    metadata_source: Path,
    metadata_path: Path,
) -> Path:
    """Materialise anomalous log files and return the metadata path."""

    anomaly_root = Path(anomaly_root)
    rows: list[dict[str, object]] = []

    for nf, entries in sorted(snippets.items()):
        nf_dir = anomaly_root / nf
        nf_dir.mkdir(parents=True, exist_ok=True)
        for idx, (issue_id, text) in enumerate(entries, start=1):
            snippet = text.strip("\n")
            if not snippet:
                continue
            lines = [line.rstrip("\n") for line in snippet.splitlines()]
            filename = f"issue-{issue_id}-sample-{idx}-{nf}.log"
            file_path = nf_dir / filename
            file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            rel_path = file_path.relative_to(metadata_path.parent)
            rows.append(
                {
                    "file": rel_path.as_posix(),
                    "nf": nf,
                    "issue_id": issue_id,
                    "lines": [
                        {"anomaly_line_number": line_no + 1}
                        for line_no in range(len(lines))
                    ],
                }
            )

    if not rows:
        raise ValueError("No anomalous log snippets were materialised for training.")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": Path(metadata_source).as_posix(),
        "rows": rows,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def prepare_defect_corpus(
    *,
    high_conf_file: Path = DEFAULT_HIGH_CONF_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_snippets_per_nf: int | None = None,
) -> Path:
    """Prepare the persistent defect corpus used during fine-tuning."""

    high_conf_file = Path(high_conf_file)
    output_dir = Path(output_dir)

    snippets = load_buggy_snippets(high_conf_file)
    if max_snippets_per_nf is not None and max_snippets_per_nf > 0:
        snippets = {
            nf: entries[:max_snippets_per_nf]
            for nf, entries in snippets.items()
            if entries
        }

    filtered_snippets = {nf: entries for nf, entries in snippets.items() if entries}
    if not filtered_snippets:
        raise ValueError("No anomalous log snippets available for defect corpus generation.")

    total_snippets = sum(len(v) for v in filtered_snippets.values())

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    metadata_path = _write_anomalous_logs(
        output_dir,
        filtered_snippets,
        metadata_source=high_conf_file,
        metadata_path=metadata_path,
    )

    LOGGER.info(
        "Materialised %d anomalous snippet(s) across %d NF(s) into %s",
        total_snippets,
        len(filtered_snippets),
        metadata_path,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "high_confidence_source": high_conf_file.as_posix(),
        "anomaly_metadata": metadata_path.relative_to(output_dir).as_posix(),
        "network_functions": sorted(filtered_snippets.keys()),
        "total_anomaly_files": total_snippets,
        "max_snippets_per_nf": max_snippets_per_nf,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return metadata_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by :func:`main`."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--high-confidence-file",
        type=Path,
        default=DEFAULT_HIGH_CONF_FILE,
        help="Path to the curated high-confidence bug dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the prepared defect corpus.",
    )
    parser.add_argument(
        "--max-snippets-per-nf",
        type=int,
        default=None,
        help="Optional limit on the number of anomalous snippets to materialise per NF.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Entry point used by the preprocessing workflow."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args(argv)
    metadata_path = prepare_defect_corpus(
        high_conf_file=resolve_path(args.high_confidence_file),
        output_dir=resolve_path(args.output_dir),
        max_snippets_per_nf=args.max_snippets_per_nf,
    )
    LOGGER.info("Defect corpus metadata written to %s", metadata_path)
    return metadata_path


if __name__ == "__main__":
    main()
