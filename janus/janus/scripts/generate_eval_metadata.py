"""Generate anomaly metadata from evaluation log files."""

from __future__ import annotations
import os
import argparse
import json
from pathlib import Path
from typing import Iterable

from janus.utils.paths import project_root
PROJECT_ROOT = project_root()

def collect_error_lines(log_path: Path) -> list[int]:
    """Return a list of 1-indexed line numbers containing the ERROR log level."""
    error_lines: list[int] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if "ERROR" in line or "FATAL" in line:
                error_lines.append(line_number)
    return error_lines


def iter_log_files(base_dir: Path) -> Iterable[Path]:
    """Yield all `.log` files under ``base_dir`` in sorted order."""
    yield from sorted(
        path for path in base_dir.rglob("*.log") if path.is_file()
    )


def generate_metadata(eval_dir: Path) -> dict[str, list[int]]:
    """Create the anomaly metadata for the given evaluation directory.

    Args:
        eval_dir: Directory containing log files to evaluate.

    Returns:
        A dictionary mapping the resolved log file path to a list of anomaly
        line numbers.
    """
    metadata: dict[str, list[int]] = {}
    for log_path in iter_log_files(eval_dir):
        metadata[str(log_path.resolve())] = collect_error_lines(log_path)
    return metadata


def main() -> None:
    """Parse arguments and write the metadata file."""
    parser = argparse.ArgumentParser(
        description="Generate anomaly metadata from evaluation log files.",
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, "data/eval_data")),
        help="Path to the directory containing evaluation log files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, "data/eval_data/eval_metadata.json")),
        help=(
            "Optional path for the metadata JSON file. Defaults to `metadata.json`"
            " in the evaluation directory."
        ),
    )
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    output_path: Path = args.output or eval_dir / "eval_metadata.json"

    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    metadata = generate_metadata(eval_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


if __name__ == "__main__":
    main()
