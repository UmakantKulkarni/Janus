#!/usr/bin/env python3
"""Bulk evaluation for Open5GS logs using the Janus inference stack."""

from __future__ import annotations
import os
import argparse
import csv
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TextIO

from janus.infer.inference import build_csv_row, load_model, run_inference
from janus.utils.paths import load_repo_config, resolve_path, project_root

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = project_root()

APP_NAMES: Tuple[str, ...] = (
    "amf",
    "smf",
    "upf",
    "ausf",
    "pcf",
    "udm",
    "udr",
    "nrf",
    "nssf",
    "bsf",
    "scp",
)


def _discover_log_files(eval_dir: Path) -> List[Path]:
    """Return all ``.log`` files found recursively under ``eval_dir``."""

    return sorted(p for p in eval_dir.rglob("*.log") if p.is_file())


def _extract_app_name(log_path: Path) -> str:
    """Extract an Open5GS application name from ``log_path``."""

    lowered = log_path.name.lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    for token in tokens:
        if token in APP_NAMES:
            return token
    for parent in log_path.parents:
        candidate = parent.name.lower()
        if candidate in APP_NAMES:
            return candidate
    raise ValueError(f"Unable to determine app_name from {log_path}")


def _prepare_calibration_dir(path: Optional[str]) -> Optional[Path]:
    """Resolve an optional calibration directory argument."""

    if not path:
        return None
    resolved = resolve_path(path)
    if not resolved.exists():
        logger.warning("Calibration directory %s does not exist; ignoring.", resolved)
        return None
    if not resolved.is_dir():
        logger.warning("Calibration path %s is not a directory; ignoring.", resolved)
        return None
    return resolved


def _open_csv(csv_path: Path, overwrite: bool) -> Tuple[csv.writer, TextIO]:
    """Create a CSV writer for ``csv_path`` handling overwrite behaviour."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        if overwrite:
            csv_path.unlink()
        else:
            raise FileExistsError(
                f"CSV file {csv_path} already exists. Use --overwrite to replace it."
            )
    handle = csv_path.open("w", newline="")
    return csv.writer(handle), handle


def _parse_args(default_seq_len: int) -> argparse.Namespace:
    """Parse CLI arguments for the evaluation script."""

    parser = argparse.ArgumentParser(
        description="Run bulk inference on Open5GS evaluation log files.",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data/eval_data"),
        help=(
            "Directory containing evaluation log files. Relative paths are resolved "
            "against the project root."
        ),
    )
    parser.add_argument(
        "--csv-log-file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "janus_inference_metrics.csv"),
        help="Destination CSV file that will collect metrics for all logs.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=default_seq_len,
        help="Maximum number of tokens per window. Defaults to config infer.seq_len.",
    )
    parser.add_argument(
        "--calibrate-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts/calibrate"),
        help="Optional directory containing calibration score distributions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the CSV file if it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of log files to evaluate.",
    )
    return parser.parse_args()


def _resolve_eval_dir(path: str) -> Path:
    """Resolve and validate the evaluation directory path."""

    eval_dir = resolve_path(path)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory {eval_dir} does not exist")
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"Evaluation path {eval_dir} is not a directory")
    return eval_dir


def main() -> None:
    """Entry point for running bulk inference across evaluation datasets."""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    config = load_repo_config()
    default_seq_len = config.get("infer", {}).get("seq_len", 1024)
    args = _parse_args(default_seq_len)

    eval_dir = _resolve_eval_dir(args.eval_dir)
    csv_path = resolve_path(args.csv_log_file)
    calibrate_dir = _prepare_calibration_dir(args.calibrate_dir)

    logger.info("Loading model once for bulk inference")
    model, tokenizer, device, pad_id = load_model(config)

    log_files = _discover_log_files(eval_dir)
    if args.limit is not None:
        log_files = log_files[: max(args.limit, 0)]
    if not log_files:
        logger.warning("No log files found under %s", eval_dir)
        return

    writer, handle = _open_csv(csv_path, overwrite=args.overwrite)
    header: Optional[Sequence[str]] = None
    insert_index: Optional[int] = None

    try:
        total = len(log_files)
        logger.info("Processing %d log files from %s", total, eval_dir)
        for index, log_path in enumerate(log_files, start=1):
            try:
                app_name = _extract_app_name(log_path)
            except ValueError as exc:
                logger.error("Skipping %s: %s", log_path, exc)
                continue

            nf = app_name.lower()
            logger.info(
                "[%d/%d] Running inference on %s (app=%s, nf=%s)",
                index,
                total,
                log_path,
                app_name,
                nf,
            )

            try:
                summary = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    pad_id=pad_id,
                    log_path=log_path,
                    nf=nf,
                    max_tokens=args.max_tokens,
                    csv_path=None,
                    calibrate_dir=calibrate_dir,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Inference failed for %s: %s", log_path, exc)
                continue

            metrics_mean = summary.get("metrics_mean")
            metrics_max = summary.get("metrics_max")
            stats = summary.get("reference_stats")
            header_base, row_base = build_csv_row(
                result=summary,
                log_path=log_path,
                nf=nf,
                num_lines=int(summary.get("num_lines", 0)),
                metrics_mean=metrics_mean,
                metrics_max=metrics_max,
                stats=stats,
            )

            row = list(row_base)
            if header is None:
                header = list(header_base)
                try:
                    insert_index = header.index("nf") + 1
                except ValueError:
                    insert_index = len(header)
                header.insert(insert_index, "app_name")
                writer.writerow(header)
            assert insert_index is not None
            row.insert(insert_index, app_name)
            writer.writerow(row)
    finally:
        handle.close()
        logger.info("Inference metrics written to %s", csv_path)


if __name__ == "__main__":
    main()
