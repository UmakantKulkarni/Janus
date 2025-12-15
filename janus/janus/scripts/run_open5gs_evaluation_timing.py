#!/usr/bin/env python3
"""
Detailed timing evaluation for Open5GS logs using the Janus inference stack.

## Timing instrumentation metrics

`janus/janus/scripts/run_open5gs_evaluation_timing.py` mirrors the bulk
inference runner while capturing fine-grained wall-clock durations for every
stage of the pipeline. The script emits one CSV row per processed log and a
batch-level summary row. All durations are recorded in milliseconds using
`time.perf_counter()` to ensure monotonic timing even across context switches.

- `record_type`: Identifies whether the row represents a single log (`log`) or
  a batch aggregate (`batch`).
- `batch_index`: 1-based index of the batch emitted by the `--batch-size`
  parameter.
- `log_index`: Position of the log inside its batch, or `*` for batch summary
  rows.
- `log_file`: Absolute path to the processed log file (or a semicolon-separated
  list of paths for a batch row).
- `app_name`: Open5GS application inferred from the log filename or directory
  (for example `amf`, `smf`, or `upf`).
- `nf`: Lower-case network-function identifier used during tokenisation.
- `num_lines`: Number of raw log lines that were read from disk for the log or
  sum of lines over the batch.
- `num_windows`: Number of token windows created for the log (or aggregated
  across the batch).
- `file_open_ms`: Time between calling `Path.open()` and obtaining a usable file
  handle, capturing file-system latency.
- `read_lines_ms`: Duration spent iterating the open file to collect raw lines
  into memory prior to any cleaning.
- `clean_lines_ms`: Latency to remove ANSI escape sequences from every raw line
  before tokenisation.
- `tagger_init_ms`: Cost to instantiate `LogTagger`, including loading any
  tokenizer tables required for tokenisation.
- `nf_prefix_tokens_ms`: Time to build the cached `NF:<name>` prefix tokens and
  their tags for the current network function.
- `nf_prefix_application_ms`: Aggregated latency for pre-pending those prefix
  tokens to every window that is constructed.
- `tokenization_ms`: Total time spent running `LogTagger.tag_line()` on each
  cleaned log line to obtain token IDs and tag IDs.
- `token_append_ms`: Duration used to push each token and tag ID into the
  current window buffer prior to tensor construction.
- `window_finalize_ms`: Overhead of flushing a full or terminal window and
  resetting the accumulation buffers.
- `window_tensor_build_ms`: Time required to wrap the accumulated integer lists
  into PyTorch tensors for tokens and tags.
- `window_range_record_ms`: Duration spent recording the `(start_line,
  end_line)` ranges for each window to support later summarisation.
- `prepare_windows_total_ms`: Wall-clock interval covering the entire window
  preparation routine, from tagger creation through tensor emission.
- `pad_sequences_ms`: Time to pad all token and tag tensors to equal length via
  `torch.nn.utils.rnn.pad_sequence` before model execution.
- `tensor_to_device_ms`: Latency for transferring the padded tensors to the
  inference device (CPU or GPU).
- `model_eval_ms`: Cost of switching the model to evaluation mode with
  `model.eval()` so dropout and other training-only layers are disabled.
- `autocast_context_ms`: Time taken to construct the device-aware autocast
  context manager controlling mixed-precision inference.
- `model_forward_ms`: Measured duration of the forward pass itself inside the
  autocast/no-grad context, including attention and classifier heads.
- `score_windows_total_ms`: Envelope timing for the entire scoring routine,
  covering padding, device transfer, eval mode, autocast setup, and forward
  execution.
- `seq_lens_tensor_ms`: Time to build a tensor of original, unpadded sequence
  lengths for downstream summary calculations.
- `summarize_ms`: Duration consumed by `_summarise_scores`, which collapses
  window logits into per-log aggregates such as mean and max probabilities.
- `load_reference_stats_ms`: Latency to load calibration statistics (e.g.
  percentile distributions) from disk for the active network function.
- `metrics_mean_ms`: Time spent feeding the mean probability into
  `_compute_anomaly_metrics` when calibration data is available. The value is
  `0.0` when the input is `NaN` or statistics are missing.
- `metrics_max_ms`: Equivalent to `metrics_mean_ms`, but using the maximum
  probability summarised from the windows.
- `metrics_total_ms`: Total duration of the anomaly-metric computation block,
  inclusive of both mean and max branches.
- `summary_enrich_ms`: Time to attach metadata (network function, path, counts)
  to the summary dictionary before CSV serialisation.
- `metrics_attach_ms`: Duration required to graft the computed metrics and
  reference statistics onto the summary structure.
- `csv_build_ms`: Time to invoke `build_csv_row` and assemble the inference
  metrics payload that will be written to the primary metrics CSV.
- `metrics_csv_write_ms`: Wall-clock time spent on the `writer.writerow()` call
  that emits the metrics CSV row for the current log. The batch summary row also
  accumulates the total write cost for every log in the batch.
- `timing_row_build_ms`: Cost to assemble the final dictionary that backs the
  timing CSV row, including copying over every measured metric.
- `timing_csv_write_ms`: Duration to flush the timing row to disk. Per-log rows
  carry `0.0` here because the value is captured after writing; the batch row
  aggregates the measured cost.
- `total_ms`: Monotonic duration from the start of per-log processing (window
  preparation) through timing-row construction, providing an end-to-end view of
  the log’s inference latency.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TextIO

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from janus.infer.inference import (
    ANSI_ESCAPE_RE,
    LogTagger,
    _autocast_context,
    _compute_anomaly_metrics,
    _load_nf_stats,
    _nf_prefix_tokens,
    _summarise_scores,
    build_csv_row,
    load_model,
)
from janus.utils.paths import load_repo_config, project_root, resolve_path

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)
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

TIMING_HEADER: Tuple[str, ...] = (
    "record_type",
    "batch_index",
    "log_index",
    "log_file",
    "app_name",
    "nf",
    "num_lines",
    "num_windows",
    "file_open_ms",
    "read_lines_ms",
    "clean_lines_ms",
    "tagger_init_ms",
    "nf_prefix_tokens_ms",
    "nf_prefix_application_ms",
    "tokenization_ms",
    "token_append_ms",
    "window_finalize_ms",
    "window_tensor_build_ms",
    "window_range_record_ms",
    "prepare_windows_total_ms",
    "pad_sequences_ms",
    "tensor_to_device_ms",
    "model_eval_ms",
    "autocast_context_ms",
    "model_forward_ms",
    "seq_lens_tensor_ms",
    "score_windows_total_ms",
    "summarize_ms",
    "load_reference_stats_ms",
    "metrics_mean_ms",
    "metrics_max_ms",
    "metrics_total_ms",
    "summary_enrich_ms",
    "metrics_attach_ms",
    "csv_build_ms",
    "metrics_csv_write_ms",
    "timing_row_build_ms",
    "timing_csv_write_ms",
    "total_ms",
)


@dataclass
class BatchContext:
    """Book-keeping for the batch currently being processed."""

    index: int
    log_paths: Sequence[Path]


def _discover_log_files(eval_dir: Path) -> List[Path]:
    """Return all ``.log`` files found recursively under ``eval_dir``."""

    return sorted(path for path in eval_dir.rglob("*.log") if path.is_file())


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
    """Parse CLI arguments for the timing evaluation script."""

    parser = argparse.ArgumentParser(
        description="Run bulk inference with detailed timing on Open5GS logs.",
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
        "--timing-csv-file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "janus_inference_timing.csv"),
        help="Destination CSV file that will collect per-batch timing information.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of log files to evaluate together when recording timings.",
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
        help="Overwrite CSV files if they already exist.",
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


def _batched(paths: Sequence[Path], batch_size: int) -> Iterator[Sequence[Path]]:
    """Yield ``paths`` grouped into batches of size ``batch_size``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(paths), batch_size):
        yield paths[start : start + batch_size]


def _prepare_windows_with_timing(
    path: Path,
    nf_name: str,
    tokenizer,
    max_tokens: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]], int, Dict[str, float]]:
    """Prepare log windows while collecting timing information."""

    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    tagger_start = time.perf_counter()
    tagger = LogTagger(tokenizer)
    timings["tagger_init_ms"] = (time.perf_counter() - tagger_start) * 1000.0

    prefix_start = time.perf_counter()
    nf_tokens, nf_tags = _nf_prefix_tokens(tagger, nf_name)
    timings["nf_prefix_tokens_ms"] = (time.perf_counter() - prefix_start) * 1000.0

    file_open_start = time.perf_counter()
    handle = path.open(encoding="utf-8", errors="ignore")
    timings["file_open_ms"] = (time.perf_counter() - file_open_start) * 1000.0

    try:
        read_start = time.perf_counter()
        raw_lines = [line.rstrip("\n") for line in handle]
        timings["read_lines_ms"] = (time.perf_counter() - read_start) * 1000.0
    finally:
        handle.close()

    clean_start = time.perf_counter()
    cleaned_lines = [ANSI_ESCAPE_RE.sub("", line) for line in raw_lines]
    timings["clean_lines_ms"] = (time.perf_counter() - clean_start) * 1000.0

    token_tensors: List[torch.Tensor] = []
    tag_tensors: List[torch.Tensor] = []
    window_line_ranges: List[Tuple[int, int]] = []

    tokenization_time = 0.0
    prefix_apply_time = 0.0
    append_time = 0.0
    finalize_time = 0.0
    tensor_build_time = 0.0
    range_record_time = 0.0

    window_tokens: List[int] = []
    window_tags: List[int] = []
    current_window_start: Optional[int] = None

    for line_number, cleaned in enumerate(cleaned_lines, start=1):
        if not window_tokens:
            current_window_start = line_number
        if nf_tokens:
            prefix_apply_start = time.perf_counter()
            window_tokens.extend(nf_tokens)
            window_tags.extend(nf_tags)
            prefix_apply_time += time.perf_counter() - prefix_apply_start
        tokenize_start = time.perf_counter()
        tagged = tagger.tag_line(cleaned)
        tokenization_time += time.perf_counter() - tokenize_start
        truncated = False
        for tagged_token in tagged:
            append_start = time.perf_counter()
            window_tokens.append(tagged_token.token_id)
            window_tags.append(tagged_token.tag_id)
            append_time += time.perf_counter() - append_start
            if len(window_tokens) >= max_tokens:
                finalize_start = time.perf_counter()
                tensor_build_start = time.perf_counter()
                token_tensors.append(torch.tensor(window_tokens, dtype=torch.long))
                tag_tensors.append(torch.tensor(window_tags, dtype=torch.long))
                tensor_build_time += time.perf_counter() - tensor_build_start
                range_record_start = time.perf_counter()
                window_line_ranges.append((current_window_start or line_number, line_number))
                range_record_time += time.perf_counter() - range_record_start
                finalize_time += time.perf_counter() - finalize_start
                window_tokens = []
                window_tags = []
                current_window_start = None
                truncated = True
                break
        if truncated:
            continue

    if window_tokens:
        finalize_start = time.perf_counter()
        tensor_build_start = time.perf_counter()
        token_tensors.append(torch.tensor(window_tokens, dtype=torch.long))
        tag_tensors.append(torch.tensor(window_tags, dtype=torch.long))
        tensor_build_time += time.perf_counter() - tensor_build_start
        range_record_start = time.perf_counter()
        window_line_ranges.append(
            (current_window_start or len(cleaned_lines), len(cleaned_lines))
        )
        range_record_time += time.perf_counter() - range_record_start
        finalize_time += time.perf_counter() - finalize_start

    timings["tokenization_ms"] = tokenization_time * 1000.0
    timings["nf_prefix_application_ms"] = prefix_apply_time * 1000.0
    timings["token_append_ms"] = append_time * 1000.0
    timings["window_finalize_ms"] = finalize_time * 1000.0
    timings["window_tensor_build_ms"] = tensor_build_time * 1000.0
    timings["window_range_record_ms"] = range_record_time * 1000.0
    timings["prepare_windows_total_ms"] = (time.perf_counter() - total_start) * 1000.0

    return token_tensors, tag_tensors, window_line_ranges, len(cleaned_lines), timings


def _score_windows_with_timing(
    model,
    token_tensors: Sequence[torch.Tensor],
    tag_tensors: Sequence[torch.Tensor],
    pad_id: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Score windows with the model while collecting timing information."""

    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    pad_start = time.perf_counter()
    ids = pad_sequence(token_tensors, batch_first=True, padding_value=pad_id)
    tags = pad_sequence(tag_tensors, batch_first=True, padding_value=0)
    timings["pad_sequences_ms"] = (time.perf_counter() - pad_start) * 1000.0

    device_start = time.perf_counter()
    ids = ids.to(device)
    tags = tags.to(device)
    timings["tensor_to_device_ms"] = (time.perf_counter() - device_start) * 1000.0

    eval_start = time.perf_counter()
    model.eval()
    timings["model_eval_ms"] = (time.perf_counter() - eval_start) * 1000.0

    autocast_start = time.perf_counter()
    ctx = _autocast_context(device)
    timings["autocast_context_ms"] = (time.perf_counter() - autocast_start) * 1000.0

    with torch.no_grad():
        with ctx:
            forward_start = time.perf_counter()
            out = model(ids, tags)
            timings["model_forward_ms"] = (time.perf_counter() - forward_start) * 1000.0

    timings["score_windows_total_ms"] = (time.perf_counter() - total_start) * 1000.0
    seq_lens_start = time.perf_counter()
    seq_lens_tensor = torch.tensor(
        [tensor.numel() for tensor in token_tensors], device=device
    )
    timings["seq_lens_tensor_ms"] = (time.perf_counter() - seq_lens_start) * 1000.0
    return {"cls_logits": out["cls"].view(-1), "seq_lens": seq_lens_tensor}, timings


def _summarise_with_timing(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    window_ranges: Sequence[Tuple[int, int]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Summarise logits into metrics while recording timing information."""

    timings: Dict[str, float] = {}
    start = time.perf_counter()
    summary = _summarise_scores(logits, seq_lens, window_ranges)
    timings["summarize_ms"] = (time.perf_counter() - start) * 1000.0
    return summary, timings


def _compute_metrics_with_timing(
    nf: str,
    summary: Dict[str, float],
    calibrate_dir: Optional[Path],
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[Dict[str, float]], Dict[str, float]]:
    """Compute anomaly metrics for ``summary`` recording timing information."""

    timings: Dict[str, float] = {}
    metrics_mean: Optional[Dict[str, float]] = None
    metrics_max: Optional[Dict[str, float]] = None
    stats: Optional[Dict[str, float]] = None

    stats_start = time.perf_counter()
    stats = _load_nf_stats(nf, calibrate_dir)
    timings["load_reference_stats_ms"] = (time.perf_counter() - stats_start) * 1000.0

    metrics_start = time.perf_counter()
    if stats:
        if not np.isnan(summary.get("cls_prob_mean", np.nan)):
            mean_start = time.perf_counter()
            metrics_mean = _compute_anomaly_metrics(summary["cls_prob_mean"], stats)
            timings["metrics_mean_ms"] = (time.perf_counter() - mean_start) * 1000.0
        else:
            timings["metrics_mean_ms"] = 0.0
        if not np.isnan(summary.get("cls_prob_max", np.nan)):
            max_start = time.perf_counter()
            metrics_max = _compute_anomaly_metrics(summary["cls_prob_max"], stats)
            timings["metrics_max_ms"] = (time.perf_counter() - max_start) * 1000.0
        else:
            timings["metrics_max_ms"] = 0.0
    else:
        timings["metrics_mean_ms"] = 0.0
        timings["metrics_max_ms"] = 0.0
    timings["metrics_total_ms"] = (time.perf_counter() - metrics_start) * 1000.0

    return metrics_mean, metrics_max, stats, timings


def _prepare_timing_row(
    batch_index: int,
    log_index: int,
    log_path: Path,
    app_name: str,
    nf: str,
    num_lines: int,
    num_windows: int,
    timings: Dict[str, float],
    metrics_csv_write_ms: float,
    timing_csv_write_ms: float,
) -> List[object]:
    """Build a row for the timing CSV using gathered timings."""

    build_start = time.perf_counter()
    row_values: Dict[str, object] = {
        "record_type": "log",
        "batch_index": batch_index,
        "log_index": log_index,
        "log_file": str(log_path),
        "app_name": app_name,
        "nf": nf,
        "num_lines": num_lines,
        "num_windows": num_windows,
        "metrics_csv_write_ms": round(metrics_csv_write_ms, 6),
        "timing_csv_write_ms": round(timing_csv_write_ms, 6),
    }
    for key, value in timings.items():
        if key == "total_ms":
            continue
        row_values[key] = round(value, 6)
    if "total_ms" in timings:
        row_values["total_ms"] = round(timings["total_ms"], 6)
    row_values["timing_row_build_ms"] = round(
        (time.perf_counter() - build_start) * 1000.0, 6
    )
    return [row_values.get(column, "") for column in TIMING_HEADER]


def _batch_summary_row(
    batch: BatchContext,
    batch_timings: Dict[str, float],
    totals: Dict[str, float],
) -> List[object]:
    """Build an aggregate summary row for the batch timing CSV."""

    row = {
        "record_type": "batch",
        "batch_index": batch.index,
        "log_index": "*",
        "log_file": ";".join(str(path) for path in batch.log_paths),
        "app_name": "",
        "nf": "",
        "num_lines": totals.get("num_lines", 0),
        "num_windows": totals.get("num_windows", 0),
        "metrics_csv_write_ms": batch_timings.get("metrics_csv_write_ms", 0.0),
        "timing_csv_write_ms": batch_timings.get("timing_csv_write_ms", 0.0),
        "total_ms": batch_timings.get("total_ms", 0.0),
    }
    row.update({key: round(value, 6) for key, value in batch_timings.items() if key.endswith("_ms")})
    return [row.get(column, "") for column in TIMING_HEADER]


def main() -> None:
    """Entry point for running bulk inference with timing instrumentation."""

    config = load_repo_config()
    default_seq_len = config.get("infer", {}).get("seq_len", 1024)
    args = _parse_args(default_seq_len)

    eval_dir = _resolve_eval_dir(args.eval_dir)
    csv_path = resolve_path(args.csv_log_file)
    timing_csv_path = resolve_path(args.timing_csv_file)
    calibrate_dir = _prepare_calibration_dir(args.calibrate_dir)

    logger.info("Loading model once for bulk inference")
    model_start = time.perf_counter()
    model, tokenizer, device, pad_id = load_model(config)
    model_load_ms = (time.perf_counter() - model_start) * 1000.0
    logger.info("Model loaded in %.2f ms", model_load_ms)

    log_files = _discover_log_files(eval_dir)
    if args.limit is not None:
        log_files = log_files[: max(args.limit, 0)]
    if not log_files:
        logger.warning("No log files found under %s", eval_dir)
        return

    writer, handle = _open_csv(csv_path, overwrite=args.overwrite)
    timing_writer, timing_handle = _open_csv(timing_csv_path, overwrite=args.overwrite)

    header: Optional[Sequence[str]] = None
    insert_index: Optional[int] = None

    try:
        total_logs = len(log_files)
        logger.info("Processing %d log files from %s", total_logs, eval_dir)
        timing_writer.writerow(TIMING_HEADER)

        for batch_index, batch_paths in enumerate(_batched(log_files, args.batch_size), start=1):
            batch_context = BatchContext(index=batch_index, log_paths=batch_paths)
            batch_start = time.perf_counter()
            batch_totals: Dict[str, float] = {"num_lines": 0.0, "num_windows": 0.0}
            batch_timings: Dict[str, float] = {
                "metrics_csv_write_ms": 0.0,
                "timing_csv_write_ms": 0.0,
            }

            for log_index, log_path in enumerate(batch_paths, start=1):
                try:
                    app_name = _extract_app_name(log_path)
                except ValueError as exc:
                    logger.error("Skipping %s: %s", log_path, exc)
                    continue

                nf = app_name.lower()
                logger.info(
                    "[batch %d][%d/%d] Running inference on %s (app=%s, nf=%s)",
                    batch_index,
                    log_index,
                    total_logs,
                    log_path,
                    app_name,
                    nf,
                )

                total_start = time.perf_counter()
                (
                    token_tensors,
                    tag_tensors,
                    window_ranges,
                    num_lines,
                    prep_timings,
                ) = _prepare_windows_with_timing(log_path, nf, tokenizer, args.max_tokens)

                batch_totals["num_lines"] += num_lines
                batch_totals["num_windows"] += len(token_tensors)

                outputs, score_timings = _score_windows_with_timing(
                    model=model,
                    token_tensors=token_tensors,
                    tag_tensors=tag_tensors,
                    pad_id=pad_id,
                    device=device,
                )

                summary, summarize_timings = _summarise_with_timing(
                    outputs["cls_logits"],
                    outputs["seq_lens"],
                    window_ranges,
                )
                summary_enrich_start = time.perf_counter()
                summary["nf"] = nf
                summary["log_file"] = str(log_path)
                summary["num_lines"] = num_lines
                summary["num_windows"] = len(token_tensors)
                summary_enrich_ms = (time.perf_counter() - summary_enrich_start) * 1000.0

                (
                    metrics_mean,
                    metrics_max,
                    stats,
                    metric_timings,
                ) = _compute_metrics_with_timing(nf, summary, calibrate_dir)

                metrics_attach_start = time.perf_counter()
                summary["metrics_mean"] = metrics_mean
                summary["metrics_max"] = metrics_max
                summary["reference_stats"] = stats
                metrics_attach_ms = (time.perf_counter() - metrics_attach_start) * 1000.0

                csv_build_start = time.perf_counter()
                header_base, row_base = build_csv_row(
                    result=summary,
                    log_path=log_path,
                    nf=nf,
                    num_lines=num_lines,
                    metrics_mean=metrics_mean,
                    metrics_max=metrics_max,
                    stats=stats,
                )
                csv_build_ms = (time.perf_counter() - csv_build_start) * 1000.0

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

                metrics_write_start = time.perf_counter()
                writer.writerow(row)
                metrics_csv_write_ms = (time.perf_counter() - metrics_write_start) * 1000.0
                batch_timings["metrics_csv_write_ms"] += metrics_csv_write_ms

                timings: Dict[str, float] = {}
                timings.update(prep_timings)
                timings.update(score_timings)
                timings.update(summarize_timings)
                timings.update(metric_timings)
                timings["summary_enrich_ms"] = summary_enrich_ms
                timings["metrics_attach_ms"] = metrics_attach_ms
                timings["csv_build_ms"] = csv_build_ms
                timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0

                timing_row = _prepare_timing_row(
                    batch_index=batch_context.index,
                    log_index=log_index,
                    log_path=log_path,
                    app_name=app_name,
                    nf=nf,
                    num_lines=num_lines,
                    num_windows=len(token_tensors),
                    timings=timings,
                    metrics_csv_write_ms=metrics_csv_write_ms,
                    timing_csv_write_ms=0.0,
                )

                timing_write_start = time.perf_counter()
                timing_writer.writerow(timing_row)
                timing_csv_write_ms = (time.perf_counter() - timing_write_start) * 1000.0
                batch_timings["timing_csv_write_ms"] += timing_csv_write_ms

            batch_duration_ms = (time.perf_counter() - batch_start) * 1000.0
            batch_timings["total_ms"] = batch_duration_ms
            batch_row = _batch_summary_row(batch_context, batch_timings, batch_totals)
            timing_writer.writerow(batch_row)

    finally:
        handle.close()
        timing_handle.close()
        logger.info("Inference metrics written to %s", csv_path)
        logger.info("Timing metrics written to %s", timing_csv_path)


if __name__ == "__main__":
    main()
