#!/usr/bin/env python3
"""Augment inference metrics CSV files with predicted and true labels."""

from __future__ import annotations
import os
import re
import argparse
import ast
import csv
import json
from bisect import bisect_left
from pathlib import Path
from typing import Sequence

from janus.utils.paths import project_root
PROJECT_ROOT = project_root()

_DEFECT_LABEL_RE = re.compile(r"label([01])\s*-\s*(normal|anomaly)", re.IGNORECASE)

def _infer_defect_true_label_from_filename(log_path: str | None) -> int | None:
    """Return 0/1 from file name like '...label0-normal...' or '...label1-anomaly...'; None if not found."""
    if not log_path:
        return None
    name = os.path.basename(log_path)
    m = _DEFECT_LABEL_RE.search(name)
    if not m:
        return None
    bit = m.group(1)
    try:
        return int(bit)
    except (TypeError, ValueError):
        return None


def _match_length(seq: list[int], target_len: int, pad_value: int = 0) -> list[int]:
    """Trim or pad a list to exactly target_len."""
    if target_len <= 0:
        return []
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [pad_value] * (target_len - len(seq))


def _parse_list(value: object) -> list[object]:
    """Return ``value`` parsed into a list using JSON or Python literals."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return [text]
    else:
        parsed = value
    if parsed is None:
        return []
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def _parse_floats(value: object) -> list[float]:
    """Return float values parsed from ``value``."""

    floats: list[float] = []
    for raw in _parse_list(value):
        try:
            floats.append(float(raw))
        except (TypeError, ValueError):
            floats.append(float("nan"))
    return floats


def _threshold_array(raw_threshold: object, count: int) -> list[float]:
    """Return ``count`` threshold values derived from ``raw_threshold``."""

    if count <= 0:
        return []
    values = [v for v in _parse_floats(raw_threshold) if not (v != v)]
    base = values[0] if values else 0.5
    if len(values) >= count:
        return values[:count]
    return [base] * count


def _parse_window_ranges(value: object) -> list[tuple[int, int]]:
    """Return window ranges as ``(start, end)`` integer tuples."""

    ranges: list[tuple[int, int]] = []
    for item in _parse_list(value):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            try:
                start, end = item  # type: ignore[misc]
            except (TypeError, ValueError):
                continue
        try:
            ranges.append((int(start), int(end)))
        except (TypeError, ValueError):
            continue
    return ranges


def _load_metadata(metadata_path: Path) -> dict[str, list[int]]:
    """Load anomaly metadata keyed by normalized log file paths."""

    with metadata_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    mapping: dict[str, list[int]] = {}
    for key, lines in raw.items():
        try:
            anomalies = sorted(int(line) for line in lines)
        except TypeError:
            anomalies = []
        mapping[key] = anomalies
        path = Path(key).expanduser()
        candidates: list[str] = []
        try:
            candidates.append(str(path.resolve(strict=False)))
        except RuntimeError:
            pass
        candidates.append(str(path))
        for candidate in candidates:
            if candidate not in mapping:
                mapping[candidate] = anomalies
    return mapping


def _lookup_anomalies(metadata: dict[str, list[int]], log_path: str | None) -> list[int]:
    """Return anomaly line numbers recorded for ``log_path``."""

    if not log_path:
        return []
    path = Path(log_path).expanduser()
    candidates: list[str] = []
    try:
        candidates.append(str(path.resolve(strict=False)))
    except RuntimeError:
        pass
    candidates.append(str(path))
    candidates.append(log_path)
    for candidate in candidates:
        if candidate in metadata:
            return metadata[candidate]
    return []


def _window_labels(
    window_ranges: Sequence[tuple[int, int]],
    anomalies: Sequence[int],
) -> list[int]:
    """Return per-window true labels using ``anomalies`` line numbers."""

    if not window_ranges:
        return []
    if not anomalies:
        return [0] * len(window_ranges)
    labels: list[int] = []
    for start, end in window_ranges:
        if start > end:
            start, end = end, start
        index = bisect_left(anomalies, start)
        label = 1 if index < len(anomalies) and anomalies[index] <= end else 0
        labels.append(label)
    return labels


def update_inference_csv(csv_path: Path, metadata: dict[str, list[int]],
                         nf_thresholds: str | None, predict_labels: int) -> None:
    """Update ``csv_path`` with predicted and true label columns."""

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    use_nf_thresholds = False
    if nf_thresholds and os.path.isfile(nf_thresholds):
        use_nf_thresholds = True
        with open(nf_thresholds, 'r') as f:
            NF_THRESHOLDS = json.load(f)
    else:
        use_nf_thresholds = False

    for row in rows:
        # Predicted labels from probabilities and thresholds
        probs = _parse_floats(row.get("cls_probs", []))
        nf = (row.get("app_name") or "").strip().lower()
        if predict_labels == 1:
            if use_nf_thresholds:
                threshold = NF_THRESHOLDS.get(nf, 0.5)
                thresholds = _threshold_array(
                    threshold,
                    len(probs),
                )
                row["threshold"] = threshold
            else:
                thresholds = _threshold_array(
                    row.get("threshold"),
                    len(probs),
                )
            predicted = [
                1 if prob > thr else 0 for prob, thr in zip(probs, thresholds)
            ]
            row["predicted_labels"] = json.dumps(predicted)

        # Determine where to get true labels
        path_str = row.get("log_file")
        is_defect = "defect_data" in path_str.lower()

        window_ranges = _parse_window_ranges(row.get("window_line_ranges", []))
        anomalies = _lookup_anomalies(metadata, path_str)
        true_labels = _window_labels(window_ranges, anomalies)
        row["true_labels"] = json.dumps(true_labels)

    for column in ("predicted_labels", "true_labels"):
        if column not in fieldnames:
            fieldnames.append(column)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""

    parser = argparse.ArgumentParser(
        description=
        "Add predicted and true labels to inference metrics CSV files.", )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to the janus inference metrics CSV file.",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=Path(
            os.path.join(PROJECT_ROOT, "data/eval_data/eval_metadata.json")),
        help=
        "Path to the eval metadata JSON file produced by generate_metadata.py.",
    )
    parser.add_argument(
        "--nf_thresholds",
        type=str,
        default=None,
        help="JSON file path to be used per-NF thresholds for classification.",
    )
    parser.add_argument(
        "--predict_labels",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to compute predicted labels from probabilities (1) or skip (0). Default is 1.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    metadata = _load_metadata(args.metadata_path)
    update_inference_csv(args.csv_path, metadata, args.nf_thresholds, args.predict_labels)


if __name__ == "__main__":
    main()
