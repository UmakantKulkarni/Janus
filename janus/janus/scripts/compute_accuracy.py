#!/usr/bin/env python3
"""Assign dataset types and accuracy metrics for inference CSV rows."""
from __future__ import annotations
import argparse
import ast
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Sequence

# --- Constants ---
_DATASET_PATTERNS: Sequence[tuple[str, str]] = (
    ("core_crisis", "fuzzing"),
    ("defect_data", "defect_data"),
    ("gnbsim_experiments", "anomaly_data"),
    ("packetrusher_experiments", "clean_samples"),
)

def _calculate_standard_metrics(predicted: Sequence[int], truths: Sequence[int]) -> dict[str, float]:
    """
    Calculate standard binary classification metrics for multi-label arrays.

    Args:
        predicted: A sequence of predicted binary labels (0 or 1).
        truths: A sequence of true binary labels (0 or 1).

    Returns:
        A dictionary containing precision, recall, f1_score, and jaccard_score.
    """
    tp = sum(p and t for p, t in zip(predicted, truths))
    fp = sum(p and not t for p, t in zip(predicted, truths))
    fn = sum(not p and t for p, t in zip(predicted, truths))

    # Precision
    if (tp + fp) == 0:
        precision = 1.0  # Convention: No positive predictions means perfect precision
    else:
        precision = tp / (tp + fp)

    # Recall
    if (tp + fn) == 0:
        recall = 1.0  # Convention: No actual positives to find
    else:
        recall = tp / (tp + fn)

    # F1-Score
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        
    # Jaccard Score
    if (tp + fp + fn) == 0:
        jaccard_score = 1.0 # Both sets are empty, so perfectly similar
    else:
        jaccard_score = tp / (tp + fp + fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "jaccard_score": jaccard_score,
    }

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
            except (SyntaxError, ValueError):
                return [text]
    else:
        parsed = value
    if parsed is None:
        return []
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]

def _parse_ints(value: object) -> list[int]:
    """Return integers parsed from ``value``."""
    ints: list[int] = []
    for raw in _parse_list(value):
        try:
            ints.append(int(raw))
        except (TypeError, ValueError):
            try:
                ints.append(1 if float(raw) else 0)
            except (TypeError, ValueError):
                continue
    return ints

def _infer_dataset_type(log_path: str | None) -> str:
    """Return the dataset type derived from ``log_path`` substrings."""
    if not log_path:
        return "unknown"
    lowered = log_path.lower()
    for needle, dataset_type in _DATASET_PATTERNS:
        if needle in lowered:
            return dataset_type
    return "unknown"

def _accuracy_scores(predicted: Sequence[int], truths: Sequence[int]) -> list[int]:
    """Return element-wise accuracy scores for ``predicted`` vs ``truths``."""
    length = max(len(predicted), len(truths))
    if length == 0:
        return []
    scores: list[int] = []
    for index in range(length):
        pred = predicted[index] if index < len(predicted) else None
        truth = truths[index] if index < len(truths) else None
        if pred is None or truth is None:
            scores.append(0)
        else:
            scores.append(1 if pred == truth else 0)
    return scores

def _mean(values: Iterable[int | float]) -> float:
    """Return the arithmetic mean of ``values`` as a float."""
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    return total / count if count else 0.0

# --- Core Logic (Updated) ---

def summarize_dataset_metrics(csv_path: Path) -> None:
    """Print the mean of all key metrics per dataset_type."""
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    metrics_to_summarize = ["mean_accuracy", "precision", "recall", "f1_score", "jaccard_score"]
    
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset_type = row.get("dataset_type", "unknown")
            counts[dataset_type] += 1
            for metric in metrics_to_summarize:
                try:
                    val = float(row.get(metric, 0.0))
                    sums[dataset_type][metric] += val
                except (TypeError, ValueError):
                    continue
                    
    print("\n--- Summary of Mean Metrics per Dataset ---")
    for ds, count in counts.items():
        print(f"\nDataset: {ds} (Total Rows: {count})")
        for metric in metrics_to_summarize:
            mean_val = sums[ds][metric] / count if count else 0.0
            print(f"  - Mean {metric:<15}: {mean_val:.6f}")

def update_dataset_metrics(csv_path: Path) -> None:
    """Enrich ``csv_path`` with dataset type and all accuracy columns."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
        
    new_metric_columns = ["precision", "recall", "f1_score", "jaccard_score"]
    
    for row in rows:
        # Original logic
        dataset_type = _infer_dataset_type(row.get("log_file"))
        row["dataset_type"] = dataset_type
        predicted = _parse_ints(row.get("predicted_labels", []))
        truths = _parse_ints(row.get("true_labels", []))
        scores = _accuracy_scores(predicted, truths)
        mean_accuracy = _mean(scores)
        row["accuracy_scores"] = json.dumps(scores)
        row["mean_accuracy"] = f"{mean_accuracy:.6f}"
        
        
        standard_metrics = _calculate_standard_metrics(predicted, truths)
        for metric_name, metric_value in standard_metrics.items():
            row[metric_name] = f"{metric_value:.6f}"

    all_new_columns = ["dataset_type", "accuracy_scores", "mean_accuracy"] + new_metric_columns
    for column in all_new_columns:
        if column not in fieldnames:
            fieldnames.append(column)
            
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Successfully updated '{csv_path.name}' with new metrics.")

def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Assign dataset types and accuracy metrics to inference CSV rows.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to the inference metrics CSV file.",
    )
    return parser.parse_args()

def main() -> None:
    """CLI entry point."""
    args = parse_args()
    update_dataset_metrics(args.csv_path)
    summarize_dataset_metrics(args.csv_path)

if __name__ == "__main__":
    main()