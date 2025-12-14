#!/usr/bin/env python3
"""Build CLS-head calibration references for each network function."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from janus.infer.inference import load_model as load_inference_model
from janus.infer.inference import run_inference
from janus.utils.paths import load_repo_config, resolve_path

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger("build_nf_reference")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Janus Calibration Builder")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the calibration configuration YAML file.",
    )
    return parser.parse_args()


def _candidate_logs(logs_root: Path, nf: str) -> List[Path]:
    """Return log files matching the naming convention for ``nf``."""

    nf_lower = nf.lower()
    patterns = [
        logs_root / "*" / f"*-{nf_lower}.log",
        logs_root / f"*-{nf_lower}.log",
    ]
    matches: List[Path] = []
    for pattern in patterns:
        globbed = sorted(Path(p) for p in glob.glob(str(pattern)))
        if globbed:
            matches.extend(globbed)
            break
    return matches


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    """Compute summary statistics for ``values``."""

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan")}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def build_nf_ref(
    cfg: Dict[str, Any],
    nf: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    pad_id: int,
    max_tokens: int,
    base_model_path: Optional[str],
    final_adapter_path: Optional[str],
) -> None:
    """Build the CLS probability reference distribution for ``nf``."""

    nf_upper = nf.upper()
    nf_lower = nf_upper.lower()
    out_dir = Path(resolve_path(cfg.get("out_dir", "artifacts/calibrate")))
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_root_value = cfg.get("logs_root", "data/raw_data/logs")
    logs_root = Path(resolve_path(logs_root_value)) if logs_root_value else None
    if logs_root is None:
        raise ValueError("logs_root must be specified in the calibration config")

    files = _candidate_logs(logs_root, nf_lower)
    if not files:
        pattern_nested = logs_root / "*" / f"*-{nf_lower}.log"
        pattern_flat = logs_root / f"*-{nf_lower}.log"
        raise FileNotFoundError(
            "No log files found for NF="
            f"{nf_upper}. Checked patterns:\n  {pattern_nested}\n  {pattern_flat}"
        )

    logger.info("Building calibration reference for NF=%s (%d files)", nf_upper, len(files))

    cls_prob_means: List[float] = []
    per_log: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for path in files:
        logger.info("Scoring log %s", path)
        try:
            summary = run_inference(
                model=model,
                tokenizer=tokenizer,
                device=device,
                pad_id=pad_id,
                log_path=path,
                nf=nf_upper,
                max_tokens=max_tokens,
                csv_path=None,
                calibrate_dir=None,
            )
        except Exception as exc:  # noqa: BLE001 - propagate context in logs
            logger.warning("Skipping %s due to inference error: %s", path, exc)
            skipped.append({"log_file": str(path), "reason": str(exc)})
            continue

        score = summary.get("cls_prob_mean")
        if score is None or np.isnan(float(score)):
            logger.warning("Skipping %s because cls_prob_mean is NaN", path)
            skipped.append({"log_file": str(path), "reason": "cls_prob_mean is NaN"})
            continue

        cls_prob_means.append(float(score))
        per_log.append(
            {
                "log_file": str(path),
                "num_windows": int(summary.get("num_windows", 0)),
                "num_lines": int(summary.get("num_lines", 0)),
                "cls_prob_mean": float(summary.get("cls_prob_mean", float("nan"))),
                "cls_prob_max": float(summary.get("cls_prob_max", float("nan"))),
                "cls_prob_std": float(summary.get("cls_prob_std", float("nan"))),
            }
        )

    if not cls_prob_means:
        raise RuntimeError(
            f"No calibration scores were generated for NF={nf_upper}. "
            "Ensure the provided logs are non-empty and compatible with training."
        )

    scores = np.asarray(cls_prob_means, dtype=np.float32)
    score_path = out_dir / f"normal_scores_{nf_lower}.npy"
    np.save(score_path, scores)
    logger.info(
        "Saved %d CLS-mean scores for NF=%s to %s",
        scores.size,
        nf_upper,
        score_path,
    )

    stats = _summary_stats(scores)
    logger.info(
        "NF=%s calibration stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
        nf_upper,
        stats["mean"],
        stats.get("std", float("nan")),
        stats.get("min", float("nan")),
        stats.get("max", float("nan")),
    )

    meta = {
        "nf": nf_upper,
        "score_metric": "cls_prob_mean",
        "count": int(scores.size),
        "max_tokens": int(max_tokens),
        "base_model_dir": base_model_path,
        "adapter_dir": final_adapter_path,
        "logs_root": str(logs_root),
        "files_considered": [str(p) for p in files],
        "per_log_summary": per_log,
        "skipped_logs": skipped,
        "score_stats": stats,
    }
    meta_path = out_dir / f"normal_meta_{nf_lower}.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    logger.info("Wrote metadata to %s", meta_path)


def main() -> None:
    """Entry point for the calibration reference builder."""

    args = parse_args()
    cfg_path = Path(args.config).expanduser()
    cfg = yaml.safe_load(cfg_path.read_text())

    repo_cfg = load_repo_config()
    model_cfg = repo_cfg.get("model", {})

    base_model_path = model_cfg.get("base_model") or os.getenv("BASE_MODEL_PATH")
    if base_model_path:
        base_model_path = str(resolve_path(base_model_path))
    final_adapter_path = (
        model_cfg.get("final_adapter_path") or os.getenv("FINAL_ADAPTER_PATH")
    )
    if final_adapter_path:
        final_adapter_path = str(resolve_path(final_adapter_path))

    model, tokenizer, device, pad_id = load_inference_model(repo_cfg)

    max_tokens = int(
        cfg.get(
            "seq_len",
            repo_cfg.get("infer", {}).get("seq_len", 1024),
        )
    )

    nf_list = cfg.get(
        "network_functions",
        [
            "amf",
            "smf",
            "nrf",
            "upf",
            "ausf",
            "pcf",
            "udm",
            "nssf",
            "bsf",
            "scp",
            # "mongodb",
            # "test",
        ],
    )

    for nf in nf_list:
        build_nf_ref(
            cfg=cfg,
            nf=str(nf),
            model=model,
            tokenizer=tokenizer,
            device=device,
            pad_id=pad_id,
            max_tokens=max_tokens,
            base_model_path=base_model_path,
            final_adapter_path=final_adapter_path,
        )

    logger.info("All calibration references generated successfully.")


if __name__ == "__main__":
    main()
