#!/usr/bin/env python3
"""Run inference on a single network-function log file."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from janus.model.janus_model import JanusModel
from janus.tagger import LogTagger
from janus.utils.paths import load_repo_config, resolve_path

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
NF_FILENAME_RE = re.compile(r"-([A-Za-z0-9]+)\.log$")
NF_PREFIX_RE = re.compile(r"^([^/\s]+)/([^/\s]+)\s+(.*)$")

NF_STATS_CACHE: Dict[Tuple[str, Path], Dict[str, Any] | None] = {}


def _compute_reference_stats(ref: np.ndarray) -> Dict[str, Any]:
    """Compute summary statistics for a reference score array."""

    ref = ref.astype(np.float32).ravel()
    n = int(ref.size)
    ref_sorted = np.sort(ref)
    mean = float(ref.mean())
    std = float(ref.std(ddof=1)) if n > 1 else 0.0
    if std <= 1e-6:
        std = 1e-6
    median = float(np.median(ref))
    mad_raw = float(np.median(np.abs(ref - median)))
    if mad_raw <= 1e-6:
        iqr = np.percentile(ref, 75) - np.percentile(ref, 25)
        mad_raw = max(mad_raw, float(iqr / 1.349), 1e-6)
    sigma_robust = float(1.4826 * mad_raw)
    return {
        "ref": ref,
        "ref_sorted": ref_sorted,
        "N": n,
        "mean": mean,
        "std": std,
        "median": median,
        "MAD_raw": mad_raw,
        "sigma_robust": sigma_robust,
    }


def _load_nf_stats(nf: str, calibrate_dir: Optional[Path]) -> Dict[str, Any] | None:
    """Load cached normal-score reference statistics for ``nf``."""

    if calibrate_dir is None:
        return None

    key = (nf.lower(), calibrate_dir)
    if key in NF_STATS_CACHE:
        return NF_STATS_CACHE[key]

    path = calibrate_dir / f"normal_scores_{nf.lower()}.npy"
    if not path.exists():
        logger.warning("Normal-score reference not found for NF=%s at %s", nf, path)
        NF_STATS_CACHE[key] = None
        return None

    ref = np.load(path).astype(np.float32)
    if ref.size == 0:
        logger.warning("Normal-score reference empty for NF=%s at %s", nf, path)
        NF_STATS_CACHE[key] = None
        return None

    stats = _compute_reference_stats(ref)
    NF_STATS_CACHE[key] = stats
    return stats


def _compute_anomaly_metrics(score: float, stats: Dict[str, Any]) -> Dict[str, float]:
    """Compute percentile, z-score and MAD-based anomalies for ``score``."""

    ref_sorted = stats["ref_sorted"]
    n = stats["N"]
    pos = int(np.searchsorted(ref_sorted, score, side="right"))
    F_val = pos / n
    A_upper = 1.0 - F_val
    A_pct = 2.0 * min(F_val, 1.0 - F_val)
    z = abs(score - stats["mean"]) / stats["std"]
    A_z = 1.0 - float(np.exp(-z))
    z_mad = abs(score - stats["median"]) / stats["sigma_robust"]
    A_mad = 1.0 - float(np.exp(-z_mad))
    return {
        "F": float(F_val),
        "A_upper": float(np.clip(A_upper, 0.0, 1.0)),
        "A_pct": float(np.clip(A_pct, 0.0, 1.0)),
        "z": float(z),
        "A_z": float(np.clip(A_z, 0.0, 1.0)),
        "z_mad": float(z_mad),
        "A_mad": float(np.clip(A_mad, 0.0, 1.0)),
    }


def _nf_from_filename(path: Path) -> str:
    """Infer network function from ``path`` using training-time convention."""

    match = NF_FILENAME_RE.search(path.name)
    if match:
        return match.group(1).upper()
    stem = path.stem
    if not stem:
        return ""
    return stem.upper()


def _nf_from_contents(path: Path, max_lines: int = 100) -> str:
    """Attempt to infer a network function name by inspecting ``path``."""

    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for _ in range(max_lines):
                raw = handle.readline()
                if not raw:
                    break
                line = raw.strip()
                if not line:
                    continue
                prefix = NF_PREFIX_RE.match(line)
                if prefix:
                    return prefix.group(2).upper()
                bracket = re.search(r"\[([A-Za-z0-9_-]+)\]", line)
                if bracket:
                    return bracket.group(1).upper()
    except OSError as exc:
        logger.warning("Unable to inspect %s for NF detection: %s", path, exc)
    return ""


def _infer_network_function(path: Path, explicit: Optional[str]) -> str:
    """Determine the network function name to use for ``path``."""

    if explicit:
        logger.info("Using network function from CLI argument: %s", explicit)
        return explicit.upper()

    nf = _nf_from_filename(path)
    if nf:
        logger.info("Inferred network function %s from filename", nf)
        return nf

    nf = _nf_from_contents(path)
    if nf:
        logger.info("Inferred network function %s from log contents", nf)
        return nf

    raise ValueError(
        "Unable to determine network function from filename or contents. "
        "Please specify --network-function explicitly."
    )


def infer_network_function(path: Path, explicit: Optional[str] = None) -> str:
    """Public wrapper returning the inferred network function for ``path``."""

    return _infer_network_function(path, explicit)


def _nf_prefix_tokens(tagger: LogTagger, nf_name: str) -> Tuple[List[int], List[int]]:
    """Return the tokens/tags for the training-style NF prefix."""

    prefix = f"NF:{nf_name}\n"
    tagged = tagger.tag_line(prefix)
    tokens = [t.token_id for t in tagged]
    tags = [t.tag_id for t in tagged]
    return tokens, tags


def _prepare_log_windows(
    path: Path,
    nf_name: str,
    tagger: LogTagger,
    max_tokens: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]], int]:
    """Convert ``path`` into training-format windows of tokens and tags."""

    nf_tokens, nf_tags = _nf_prefix_tokens(tagger, nf_name)
    window_tokens: List[int] = []
    window_tags: List[int] = []
    token_tensors: List[torch.Tensor] = []
    tag_tensors: List[torch.Tensor] = []
    window_line_ranges: List[Tuple[int, int]] = []
    current_window_start: Optional[int] = None
    line_count = 0

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line_count += 1
            if not window_tokens:
                current_window_start = line_count
            cleaned = ANSI_ESCAPE_RE.sub("", raw_line.rstrip("\n"))
            if nf_tokens:
                window_tokens.extend(nf_tokens)
                window_tags.extend(nf_tags)
            tagged = tagger.tag_line(cleaned)
            truncated = False
            for tt in tagged:
                window_tokens.append(tt.token_id)
                window_tags.append(tt.tag_id)
                if len(window_tokens) >= max_tokens:
                    token_tensors.append(torch.tensor(window_tokens, dtype=torch.long))
                    tag_tensors.append(torch.tensor(window_tags, dtype=torch.long))
                    start_line = (
                        current_window_start if current_window_start is not None else line_count
                    )
                    window_line_ranges.append((start_line, line_count))
                    window_tokens = []
                    window_tags = []
                    current_window_start = None
                    truncated = True
                    break
            if truncated:
                # Training re-attaches the NF prefix after each emitted window.
                continue

    if window_tokens:
        token_tensors.append(torch.tensor(window_tokens, dtype=torch.long))
        tag_tensors.append(torch.tensor(window_tags, dtype=torch.long))
        start_line = (
            current_window_start if current_window_start is not None else max(line_count, 1)
        )
        window_line_ranges.append((start_line, line_count))

    return token_tensors, tag_tensors, window_line_ranges, line_count


def _autocast_context(device: torch.device) -> torch.autocast | nullcontext:
    """Return an autocast context suitable for ``device``."""

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _score_windows(
    model: JanusModel,
    token_tensors: Sequence[torch.Tensor],
    tag_tensors: Sequence[torch.Tensor],
    pad_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run the model on the prepared windows."""

    if not token_tensors:
        raise ValueError("No tokens were produced from the provided log file.")

    ids = pad_sequence(token_tensors, batch_first=True, padding_value=pad_id).to(device)
    tags = pad_sequence(tag_tensors, batch_first=True, padding_value=0).to(device)

    logger.debug("Scoring %d windows (max seq len=%d)", ids.size(0), ids.size(1))

    model.eval()
    with torch.no_grad():
        ctx = _autocast_context(device)
        with ctx:
            out = model(ids, tags)

    return {
        "cls_logits": out["cls"].view(-1),
        "seq_lens": torch.tensor([t.numel() for t in token_tensors], device=device),
    }


def load_model(config: Dict[str, Any]) -> Tuple[JanusModel, Any, torch.device, int]:
    """Load the full model stack (base + adapters + heads) for inference."""

    logger.info("Loading combined model for inference")

    base_model_path = config.get("model", {}).get("base_model") or os.getenv("BASE_MODEL_PATH")
    if base_model_path:
        base_model_path = resolve_path(base_model_path)
    final_adapter_path = config.get("model", {}).get("final_adapter_path") or os.getenv("FINAL_ADAPTER_PATH")
    if final_adapter_path:
        final_adapter_path = resolve_path(final_adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer `pad_token` was None, setting it to `eos_token`.")
    pad_id = tokenizer.pad_token_id

    logger.info("Step 0: Loading base model from %s", base_model_path)
    torch_dtype = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        device_map=None,
        torch_dtype=torch_dtype,
    )
    logger.info("Step 0: Base model loaded:\n %s", base_model)

    peft_model = PeftModel.from_pretrained(
        base_model,
        final_adapter_path,
        device_map="auto",
        inference_mode=True,
        is_trainable=False,
        use_safetensors=True,
        torch_dtype=torch_dtype,
    )
    logger.info("Step 1: PEFT model loaded from %s", final_adapter_path)

    model = JanusModel(
        base_model=peft_model,
        dual_mask=config["model"].get("dual_mask", True),
        force_local_attention_only=config["model"].get("force_local_only", False),
    )
    logger.info("Step 2: JanusModel initialized")

    model.tokenizer = tokenizer
    logger.info("Step 3: JanusModel initialized with tokenizer from %s", base_model_path)

    if final_adapter_path:
        for head_name in ("cls_head",):
            head_file = os.path.join(final_adapter_path, f"{head_name}_tensor.pth")
            if os.path.exists(head_file):
                logger.info("Loading full head %s from %s", head_name, head_file)
                state_dict = torch.load(head_file, map_location="cpu")
                model_head = getattr(model, head_name)
                missing, unexpected = model_head.load_state_dict(state_dict, strict=True)
                logger.info("Loaded full %s from saved model.", head_name)
                logger.info("%s.weight.shape = %s", head_name, tuple(model_head.weight.shape))
                logger.info("%s.bias.shape   = %s", head_name, tuple(model_head.bias.shape))
                if missing or unexpected:
                    logger.warning("%s state_dict issues:", head_name)
                    if missing:
                        logger.warning("%s missing keys:    %s", head_name, missing)
                    if unexpected:
                        logger.warning("%s unexpected keys: %s", head_name, unexpected)
    logger.info("Step 4: Model heads loaded from saved model.")

    adapter_path = config["model"].get("final_adapter_path", "")
    if adapter_path and os.path.isdir(adapter_path):
        head = "cls_head"
        ckpt_file = os.path.join(adapter_path, f"{head}_tensor.pth")
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location="cpu")
            w_loaded = model.__getattr__(head).weight.detach().cpu()
            b_loaded = model.__getattr__(head).bias.detach().cpu()
            w_ckpt = ckpt["weight"]
            b_ckpt = ckpt["bias"]
            logger.info("  %s ckpt mean=%.6f, model mean=%.6f", head, w_ckpt.mean(), w_loaded.mean())
            logger.info("  %s bias ckpt=%.6f, model bias=%.6f", head, b_ckpt.mean(), b_loaded.mean())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()
    logger.info("Using device: %s", device)

    use_dp = device_count > 1 and os.getenv("USE_DATAPARALLEL", "0") == "1"
    if use_dp:
        logger.info("Detected %d GPUs, using DataParallel.", device_count)
        model = torch.nn.DataParallel(model)
        if hasattr(model.module, "tokenizer"):
            model.module.tokenizer = tokenizer
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully and is ready for inference.")
    return model, tokenizer, device, pad_id


def _summarise_scores(
    cls_logits: torch.Tensor,
    seq_lens: torch.Tensor,
    window_line_ranges: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    """Aggregate per-window outputs into summary statistics."""

    cls_logits_np = cls_logits.detach().cpu().numpy()
    cls_probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
    seq_lengths = seq_lens.detach().cpu().numpy()

    if len(window_line_ranges) != cls_probs.size:
        raise ValueError(
            "Mismatch between number of windows and recorded line ranges: "
            f"{cls_probs.size} vs {len(window_line_ranges)}"
        )

    def _safe_stat(arr: np.ndarray, func, default: float = float("nan")) -> float:
        return float(func(arr)) if arr.size else default

    def _quantile(arr: np.ndarray, q: float) -> float:
        return _safe_stat(arr, lambda a: np.quantile(a, q))

    entropy = np.zeros_like(cls_probs)
    if cls_probs.size:
        clipped = np.clip(cls_probs, 1e-6, 1 - 1e-6)
        entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))

    summary = {
        "cls_logits": cls_logits_np.tolist(),
        "cls_probs": cls_probs.tolist(),
        "cls_prob_mean": _safe_stat(cls_probs, np.mean),
        "cls_prob_max": _safe_stat(cls_probs, np.max),
        "cls_prob_min": _safe_stat(cls_probs, np.min),
        "cls_prob_std": _safe_stat(cls_probs, np.std),
        "cls_prob_var": _safe_stat(cls_probs, np.var),
        "cls_prob_median": _safe_stat(cls_probs, np.median),
        "cls_prob_p25": _quantile(cls_probs, 0.25),
        "cls_prob_p75": _quantile(cls_probs, 0.75),
        "cls_prob_ge_0_5_ratio": float(np.mean(cls_probs >= 0.5)) if cls_probs.size else float("nan"),
        "cls_prob_ge_0_8_ratio": float(np.mean(cls_probs >= 0.8)) if cls_probs.size else float("nan"),
        "cls_prob_ge_0_9_ratio": float(np.mean(cls_probs >= 0.9)) if cls_probs.size else float("nan"),
        "cls_entropy_mean": _safe_stat(entropy, np.mean),
        "cls_entropy_std": _safe_stat(entropy, np.std),
        "cls_logit_mean": _safe_stat(cls_logits_np, np.mean),
        "cls_logit_max": _safe_stat(cls_logits_np, np.max),
        "cls_logit_min": _safe_stat(cls_logits_np, np.min),
        "cls_logit_std": _safe_stat(cls_logits_np, np.std),
        "cls_logit_var": _safe_stat(cls_logits_np, np.var),
        "cls_logit_median": _safe_stat(cls_logits_np, np.median),
        "cls_logit_p25": _quantile(cls_logits_np, 0.25),
        "cls_logit_p75": _quantile(cls_logits_np, 0.75),
        "seq_lengths": seq_lengths.tolist(),
        "window_line_ranges": [[int(start), int(end)] for start, end in window_line_ranges],
    }
    return summary


def _build_csv_row(
    result: Dict[str, Any],
    log_path: Path,
    nf: str,
    num_lines: int,
    metrics_mean: Optional[Dict[str, float]],
    metrics_max: Optional[Dict[str, float]],
    stats: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Return CSV header and row for the inference summary."""

    header = [
        "timestamp",
        "log_file",
        "nf",
        "threshold",
        "num_lines",
        "num_windows",
        "seq_lengths",
        "window_line_ranges",
        "cls_logits",
        "cls_probs",
        "cls_prob_mean",
        "cls_prob_max",
        "cls_prob_min",
        "cls_prob_std",
        "cls_prob_var",
        "cls_prob_median",
        "cls_prob_p25",
        "cls_prob_p75",
        "cls_prob_ge_0_5_ratio",
        "cls_prob_ge_0_8_ratio",
        "cls_prob_ge_0_9_ratio",
        "cls_entropy_mean",
        "cls_entropy_std",
        "cls_logit_mean",
        "cls_logit_max",
        "cls_logit_min",
        "cls_logit_std",
        "cls_logit_var",
        "cls_logit_median",
        "cls_logit_p25",
        "cls_logit_p75",
        "F_mean",
        "A_upper_mean",
        "A_pct_mean",
        "z_mean",
        "A_z_mean",
        "z_mad_mean",
        "A_mad_mean",
        "F_max",
        "A_upper_max",
        "A_pct_max",
        "z_max",
        "A_z_max",
        "z_mad_max",
        "A_mad_max",
        "ref_N",
        "ref_mean",
        "ref_std",
        "ref_median",
        "ref_MAD",
    ]

    timestamp = datetime.utcnow().isoformat()
    seq_lengths = list(int(v) for v in result["seq_lengths"])
    window_ranges_str = json.dumps(result.get("window_line_ranges", []))

    def _fmt_float(value: Optional[float]) -> str:
        if value is None:
            return "nan"
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "nan"
        if np.isnan(val):
            return "nan"
        return f"{val:.6f}"

    row: List[str] = [
        timestamp,
        str(log_path),
        nf,
        "0.5",
        str(num_lines),
        str(result.get("num_windows", 0)),
        seq_lengths,
        window_ranges_str,
        result.get("cls_logits", []),
        result.get("cls_probs", []),
        _fmt_float(result.get("cls_prob_mean")),
        _fmt_float(result.get("cls_prob_max")),
        _fmt_float(result.get("cls_prob_min")),
        _fmt_float(result.get("cls_prob_std")),
        _fmt_float(result.get("cls_prob_var")),
        _fmt_float(result.get("cls_prob_median")),
        _fmt_float(result.get("cls_prob_p25")),
        _fmt_float(result.get("cls_prob_p75")),
        _fmt_float(result.get("cls_prob_ge_0_5_ratio")),
        _fmt_float(result.get("cls_prob_ge_0_8_ratio")),
        _fmt_float(result.get("cls_prob_ge_0_9_ratio")),
        _fmt_float(result.get("cls_entropy_mean")),
        _fmt_float(result.get("cls_entropy_std")),
        _fmt_float(result.get("cls_logit_mean")),
        _fmt_float(result.get("cls_logit_max")),
        _fmt_float(result.get("cls_logit_min")),
        _fmt_float(result.get("cls_logit_std")),
        _fmt_float(result.get("cls_logit_var")),
        _fmt_float(result.get("cls_logit_median")),
        _fmt_float(result.get("cls_logit_p25")),
        _fmt_float(result.get("cls_logit_p75")),
    ]

    def _metric_row(metrics: Optional[Dict[str, float]]) -> List[str]:
        if not metrics:
            return ["nan"] * 7
        return [
            _fmt_float(metrics.get("F")),
            _fmt_float(metrics.get("A_upper")),
            _fmt_float(metrics.get("A_pct")),
            _fmt_float(metrics.get("z")),
            _fmt_float(metrics.get("A_z")),
            _fmt_float(metrics.get("z_mad")),
            _fmt_float(metrics.get("A_mad")),
        ]

    row.extend(_metric_row(metrics_mean))
    row.extend(_metric_row(metrics_max))

    if not stats:
        row.extend(["nan"] * 5)
    else:
        row.extend(
            [
                str(stats.get("N", "nan")),
                _fmt_float(stats.get("mean")),
                _fmt_float(stats.get("std")),
                _fmt_float(stats.get("median")),
                _fmt_float(stats.get("MAD_raw")),
            ]
        )

    return header, row


def build_csv_row(
    result: Dict[str, Any],
    log_path: Path,
    nf: str,
    num_lines: int,
    metrics_mean: Optional[Dict[str, float]],
    metrics_max: Optional[Dict[str, float]],
    stats: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Public helper returning the CSV header and row for ``result``."""

    return _build_csv_row(result, log_path, nf, num_lines, metrics_mean, metrics_max, stats)


def _write_csv(csv_path: Path, header: Sequence[str], row: Sequence[str]) -> None:
    """Append ``row`` to ``csv_path`` writing ``header`` if necessary."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(list(header))
        writer.writerow(list(row))


def run_inference(
    model: JanusModel,
    tokenizer,
    device: torch.device,
    pad_id: int,
    log_path: Path,
    nf: str,
    max_tokens: int,
    csv_path: Optional[Path],
    calibrate_dir: Optional[Path],
) -> Dict[str, Any]:
    """Score ``log_path`` and optionally append results to ``csv_path``."""

    tagger = LogTagger(tokenizer)
    logger.info("Preparing log %s for inference (NF=%s)", log_path, nf)
    (
        token_tensors,
        tag_tensors,
        window_line_ranges,
        num_lines,
    ) = _prepare_log_windows(log_path, nf, tagger, max_tokens)
    logger.info("Extracted %d windows across %d log lines", len(token_tensors), num_lines)

    outputs = _score_windows(model, token_tensors, tag_tensors, pad_id, device)
    summary = _summarise_scores(
        outputs["cls_logits"],
        outputs["seq_lens"],
        window_line_ranges,
    )
    summary["nf"] = nf
    summary["log_file"] = str(log_path)
    summary["num_lines"] = num_lines
    summary["num_windows"] = len(token_tensors)

    stats = _load_nf_stats(nf, calibrate_dir)
    metrics_mean: Optional[Dict[str, float]] = None
    metrics_max: Optional[Dict[str, float]] = None
    if stats:
        if not np.isnan(summary["cls_prob_mean"]):
            metrics_mean = _compute_anomaly_metrics(summary["cls_prob_mean"], stats)
        if not np.isnan(summary["cls_prob_max"]):
            metrics_max = _compute_anomaly_metrics(summary["cls_prob_max"], stats)
        summary["metrics_mean"] = metrics_mean
        summary["metrics_max"] = metrics_max
        summary["reference_stats"] = {
            "N": stats["N"],
            "mean": stats["mean"],
            "std": stats["std"],
            "median": stats["median"],
            "MAD_raw": stats["MAD_raw"],
        }
    else:
        summary["metrics_mean"] = None
        summary["metrics_max"] = None
        summary["reference_stats"] = None

    logger.info(
        "NF=%s | windows=%d | cls_prob(mean/max)=%.4f / %.4f",
        nf,
        summary["num_windows"],
        summary.get("cls_prob_mean", float("nan")),
        summary.get("cls_prob_max", float("nan")),
    )

    if csv_path is not None:
        header, row = _build_csv_row(
            summary,
            log_path,
            nf,
            num_lines,
            metrics_mean,
            metrics_max,
            stats,
        )
        _write_csv(csv_path, header, row)
        logger.info("Appended inference results to %s", csv_path)

    return summary


def main() -> None:
    """CLI entry point."""

    config = load_repo_config()

    parser = argparse.ArgumentParser(description="Janus Inference")
    parser.add_argument("log_file", type=str, help="Path to the network function log file")
    parser.add_argument(
        "--network-function",
        type=str,
        default=None,
        help="Network function name. Inferred from the log when omitted.",
    )
    parser.add_argument(
        "--csv-log-file",
        type=str,
        default="/tmp/inference_results.csv",
        help="Path to CSV results file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.get("infer", {}).get("seq_len", 1024),
        help="Maximum number of tokens per window (should match training seq_len)",
    )
    parser.add_argument(
        "--calibrate-dir",
        type=str,
        default="artifacts/calibrate",
        help="Optional directory containing calibration score distributions",
    )
    args = parser.parse_args()

    model, tokenizer, device, pad_id = load_model(config)

    log_path = Path(resolve_path(args.log_file))
    if not log_path.exists():
        raise FileNotFoundError(f"Log file {log_path} does not exist")

    nf = _infer_network_function(log_path, args.network_function)

    csv_path = Path(args.csv_log_file) if args.csv_log_file else None

    calibrate_dir: Optional[Path] = None
    if args.calibrate_dir:
        calibrate_dir = Path(resolve_path(args.calibrate_dir))
        if not calibrate_dir.exists():
            logger.warning("Calibrate directory %s does not exist", calibrate_dir)
            calibrate_dir = None

    summary = run_inference(
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_id=pad_id,
        log_path=log_path,
        nf=nf,
        max_tokens=args.max_tokens,
        csv_path=csv_path,
        calibrate_dir=calibrate_dir,
    )

    logger.info("Inference complete for %s", log_path)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
