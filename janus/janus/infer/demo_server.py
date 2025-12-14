#!/usr/bin/env python3
"""Simple FastAPI inference server"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
import urllib.request
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from janus.model.janus_model import JanusModel
from janus.infer.sliding_window import SlidingWindow
from janus.infer.retriever import ExplainabilityRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from janus.utils.paths import load_repo_config, resolve_path


def _dirichlet_probs(evidential_logits: torch.Tensor) -> torch.Tensor:
    """Return posterior mean probabilities under a Dirichlet evidence model."""

    evidence = F.softplus(evidential_logits)
    alpha = evidence + 1
    return alpha / alpha.sum(dim=1, keepdim=True)


def get_dirichlet_expected_score(evidential_logits: torch.Tensor) -> torch.Tensor:
    """Expected severity score from evidential head logits."""

    probs = _dirichlet_probs(evidential_logits)
    levels = torch.arange(probs.size(1), device=probs.device, dtype=probs.dtype)
    return (probs * levels).sum(dim=1)

# ──────────────────────────────────────────────────────────────────────────────
# Logging & Globals
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

MAX_BATCH_SIZE:     int
BATCH_LOG_FILE:     Path
CSV_LOG_FILE:       Path
MAX_TOKENS:         int
SCORE_POST_URL:     str
SCORE_STALE_SECONDS:int

LAST_SCORE_TIME = time.time()
score_lock     = threading.Lock()
csv_lock       = threading.Lock()

fifo_log_queue      : queue.Queue[str]               = queue.Queue()
application_queues: defaultdict[str, queue.Queue[str]] = defaultdict(queue.Queue)
application_lock    = threading.Lock()

model      : JanusModel | None            = None
tokenizer  = None
retriever  : ExplainabilityRetriever | None= None
window     = SlidingWindow()
CFG        : Dict[str, Any]               = {}

# ──────────────────────────────────────────────────────────────────────────────
# Configuration Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    """Load configuration using :mod:`janus.utils.paths`."""
    return load_repo_config()

CFG = load_config()

CSV_LOG_FILE = Path(tempfile.gettempdir()) / "batch_results.csv"

# Cache for NF-specific normal-score references
CALIBRATE_DIR = Path(
    resolve_path(CFG.get("calibrate", {}).get("out_dir", "artifacts/calibrate"))
)
NF_STATS_CACHE: Dict[str, Dict[str, Any] | None] = {}
NF_STATS_PATH = CALIBRATE_DIR / "nf_stats.json"


def _compute_reference_stats(ref: np.ndarray) -> Dict[str, Any]:
    """Compute summary statistics for a reference score array."""

    ref = ref.astype(np.float32).ravel()
    n = int(ref.size)
    ref_sorted = np.sort(ref)
    mean = float(ref.mean())
    std = float(ref.std(ddof=1))
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


def _load_nf_stats(nf: str) -> Dict[str, Any] | None:
    """Load reference scores and statistics for ``nf``.

    Parameters
    ----------
    nf:
        Network function name.
    Returns
    -------
    dict | None
        Statistics dictionary or ``None`` if reference is missing/empty.
    """

    nf = nf.lower()
    if nf in NF_STATS_CACHE:
        return NF_STATS_CACHE[nf]

    path = CALIBRATE_DIR / f"normal_scores_{nf}.npy"
    if not path.exists():
        logger.warning("Normal-score reference not found for NF=%s at %s", nf, path)
        NF_STATS_CACHE[nf] = None
        return None

    ref = np.load(path).astype(np.float32)
    if ref.size == 0:
        logger.warning("Normal-score reference empty for NF=%s at %s", nf, path)
        NF_STATS_CACHE[nf] = None
        return None

    stats = _compute_reference_stats(ref)
    NF_STATS_CACHE[nf] = stats

    try:
        NF_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if NF_STATS_PATH.exists():
            with NF_STATS_PATH.open() as f:
                existing = json.load(f)
        existing[nf] = {
            "N": stats["N"],
            "mean": stats["mean"],
            "std": stats["std"],
            "median": stats["median"],
            "MAD_raw": stats["MAD_raw"],
        }
        with NF_STATS_PATH.open("w") as f:
            json.dump(existing, f)
    except Exception as exc:  # pragma: no cover - cache failures are non-fatal
        logger.debug("Failed to update stats cache: %s", exc)

    return stats


def _compute_anomaly_metrics(score: float, stats: Dict[str, Any]) -> Dict[str, float]:
    """Compute percentile, z-score and MAD-based anomalies."""

    ref_sorted = stats["ref_sorted"]
    n = stats["N"]
    pos = int(np.searchsorted(ref_sorted, score, side="right"))
    F = pos / n
    A_upper = 1.0 - F
    A_pct = 2.0 * min(F, 1.0 - F)
    z = abs(score - stats["mean"]) / stats["std"]
    A_z = 1.0 - float(np.exp(-z))
    z_mad = abs(score - stats["median"]) / stats["sigma_robust"]
    A_mad = 1.0 - float(np.exp(-z_mad))
    return {
        "F": float(F),
        "A_upper": float(np.clip(A_upper, 0.0, 1.0)),
        "A_pct": float(np.clip(A_pct, 0.0, 1.0)),
        "z": float(z),
        "A_z": float(np.clip(A_z, 0.0, 1.0)),
        "z_mad": float(z_mad),
        "A_mad": float(np.clip(A_mad, 0.0, 1.0)),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Model & Retriever Initialization
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown via lifespan instead of on_event."""
    global model, tokenizer, retriever

    logger.info("Loading combined model for inference")

    base_model_path = CFG.get("model", {}).get("base_model") or os.getenv("BASE_MODEL_PATH")
    if base_model_path:
        base_model_path = resolve_path(base_model_path)
    final_adapter_path = CFG.get("model", {}).get("final_adapter_path") or os.getenv("FINAL_ADAPTER_PATH")
    if final_adapter_path:
        final_adapter_path = resolve_path(final_adapter_path)

    logger.info("Step 0: Loading base model from %s", base_model_path)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        device_map=None,
        torch_dtype=torch_dtype,
    )
    logger.info(f"Step 0: Base model loaded:\n {base_model}")

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
        dual_mask=CFG["model"].get("dual_mask", True),
        force_local_attention_only=CFG["model"].get("force_local_only", False),
    )
    logger.info("Step 2: JanusModel initialized")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model.tokenizer = tokenizer
    logger.info("Step 3: JanusModel initialized with tokenizer from %s", base_model_path)

    for head_name in ("cls_head", "evidential_head"):
        head_file = os.path.join(final_adapter_path, f"{head_name}_tensor.pth")
        if os.path.exists(head_file):
            logger.info(f"Loading full head {head_name} from {head_file}")
            state_dict = torch.load(head_file, map_location="cpu")
            model_head = getattr(model, head_name)
            missing, unexpected = model_head.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded full {head_name} from saved model.")
            logger.info(f"{head_name}.weight.shape = {tuple(model_head.weight.shape)}")
            logger.info(f"{head_name}.bias.shape   = {tuple(model_head.bias.shape)}")
            if missing or unexpected:
                logger.warning(f"{head_name} state_dict issues:")
                if missing:
                    logger.warning(f"{head_name} missing keys:    %s", missing)
                if unexpected:
                    logger.warning(f"{head_name} unexpected keys: %s", unexpected)
    logger.info("Step 4: Model heads loaded from saved model.")

    # (optional) compare to original adapter files for extra confidence
    adapter_path = CFG["model"].get("final_adapter_path", "")
    if adapter_path and os.path.isdir(adapter_path):
        for head in ("cls_head", "evidential_head"):
            ckpt = torch.load(os.path.join(adapter_path, f"{head}_tensor.pth"), map_location="cpu")
            w_loaded = model.__getattr__(head).weight.detach().cpu()
            b_loaded = model.__getattr__(head).bias.detach().cpu()
            w_ckpt   = ckpt["weight"]
            b_ckpt   = ckpt["bias"]
            def to_cpu(t):
                return t.to("cpu") if not isinstance(t, torch.Tensor) else t.cpu()
            # simple stats
            logger.info(f"  {head} ckpt mean={w_ckpt.mean():.6f}, model mean={w_loaded.mean():.6f}")
            logger.info(f"  {head} bias ckpt={b_ckpt.mean():.6f}, model bias={b_loaded.mean():.6f}")

    # Move model to GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()
    logger.info(f"Using device: {device}")

    use_dp = device_count > 1 and os.getenv("USE_DATAPARALLEL", "0") == "1"
    if use_dp:
        logger.info(f"Detected {device_count} GPUs, using DataParallel.")
        model = torch.nn.DataParallel(model)
        if hasattr(model.module, "tokenizer"):
            model.module.tokenizer = tokenizer
    model.to(device)
    model.eval()

    index_dir = CFG.get("explainability", {}).get("indexes_dir") or os.getenv("EXPLAINABILITY_INDEXES_DIR")
    if index_dir:
        index_dir = resolve_path(index_dir)
    if index_dir and Path(index_dir).exists():
        retriever = ExplainabilityRetriever(Path(index_dir))
    else:
        logger.warning("Explainability indexes not found: %s", index_dir)
        retriever = None

    logger.info("Model loaded successfully and is ready for inference.")

    # Start dispatcher/assembler threads
    queue_mode = os.getenv("QUEUE_MODE", "fifo").lower()
    logger.info("Starting workers in %s mode", queue_mode)
    if queue_mode == "application":
        threading.Thread(target=application_dispatcher_thread, daemon=True).start()
    else:
        threading.Thread(target=fifo_batch_assembler_thread, daemon=True).start()

    threading.Thread(target=score_keepalive_thread, daemon=True).start()

    yield
    # (optional) shutdown logic here

def create_app() -> FastAPI:
    """Assemble FastAPI app with custom lifespan."""
    app = FastAPI(lifespan=lifespan)
    register_routes(app)
    return app

# ──────────────────────────────────────────────────────────────────────────────
# Utility & Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request payload for log ingestion."""
    lines: List[str]

def _build_diagnostic_report(score: float, data: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a diagnostic report using retrieved context."""
    procedure = data.get("procedure") or {}
    code = data.get("code") or []
    spec = data.get("spec") or {}
    summary = f"Possible violation of {procedure.get('description', 'unknown procedure')}"
    action = "Investigate logic in "
    if code:
        action += code[0]["function"]
    else:
        action += "unknown function"
    return {
        "score": score,
        "procedure": procedure,
        "spec": spec,
        "code": code,
        "summary": summary,
        "suggested_action": action,
    }


def _extract_app_name(line: str) -> str:
    """Return an application name guessed from ``line``."""
    token = line.split()[0] if line.split() else ""
    if "/" in token:
        return token.split("/")[-1]
    match = re.search(r"\[(\w+)\]", line)
    if match:
        return match.group(1)
    return "default"


NF_PREFIX_RE = re.compile(r"^([^/\s]+)/([^/\s]+)\s+(.*)$")


def _split_nf_line(line: str) -> tuple[str, str]:
    """Extract network function name and strip pod/container prefix.

    Parameters
    ----------
    line:
        Original log line which may begin with ``pod/container``.

    Returns
    -------
    tuple[str, str]
        ``(network_function, remaining_log_line)`` where ``network_function`` is
        the container name. If the prefix cannot be parsed, the application name
        is inferred via :func:`_extract_app_name` and the original line is
        returned unchanged.
    """

    match = NF_PREFIX_RE.match(line.strip())
    if match:
        return match.group(2), match.group(3)
    nf = _extract_app_name(line)
    return nf, line

def _log_batch(name: str, lines: List[str], output: Dict[str, Any]) -> None:
    """Persist batch input and output to ``BATCH_LOG_FILE``."""
    with BATCH_LOG_FILE.open("a") as f:
        f.write(f"## Batch {name}\n")
        f.write("# Input:\n")
        for line in lines:
            f.write(line + "\n")
        f.write("# Output:\n")
        f.write(json.dumps(output) + "\n\n")


def _log_scores_csv(
    nf: str,
    num_lines: int,
    probs: List[float] | None,
    score: float | None,
    metrics: Dict[str, float] | None,
    stats: Dict[str, Any] | None,
    seq_len: int,
) -> None:
    """Append per-NF scoring results to :data:`CSV_LOG_FILE`."""

    timestamp = datetime.utcnow().isoformat()
    row: List[str] = [timestamp, nf, str(num_lines)]
    if probs is not None and score is not None:
        row += [f"{probs[0]:.6f}", f"{probs[1]:.6f}", f"{probs[2]:.6f}", f"{score:.6f}"]
    else:
        row += ["", "", "", ""]
    if metrics is not None:
        row += [
            f"{metrics['F']:.6f}",
            f"{metrics['A_upper']:.6f}",
            f"{metrics['A_pct']:.6f}",
            f"{metrics['z']:.6f}",
            f"{metrics['A_z']:.6f}",
            f"{metrics['z_mad']:.6f}",
            f"{metrics['A_mad']:.6f}",
        ]
    else:
        row += [""] * 7
    if stats is not None:
        row += [
            str(stats["N"]),
            f"{stats['mean']:.6f}",
            f"{stats['std']:.6f}",
            f"{stats['median']:.6f}",
            f"{stats['MAD_raw']:.6f}",
        ]
    else:
        row += [""] * 5
    row.append(str(seq_len))

    with csv_lock:
        exists = CSV_LOG_FILE.exists()
        CSV_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CSV_LOG_FILE.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not exists:
                writer.writerow(
                    [
                        "timestamp",
                        "nf",
                        "num_lines",
                        "e0",
                        "e1",
                        "e2",
                        "S",
                        "F",
                        "A_upper",
                        "A_pct",
                        "z",
                        "A_z",
                        "z_mad",
                        "A_mad",
                        "ref_N",
                        "ref_mean",
                        "ref_std",
                        "ref_median",
                        "ref_MAD",
                        "seq_len",
                    ]
                )
            writer.writerow(row)


def _post_score(score: float) -> None:
    """Send ``score`` to ``SCORE_POST_URL`` if configured. Example - http://127.0.0.1:8080/score"""
    global LAST_SCORE_TIME

    with score_lock:
        LAST_SCORE_TIME = time.time()

    if not SCORE_POST_URL:
        return
    try:
        data = json.dumps({"score": score}).encode()
        req = urllib.request.Request(
            SCORE_POST_URL,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3)
        logger.info("Posted score %.2f to %s", score, SCORE_POST_URL)
    except Exception as exc:
        logger.warning("Failed to post score to %s: %s", SCORE_POST_URL, exc)


def score_keepalive_thread() -> None:
    """Send a low score if no updates are posted for a while."""
    while True:
        time.sleep(1)
        with score_lock:
            idle = time.time() - LAST_SCORE_TIME
        if idle >= SCORE_STALE_SECONDS:
            _post_score(0.1)

# ──────────────────────────────────────────────────────────────────────────────
# Batch Processing & Thread Workers
# ──────────────────────────────────────────────────────────────────────────────

def process_batch(batch_lines: List[str], batch_identifier: str = "default") -> None:
    """Run inference on ``batch_lines`` per network function."""
    global model, tokenizer, retriever

    if model is None:
        logger.warning("Model not loaded, dropping batch %s", batch_identifier)
        return
    if tokenizer is None:
        logger.warning("Tokenizer not loaded, dropping batch %s", batch_identifier)
        return

    device = next(model.parameters()).device
    logger.debug("Processing %d lines for %s", len(batch_lines), batch_identifier)

    nf_groups: Dict[str, List[str]] = defaultdict(list)
    for raw_line in batch_lines:
        nf, cleaned = _split_nf_line(raw_line)
        nf_groups[nf].append(cleaned)

    for nf, lines in nf_groups.items():
        windowed: List[str] = []
        for line in lines:
            windowed.extend(window.add(nf, line))

        pl = "\n".join(windowed)
        #logger.info("input log lines for %s: %s", nf, pl)
        tokens = tokenizer(
            pl,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
        ).to(device)
        tags = torch.zeros_like(tokens.input_ids).to(device)

        try:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                out = model(tokens.input_ids, tags)
                evidential_logits = out["evidential"]
                if not torch.isfinite(evidential_logits).all():
                    raise ValueError("non-finite logits")
                probs = _dirichlet_probs(evidential_logits)
                evidential_scores = probs[0].tolist()
                score = float(get_dirichlet_expected_score(evidential_logits)[0])
        except RuntimeError as exc:  # handle OOM across PyTorch versions
            if "out of memory" in str(exc).lower():
                logger.error(
                    "%s OOM in batch %s: %s",
                    device.type.upper(),
                    batch_identifier,
                    exc,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                _log_batch(batch_identifier, lines, {"error": f"{device.type}_oom"})
                continue
            raise
        except ValueError:
            logger.error("Invalid logits for NF=%s", nf)
            _log_scores_csv(nf, len(lines), None, None, None, None, tokens.input_ids.size(1))
            continue
        stats = _load_nf_stats(nf)
        metrics: Dict[str, float] | None = None
        if stats is None:
            logger.warning("Reference not found for NF=%s", nf)
        elif stats["N"] < 5:
            logger.warning("Insufficient reference for NF=%s (N=%d)", nf, stats["N"])
        else:
            metrics = _compute_anomaly_metrics(score, stats)

        probs_str = "[" + " ".join(f"{p:.3f}" for p in evidential_scores) + "]"
        if metrics is None:
            logger.info(
                "%s | probs=%s | S=%.4f | "
                "F=N/A A_pct=N/A A_upper=N/A | "
                "z=N/A A_z=N/A | zMAD=N/A A_mad=N/A | "
                "N=%s μ=%s σ=%s m=%s MAD=%s",
                nf,
                probs_str,
                score,
                stats["N"] if stats else "N/A",
                f"{stats['mean']:.3f}" if stats else "N/A",
                f"{stats['std']:.3f}" if stats else "N/A",
                f"{stats['median']:.3f}" if stats else "N/A",
                f"{stats['MAD_raw']:.3f}" if stats else "N/A",
            )
        else:
            logger.info(
                "%s | probs=%s | S=%.4f | F=%.4f A_pct=%.3f A_upper=%.3f | "
                "z=%.2f A_z=%.3f | zMAD=%.2f A_mad=%.3f | "
                "N=%d μ=%.3f σ=%.3f m=%.3f MAD=%.3f",
                nf,
                probs_str,
                score,
                metrics["F"],
                metrics["A_pct"],
                metrics["A_upper"],
                metrics["z"],
                metrics["A_z"],
                metrics["z_mad"],
                metrics["A_mad"],
                stats["N"],
                stats["mean"],
                stats["std"],
                stats["median"],
                stats["MAD_raw"],
            )

        if retriever is not None and score > 0.5 and metrics is not None:
            diag = _build_diagnostic_report(score, retriever.explain(pl))
            diag.update(metrics)
            diag["evidential"] = evidential_scores
            diag["nf"] = nf
            output = diag
        else:
            output = {
                "evidential": evidential_scores,
                "score": score,
                "nf": nf,
                "metrics": metrics,
            }

        _post_score(score)
        _log_batch(batch_identifier, lines, output)
        _log_scores_csv(
            nf,
            len(lines),
            evidential_scores,
            score,
            metrics,
            stats,
            tokens.input_ids.size(1),
        )
    #logger.info("Batch %s output: %s", batch_identifier, output)


def fifo_batch_assembler_thread() -> None:
    """Assemble batches from the global FIFO queue."""
    buffer: List[str] = []
    batch_idx = 0
    while True:
        try:
            line = fifo_log_queue.get(timeout=1)
            buffer.append(line)
        except queue.Empty:
            if buffer:
                process_batch(buffer, f"fifo-{batch_idx}")
                buffer = []
                batch_idx += 1
            continue

        if len(buffer) >= MAX_BATCH_SIZE:
            process_batch(buffer, f"fifo-{batch_idx}")
            buffer = []
            batch_idx += 1


def application_batch_assembler_thread(app_name: str, app_queue: queue.Queue[str]) -> None:
    """Process logs for a specific application."""
    buffer: List[str] = []
    batch_idx = 0
    while True:
        try:
            line = app_queue.get(timeout=1)
            buffer.append(line)
        except queue.Empty:
            if buffer:
                process_batch(buffer, f"{app_name}-{batch_idx}")
                buffer = []
                batch_idx += 1
            continue

        if len(buffer) >= MAX_BATCH_SIZE:
            process_batch(buffer, f"{app_name}-{batch_idx}")
            buffer = []
            batch_idx += 1


def application_dispatcher_thread() -> None:
    """Spawn batch threads for new applications."""
    workers: Dict[str, threading.Thread] = {}
    while True:
        with application_lock:
            for app_name, app_queue in list(application_queues.items()):
                if app_name not in workers:
                    t = threading.Thread(
                        target=application_batch_assembler_thread,
                        args=(app_name, app_queue),
                        daemon=True,
                    )
                    t.start()
                    workers[app_name] = t
        time.sleep(1)

# ──────────────────────────────────────────────────────────────────────────────
# HTTP Routes
# ──────────────────────────────────────────────────────────────────────────────

def register_routes(app: FastAPI):
    @app.post("/logs", response_model=None)
    async def logs(request: Request) -> Dict[str, Any]:
        """Ingest log lines and place them on the selected queue."""
        body = await request.body()
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                if "lines" in data:
                    lines = data["lines"]
                elif "line" in data:
                    lines = [data["line"]]
                else:
                    lines = []
            elif isinstance(data, list):
                lines = [str(item) for item in data]
            else:
                lines = []
        except Exception:
            lines = body.decode().splitlines()

        queue_mode = os.getenv("QUEUE_MODE", "fifo").lower()
        if queue_mode == "application":
            for line in lines:
                app_name = _extract_app_name(line)
                with application_lock:
                    q = application_queues[app_name]
                q.put(line)
        else:
            for line in lines:
                fifo_log_queue.put(line)

        return {"status": "accepted", "queued": len(lines)}

    @app.post("/ingest", response_model=None)
    async def ingest(request: Request) -> Dict[str, Any]:
        """Alias for the `/logs` endpoint for backward compatibility."""
        return await logs(request=request)


# ──────────────────────────────────────────────────────────────────────────────
# Main Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global MAX_BATCH_SIZE, BATCH_LOG_FILE, CSV_LOG_FILE, MAX_TOKENS, SCORE_POST_URL
    global SCORE_STALE_SECONDS, CFG

    CFG = load_config()

    parser = argparse.ArgumentParser(description="Janus FastAPI inference server")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=int(os.getenv("MAX_BATCH_SIZE", "32")),
        help="Max lines per batch",
    )
    parser.add_argument(
        "--batch-log-file",
        type=str,
        default=os.getenv("BATCH_LOG_FILE", "batch_results.log"),
        help="Path to batch log file",
    )
    parser.add_argument(
        "--csv-log-file",
        type=str,
        default=os.path.join(tempfile.gettempdir(), "batch_results.csv"),
        help="Path to CSV results file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", "1024")),
        help="Max tokens for tokenizer",
    )
    parser.add_argument(
        "--score-post-url",
        type=str,
        default=os.getenv("SCORE_POST_URL", ""),
        help="URL to post scores",
    )
    parser.add_argument(
        "--score-stale-seconds",
        type=int,
        default=int(os.getenv("SCORE_STALE_SECONDS", "60")),
        help="Seconds before sending idle score",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9000, help="Server port")
    args = parser.parse_args()

    MAX_BATCH_SIZE     = args.max_batch_size
    BATCH_LOG_FILE     = Path(args.batch_log_file)
    CSV_LOG_FILE       = Path(args.csv_log_file)
    MAX_TOKENS         = args.max_tokens
    SCORE_POST_URL     = args.score_post_url
    SCORE_STALE_SECONDS = args.score_stale_seconds

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
else:
    # Expose a ready-to-use FastAPI application for importers.
    app = create_app()