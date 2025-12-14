#!/usr/bin/env python3

from __future__ import annotations

import re
import argparse
import os
import json

# Disable tokenizers parallelism before any tokenizers module is loaded.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml
import logging
import math
import csv
from datetime import datetime, timezone
from pathlib import Path
from functools import partial
from collections import deque
from typing import Union
import copy
import random
import torch
import torch.distributed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed.tensor

from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs
)
from accelerate.utils import set_seed
from transformers import (
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
)
from torch.optim import AdamW

from peft import PeftModel

from janus.dataset import PerNFDataset, PerNFDatasetDefect, nf_from_filename
from janus.model.janus_model import JanusModel
from janus.tagger import build_tokenizer
from janus.train.utils import save_full_checkpoint
from janus.utils.paths import load_repo_config, resolve_path

# Set up basic logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

DatasetLike = Union[PerNFDataset, PerNFDatasetDefect]

def freeze_global_adapters(model):
    """
    Freeze any parameters that belong to the 'global' adapter path.
    Adjust the patterns to match your JanusModel module names.
    """
    # heuristics: names containing 'global' or an explicit list of submodules
    patterns = [r"\bglobal\b", r"global_adapter", r"global_lora", r"global_mask"]
    keep_patterns = [r"^cls_head\.", r"^evidential_head\.", r"\blocal\b"]
    re_global = re.compile("|".join(patterns))
    re_keep   = re.compile("|".join(keep_patterns))
    total_elems = trainable_elems = frozen_elems = 0
    for n, p in model.named_parameters():
        elems = p.numel()
        total_elems += elems
        if not p.requires_grad:
            continue
        trainable_elems += elems
        if re_global.search(n) and not re_keep.search(n):
            p.requires_grad = False
            frozen_elems += elems
    logger.info(f"[freeze_global_adapters] total elems={total_elems} | "
                f"trainable before={trainable_elems} | frozen now={frozen_elems}")

# -----------------------------
# Batch sampler (PN-balanced)
# -----------------------------
class PerNFBatchSampler(torch.utils.data.Sampler):
    """
    Round-robin over NFs, yielding batched indices.

    Modes:
      - shard_mode="batch_mod": (default) All ranks build the same global batch
        sequence but each rank only *yields* batches whose global_batch_id % world_size == rank.
        With drop_last=True, every rank gets the same number of batches -> perfectly aligned steps.
      - shard_mode="none": every rank gets the full stream (useful for debugging only).

    When lambda_pn > 0 and both classes exist for an NF, each batch is half normal, half anomaly.
    """

    def __init__(
        self,
        nf_to_indices: dict[str, list[int]],
        labels: list[int],
        batch_size: int,
        *,
        lambda_pn: float = 0.0,
        rank: int = 0,
        world_size: int = 1,
        shard_mode: str = "batch_mod",  # "batch_mod" or "none"
        drop_last: bool = True,
    ) -> None:
        assert batch_size % 2 == 0, "batch_size must be even"
        assert shard_mode in ("batch_mod", "none")
        self.batch_size = int(batch_size)
        self.lambda_pn = float(lambda_pn)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shard_mode = shard_mode
        self.drop_last = bool(drop_last)

        self.assigned_nfs = sorted(nf_to_indices.keys())   # same order on all ranks
        self.nf_to_indices = {nf: nf_to_indices[nf] for nf in self.assigned_nfs}
        self.nf_neg: dict[str, list[int]] = {}
        self.nf_pos: dict[str, list[int]] = {}
        for nf, idxs in self.nf_to_indices.items():
            self.nf_neg[nf] = [i for i in idxs if labels[i] == 0]
            self.nf_pos[nf] = [i for i in idxs if labels[i] == 1]

        self.epoch = 0
        self._warned: set[str] = set()
        self._reset_pointers()

        # Pre-compute total number of global batches for __len__ and drop_last logic
        self._global_batches = self._compute_total_global_batches()

    def _reset_pointers(self) -> None:
        self.ptr = {nf: 0 for nf in self.assigned_nfs}
        self.ptr_pos = {nf: 0 for nf in self.assigned_nfs}
        self.ptr_neg = {nf: 0 for nf in self.assigned_nfs}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._reset_pointers()

    def get_nf_list(self) -> list[str]:
        return list(self.assigned_nfs)

    def _compute_total_global_batches(self) -> int:
        total = 0
        for nf, idxs in self.nf_to_indices.items():
            if self.lambda_pn > 0 and self.nf_pos[nf] and self.nf_neg[nf]:
                half = self.batch_size // 2
                max_len = max(len(self.nf_pos[nf]), len(self.nf_neg[nf]))
                total += math.ceil(max_len / half)
            else:
                total += math.ceil(len(idxs) / self.batch_size)
        if self.shard_mode == "batch_mod" and self.drop_last:
            total = (total // self.world_size) * self.world_size
        return total

    def __iter__(self):
        self._reset_pointers()
        self._global_batches = self._compute_total_global_batches()
        nfs = self.assigned_nfs
        if not nfs:
            return iter(())

        start = self.epoch % len(nfs)
        order = nfs[start:] + nfs[:start]
        finished = {nf: False for nf in order}

        global_batch_id = 0
        emitted_by_this_rank = 0
        target_emits = None
        if self.shard_mode == "batch_mod":
            # Each rank will emit exactly total/world_size batches when drop_last=True
            if self.drop_last:
                target_emits = self._global_batches // self.world_size
            else:
                # uneven tail possible; ranks may differ by at most 1
                target_emits = math.ceil(self._global_batches / self.world_size)

        while not all(finished.values()):
            for nf in order:
                if finished[nf]:
                    continue
                idxs = self.nf_to_indices[nf]

                # Always create batches by taking the next sequential slice from the current NF.
                # This respects the time-series nature of the data.
                start_idx = self.ptr[nf]
                end_idx = min(start_idx + self.batch_size, len(idxs))
                if start_idx >= end_idx:
                    finished[nf] = True
                    continue
                batch = idxs[start_idx:end_idx]
                self.ptr[nf] = end_idx

                # If batch shorter than batch_size (tail), we still count it as one global batch.
                # Shard by global batch id if enabled.
                take_here = True
                if self.shard_mode == "batch_mod":
                    take_here = (global_batch_id % self.world_size) == self.rank

                # Stop if we already reached the per-rank target (only with drop_last)
                if self.shard_mode == "batch_mod" and self.drop_last and emitted_by_this_rank >= target_emits:
                    return

                if take_here:
                    emitted_by_this_rank += 1
                    yield batch

                global_batch_id += 1

                if self.ptr[nf] >= len(idxs):
                    finished[nf] = True

                # Stop globally if we’ve reached the capped global batches for drop_last
                if self.shard_mode == "batch_mod" and self.drop_last and global_batch_id >= self._global_batches:
                    return

        # If we fall out naturally and drop_last is False, we're done.

    def __len__(self) -> int:
        if self.shard_mode == "batch_mod":
            if self.drop_last:
                return self._global_batches // self.world_size
            else:
                return math.ceil(self._global_batches / self.world_size)
        else:
            return self._global_batches

# -----------------------------
# Paired sampler
# -----------------------------
class PairedBatchSampler(torch.utils.data.Sampler):
    """Round-robin sampler over paired indices grouped by NF.

    The ``pairs_by_nf`` mapping must contain pre-sorted index pairs for each
    network function. Pairs are yielded in the provided order without internal
    reordering.
    """

    def __init__(
        self,
        pairs_by_nf: dict[str, list[tuple[int, int]]],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        assert batch_size % 2 == 0, "batch_size must be even"
        self.batch_size = int(batch_size)
        self.per_batch = self.batch_size // 2
        self.shuffle = shuffle
        self.drop_last = bool(drop_last)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.pairs_by_nf = {nf: list(pairs) for nf, pairs in pairs_by_nf.items()}
        self.assigned_nfs = sorted(self.pairs_by_nf.keys())
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def get_nf_list(self) -> list[str]:  # pragma: no cover - simple
        return list(self.assigned_nfs)

    def __len__(self) -> int:
        total_pairs = sum(len(p) for p in self.pairs_by_nf.values())
        per_rank = total_pairs // self.world_size if self.drop_last else math.ceil(total_pairs / self.world_size)
        return per_rank // self.per_batch if self.drop_last else math.ceil(per_rank / self.per_batch)

    def __iter__(self):
        nfs = self.assigned_nfs[:]
        if self.shuffle:
            random.shuffle(nfs)
        start = self.epoch % len(nfs) if nfs else 0
        order = nfs[start:] + nfs[:start]
        batches: list[list[int]] = []
        for nf in order:
            pairs = self.pairs_by_nf[nf]
            if self.shuffle:
                random.shuffle(pairs)
            for i in range(0, len(pairs), self.per_batch):
                batch_pairs = pairs[i : i + self.per_batch]
                if len(batch_pairs) < self.per_batch and self.drop_last:
                    logger.info(
                        "[PAIR-DROP] nf=%s dropped_pairs=%d",
                        nf,
                        len(batch_pairs),
                    )
                    continue
                batch: list[int] = []
                for first, second in batch_pairs:
                    batch.extend([first, second])
                batches.append(batch)
        batches = batches[self.rank :: self.world_size]
        for b in batches:
            yield b

# -----------------------------
# Misc helpers
# -----------------------------
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Unified Janus Training Script")
    p.add_argument("--config", type=str, required=True, help="Path to the training configuration YAML file.")
    return p.parse_args()

def _collate(batch: list[dict], pad_id: int) -> dict:
    """Pad batch elements and stack them into tensors."""
    ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    tags = pad_sequence([b["tags"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    nf = batch[0]["nf"]
    nf_idx = torch.tensor([b["nf_idx"] for b in batch], dtype=torch.long)
    return {"input_ids": ids, "tags": tags, "label": labels, "nf": nf, "nf_idx": nf_idx}


def _subset_dataset(ds: DatasetLike, indices: list[int]) -> DatasetLike:
    """Return a shallow copy of ``ds`` containing only ``indices``."""
    sub = copy.copy(ds)
    sub.items = [ds.items[i] for i in indices]
    sub.nf_to_indices = {}
    for new_idx, orig_idx in enumerate(indices):
        nf = ds.items[orig_idx]["nf"]
        sub.nf_to_indices.setdefault(nf, []).append(new_idx)
    if getattr(ds, "pairs", None):
        index_map = {orig: i for i, orig in enumerate(indices)}
        sub.pairs = [
            (index_map[a], index_map[b])
            for a, b in ds.pairs
            if a in index_map and b in index_map
        ]
        sub.pairs.sort(key=lambda p: p[0])
        sub.pairs_by_nf = {}
        for nf, pairs in ds.pairs_by_nf.items():
            filt = [
                (index_map[a], index_map[b])
                for a, b in pairs
                if a in index_map and b in index_map
            ]
            if filt:
                filt.sort(key=lambda p: p[0])
                sub.pairs_by_nf[nf] = filt
    else:
        sub.pairs = []
        sub.pairs_by_nf = {}
    return sub

def _split_dataset(ds: DatasetLike, val_pct: float) -> tuple[DatasetLike, DatasetLike]:
    """Split ``ds`` into train/validation subsets.

    When ``ds`` contains paired windows (one normal and one anomalous window
    from the same file) the split is performed **per pair** to preserve the
    one-to-one mapping.  Unpaired windows are dropped entirely in this mode to
    keep class counts balanced.  Otherwise, the split operates per network
    function (NF) using simple index slicing.  In both cases we ensure that an
    NF with multiple elements contributes at least one item to the training
    portion.
    """

    # Pair-aware splitting keeps normal/anomaly counts aligned.
    if ds.pairs_by_nf:
        train_idx: list[int] = []
        val_idx: list[int] = []
        for nf, pairs in ds.pairs_by_nf.items():
            split_pairs = int(len(pairs) * val_pct / 100)
            if len(pairs) > 1:
                split_pairs = min(max(split_pairs, 1), len(pairs) - 1)
            else:
                split_pairs = 0
            val_pairs = pairs[:split_pairs]
            train_pairs = pairs[split_pairs:]
            val_idx.extend(i for pair in val_pairs for i in pair)
            train_idx.extend(i for pair in train_pairs for i in pair)
        train_ds, val_ds = _subset_dataset(ds, train_idx), _subset_dataset(ds, val_idx)
        # _log_sample_pairs(train_ds, "train")
        # _log_sample_pairs(val_ds, "validation")
        return train_ds, val_ds

    # Fallback: simple per-NF split of unpaired indices.
    train_idx = []
    val_idx = []
    for nf, idxs in ds.nf_to_indices.items():
        split = int(len(idxs) * val_pct / 100)
        if len(idxs) > 1:
            split = min(max(split, 1), len(idxs) - 1)
        else:
            split = 0
        val_idx.extend(idxs[:split])
        train_idx.extend(idxs[split:])
    train_ds, val_ds = _subset_dataset(ds, train_idx), _subset_dataset(ds, val_idx)
    # _log_sample_pairs(train_ds, "train")
    # _log_sample_pairs(val_ds, "validation")
    return train_ds, val_ds


def _dataset_label_stats(ds: DatasetLike) -> tuple[int, int]:
    """Return counts of anomaly and normal samples in a dataset."""
    pos = sum(int(item["label"]) == 1 for item in ds)
    neg = len(ds) - pos
    return pos, neg


def _log_sample_pairs(ds: DatasetLike, name: str) -> None:
    """Log one example normal/anomaly pair per NF for ``ds``."""

    logged: set[str] = set()
    pairs_by_nf = getattr(ds, "pairs_by_nf", None) or {}
    for nf, pairs in pairs_by_nf.items():
        if not pairs:
            continue
        a_idx, b_idx = pairs[0]
        item_a = ds.items[a_idx]
        item_b = ds.items[b_idx]
        if int(item_a["label"]) == 0:
            normal, anomaly = item_a, item_b
        else:
            normal, anomaly = item_b, item_a
        logger.info(
            "[%s] sample pair nf=%s normal=%s anomaly=%s",
            name,
            nf,
            normal.get("file"),
            anomaly.get("file"),
        )
        logged.add(nf)

    # Fallback: if no explicit pairs exist for an NF, surface the first
    # available normal/anomaly combination based on labels.
    remaining_nfs = set(ds.nf_to_indices.keys()) - logged
    for nf in sorted(remaining_nfs):
        indices = ds.nf_to_indices.get(nf, [])
        normal_idx = next(
            (i for i in indices if int(ds.items[i]["label"]) == 0),
            None,
        )
        anomaly_idx = next(
            (i for i in indices if int(ds.items[i]["label"]) == 1),
            None,
        )
        if normal_idx is None or anomaly_idx is None:
            continue
        normal = ds.items[normal_idx]
        anomaly = ds.items[anomaly_idx]
        logger.info(
            "[%s] sample pair nf=%s normal=%s anomaly=%s",
            name,
            nf,
            normal.get("file"),
            anomaly.get("file"),
        )


def _log_nf_sources(name: str, sampler: torch.utils.data.Sampler) -> None:
    """Log network functions contributing batches for ``sampler``."""

    getter = getattr(sampler, "get_nf_list", None)
    if getter is None:  # pragma: no cover - defensive
        return
    nfs = getter()
    if nfs:
        logger.info("[%s] nfs=%s", name, ",".join(sorted(nfs)))


def _log_pair_counts(ds: DatasetLike, name: str) -> None:
    """Log number of pairs per network function for ``ds``."""

    pairs = getattr(ds, "pairs_by_nf", None)
    if not pairs:
        return
    for nf in sorted(pairs):
        logger.info("[%s] nf=%s pairs=%d", name, nf, len(pairs[nf]))


def _log_first_batch(
    loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    name: str,
) -> None:
    """Log example clean and anomaly logs from the first batch of ``loader``."""
    try:
        batch = next(iter(loader))
    except StopIteration:
        logger.warning("[DATA] %s loader is empty", name)
        return
    labels = batch["label"].tolist()
    inputs = batch["input_ids"]
    clean_ex, anom_ex = None, None
    for i, lbl in enumerate(labels):
        text = tokenizer.decode(inputs[i].tolist(), skip_special_tokens=True)
        if lbl == 0 and clean_ex is None:
            clean_ex = text
        elif lbl == 1 and anom_ex is None:
            anom_ex = text
        if clean_ex and anom_ex:
            break
    if clean_ex is not None:
        logger.info("[DATA][%s] example clean log: %s", name, clean_ex)
    if anom_ex is not None:
        logger.info("[DATA][%s] example anomaly log: %s", name, anom_ex)

def get_dirichlet_expected_score(evidential_logits: torch.Tensor) -> torch.Tensor:
    """
    Expected severity score from evidential head.
    Use softplus (not ReLU) for smoother gradients on negative logits.
    """
    evidence = F.softplus(evidential_logits)
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    severity_levels = torch.arange(alpha.size(1), device=alpha.device, dtype=alpha.dtype)
    return (probs * severity_levels).sum(dim=1)

def pairwise_logistic(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable pairwise logistic (AUC proxy): softplus(normal - anomaly).
    Encourages anomaly scores > normal scores.
    """
    normal  = scores[labels == 0]
    anomaly = scores[labels == 1]
    if normal.numel() == 0 or anomaly.numel() == 0:
        return scores.new_tensor(0.0)
    n = min(len(normal), len(anomaly), 64)
    diff = normal[:n].unsqueeze(0) - anomaly[:n].unsqueeze(1)
    return F.softplus(diff).mean()

def evidential_ce_loss(
    evidential_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    label_smoothing: float,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    p = F.softmax(evidential_logits / temperature, dim=1)
    eps = label_smoothing
    t_norm = evidential_logits.new_tensor([1 - eps, eps / 2, eps / 2])
    t_anom = evidential_logits.new_tensor([eps / 2, eps / 2, 1 - eps])
    targets = torch.where(labels.unsqueeze(1) == 0, t_norm, t_anom)
    log_p = torch.log(p.to(torch.float32) + 1e-8)
    ce = -(targets * log_p).sum(dim=1)
    if focal_gamma > 0:
        idx = (labels * 2).unsqueeze(1)
        max_prob = p.gather(1, idx).squeeze(1)
        ce = ce * (1 - max_prob).pow(focal_gamma)
    return ce.mean()

def get_current_masking_rate(initial_rate: float, final_rate: float, decay_steps: int, current_step: int) -> float:
    if current_step >= decay_steps:
        return final_rate
    if initial_rate <= final_rate:
        return initial_rate
    decay_rate = -math.log(final_rate / initial_rate) / max(decay_steps, 1)
    return initial_rate * math.exp(-decay_rate * current_step)

def get_monotone_schedule(initial: float, final: float, steps: int, t: int) -> float:
    if steps <= 0 or t >= steps:
        return final
    if initial == final:
        return initial
    # exponential bridge that grows or decays depending on final/initial
    ratio = final / initial if initial != 0 else 0.0
    if ratio <= 0:
        # fall back to linear if signs or zeros are weird
        return initial + (final - initial) * (t / steps)
    return initial * (ratio ** (t / steps))

# ---- synchronized eval trigger (rank 0 decides) ----
def _broadcast_eval_flag(eval_flag: bool, device) -> bool:
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    t = torch.tensor([1 if eval_flag else 0], device=device, dtype=torch.int32)
    if is_dist:
        src = 0
        if torch.distributed.get_rank() != src:
            t.zero_()
        torch.distributed.broadcast(t, src=src)
    return bool(int(t.item()))


def evaluate(
    model,
    dataloader,
    accelerator,
    pad_id,
    margin,
    auc_weight,
    lambda_pn,
    *,
    compute_auc: bool = False,
    pair_windows: bool = False,
    return_scores: bool = False,
):
    """Evaluate model and return global and per-NF metrics.

    Runs locally on every rank (no barriers/collectives here). The caller
    will all-reduce the returned scalars across processes.

    Returns
    -------
    tuple
        ``(clm_num, clm_den, pn_num, pn_den, auc_anom_val, auc_anom_count,
        auc_norm_val, auc_norm_count, s_norm_sum, s_norm_count, s_anom_sum,
        s_anom_count, per_nf, scores_cat, labels_cat)``.
    """
    model.eval()
    device = accelerator.device

    # per-rank accumulators
    clm_num = 0.0
    clm_den = 0.0
    pn_num = 0.0
    pn_den = 0
    auc_anom_val = float("nan")
    auc_norm_val = float("nan")
    auc_anom_count = 0
    auc_norm_count = 0
    s_norm_sum = 0.0
    s_norm_count = 0
    s_anom_sum = 0.0
    s_anom_count = 0

    per_nf: dict[str, dict[str, float]] = {}
    mixed_batches: dict[str, int] = {}
    total_batches: dict[str, int] = {}

    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            nf = batch.pop("nf")
            # move to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # forward
            with accelerator.autocast():
                out = model(batch["input_ids"], batch["tags"])
                logits = out["logits"]

            # CLM loss (token average)
            logits_shifted = logits[:, :-1, :].transpose(1, 2)  # [B, V, T-1]
            labels_shifted = batch["input_ids"][:, 1:].clone()  # [B, T-1]
            anom_rows = (batch["label"] == 1).nonzero(as_tuple=False).flatten()
            if anom_rows.numel() > 0:
                labels_shifted[anom_rows, :] = pad_id
            ce_tok = F.cross_entropy(
                logits_shifted, labels_shifted, ignore_index=pad_id, reduction="none"
            )
            tok_mask = (labels_shifted != pad_id).float()
            if lambda_pn > 0:
                # With long sequences, CLM gradients swamp PN. Compute CE per sequence,
                # then average across the batch.
                ce_per_seq = (
                    (ce_tok * tok_mask).sum(dim=1)
                    / tok_mask.sum(dim=1).clamp_min(1.0)
                )
                loss_clm = ce_per_seq.mean()
            else:
                loss_clm = (ce_tok * tok_mask).sum() / tok_mask.sum().clamp_min(1.0)

            valid_tokens_local = int(tok_mask.sum().item())
            clm_num += float(loss_clm.item() * valid_tokens_local)
            clm_den += float(valid_tokens_local)

            rec = per_nf.setdefault(
                nf,
                {
                    "clm_num": 0.0,
                    "clm_den": 0.0,
                    "pn_num": 0.0,
                    "pn_den": 0,
                    "s_norm_sum": 0.0,
                    "s_norm_count": 0,
                    "s_anom_sum": 0.0,
                    "s_anom_count": 0,
                    "seq_num": 0.0,
                    "seq_den": 0.0,
                },
            )
            if valid_tokens_local > 0:
                rec["clm_num"] += float(loss_clm.item() * valid_tokens_local)
                rec["clm_den"] += float(valid_tokens_local)
            else:
                # Fallback: if shifting left produced zero valid tokens (e.g., very short windows),
                # accumulate per-sequence instead so NF metrics are not dropped.
                bs_fallback = int(batch["input_ids"].size(0))
                rec["clm_num"] += float(loss_clm.item() * bs_fallback)
                rec["clm_den"] += float(bs_fallback)

            # sequence-granular accumulation for per-NF
            bs = int(batch["input_ids"].size(0))
            rec["seq_num"] += float(loss_clm.item() * bs)
            rec["seq_den"] += float(bs)

            total_batches[nf] = total_batches.get(nf, 0) + 1

            # Use the binary classification head for anomaly scores.
            # By convention: higher score = more anomalous.
            scores = out["cls"].squeeze(1).to(torch.float32)

            labels = batch["label"]
            if pair_windows:
                if labels.numel() % 2 != 0 or not torch.all(
                    labels.view(-1, 2).sum(dim=1) == 1
                ):
                    raise AssertionError("paired batch requires alternating labels")
            normal_mask = labels == 0
            anomaly_mask = labels == 1

            if compute_auc:
                all_scores.append(scores.detach())
                all_labels.append(labels.detach())
                if normal_mask.any():
                    s_norm_sum += float(scores[normal_mask].sum().item())
                    s_norm_count += int(normal_mask.sum().item())
                    rec["s_norm_sum"] += float(scores[normal_mask].sum().item())
                    rec["s_norm_count"] += int(normal_mask.sum().item())
                if anomaly_mask.any():
                    s_anom_sum += float(scores[anomaly_mask].sum().item())
                    s_anom_count += int(anomaly_mask.sum().item())
                    rec["s_anom_sum"] += float(scores[anomaly_mask].sum().item())
                    rec["s_anom_count"] += int(anomaly_mask.sum().item())

            if pair_windows:
                pair_losses = []
                for i in range(0, labels.size(0) - 1, 2):
                    l1, l2 = labels[i], labels[i + 1]
                    if l1 == l2:
                        continue
                    s1, s2 = scores[i], scores[i + 1]
                    if l1 == 1:
                        pair_losses.append(torch.relu(margin - s1 + s2))
                    else:
                        pair_losses.append(torch.relu(margin - s2 + s1))
                if pair_losses:
                    mixed_batches[nf] = mixed_batches.get(nf, 0) + 1
                    hinge = torch.stack(pair_losses).mean()
                    auc_proxy = pairwise_logistic(scores, labels)
                    pn_loss = hinge + (auc_weight * auc_proxy)
                    bs = len(pair_losses)
                    pn_num += float(pn_loss.item() * bs)
                    pn_den += bs
                    rec["pn_num"] += float(pn_loss.item() * bs)
                    rec["pn_den"] += bs
            else:
                loss_anom = (
                    F.relu(margin - scores[anomaly_mask]).mean()
                    if anomaly_mask.any()
                    else torch.tensor(0.0, device=accelerator.device)
                )
                loss_norm = (
                    F.relu(margin + scores[normal_mask]).mean()
                    if normal_mask.any()
                    else torch.tensor(0.0, device=accelerator.device)
                )
                if anomaly_mask.any() or normal_mask.any():
                    mixed_batches[nf] = mixed_batches.get(nf, 0) + 1
                hinge = loss_anom + loss_norm
                auc_proxy = pairwise_logistic(scores, labels)
                pn_loss = hinge + (auc_weight * auc_proxy)
                bs = int(batch["input_ids"].size(0))
                pn_num += float(pn_loss.item() * bs)
                pn_den += bs
                rec["pn_num"] += float(pn_loss.item() * bs)
                rec["pn_den"] += bs

    # AUC on this rank only; caller will average across ranks.
    if compute_auc and len(all_scores) > 0:
        scores_np = torch.cat(all_scores).cpu().numpy()
        labels_np = torch.cat(all_labels).to(torch.long).cpu().numpy()
        try:
            from sklearn.metrics import roc_auc_score

            auc_anom_val = float(roc_auc_score(labels_np, scores_np))
            auc_norm_val = float(roc_auc_score(labels_np, -scores_np))
        except Exception:
            order = scores_np.argsort()
            ranks = order.argsort() + 1
            pos = labels_np == 1
            neg = labels_np == 0
            n_pos = int(pos.sum())
            n_neg = int(neg.sum())
            if n_pos > 0 and n_neg > 0:
                pos_ranks = int(ranks[pos].sum())
                U = pos_ranks - n_pos * (n_pos + 1) / 2
                auc_anom_val = float(U / (n_pos * n_neg))
                auc_norm_val = 1.0 - auc_anom_val
            else:
                auc_anom_val = float("nan")
                auc_norm_val = float("nan")
        if math.isnan(auc_anom_val):
            auc_anom_count = 0
            auc_anom_val = 0.0
        else:
            auc_anom_count = accelerator.num_processes
        if math.isnan(auc_norm_val):
            auc_norm_count = 0
            auc_norm_val = 0.0
        else:
            auc_norm_count = accelerator.num_processes

    # Per-NF logging on main process only (purely cosmetic)
    if accelerator.is_main_process:
        for nf, rec in per_nf.items():
            val_pn_nf = (
                rec["pn_num"] / rec["pn_den"] if rec["pn_den"] > 0 else float("nan")
            )
            mean_norm = (
                rec["s_norm_sum"] / rec["s_norm_count"]
                if rec.get("s_norm_count", 0) > 0
                else float("nan")
            )
            mean_anom = (
                rec["s_anom_sum"] / rec["s_anom_count"]
                if rec.get("s_anom_count", 0) > 0
                else float("nan")
            )
            logger.info(
                (
                    "[PN-EVAL] nf=%s mixed_batches=%d total_batches=%d val_pn_nf=%s "
                    "mean_margin_norm=%s mean_margin_anom=%s"
                ),
                nf,
                mixed_batches.get(nf, 0),
                total_batches.get(nf, 0),
                f"{val_pn_nf:.6f}" if not math.isnan(val_pn_nf) else "nan",
                f"{mean_norm:.6f}" if not math.isnan(mean_norm) else "nan",
                f"{mean_anom:.6f}" if not math.isnan(mean_anom) else "nan",
            )

    scores_cat = None
    labels_cat = None
    if compute_auc and return_scores and all_scores:
        scores_cat = torch.cat(all_scores).cpu()
        labels_cat = torch.cat(all_labels).to(torch.long).cpu()

    return (
        clm_num,
        clm_den,
        pn_num,
        pn_den,
        auc_anom_val,
        auc_anom_count,
        auc_norm_val,
        auc_norm_count,
        s_norm_sum,
        s_norm_count,
        s_anom_sum,
        s_anom_count,
        per_nf,
        scores_cat,
        labels_cat,
    )


def _gather_per_nf(per_nf: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Gather and merge per-NF metric dictionaries across all ranks.

    Each rank contributes its local ``per_nf`` mapping. The dictionaries are
    gathered via :func:`torch.distributed.all_gather_object` and merged by
    summing the relevant fields for every NF.
    """
    fields = (
        "clm_num",
        "clm_den",
        "pn_num",
        "pn_den",
        "seq_num",
        "seq_den",
        "s_norm_sum",
        "s_norm_count",
        "s_anom_sum",
        "s_anom_count",
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, per_nf)
    else:
        gathered = [per_nf]

    merged: dict[str, dict[str, float]] = {}
    for d in gathered:
        for nf, rec in d.items():
            acc = merged.setdefault(nf, {k: 0.0 for k in fields})
            for k in fields:
                acc[k] += float(rec.get(k, 0.0))
    return merged


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    cfg_path = os.path.abspath(args.config)
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(int(cfg.get("seed", 1592)), device_specific=False)

    approach = cfg.get("approach", "warmup")
    approach_step = cfg.get("approach_step", 1)
    PFX = f"[{approach}:{approach_step}]"

    repo_cfg = load_repo_config()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # fsdp_plugin = FullyShardedDataParallelPlugin(
    #     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    # )

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision")
        or repo_cfg.get("train", {}).get("mixed_precision", "no"),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False,
                                                  split_batches=False),
        kwargs_handlers=[ddp_kwargs],
    )

    cpu_threads = int(
        repo_cfg.get("train", {}).get("cpu_threads",
                                      os.cpu_count() or 1))
    cpu_count = os.cpu_count() or 1
    device_type = accelerator.device.type
    logger.info(
        f"{PFX} Using {accelerator.num_processes} {device_type} device(s)")
    if device_type == "cpu":
        torch.set_num_threads(cpu_threads)
        logger.info(f"{PFX} Configuring {cpu_threads} CPU thread(s)")
        cpu_count = cpu_threads

    # --- Tokenizer & dataset ---
    base_model_path = cfg.get("base_model") or repo_cfg.get("model", {}).get(
        "base_model", "")
    tokenizer = build_tokenizer(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "Tokenizer `pad_token` was None, setting it to `eos_token`.")
    pad_id = tokenizer.pad_token_id

    exclude_anomalous_logs = bool(cfg.get("exclude_anomalous_logs", False))
    keep_clean_in_anomalous = bool(cfg.get("keep_clean_in_anomalous", False))
    if exclude_anomalous_logs:
        logger.warning(
            "[DATA] `exclude_anomalous_logs` is incompatible with defect "
            "training; including anomalous snippets instead."
        )
        exclude_anomalous_logs = False

    clean_logs_source = resolve_path(
        repo_cfg.get("clean_logs_dir", "data/raw_data/logs")
    )
    defect_data_root = resolve_path(
        repo_cfg.get("defect_logs_dir", "data/preprocessed_data/defect_data")
    )

    max_clean_cfg = cfg.get("max_clean_logs_per_nf")
    if max_clean_cfg is not None:
        logger.warning(
            "[DATA] max_clean_logs_per_nf=%s is configured but the defect corpus "
            "is precomputed. Regenerate the corpus to change this limit.",
            max_clean_cfg,
        )
    max_snippets_cfg = cfg.get("max_anomalous_snippets_per_nf")
    if max_snippets_cfg is not None:
        logger.warning(
            "[DATA] max_anomalous_snippets_per_nf=%s is configured but the defect "
            "corpus controls this value. Regenerate the corpus to change it.",
            max_snippets_cfg,
        )

    anomaly_meta = defect_data_root / "metadata.json"

    if not defect_data_root.exists():
        raise FileNotFoundError(
            f"Defect corpus directory not found: {defect_data_root}"
        )
    if not clean_logs_source.exists():
        raise FileNotFoundError(
            f"Clean log directory not found: {clean_logs_source}"
        )
    if not anomaly_meta.exists():
        raise FileNotFoundError(
            f"Anomaly metadata not found: {anomaly_meta}"
        )

    with anomaly_meta.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    rows = metadata.get("rows", [])
    if not rows:
        raise ValueError(
            f"No anomalous log snippets found in metadata file {anomaly_meta}."
        )

    anomaly_files: list[Path] = []
    for row in rows:
        rel = row.get("file")
        if not rel:
            continue
        anomaly_path = defect_data_root / rel
        anomaly_files.append(anomaly_path)
    missing_files = [path for path in anomaly_files if not path.exists()]
    if missing_files:
        missing_preview = ", ".join(str(p) for p in missing_files[:3])
        raise FileNotFoundError(
            f"Defect corpus metadata references missing anomaly logs: {missing_preview}"
        )

    target_nfs = sorted(
        {str(row.get("nf", "")).lower() for row in rows if row.get("nf")}
    )
    total_snippets = len(rows)

    clean_nf_counts: dict[str, int] = {}
    for fp in sorted(clean_logs_source.glob("**/*.log")):
        nf = nf_from_filename(fp).lower()
        if target_nfs and nf not in target_nfs:
            continue
        clean_nf_counts[nf] = clean_nf_counts.get(nf, 0) + 1

    if not clean_nf_counts:
        raise ValueError(
            "No clean logs are available for the network functions present in the defect corpus."
        )

    total_clean_files = sum(clean_nf_counts.values())

    logger.info(
        "[DATA] Loaded %d anomalous snippet(s) across %d NF(s) from %s",
        total_snippets,
        len(target_nfs),
        anomaly_meta,
    )
    logger.info(
        "[DATA] Located %d clean log file(s) across %d NF(s) under %s",
        total_clean_files,
        len(clean_nf_counts),
        clean_logs_source,
    )
    logger.info("[DATA] Clean log source directory: %s", clean_logs_source)

    root = clean_logs_source

    cfg_map_path = resolve_path(
        repo_cfg.get("log_code_cfg_map",
                     "data/preprocessed_data/log_code_cfg_map.json"))
    if not cfg_map_path.exists():
        cfg_map_path = None

    max_seq_len = int(cfg.get("max_seq_len", 2048))
    stride = int(cfg.get("stride", 256))

    logger.info("[DATA] Using clean logs from %s", clean_logs_source)

    corruption_prob = 0.0 if anomaly_meta else float(
        cfg.get("corruption_prob", 0.0))
    hard_neg_prob = 0.0 if anomaly_meta else float(
        cfg.get("hard_neg_prob", 0.35))
    pair_windows = True
    exclude_nfs = repo_cfg.get("exclude_nf", [])

    base_dataset = PerNFDatasetDefect(
        root=root,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        corruption_prob=corruption_prob,
        hard_neg_prob=hard_neg_prob,
        cfg_map_path=cfg_map_path,
        anomaly_metadata_path=anomaly_meta,
        pair_windows=pair_windows,
        keep_clean_in_anomalous=keep_clean_in_anomalous,
        exclude_nfs=exclude_nfs,
        additional_roots=[defect_data_root],
    )

    # If pairing was requested but the dataset contains only normal windows,
    # ``pairs`` will be empty.  In that case disable pair-wise logic so that
    # warm-up training on clean logs proceeds without assertions.
    if pair_windows and not base_dataset.pairs:
        pair_windows = False
        base_dataset.pair_windows = False

    val_pct = float(cfg.get("validation_split_percentage", 20))
    train_ds, val_ds = _split_dataset(base_dataset, val_pct)
    if accelerator.is_main_process:
        _log_sample_pairs(train_ds, "train")
        _log_sample_pairs(val_ds, "validation")
    val_clean_indices = [
        i for i, d in enumerate(val_ds) if int(d["label"]) == 0
    ]
    val_clean_ds = _subset_dataset(val_ds, val_clean_indices)

    pn_dataset = PerNFDatasetDefect(
        root=root,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        corruption_prob=corruption_prob,
        hard_neg_prob=0.0,
        cfg_map_path=cfg_map_path,
        anomaly_metadata_path=anomaly_meta,
        pair_windows=pair_windows,
        keep_clean_in_anomalous=keep_clean_in_anomalous,
        exclude_nfs=exclude_nfs,
        additional_roots=[defect_data_root],
    )
    #_log_pair_counts(pn_dataset, "PN_TOTAL")
    pn_train_ds, val_pn_ds = _split_dataset(pn_dataset, val_pct)
    #_log_pair_counts(pn_train_ds, "PN_TRAIN")
    #_log_pair_counts(val_pn_ds, "PN_VAL_BEFORE_FILTER")
    val_pn_labels_all = [int(d["label"]) for d in val_pn_ds]
    keep_idx: list[int] = []
    for nf, idxs in val_pn_ds.nf_to_indices.items():
        lbls = [val_pn_labels_all[i] for i in idxs]
        if any(lbl == 0 for lbl in lbls) and any(lbl == 1 for lbl in lbls):
            keep_idx.extend(idxs)
        else:
            logger.info("[PN-FILTER] dropping nf=%s lacking class", nf)
    val_pn_ds = _subset_dataset(val_pn_ds, keep_idx)
    #_log_pair_counts(val_pn_ds, "PN_VAL")

    try:
        logger.info(
            "%s Train windows ~%d | Val windows ~%d | ValPN windows ~%d",
            PFX,
            len(train_ds),
            len(val_clean_ds),
            len(val_pn_ds),
        )
    except TypeError:
        logger.info("%s Dataset length unavailable", PFX)

    collate_fn = partial(_collate, pad_id=pad_id)
    batch_size = int(cfg.get("batch_size", 1))
    base_num_workers = max(1, min(8, cpu_count // 2))
    train_num_workers = 0 if len(
        train_ds.files) < base_num_workers else base_num_workers
    val_num_workers = 0 if len(
        val_clean_ds.files) < base_num_workers else base_num_workers

    training_cfg = cfg.get("training", {})
    lambda_clm = float(training_cfg.get("lambda_clm", 1.0))
    lambda_pn_base = float(training_cfg.get("lambda", 0.0))
    margin = float(training_cfg.get("margin", 0.5))
    auc_weight = float(training_cfg.get("auc_weight", 0.0))
    cm_cfg = training_cfg.get("curriculum_masking", {})
    lambda_sched = training_cfg.get("lambda_schedule", {})
    evi_cfg = training_cfg.get("evidential_supervision", {})
    lambda_evi = float(evi_cfg.get("lambda_evi", 0.0))
    temperature = float(evi_cfg.get("temperature", 2.0))
    label_smoothing = float(evi_cfg.get("label_smoothing", 0.05))
    focal_gamma = float(evi_cfg.get("focal_gamma", 0.0))
    adapter_cfg = training_cfg.get("adapter_config", {})
    class_map = evi_cfg.get("class_map")
    if class_map and (class_map.get("normal") != 0
                      or class_map.get("anomaly") != 2):
        raise ValueError(
            "training.evidential_supervision.class_map must map normal:0, anomaly:2"
        )
    patience = cfg.get("early_stopping_patience", 5)
    eval_steps = cfg.get("eval_steps", 100)
    clip_norm = float(cfg.get("clip_grad_norm", 0))
    epochs = int(cfg.get("epochs", 1))
    best_val = float("inf")
    patience_ctr = 0
    gstep = 0
    best_step = None
    save_path = resolve_path(
        cfg.get("save_path")) if cfg.get("save_path") else None
    early_stop_metric = cfg.get("early_stop_metric", "auroc")
    early_stop_gamma = float(cfg.get("early_stop_gamma", 0.25))
    early_stop_beta = float(cfg.get("early_stop_beta", 0.0))
    dry_run = cfg.get("dry_run")
    dry_run_steps = int(cfg.get("dry_run_steps", 3))

    if base_num_workers > 0 and (train_num_workers == 0
                                 or val_num_workers == 0):
        logger.warning(
            "%s Dataset smaller than num_workers; disabling workers to avoid deadlock.",
            PFX)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    pin_mem = (accelerator.device.type == "cuda")

    labels = [int(d["label"]) for d in train_ds]

    if pair_windows and train_ds.pairs:
        sampler = PairedBatchSampler(
            train_ds.pairs_by_nf,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )
    else:
        sampler = PerNFBatchSampler(
            train_ds.nf_to_indices,
            labels,
            batch_size=batch_size,
            lambda_pn=1.0,
            rank=rank,
            world_size=world_size,
            shard_mode="batch_mod",
            drop_last=True,
        )
    _log_nf_sources("TRAIN", sampler)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=train_num_workers,
        pin_memory=pin_mem,
    )

    val_labels_clean = [int(d["label"]) for d in val_clean_ds]
    if pair_windows and val_clean_ds.pairs:
        val_clean_sampler = PairedBatchSampler(
            val_clean_ds.pairs_by_nf,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            rank=rank,
            world_size=world_size,
        )
    else:
        val_clean_sampler = PerNFBatchSampler(
            val_clean_ds.nf_to_indices,
            val_labels_clean,
            batch_size=batch_size,
            lambda_pn=0.0,
            rank=rank,
            world_size=world_size,
            shard_mode="batch_mod",
            drop_last=False,
        )
    _log_nf_sources("VAL_CLEAN", val_clean_sampler)
    val_clean_loader = DataLoader(
        val_clean_ds,
        batch_sampler=val_clean_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=pin_mem,
    )

    val_pn_labels = [int(d["label"]) for d in val_pn_ds]
    if pair_windows and val_pn_ds.pairs:
        val_pn_sampler = PairedBatchSampler(
            val_pn_ds.pairs_by_nf,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            rank=rank,
            world_size=world_size,
        )
    else:
        val_pn_sampler = PerNFBatchSampler(
            val_pn_ds.nf_to_indices,
            val_pn_labels,
            batch_size=batch_size,
            lambda_pn=1.0,
            rank=rank,
            world_size=world_size,
            shard_mode="batch_mod",
            drop_last=False,
        )
    _log_nf_sources("VAL_PN", val_pn_sampler)
    val_pn_loader = DataLoader(
        val_pn_ds,
        batch_sampler=val_pn_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=pin_mem,
    )

    eval_nfs = sorted(
        set(val_clean_sampler.get_nf_list())
        | set(val_pn_sampler.get_nf_list())
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered_nfs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_nfs, eval_nfs)
        eval_nfs = sorted({nf for sub in gathered_nfs for nf in sub})

    auc_scores_buffer: deque[torch.Tensor] = deque(maxlen=5)
    auc_labels_buffer: deque[torch.Tensor] = deque(maxlen=5)

    if accelerator.is_main_process:
        train_pos, train_neg = _dataset_label_stats(train_ds)
        val_pos, val_neg = _dataset_label_stats(val_clean_ds)
        val_pn_pos, val_pn_neg = _dataset_label_stats(val_pn_ds)
        logger.info(
            "[DATA] Train dataset: total=%d | anomaly=%d | normal=%d | "
            "max_seq_len=%d | stride=%d | batch_size=%d",
            len(train_ds),
            train_pos,
            train_neg,
            train_ds.max_seq_len,
            train_ds.stride,
            batch_size,
        )
        logger.info(
            "[DATA] Validation clean dataset: total=%d | anomaly=%d | normal=%d | "
            "max_seq_len=%d | stride=%d | batch_size=%d",
            len(val_clean_ds),
            val_pos,
            val_neg,
            val_clean_ds.max_seq_len,
            val_clean_ds.stride,
            batch_size,
        )
        logger.info(
            "[DATA] Validation PN dataset: total=%d | anomaly=%d | normal=%d | "
            "max_seq_len=%d | stride=%d | batch_size=%d",
            len(val_pn_ds),
            val_pn_pos,
            val_pn_neg,
            val_pn_ds.max_seq_len,
            val_pn_ds.stride,
            batch_size,
        )
        # _log_first_batch(train_loader, tokenizer, "train")
        # _log_first_batch(val_clean_loader, tokenizer, "validation")

    if dry_run:
        steps = dry_run_steps
        for i, batch in enumerate(train_loader):
            nf = batch["nf"]
            idxs = batch["nf_idx"].tolist()
            labels = batch["label"].tolist()
            logger.info(
                "[DRY-RUN] batch=%d nf=%s indices=%s labels=%s",
                i,
                nf,
                idxs,
                labels,
            )
            if i + 1 >= steps:
                break
        return

    # --- Base/PEFT model ---
    base_model_path = resolve_path(base_model_path)
    logger.info(f"{PFX} Loading base model {base_model_path}")
    if accelerator.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = None
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                      low_cpu_mem_usage=True,
                                                      device_map=None,
                                                      torch_dtype=torch_dtype)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    try:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        logger.warning(
            f"{PFX} Gradient checkpointing not supported with use_reentrant")
        base_model.gradient_checkpointing_enable()
    if accelerator.is_main_process:
        logger.info(f"{PFX} Base model loaded:\n {base_model}")

    peft_model = None
    load_adapter_path = cfg.get("load_adapter_path")
    if load_adapter_path:
        if os.path.isdir(load_adapter_path):
            peft_model = PeftModel.from_pretrained(base_model,
                                                   load_adapter_path,
                                                   is_trainable=True,
                                                   device_map=None,
                                                   inference_mode=False,
                                                   use_safetensors=True,
                                                   torch_dtype=torch_dtype)
            logger.info(f"{PFX} Loaded PEFT adapter from {load_adapter_path}")
        else:
            peft_model = base_model
            logger.info(f"{PFX} Using base model without PEFT adapter")
    else:
        peft_model = base_model
        logger.info(f"{PFX} Using base model without PEFT adapter")
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    if accelerator.is_main_process:
        logger.info(f"{PFX} PEFT model loaded:\n {peft_model}")

    if accelerator.is_main_process and hasattr(peft_model,
                                               "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    logger.info(f"{PFX} Wrapping with JanusModel")
    model = JanusModel(
        base_model=peft_model,
        dual_mask=cfg.get("dual_mask", True),
        force_local_attention_only=adapter_cfg.get("local_only", False),
    )

    if adapter_cfg.get("local_only", False):
        freeze_global_adapters(model)

    if accelerator.is_main_process:
        global_lora_names = [
            n for n, _ in model.named_parameters() if "lora_" in n and (
                "global" in n.lower() or "global_adapter" in n.lower())
        ]
        logger.info(
            "%s Global-tagged LoRA params: %d%s", PFX, len(global_lora_names),
            f" | examples: {global_lora_names[:8]}"
            if global_lora_names else "")

    # Load custom heads
    if load_adapter_path and os.path.isdir(load_adapter_path):
        for head_name in ("cls_head", "evidential_head"):
            head_file = os.path.join(load_adapter_path,
                                     f"{head_name}_tensor.pth")
            if os.path.exists(head_file):
                logger.info(
                    f"{PFX} Loading full head {head_name} from {head_file}")
                state_dict = torch.load(head_file, map_location="cpu")
                getattr(model, head_name).load_state_dict(state_dict,
                                                          strict=True)
                logger.info(f"{PFX} Loaded {head_name} from previous phase.")
    logger.info(f"{PFX} Model ready.")

    lr = float(cfg.get("learning_rate", 1e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = n.startswith("cls_head.") or n.startswith("evidential_head.")
        is_lora = "lora_" in n
        if is_head or is_lora:
            if n.endswith("bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        else:
            no_decay.append(p)
    if accelerator.is_main_process:
        logger.info(f"{PFX} Wrapped JanusModel:\n {model}")

    optimizer = AdamW(
        [{
            "params": decay,
            "weight_decay": wd
        }, {
            "params": no_decay,
            "weight_decay": 0.0
        }],
        lr=lr,
    )

    model, optimizer, train_loader, val_clean_loader, val_pn_loader = accelerator.prepare(
        model, optimizer, train_loader, val_clean_loader, val_pn_loader)

    try:
        num_update_steps_per_epoch = math.ceil(
            len(train_loader) / accelerator.gradient_accumulation_steps)
        max_train_steps = epochs * num_update_steps_per_epoch
    except TypeError:
        logger.warning(
            f"{PFX} Could not infer max_train_steps, using default 100000")
        max_train_steps = 100000

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.get("warmup_steps", 0)),
        num_training_steps=max_train_steps)

    accelerator.wait_for_everyone()

    loss_log_path = resolve_path(
        cfg.get("loss_log_path", f"artifacts/training_loss_{approach}.csv"))
    loss_writer = None
    loss_file = None
    if accelerator.is_main_process:
        loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        loss_file = loss_log_path.open("w", newline="")
        loss_writer = csv.writer(loss_file)
        loss_writer.writerow([
            "timestamp",
            "approach",
            "approach_step",
            "epoch",
            "step",
            "avg_train_loss",
            "val_loss_clean_clm",
            "val_pn_metric",
            "avg_pn_loss",
            "avg_clm_loss",
            "avg_evi_loss",
            "mask_rate",
            "lambda_pn",
            "auc_weight",
            "metric_name",
            "val_metric",
            "val_auroc_s",
            "auc_pos_anom",
            "auc_pos_norm",
            "val_s_norm",
            "val_s_anom",
            "delta_s",
            "nf",
            "val_clm_nf",
            "val_pn_nf",
        ])

    accumulated_train_loss = 0.0
    accumulated_pn_loss = 0.0
    accumulated_clm_loss = 0.0
    accumulated_evi_loss = 0.0
    logging_steps = 0

    # -----------------------------
    # Training loop
    # -----------------------------
    gstep = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        val_pn_sampler.set_epoch(epoch)
        if pair_windows and hasattr(val_clean_sampler, "set_epoch"):
            val_clean_sampler.set_epoch(epoch)
        nf_list = sorted(train_ds.nf_to_indices)
        counts = [len(train_ds.nf_to_indices[nf]) for nf in nf_list]
        logger.info("[EPOCH] rank=%d nf_list=%s counts=%s", rank, nf_list,
                    counts)

        logger.info(f"{PFX} Epoch {epoch + 1}/{epochs}")
        model.train()

        for batch in train_loader:
            nf = batch.pop("nf")
            nf_idx = batch["nf_idx"]
            batch = {
                k: v.to(accelerator.device, non_blocking=True)
                for k, v in batch.items()
            }

            with accelerator.accumulate(model):
                mask_rate = get_current_masking_rate(
                    cm_cfg.get("initial_rate", 0.30),
                    cm_cfg.get("final_rate", 0.20),
                    cm_cfg.get("decay_steps", 500),
                    gstep,
                )
                if hasattr(train_loader.dataset, 'set_corruption_prob'):
                    train_loader.dataset.set_corruption_prob(mask_rate)

                if lambda_sched:
                    lambda_pn = get_monotone_schedule(
                        lambda_sched.get("start", lambda_pn_base),
                        lambda_sched.get("end", lambda_pn_base),
                        lambda_sched.get("steps", 1),
                        gstep,
                    )
                else:
                    lambda_pn = lambda_pn_base

                with accelerator.autocast():
                    out = model(batch["input_ids"], batch["tags"])
                    logits = out["logits"]
                    loss_clm = torch.tensor(0.0, device=accelerator.device)
                    if lambda_clm > 0.0:
                        logits_shifted = logits[:, :-1, :].transpose(1, 2)
                        labels_shifted = batch["input_ids"][:, 1:].clone()
                        anom_rows = (batch["label"] == 1).nonzero(as_tuple=False).flatten()
                        if anom_rows.numel() > 0:
                            labels_shifted[anom_rows, :] = pad_id
                        ce_tok = F.cross_entropy(
                            logits_shifted, labels_shifted, ignore_index=pad_id, reduction="none"
                        )
                        tok_mask = (labels_shifted != pad_id).float()
                        if lambda_pn > 0:
                            ce_per_seq = (
                                (ce_tok * tok_mask).sum(dim=1)
                                / tok_mask.sum(dim=1).clamp_min(1.0)
                            )
                            loss_clm = ce_per_seq.mean()
                        else:
                            loss_clm = (ce_tok * tok_mask).sum() / tok_mask.sum().clamp_min(1.0)

                    # CLS-based anomaly score (higher = more anomalous)
                    scores = out["cls"].squeeze(1).to(torch.float32)
                    
                    # Ignore Evidential loss
                    if lambda_evi > 0.0:
                        evi_loss = evidential_ce_loss(
                            out["evidential"],
                            batch["label"],
                            temperature,
                            label_smoothing,
                            focal_gamma,
                        )
                    else:
                        evi_loss = torch.tensor(0.0, device=accelerator.device)
                    
                labels = batch["label"]
                if pair_windows:
                    if labels.numel() % 2 != 0 or not torch.all(
                            labels.view(-1, 2).sum(dim=1) == 1):
                        raise AssertionError(
                            "paired batch requires alternating labels")
                    pair_losses = []
                    for i in range(0, labels.size(0) - 1, 2):
                        l1, l2 = labels[i], labels[i + 1]
                        if l1 == l2:
                            continue
                        s1, s2 = scores[i], scores[i + 1]
                        if l1 == 1:
                            pair_losses.append(torch.relu(margin - s1 + s2))
                        else:
                            pair_losses.append(torch.relu(margin - s2 + s1))
                    if pair_losses:
                        hinge = torch.stack(pair_losses).mean()
                        auc = pairwise_logistic(scores, labels)
                        pn_loss = hinge + (auc_weight *
                                           auc if lambda_pn > 0 else 0.0)
                    else:
                        pn_loss = torch.tensor(0.0, device=accelerator.device)
                else:
                    normal_mask = labels == 0
                    anomaly_mask = labels == 1
                    loss_anom = (
                        F.relu(margin - scores[anomaly_mask]).mean()
                        if anomaly_mask.any()
                        else torch.tensor(0.0, device=accelerator.device)
                    )
                    loss_norm = (
                        F.relu(margin + scores[normal_mask]).mean()
                        if normal_mask.any()
                        else torch.tensor(0.0, device=accelerator.device)
                    )
                    hinge = loss_anom + loss_norm
                    auc = pairwise_logistic(scores, labels)
                    pn_loss = hinge + (
                        auc_weight * auc if lambda_pn > 0 else 0.0
                    )

                loss = (lambda_clm * loss_clm) + (lambda_pn * pn_loss) + (lambda_evi * evi_loss)
                accelerator.backward(loss)

                accumulated_train_loss += loss.item()
                accumulated_pn_loss += pn_loss.item()
                accumulated_clm_loss += loss_clm.item()
                if lambda_evi > 0.0:
                    accumulated_evi_loss += evi_loss.item()
                logging_steps += 1

                if accelerator.sync_gradients:
                    if clip_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(),
                                                    clip_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    gstep += 1
                    first_idx = int(nf_idx.min().item())
                    last_idx = int(nf_idx.max().item())
                    num_norm = int((batch["label"] == 0).sum().item())
                    num_anom = int((batch["label"] == 1).sum().item())
                    if not torch.all(nf_idx[1:] >= nf_idx[:-1]):
                        raise AssertionError("window indices not increasing")
                    logger.info(
                        "[TRAIN] epoch=%d step=%d nf=%s batch_size=%d first_index_in_nf=%d last_index_in_nf=%d num_normals=%d num_anomalies=%d pn_lambda=%.3f margin=%.2f auc_weight=%.2f",
                        epoch + 1,
                        gstep,
                        nf,
                        batch["label"].size(0),
                        first_idx,
                        last_idx,
                        num_norm,
                        num_anom,
                        lambda_pn,
                        margin,
                        auc_weight,
                    )

                    # ---- Synchronized Evaluation ----
                    do_eval_local = gstep > 0 and (gstep % int(eval_steps)
                                                   == 0)
                    do_eval = _broadcast_eval_flag(do_eval_local,
                                                   accelerator.device)
                    if do_eval:
                        if torch.distributed.is_available(
                        ) and torch.distributed.is_initialized():
                            torch.cuda.synchronize()
                            torch.distributed.barrier()
                        if accelerator.is_main_process:
                            logger.debug("%s Running evaluation at step %d",
                                         PFX, gstep)

                        # (A) CLEAN-ONLY CLM
                        (clm_num_c, clm_den_c, _, _, _, _, _, _, _, _, _, _,
                         per_nf_clean, _, _) = evaluate(
                            model,
                            val_clean_loader,
                            accelerator,
                            pad_id,
                             margin,
                             0.0,
                             lambda_pn,
                             compute_auc=False,
                             pair_windows=False,
                         )
                        per_nf_clean = _gather_per_nf(per_nf_clean)

                        # Debug: if an NF is in eval_nfs but has zero clean denominator, log it once.
                        try:
                            missing_clean = [nf for nf in eval_nfs if float(per_nf_clean.get(nf, {}).get("clm_den", 0.0)) <= 0.0]
                            if missing_clean:
                                logger.info("[VAL_CLEAN] nfs with zero clean-denominator this eval: %s", ",".join(missing_clean))
                        except Exception:
                            pass

                        # (B) MIXED PN tracking
                        (
                            _,
                            _,
                            pn_num_e,
                            pn_den_e,
                            auc_val_tmp_anom,
                            auc_cnt_anom,
                            auc_val_tmp_norm,
                            auc_cnt_norm,
                            s_norm_sum,
                            s_norm_count,
                            s_anom_sum,
                            s_anom_count,
                            per_nf_pn,
                            scores_buf,
                            labels_buf,
                        ) = evaluate(
                            model,
                            val_pn_loader,
                            accelerator,
                            pad_id,
                            margin,
                            auc_weight,
                            lambda_pn,
                            compute_auc=True,
                            pair_windows=False,
                            return_scores=True,
                        )
                        per_nf_pn = _gather_per_nf(per_nf_pn)
                        if scores_buf is not None and labels_buf is not None:
                            auc_scores_buffer.append(scores_buf)
                            auc_labels_buffer.append(labels_buf)
                            try:
                                from sklearn.metrics import roc_auc_score

                                scores_cat = torch.cat(
                                    list(auc_scores_buffer)).cpu().numpy()
                                labels_cat = torch.cat(
                                    list(auc_labels_buffer)).cpu().numpy()
                                auc_pos_anom = float(
                                    roc_auc_score(labels_cat, scores_cat))
                                auc_pos_norm = float(
                                    roc_auc_score(labels_cat, -scores_cat))
                            except Exception:
                                auc_pos_anom = float("nan")
                                auc_pos_norm = float("nan")
                        else:
                            auc_pos_anom = auc_val_tmp_anom
                            auc_pos_norm = auc_val_tmp_norm
                        if accelerator.is_main_process and not (
                            math.isnan(auc_pos_anom)
                            or math.isnan(auc_pos_norm)
                        ):
                            auc_cnt = 1.0
                            auc_val_to_reduce = max(auc_pos_anom, auc_pos_norm)
                            auc_pos_anom_val = auc_pos_anom
                            auc_pos_norm_val = auc_pos_norm
                            if auc_pos_norm_val > auc_pos_anom_val:
                                logger.warning(
                                    "%s AUROC polarity inverted; using normal-positive score",
                                    PFX,
                                )
                        else:
                            auc_cnt = 0.0
                            auc_val_to_reduce = 0.0
                            auc_pos_anom_val = float("nan")
                            auc_pos_norm_val = float("nan")

                        # ---- Reductions (single packed float32 all_reduce; NCCL-safe) ----
                        vec_local = torch.tensor(
                            [
                                float(clm_num_c),
                                float(clm_den_c),
                                float(pn_num_e),
                                float(pn_den_e),
                                float(auc_val_to_reduce),
                                float(auc_cnt),
                                float(s_norm_sum),
                                float(s_norm_count),
                                float(s_anom_sum),
                                float(s_anom_count),
                            ],
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                        vec_global = vec_local.clone()
                        if torch.distributed.is_available(
                        ) and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                vec_global, op=torch.distributed.ReduceOp.SUM)

                        g_clm_num_c = vec_global[0].item()
                        g_clm_den_c = vec_global[1].item()
                        g_pn_num = vec_global[2].item()
                        g_pn_den = vec_global[3].item()
                        g_auc_sum = vec_global[4].item()
                        g_auc_count = vec_global[5].item()
                        g_s_norm_sum = vec_global[6].item()
                        g_s_norm_count = vec_global[7].item()
                        g_s_anom_sum = vec_global[8].item()
                        g_s_anom_count = vec_global[9].item()

                        clm_val = g_clm_num_c / max(1.0, g_clm_den_c)
                        pn_val = g_pn_num / max(1.0, g_pn_den)
                        auroc_val = g_auc_sum / max(1.0, g_auc_count)
                        s_norm_mean = (
                            g_s_norm_sum / max(1.0, g_s_norm_count)
                        ) if g_s_norm_count > 0.5 else float("nan")
                        s_anom_mean = (
                            g_s_anom_sum / max(1.0, g_s_anom_count)
                        ) if g_s_anom_count > 0.5 else float("nan")
                        delta_s = (s_anom_mean - s_norm_mean) if not (
                            math.isnan(s_anom_mean)
                            or math.isnan(s_norm_mean)) else float("nan")

                        if accelerator.is_main_process:
                            logger.info(
                                f"[PN-EVAL] reduced pn_den={int(g_pn_den)} pn_num={g_pn_num:.6e} "
                                f"raw_pn_val={pn_val:.9f}")

                        auc_for_stop = auroc_val if not math.isnan(
                            auroc_val) else 0.5
                        metric = early_stop_metric
                        if metric == "composite":
                            criterion_val = clm_val + early_stop_beta * pn_val
                            tracked_name = "composite"
                        elif metric == "auroc":
                            criterion_val = 1.0 - auc_for_stop
                            tracked_name = "1-AUROC"
                        elif metric == "composite_evi":
                            criterion_val = (clm_val +
                                             early_stop_beta * pn_val +
                                             early_stop_gamma *
                                             (1.0 - auc_for_stop))
                            tracked_name = "composite_evi"
                        else:
                            criterion_val = clm_val
                            tracked_name = "clean-clm"

                        if torch.distributed.is_available(
                        ) and torch.distributed.is_initialized():
                            torch.cuda.synchronize()
                            torch.distributed.barrier()
                        model.train()

                        avg_train_loss = accumulated_train_loss / max(
                            1, logging_steps)
                        avg_pn_loss = accumulated_pn_loss / max(
                            1, logging_steps)
                        avg_clm_loss = accumulated_clm_loss / max(
                            1, logging_steps)
                        avg_evi_loss = accumulated_evi_loss / max(
                            1, logging_steps)
                        accumulated_train_loss = accumulated_pn_loss = 0.0
                        accumulated_clm_loss = accumulated_evi_loss = 0.0
                        logging_steps = 0

                        should_save = False
                        should_stop = False
                        if accelerator.is_main_process:
                            logger.info(
                                f"{PFX} E{epoch + 1} S{gstep} | CLM={clm_val:.4f} | "
                                f"PN={pn_val:.4f} | AUC(S)={auroc_val:.4f} "
                                f"(anom={auc_pos_anom_val:.4f} norm={auc_pos_norm_val:.4f}) "
                                f"| delta_s={delta_s:.3f} | S_norm={s_norm_mean:.3f} "
                                f"| S_anom={s_anom_mean:.3f} | CE_evi(avg)={avg_evi_loss:.4f}"
                            )
                            if loss_writer:
                                loss_writer.writerow([
                                    datetime.now(timezone.utc).isoformat(),
                                    approach,
                                    approach_step,
                                    epoch + 1,
                                    gstep,
                                    f"{avg_train_loss:.6f}",
                                    f"{clm_val:.6f}",
                                    f"{pn_val:.6f}",
                                    f"{avg_pn_loss:.6f}",
                                    f"{avg_clm_loss:.6f}",
                                    f"{avg_evi_loss:.6f}",
                                    f"{mask_rate:.4f}",
                                    f"{lambda_pn:.3f}",
                                    f"{auc_weight:.3f}",
                                    tracked_name,
                                    f"{criterion_val:.6f}",
                                    f"{auroc_val:.6f}",
                                    f"{auc_pos_anom_val:.6f}",
                                    f"{auc_pos_norm_val:.6f}",
                                    f"{s_norm_mean:.6f}",
                                    f"{s_anom_mean:.6f}",
                                    f"{delta_s:.6f}",
                                    "ALL",
                                    f"{clm_val:.6f}",
                                    f"{pn_val:.6f}",
                                ])
                                for nf in eval_nfs:
                                    rec_c = per_nf_clean.get(
                                        nf, {
                                            "clm_num": 0.0,
                                            "clm_den": 0.0,
                                            "seq_num": 0.0,
                                            "seq_den": 0.0
                                        })
                                    rec_p = per_nf_pn.get(
                                        nf, {
                                            "pn_num": 0.0,
                                            "pn_den": 0.0
                                        })
                                    val_clm_nf = (rec_c["seq_num"] / rec_c["seq_den"]) if rec_c["seq_den"] > 0 else float("nan")
                                    val_pn_nf = (rec_p["pn_num"] / rec_p["pn_den"]) if rec_p["pn_den"] > 0 else float("nan")
                                    loss_writer.writerow([
                                        datetime.now(timezone.utc).isoformat(),
                                        approach,
                                        approach_step,
                                        epoch + 1,
                                        gstep,
                                        f"{avg_train_loss:.6f}",
                                        f"{clm_val:.6f}",
                                        f"{pn_val:.6f}",
                                        f"{avg_pn_loss:.6f}",
                                        f"{avg_clm_loss:.6f}",
                                        f"{avg_evi_loss:.6f}",
                                        f"{mask_rate:.4f}",
                                        f"{lambda_pn:.3f}",
                                        f"{auc_weight:.3f}",
                                        tracked_name,
                                        f"{criterion_val:.6f}",
                                        f"{auroc_val:.6f}",
                                        f"{auc_pos_anom_val:.6f}",
                                        f"{auc_pos_norm_val:.6f}",
                                        f"{s_norm_mean:.6f}",
                                        f"{s_anom_mean:.6f}",
                                        f"{delta_s:.6f}",
                                        nf,
                                        f"{val_clm_nf:.6f}",
                                        f"{val_pn_nf:.6f}",
                                    ])
                                loss_file.flush()

                            tol = 1e-4
                            if criterion_val + tol < best_val:
                                best_val = criterion_val
                                patience_ctr = 0
                                logger.info(
                                    f"{PFX} New best {tracked_name}: {best_val:.4f}"
                                )
                                should_save = True
                            else:
                                patience_ctr += 1
                                logger.info(
                                    f"{PFX} No {tracked_name} improvement. Patience {patience_ctr}/{patience}"
                                )
                                if patience_ctr >= patience:
                                    logger.info("%s Early stopping triggered.",
                                                PFX)
                                    should_stop = True

                        # flags via MAX all_reduce on float32 (safe)
                        flag_vec = torch.tensor([
                            1.0 if should_save else 0.0,
                            1.0 if should_stop else 0.0
                        ],
                                                device=accelerator.device,
                                                dtype=torch.float32)
                        if torch.distributed.is_available(
                        ) and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                flag_vec, op=torch.distributed.ReduceOp.MAX)
                        save_flag = int(flag_vec[0].item())
                        stop_flag = int(flag_vec[1].item())

                        if save_flag == 1 and save_path:
                            best_step = gstep
                            best_save_path = os.path.join(
                                Path(save_path).parent,
                                f"{Path(save_path).stem}_final")
                            save_full_checkpoint(model, best_save_path,
                                                 accelerator)

                        if stop_flag == 1:
                            patience_ctr = patience
                            if torch.distributed.is_available(
                            ) and torch.distributed.is_initialized():
                                torch.cuda.synchronize()
                                torch.distributed.barrier()
                            break  # inner loop

        if patience_ctr >= patience:
            logger.info(
                f"%s Rank {accelerator.process_index} breaking epoch loop (early stopping).",
                PFX)
            break
        logger.debug("%s Epoch %d completed", PFX, epoch + 1)

    if save_path and patience_ctr < patience and best_step is None:
        logger.info(
            f"{PFX} Training completed without early stopping. Saving final model state."
        )
        final_save_path = os.path.join(
            Path(save_path).parent, f"{Path(save_path).stem}_final")
        save_full_checkpoint(model, final_save_path, accelerator)

    accelerator.wait_for_everyone()
    logger.info("%s Training complete.", PFX)
    logger.debug("%s Final step count: %d", PFX, gstep)
    accelerator.end_training()
    if accelerator.is_main_process and loss_file:
        loss_file.close()


# Optional reference:
# pairwise = scores[anomaly_mask][:n].unsqueeze(1) - scores[normal_mask][:n].unsqueeze(0)
# hinge = F.relu(margin - pairwise).mean()

if __name__ == "__main__":
    main()
