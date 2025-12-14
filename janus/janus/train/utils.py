#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
from pathlib import Path
import torch
from accelerate import Accelerator
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Replicate

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def find_janus(m, depth=0):
    if all(hasattr(m, a) for a in ("model", "cls_head", "evidential_head")):
        return m
    for attr in ("module", "_orig_mod", "_fully_sharded_module", "_wrapped_module"):
        if hasattr(m, attr):
            found = find_janus(getattr(m, attr), depth + 1)
            if found:
                return found
    for _, child in m.named_children():
        found = find_janus(child, depth + 1)
        if found:
            return found
    return None


def state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Dump a module’s parameters (whether plain Tensors or DTensors)
    into a CPU‐only dict of torch.Tensor.
    """
    sd = module.state_dict()
    new_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        # If this is a DTensor shard, gather into a full tensor
        if isinstance(v, DTensor):
            try:
                # try gathering from shards via the original mesh
                v = v.redistribute(device_mesh=v.device_mesh,
                                   placements=[Replicate()]) \
                      .to_local()
            except RuntimeError as e:
                if "No backend type associated with device type cpu" in str(e):
                    # local shard is already on CPU and (we hope) already full
                    v = v.to_local()
                else:
                    raise
        # Finally detach & move to CPU
        if isinstance(v, torch.Tensor):
            new_sd[k] = v.detach().cpu()
        else:
            # any non‐tensor state (unlikely), keep as‐is
            new_sd[k] = v
    return new_sd


def save_full_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    accelerator: Accelerator,
) -> None:
    """
    Saves a complete checkpoint for a Janus phase (main process only).
    Produces in `path/`:
      - a PEFT adapter folder (via save_pretrained)
      - cls_head.pth, evidential_head.pth  (the *full* weights)
      - optimizer.pt
    """
    # This is a collective call, all processes must participate.
    # The log message the user sees is the one BEFORE this call in train.py.
    state_dict = accelerator.get_state_dict(model)

    # This is a collective operation and must be called by everyone.
    unwrapped = accelerator.unwrap_model(model)

    # All file I/O should be handled by the main process to prevent conflicts.
    if accelerator.is_main_process:

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"State dictionary gathered. Saving checkpoint to {save_dir} on main process...")

        janus = find_janus(unwrapped)
        if not janus:
            logger.error("Could not find JanusModel to save, aborting checkpoint.")
            accelerator.wait_for_everyone()
            return
        peft_model = janus.model
        logger.info("Extracted peft_model from JanusModel.")

        # The state_dict from the full JanusModel has keys prefixed with 'model.'.
        # We must strip this prefix before passing the state_dict to PEFT's save function.
        peft_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                peft_state_dict[key[len("model."):]] = value
            else:
                # This case should not be hit for PEFT parameters, but is safe to include
                peft_state_dict[key] = value

        # Save the PEFT adapter using the corrected state dictionary
        # Pass the complete state_dict to the save function.
        peft_model.save_pretrained(
            str(save_dir),
            safe_serialization=True,
            state_dict=peft_state_dict
        )
        logger.info(f"Saved PEFT adapter to {save_dir}")

        # 2) Save the full heads
        for head in ("cls_head", "evidential_head"):
            module = getattr(unwrapped, head, None)
            if module is not None:
                head_path = save_dir / f"{head}_dtensor.pth"
                torch.save(module.state_dict(), head_path)
                logger.info(f"Saved DTensor {head} to {head_path}")

    # All processes wait here. This ensures rank 0 finishes writing to disk
    # before any process moves on or exits the script.
    accelerator.wait_for_everyone()
