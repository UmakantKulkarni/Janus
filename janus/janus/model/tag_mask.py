#!/usr/bin/env python3
"""
Utility functions for creating attention masks used by Janus:
- Full causal mask (with PAD keys masked)
- Field-semantic tag mask (same-tag, optional causal, no-PAD)
Both masks guarantee at least one allowed key per query row (diagonal),
which prevents softmax NaNs when a row would otherwise be all -inf.
"""

from __future__ import annotations
import torch
from torch import Tensor

def _to_additive(mask_bool: Tensor) -> Tensor:
    """
    Convert a boolean allow-mask to an additive attention mask:
    True (allow)  -> 0.0
    False (block) -> -inf
    Shape preserved.
    """
    # Use the device of the mask, constants are created on the same device.
    zero = torch.tensor(0.0, device=mask_bool.device)
    neg_inf = torch.tensor(float("-inf"), device=mask_bool.device)
    return torch.where(mask_bool, zero, neg_inf)

def make_full_mask(ids: Tensor, pad_id: int) -> Tensor:
    """
    Standard causal mask with PAD keys masked, plus a guaranteed diagonal.
    Args:
        ids:    LongTensor [B, S] of token ids
        pad_id: int token id used for padding in 'ids'
    Returns:
        additive mask FloatTensor [B, 1, S, S] with 0.0 allowed, -inf blocked
    """
    bsz, seq_len = ids.shape
    device = ids.device

    # causal lower triangle (include diagonal so token can attend to itself)
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))  # [S,S]
    causal = causal.view(1, 1, seq_len, seq_len).expand(bsz, 1, seq_len, seq_len)        # [B,1,S,S]

    # valid keys are non-PAD positions (mask applies to keys/columns)
    valid_keys = (ids != pad_id).view(bsz, 1, 1, seq_len).expand(bsz, 1, seq_len, seq_len)  # [B,1,S,S]

    allow = causal & valid_keys  # block future & PAD keys

    # allow self on the diagonal for every row (even if the key is PAD)
    eye = torch.eye(seq_len, dtype=torch.bool, device=device).view(1, 1, seq_len, seq_len).expand(bsz, 1, seq_len, seq_len)
    allow = allow | eye

    return _to_additive(allow)

def make_tag_mask(
    ids: Tensor,
    tags: Tensor,
    pad_id: int,
    tag_pad_id: int = -100,
    include_causal: bool = True,
    ensure_self: bool = True,  # kept for API compatibility; diagonal is enforced unconditionally below
) -> Tensor:
    """
    Field-semantic local mask:
      - Only same-tag keys are visible
      - Optionally causal (recommended)
      - PAD keys and PAD tags are blocked
      - Diagonal is always allowed to avoid all -inf rows (prevents softmax NaNs)

    Args:
        ids:            LongTensor [B, S] token ids (to detect PAD tokens)
        tags:           LongTensor [B, S] tag ids for each token
        pad_id:         int token id used for padding in 'ids'
        tag_pad_id:     int pad/ignore id used in 'tags' (defaults to -100)
        include_causal: if True, also enforce lower-triangular causality
        ensure_self:    kept for backward-compat; diagonal is always allowed

    Returns:
        additive mask FloatTensor [B, 1, S, S] with 0.0 allowed, -inf blocked
    """
    bsz, seq_len = tags.shape
    device = tags.device

    # same-tag equality, excluding tag PAD positions from both query and key sides
    same_tag = tags.unsqueeze(1) == tags.unsqueeze(2)                                # [B,S,S]
    valid_tag_q = (tags != tag_pad_id).unsqueeze(2).expand(-1, -1, seq_len)          # [B,S,S]
    valid_tag_k = (tags != tag_pad_id).unsqueeze(1).expand(-1, seq_len, -1)          # [B,S,S]
    same_tag = same_tag & valid_tag_q & valid_tag_k

    # causal constraint (optional)
    if include_causal:
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))  # [S,S]
        causal = causal.unsqueeze(0)                                                          # [1,S,S]
        tag_allow = same_tag & causal
    else:
        tag_allow = same_tag

    # block PAD keys (columns)
    valid_keys = (ids != pad_id).unsqueeze(1).expand(-1, seq_len, -1)               # [B,S,S]
    tag_allow = tag_allow & valid_keys

    # allow self on the diagonal for every row (even PAD queries)
    # Do this AFTER key masking so the diagonal survives even if key is PAD.
    idx = torch.arange(seq_len, device=device)
    tag_allow[:, idx, idx] = True

    # reshape to [B,1,S,S] and convert to additive mask
    tag_allow = tag_allow.view(bsz, 1, seq_len, seq_len)
    return _to_additive(tag_allow)