#!/usr/bin/env python3
"""
This module defines the Janus model architecture, including the core
Single-Pass Dual-Mask Attention mechanism.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers.models.llama.modeling_llama import LlamaAttention

from janus.model.tag_mask import make_full_mask, make_tag_mask

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class DualMaskAttention(LlamaAttention):
    """
    A LlamaAttention layer modified to support Single-Pass Dual-Masking.
    It can also be forced to use only local attention for all heads.
    """
    def __init__(self, *args, **kwargs):
        self.force_local_only = kwargs.pop("force_local_only", False)
        super().__init__(*args, **kwargs)
        # Newer versions of transformers do not expose num_heads attributes
        self.num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
        self.num_key_value_heads = getattr(
            self, "num_key_value_heads", self.config.num_key_value_heads
        )
        self.num_key_value_groups = getattr(
            self,
            "num_key_value_groups",
            self.num_heads // self.num_key_value_heads,
        )
        self.hidden_size = getattr(self, "hidden_size", self.config.hidden_size)

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the standard KV repetition method from the Llama implementation,
        used for Grouped-Query Attention.
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None and attention_mask.dim() == 4 and attention_mask.shape[1] == 2:
            full_mask = attention_mask[:, 0:1, :, :]
            tag_mask = attention_mask[:, 1:2, :, :]
        else:
            full_mask = attention_mask
            tag_mask = full_mask

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.force_local_only:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
            attn_output = self._masked_sdpa(query_states, key_states, value_states, tag_mask)
        else:
            num_global_q_heads = self.num_heads // 2
            num_local_q_heads = self.num_heads - num_global_q_heads
            num_global_kv_heads = self.num_key_value_heads // 2
            num_local_kv_heads = self.num_key_value_heads - num_global_kv_heads

            q_g, q_l = torch.split(query_states, [num_global_q_heads, num_local_q_heads], dim=1)
            k_g, k_l = torch.split(key_states, [num_global_kv_heads, num_local_kv_heads], dim=1)
            v_g, v_l = torch.split(value_states, [num_global_kv_heads, num_local_kv_heads], dim=1)

            kv_group_ratio_g = num_global_q_heads // num_global_kv_heads
            kv_group_ratio_l = num_local_q_heads // num_local_kv_heads

            k_g = self._repeat_kv(k_g, kv_group_ratio_g)
            v_g = self._repeat_kv(v_g, kv_group_ratio_g)
            k_l = self._repeat_kv(k_l, kv_group_ratio_l)
            v_l = self._repeat_kv(v_l, kv_group_ratio_l)

            attn_output_g = self._masked_sdpa(q_g, k_g, v_g, full_mask)
            attn_output_l = self._masked_sdpa(q_l, k_l, v_l, tag_mask)

            attn_output = torch.cat([attn_output_g, attn_output_l], dim=1)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _masked_sdpa(self, query, key, value, mask):
        """Helper for standard scaled dot-product attention with a mask."""
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim**0.5)
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value)


class JanusModel(nn.Module):
    """
    The main Janus model, wrapping a pre-configured backbone (e.g., a PEFT model)
    with custom attention and classification heads.
    """
    def __init__(
        self,
        base_model: nn.Module,
        dual_mask: bool = True,
        force_local_attention_only: bool = False,
    ):
        """
        Initializes the Janus model.

        Args:
            base_model: A pre-configured model, which may already be a PEFT model.
            dual_mask: Whether to replace the standard attention with DualMaskAttention.
            force_local_attention_only: If using dual_mask, forces all heads to use local attention.
        """
        super().__init__()
        self.model = base_model
        self.dual_mask = dual_mask
        self.force_local_attention_only = force_local_attention_only

        if self.dual_mask:
            logger.info("Replacing LlamaAttention with DualMaskAttention.")
            self._replace_attention_layers(self.model)

        # The base_model is expected to have a 'config' attribute
        hidden_size = self.model.config.hidden_size
        self.cls_head = nn.Linear(hidden_size, 1)
        self.evidential_head = nn.Linear(hidden_size, 3)

        cfg_pad = getattr(base_model.config, "pad_token_id", None)  # base_hf_model = your underlying AutoModelForCausalLM
        if cfg_pad is None:
            # fall back to eos if pad is unset; you already set pad_token_id elsewhere in training, so this is just a guard
            cfg_pad = getattr(base_model.config, "eos_token_id", 0)
        self.pad_id = int(cfg_pad)

    def _replace_attention_layers(self, module: nn.Module):
        """
        Recursively find LlamaAttention layers and replace them with 
        DualMaskAttention, while preserving the PEFT-modified sub-modules.
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, LlamaAttention):
                # Create the new DualMaskAttention module. It will have vanilla, non-PEFT layers for now.
                new_attn = DualMaskAttention(
                    config=child_module.config,
                    layer_idx=getattr(child_module, 'layer_idx', 0),
                    force_local_only=self.force_local_attention_only
                )

                # Copy the PEFT-modified layers from the old module to the new one.
                # This ensures the LoRA adapters are preserved.
                new_attn.q_proj = child_module.q_proj
                new_attn.k_proj = child_module.k_proj
                new_attn.v_proj = child_module.v_proj
                new_attn.o_proj = child_module.o_proj

                # Also preserve the rotary embeddings
                if hasattr(child_module, 'rotary_emb'):
                    new_attn.rotary_emb = child_module.rotary_emb

                # Now, perform the replacement in the model graph
                setattr(module, name, new_attn)
                logger.info(f"Replaced attention layer '{name}' and preserved its LoRA weights.")
            else:
                # Recurse into sub-modules
                self._replace_attention_layers(child_module)

    def forward(self, ids: Tensor, tags: Tensor) -> dict[str, Tensor]:
        """
        The main forward pass for Janus. It constructs the appropriate attention
        mask(s) and passes them to the underlying model.
        """
        # Find the index of the last non-padding token for each sequence
        # Assumes pad_token_id is accessible, often from model.config.pad_token_id
        
        # Find the last token that is not a pad token
        is_not_pad = (ids != self.pad_id).long()
        last_token_indices = torch.sum(is_not_pad, dim=1) - 1
        # Ensure indices are not negative for sequences that might be all padding
        last_token_indices.clamp_(min=0)

        full_mask = make_full_mask(ids, pad_id=self.pad_id)

        if self.dual_mask:
            tag_mask = make_tag_mask(ids=ids, tags=tags, pad_id=self.pad_id)
            # Combine masks along a new dimension for DualMaskAttention to split
            combined_mask = torch.cat([full_mask, tag_mask], dim=1)
            attention_mask = combined_mask
        else:
            attention_mask = full_mask

        # The underlying PEFT model will handle passing the mask to the attention layers
        model_output = self.model(
            input_ids=ids,
            attention_mask=attention_mask,
            output_hidden_states=True # Ensure hidden states are returned
        )

        hidden = model_output.hidden_states[-1]

        # Use the last token's hidden state for classification heads
        # Gather the hidden states of the last non-padded token for each sequence
        # batch_indices = torch.arange(ids.shape[0], device=ids.device)
        # pooled_hidden = hidden[batch_indices, last_token_indices, :]

        # Better sequence representation: masked mean pooling over non-PAD tokens
        valid = (ids != self.pad_id).float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_hidden = (hidden * valid.unsqueeze(-1)).sum(dim=1) / denom
        
        cls_output = self.cls_head(pooled_hidden.to(self.cls_head.weight.dtype))
        evidential_output = self.evidential_head(pooled_hidden.to(self.evidential_head.weight.dtype))

        # The logits are part of the main output object when calling the PEFT model
        logits = model_output.logits

        return {
            "logits": logits,
            "cls": cls_output,
            "evidential": evidential_output,
        }
