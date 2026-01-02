# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Qwen3 MoE Grouped GEMM Optimization Patches

This module patches Qwen3 MoE models to use efficient grouped GEMM kernels
for the Mixture of Experts forward pass. Supports both:
- Transformers v5+ (native stacked expert weights)
- Transformers v4.x (nn.ModuleList, via compatibility layer)

The patch automatically selects the best available backend:
1. torch._grouped_mm (PyTorch native, fastest)
2. Unsloth Triton kernels (custom optimized)
3. Pure PyTorch loop (fallback)
"""

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import torch_compile
from .utils import patch_function, raise_error, logger

# MoE backend utilities
from .moe_utils import (
    _check_grouped_gemm_available,
    _TORCH_GROUPED_MM_AVAILABLE,
    forward_native_grouped_mm,
    forward_triton_grouped_gemm,
    select_moe_backend,
)


# ============================================================================
# Helper: Check if transformers uses old (v4) or new (v5) MoE format
# ============================================================================

def _is_old_transformers():
    """
    Check if transformers uses the old v4 MoE format (nn.ModuleList).

    Returns True for transformers < 5.0, False for >= 5.0.
    """
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        # New transformers has Qwen3MoeExperts class
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts
        return False
    except AttributeError:
        return True


# ============================================================================
# Expert Forward Implementations (shared by both v4 and v5)
# ============================================================================

def _create_experts_forward(backend):
    """
    Create the experts forward function based on selected backend.

    Args:
        backend: One of "grouped_mm", "unsloth_triton", or "native_torch"

    Returns:
        Forward function for Qwen3MoeExperts
    """
    if backend == "grouped_mm":
        def experts_forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            """Native Pytorch grouped GEMM MoE forward pass."""
            return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)
        return experts_forward

    elif backend == "unsloth_triton":
        from grouped_gemm.interface import grouped_gemm, supports_tma

        def experts_forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            """Grouped GEMM MoE forward pass using Triton kernels."""
            return forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)
        return experts_forward

    else:
        # Fallback: Pure PyTorch loop-based implementation
        @torch.compiler.disable
        def experts_forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            """
            Loop-based MoE forward pass for stacked expert format.
            Uses @torch.compiler.disable because the loop is data-dependent.
            """
            final_hidden_states = torch.zeros_like(hidden_states)

            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue

                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]

                # Use stacked weights
                gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

            return final_hidden_states
        return experts_forward


# ============================================================================
# SparseMoeBlock Forward (shared structure, handles both gate types)
# ============================================================================

@torch.compiler.disable
def _sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    SparseMoeBlock forward that handles both gate output formats.

    Handles:
    - TopKRouter: returns (router_logits, routing_weights, selected_experts)
    - nn.Linear: returns just logits, routing computed manually
    """
    if getattr(self, '_unsloth_needs_conversion', False):
        self._convert_to_stacked_experts()

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    # Handle both old (nn.Linear) and new (TopKRouter) gate formats
    gate_output = self.gate(hidden_states_reshaped)

    if isinstance(gate_output, tuple):
        # TopKRouter returns (router_logits, routing_weights, selected_experts)
        if len(gate_output) == 3:
            router_logits, routing_weights, selected_experts = gate_output
        elif len(gate_output) == 2:
            routing_weights, selected_experts = gate_output
            router_logits = None
        else:
            routing_weights, selected_experts = gate_output[1], gate_output[2]
            router_logits = gate_output[0] if hasattr(gate_output[0], "shape") else None
    else:
        # nn.Linear gate - compute routing manually
        router_logits = gate_output
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k = getattr(self, 'top_k', getattr(self, 'num_experts_per_tok', 8))
        norm_topk_prob = getattr(self, 'norm_topk_prob', True)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

    # Check if experts is stacked format (Qwen3MoeExperts) or ModuleList
    if isinstance(self.experts, nn.ModuleList):
        # Legacy path - use original loop
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states_reshaped[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    else:
        # Stacked format - use optimized grouped GEMM
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


# ============================================================================
# Qwen3MoeForCausalLM Forward Patch (hidden states optimization)
# ============================================================================

def _create_causal_lm_forward():
    """Create forward function for Qwen3MoeForCausalLM with hidden states optimization."""
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def forward_causal_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        # If we need hidden states (Unsloth optimization), return them directly
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            output_hidden_states = True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = outputs[0]

        # Unsloth Fast Inference Path
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            if num_logits_to_keep != 0:
                hidden_states = hidden_states[:, -num_logits_to_keep:, :]

            return CausalLMOutputWithPast(
                loss=None,
                logits=hidden_states,  # Return hidden states in logits field!
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # Normal Path
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return forward_causal_lm


def _create_from_pretrained_wrapper(OriginalFromPretrained):
    """Create from_pretrained wrapper that forces expert conversion."""
    @classmethod
    def from_pretrained_wrapper(cls, *args, **kwargs):
        model = OriginalFromPretrained(*args, **kwargs)

        # Perform conversion here
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            from unsloth.models.llama import logger
            logger.warning("Unsloth: Converting Qwen3 MoE experts to stacked format...")

            for layer in model.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "_convert_to_stacked_experts"):
                    layer.mlp._convert_to_stacked_experts()

            # Final cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model

    return from_pretrained_wrapper


# ============================================================================
# Main Patch Function
# ============================================================================

def patch_qwen3_moe():
    """
    Apply grouped GEMM optimization patches to Qwen3 MoE models.

    Supports both transformers v4.x and v5+. Automatically selects the best
    available backend (grouped_mm > triton > native torch).
    """
    # Verify Qwen3 MoE is available
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_module
        qwen3_module.Qwen3MoeSparseMoeBlock
    except Exception as e:
        return raise_error("transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock", e)

    old_transformers = _is_old_transformers()
    backend = select_moe_backend()

    # Get the experts forward implementation
    experts_forward = _create_experts_forward(backend)

    if old_transformers:
        # ================================================================
        # Transformers v4.x: Apply compatibility layer
        # ================================================================
        from .transformers_v4_compat import (
            Qwen3MoeExperts,
            Qwen3MoeTopKRouter,
            apply_v4_compatibility_patches,
        )

        # Apply v4 compatibility patches (init, state dict loading, etc.)
        apply_v4_compatibility_patches(qwen3_module)

        # Patch expert forward
        Qwen3MoeExperts.forward = experts_forward

        # Patch SparseMoeBlock forward
        patch_function(
            qwen3_module.Qwen3MoeSparseMoeBlock,
            "forward",
            _sparse_moe_block_forward
        )
    else:
        # ================================================================
        # Transformers v5+: Native stacked expert format
        # ================================================================
        patch_function(
            qwen3_module.Qwen3MoeExperts,
            "forward",
            experts_forward
        )
        patch_function(
            qwen3_module.Qwen3MoeSparseMoeBlock,
            "forward",
            _sparse_moe_block_forward
        )

    # ================================================================
    # Patch Qwen3MoeForCausalLM for hidden states optimization
    # ================================================================
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

        forward_causal_lm = _create_causal_lm_forward()

        # Patch from_pretrained to force conversion
        Qwen3MoeForCausalLM.from_pretrained = _create_from_pretrained_wrapper(
            Qwen3MoeForCausalLM.from_pretrained
        )

        patch_function(Qwen3MoeForCausalLM, "forward", forward_causal_lm)
        # Brute force patch in case verify_function fails or behaves safely
        Qwen3MoeForCausalLM.forward = forward_causal_lm

    except Exception:
        pass  # If imports fail, just ignore

    qwen3_module.__UNSLOTH_PATCHED__ = True


# Apply patch on module import
patch_qwen3_moe()
