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

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from .common import (
    TEMPORARY_PATCHES,
    torch_compile,
    _torch_compile,
    get_torch_compile_options,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    patch_function_past_key_values,
    dedent,
    KWARGS_TYPE,
    raise_error,
    logger,
    Cache,
    process_return,
)


def patch_qwen3_moe():
    # https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L213
    # Transformers >= 5       uses self.gate_up_proj = nn.Parameter(...)
    # whilst old transformers uses self.experts = nn.ModuleList(...)
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
    except Exception as e:
        return raise_error("transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock", e)
    old_transformers = True
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts # New transformers has this
        old_transformers = False
    except Exception as e:
        old_transformers = True
    import torch

    if old_transformers:
        def old_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        @torch_compile(dynamic = True, fullgraph = True)
        def router_forward(self, hidden_states):
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)
            router_scores = torch.zeros_like(router_logits, dtype = hidden_states.dtype).scatter_(1, selected_experts, routing_weights)
            return router_scores, selected_experts, router_logits

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Fast + memory-efficient + torch.compile compatible MoE."""
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            n_tokens = batch_size * sequence_length
            hidden_states = hidden_states.view(-1, hidden_dim)

            # Get routing decisions
            router_scores, selected_experts, router_logits = router_forward(self, hidden_states)

            final_hidden_states = torch.zeros(
                (n_tokens, hidden_dim), dtype=torch.float32, device=hidden_states.device
            )

            # Static loop over ALL experts (torch.compile compatible)
            # Empty tensor operations are no-ops automatically
            for expert_idx in range(self.num_experts):
                token_idx, _ = torch.where(selected_experts == expert_idx)

                # No if check needed - empty tensors are handled efficiently
                # Compute expert output
                expert_layer = self.experts[expert_idx]
                current_state = hidden_states[token_idx]
                current_hidden_states = expert_layer(current_state) * router_scores[token_idx, expert_idx, None]

                # Accumulate results
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(torch.float32))

            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states.to(hidden_states.dtype), router_logits
    else:
        # ============================================================
        # Optimized Sparse MoE (torch.compile compatible)
        # - Static loop over experts (no data-dependent shapes)
        # - Dense bmm for decode (6x faster for small batches)
        # - Maintains true sparsity for all paths
        # ============================================================

        def _moe_static_sparse(
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
            gate_up_proj: torch.Tensor,
            down_proj: torch.Tensor,
            act_fn,
        ) -> torch.Tensor:
            """
            Static loop sparse MoE - fully torch.compile compatible.
            NO data-dependent control flow. Empty tensors handled as no-ops.
            """
            n_tokens, hidden_dim = hidden_states.shape
            num_experts = gate_up_proj.shape[0]

            final_hidden = torch.zeros_like(hidden_states)

            # Static loop over ALL experts (torch.compile compatible)
            # NO if checks - empty tensor operations are no-ops automatically
            for expert_idx in range(num_experts):
                # Find tokens assigned to this expert
                token_idx, k_pos = torch.where(top_k_index == expert_idx)

                # Gather tokens for this expert (empty tensor if no tokens)
                current_state = hidden_states[token_idx]

                # Expert computation (no-op for empty tensors)
                gate_up = F.linear(current_state, gate_up_proj[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = act_fn(gate) * up
                expert_output = F.linear(intermediate, down_proj[expert_idx])

                # Apply routing weights and scatter back (no-op for empty tensors)
                weighted_output = expert_output * top_k_weights[token_idx, k_pos, None]
                final_hidden.index_add_(0, token_idx, weighted_output.to(final_hidden.dtype))

            return final_hidden

        def _moe_dense(
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
            gate_up_proj: torch.Tensor,
            down_proj: torch.Tensor,
            act_fn,
        ) -> torch.Tensor:
            """Dense bmm - optimal for decode (6x faster for 1 token)."""
            n_tokens, hidden_dim = hidden_states.shape
            num_experts = gate_up_proj.shape[0]
            dtype = hidden_states.dtype
            device = hidden_states.device

            # All tokens through all experts
            hidden_expanded = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
            gate_up = torch.bmm(hidden_expanded, gate_up_proj.transpose(1, 2))
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = act_fn(gate) * up
            all_outputs = torch.bmm(intermediate, down_proj.transpose(1, 2))

            # Apply sparse routing mask
            routing_mask = torch.zeros((num_experts, n_tokens), dtype=dtype, device=device)
            routing_mask.scatter_(0, top_k_index.T, top_k_weights.T)

            return (all_outputs * routing_mask.unsqueeze(-1)).sum(dim=0)

        # Threshold for switching between dense and sparse
        _HYBRID_THRESHOLD = 32

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            """
            Optimized Hybrid Sparse MoE (torch.compile compatible):
            - Dense bmm for decode/small batches (6x faster)
            - Static loop for training/large batches (compile-friendly)
            """
            n_tokens = hidden_states.shape[0]

            # Use dense for small batches (decode/short prefill)
            # This is faster due to reduced kernel launch overhead
            if n_tokens <= _HYBRID_THRESHOLD and not self.training:
                return _moe_dense(
                    hidden_states, top_k_index, top_k_weights,
                    self.gate_up_proj, self.down_proj, self.act_fn
                )

            # Use static sparse loop for training and large batches
            return _moe_static_sparse(
                hidden_states, top_k_index, top_k_weights,
                self.gate_up_proj, self.down_proj, self.act_fn
            )


    # For old transformers, patch Qwen3MoeSparseMoeBlock
    # For new transformers, patch Qwen3MoeExperts (which has the expert loop)
    if old_transformers:
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock, "forward", forward)
    else:
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts, "forward", forward)
pass
TEMPORARY_PATCHES.append(patch_qwen3_moe)
