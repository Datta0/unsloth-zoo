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
)
from .utils import (
    patch_function,
    patch_function,
    raise_error,
    logger,
)


# ============================================================================
# Grouped GEMM kernel integration for MoE training acceleration
# ============================================================================

from .moe_utils import (
    _check_grouped_gemm_available,
    _TORCH_GROUPED_MM_AVAILABLE,
    forward_native_grouped_mm,
    forward_triton_grouped_gemm,
    select_moe_backend,
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

    if old_transformers:
        # ====================================================================
        # Transformers v4.x Compatibility Layer
        # Convert nn.ModuleList[Qwen3MoeMLP] to stacked tensor format for grouped GEMM
        # ====================================================================
        from transformers.activations import ACT2FN

        class Qwen3MoeExperts(nn.Module):
            """
            v5-compatible Experts class for use with v4 transformers.
            Stores expert weights as stacked 3D tensors for efficient grouped GEMM.
            """
            def __init__(self, config):
                super().__init__()
                self.num_experts = config.num_experts
                self.hidden_dim = config.hidden_size
                self.intermediate_dim = config.moe_intermediate_size
                # Stacked weights: [E, 2*I, H] and [E, H, I]
                self.gate_up_proj = nn.Parameter(
                    torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
                )
                self.down_proj = nn.Parameter(
                    torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
                )
                self.act_fn = ACT2FN[config.hidden_act]

            @classmethod
            def from_modulelist(cls, experts_list, config, dtype=None):
                """
                Convert nn.ModuleList[Qwen3MoeMLP] to stacked format.
                Memory-efficient: stacks directly without intermediate copies.
                """
                instance = cls.__new__(cls)
                nn.Module.__init__(instance)

                instance.num_experts = config.num_experts
                instance.hidden_dim = config.hidden_size
                instance.intermediate_dim = config.moe_intermediate_size
                instance.act_fn = ACT2FN[config.hidden_act]

                # Determine dtype from first expert if not specified
                if dtype is None:
                    dtype = experts_list[0].gate_proj.weight.dtype
                device = experts_list[0].gate_proj.weight.device

                # Stack weights efficiently
                # gate_up_proj: [E, 2*I, H] - concat gate and up per expert, then stack
                # down_proj: [E, H, I]
                gate_up_list = []
                down_list = []

                for i, expert in enumerate(experts_list):
                    # gate_proj.weight: [I, H], up_proj.weight: [I, H]
                    gate_up = torch.cat([expert.gate_proj.weight.data, expert.up_proj.weight.data], dim=0)
                    gate_up_list.append(gate_up)
                    # down_proj.weight: [H, I]
                    down_list.append(expert.down_proj.weight.data)

                instance.gate_up_proj = nn.Parameter(torch.stack(gate_up_list, dim=0))
                instance.down_proj = nn.Parameter(torch.stack(down_list, dim=0))

                # Clear the list to free memory
                gate_up_list.clear()
                down_list.clear()

                return instance

        class Qwen3MoeTopKRouter(nn.Module):
            """
            v5-compatible TopK Router for use with v4 transformers.
            """
            def __init__(self, config):
                super().__init__()
                self.top_k = config.num_experts_per_tok
                self.num_experts = config.num_experts
                self.norm_topk_prob = config.norm_topk_prob
                self.hidden_dim = config.hidden_size
                self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

            @classmethod
            def from_linear(cls, gate_linear, config):
                """Convert nn.Linear gate to TopKRouter."""
                instance = cls.__new__(cls)
                nn.Module.__init__(instance)

                instance.top_k = config.num_experts_per_tok
                instance.num_experts = config.num_experts
                instance.norm_topk_prob = config.norm_topk_prob
                instance.hidden_dim = config.hidden_size

                # gate_linear.weight: [num_experts, hidden_dim]
                instance.weight = nn.Parameter(gate_linear.weight.data.clone())

                return instance

            def forward(self, hidden_states):
                hidden_states = hidden_states.reshape(-1, self.hidden_dim)
                router_logits = F.linear(hidden_states, self.weight)
                router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
                if self.norm_topk_prob:
                    router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
                router_top_value = router_top_value.to(hidden_states.dtype)
                return router_logits, router_top_value, router_indices

        # Store original __init__ for reference
        OriginalSparseMoeBlock = transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
        _original_sparse_moe_init = OriginalSparseMoeBlock.__init__

        def _lazy_convert_pre_hook(module, args):
            """
            Pre-forward hook that converts nn.ModuleList to stacked format on first forward.
            This handles the case where accelerate/device_map loading bypasses _load_from_state_dict.
            """
            if not getattr(module, '_unsloth_needs_conversion', False):
                return

            if not hasattr(module, 'experts') or not isinstance(module.experts, nn.ModuleList):
                module._unsloth_needs_conversion = False
                return

            # Check if weights are actually loaded (not on meta device)
            try:
                first_weight = module.experts[0].gate_proj.weight
                if first_weight is None or first_weight.device.type == 'meta':
                    return  # Weights not loaded yet
            except (AttributeError, IndexError):
                return

            config = module._unsloth_config

            # Convert experts
            old_experts = module.experts
            module.experts = Qwen3MoeExperts.from_modulelist(old_experts, config)
            del old_experts

            # Convert gate to TopKRouter
            if hasattr(module, 'gate') and isinstance(module.gate, nn.Linear):
                old_gate = module.gate
                module.gate = Qwen3MoeTopKRouter.from_linear(old_gate, config)
                del old_gate

            module._unsloth_needs_conversion = False

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log conversion
            logger.info("Unsloth: Converted MoE experts to stacked format for grouped GEMM optimization")

        def _new_sparse_moe_init(self, config):
            """
            New __init__ that stores config for potential future conversion.
            Conversion is optional - patched forward handles both ModuleList and stacked format.
            """
            # Call original init - this creates nn.ModuleList
            _original_sparse_moe_init(self, config)

            # Store config for potential manual conversion later
            self._unsloth_config = config
            self._unsloth_needs_conversion = True  # Enabled - eager conversion during loading

            # NOTE: Lazy conversion disabled due to memory concerns (doubles expert memory).
            # The patched forward handles both ModuleList (loop-based) and stacked format (grouped GEMM).
            # Conversion can be manually triggered via `model.layers[i].mlp._convert_to_stacked_experts()` if needed.

        def _convert_to_stacked_experts(self):
            """
            Convert nn.ModuleList to stacked Qwen3MoeExperts after weights are loaded.
            Should be called once after model.load_state_dict() or from_pretrained().
            """
            if not getattr(self, '_unsloth_needs_conversion', False):
                return

            if not hasattr(self, 'experts') or not isinstance(self.experts, nn.ModuleList):
                self._unsloth_needs_conversion = False
                return

            config = self._unsloth_config

            # Convert experts
            old_experts = self.experts
            self.experts = Qwen3MoeExperts.from_modulelist(old_experts, config)
            del old_experts

            # Convert gate to TopKRouter
            if hasattr(self, 'gate') and isinstance(self.gate, nn.Linear):
                old_gate = self.gate
                self.gate = Qwen3MoeTopKRouter.from_linear(old_gate, config)
                del old_gate

            self._unsloth_needs_conversion = False

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Custom _load_from_state_dict that handles v4 checkpoint format
        _original_load_from_state_dict = OriginalSparseMoeBlock._load_from_state_dict if hasattr(OriginalSparseMoeBlock, '_load_from_state_dict') else nn.Module._load_from_state_dict

        def _load_from_state_dict_v4_compat(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            """
            Custom state dict loading that detects v4 or v5 checkpoint format
            and handles accordingly.
            """
            # Check if checkpoint has v4 format (experts.0.gate_proj.weight)
            v4_keys = [k for k in state_dict.keys() if k.startswith(f"{prefix}experts.") and ".gate_proj." in k]

            if v4_keys and getattr(self, '_unsloth_needs_conversion', False):
                # v4 checkpoint format detected
                # We need to convert the state_dict keys to v5 format before loading
                config = self._unsloth_config
                num_experts = config.num_experts

                gate_up_list = []
                down_list = []

                for i in range(num_experts):
                    gate_key = f"{prefix}experts.{i}.gate_proj.weight"
                    up_key = f"{prefix}experts.{i}.up_proj.weight"
                    down_key = f"{prefix}experts.{i}.down_proj.weight"

                    if gate_key in state_dict and up_key in state_dict:
                        gate = state_dict.pop(gate_key)
                        up = state_dict.pop(up_key)
                        down = state_dict.pop(down_key)

                        gate_up_list.append(torch.cat([gate, up], dim=0))
                        down_list.append(down)

                if gate_up_list:
                    # Add converted tensors to state_dict
                    # Ensure they are the same dtype as the loaded weights
                    dtype = gate_up_list[0].dtype
                    state_dict[f"{prefix}experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0).to(dtype)
                    state_dict[f"{prefix}experts.down_proj"] = torch.stack(down_list, dim=0).to(dtype)

                    # Clear lists
                    gate_up_list.clear()
                    down_list.clear()

                # Convert gate.weight to gate.weight (already correct key for TopKRouter)
                # No change needed - both use gate.weight

                # Now we need to replace self.experts with Qwen3MoeExperts BEFORE loading
                # Create empty Qwen3MoeExperts
                if isinstance(self.experts, nn.ModuleList):
                    old_experts = self.experts
                    self.experts = Qwen3MoeExperts(config)
                    # Move to same device/dtype if possible
                    if hasattr(old_experts[0].gate_proj, 'weight') and old_experts[0].gate_proj.weight is not None:
                        device = old_experts[0].gate_proj.weight.device
                        dtype = old_experts[0].gate_proj.weight.dtype
                        self.experts.to(device=device, dtype=dtype)
                    del old_experts

                # Convert gate
                if isinstance(self.gate, nn.Linear):
                    old_gate = self.gate
                    self.gate = Qwen3MoeTopKRouter(config)
                    if old_gate.weight is not None:
                        self.gate.to(device=old_gate.weight.device, dtype=old_gate.weight.dtype)
                    del old_gate

                self._unsloth_needs_conversion = False

            # Call parent implementation
            return _original_load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        # Apply patches to Qwen3MoeSparseMoeBlock
        OriginalSparseMoeBlock.__init__ = _new_sparse_moe_init
        OriginalSparseMoeBlock._load_from_state_dict = _load_from_state_dict_v4_compat
        OriginalSparseMoeBlock._convert_to_stacked_experts = _convert_to_stacked_experts

        # Register the new classes in the module for proper pickling/unpickling
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts = Qwen3MoeExperts
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeTopKRouter = Qwen3MoeTopKRouter

        # Select backend for forward implementations
        backend = select_moe_backend()

        if backend == "grouped_mm":
            def experts_forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                """Native Pytorch grouped GEMM MoE forward pass."""
                return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)

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

        else:
            # Fallback: Pure PyTorch loop-based implementation
            @torch.compiler.disable
            def experts_forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                """Loop-based MoE forward pass for stacked expert format."""
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

        # SparseMoeBlock forward for v4 with converted experts
        @torch.compiler.disable
        def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

        # Patch the forward methods
        Qwen3MoeExperts.forward = experts_forward
        patch_function(OriginalSparseMoeBlock, "forward", sparse_moe_block_forward)

    else:
    # ====================================================================
        # New transformers (5.0+) with stacked expert weights
        # Uses Triton grouped GEMM kernels for high performance
        # ====================================================================

        backend = select_moe_backend()

        if backend == "grouped_mm":

            def forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                """
                Native Pytorch grouped GEMM MoE forward pass.
                Uses torch._grouped_mm which is significantly faster than loop and works without Triton dependencies.
                """
                return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)

        elif backend == "unsloth_triton":
            # Import grouped GEMM interface (sys.path was set by _check_grouped_gemm_available)
            from grouped_gemm.interface import grouped_gemm, supports_tma
            # Import autotune cache
            # from unsloth.kernels.moe.autotune_cache import get_or_autotune_moe_kernels

            def forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                """
                Grouped GEMM MoE forward pass using Triton kernels.

                Uses fused permutation (permute_x for first GEMM, permute_y for second GEMM)
                to minimize memory traffic and achieve high GPU utilization.

                Uses cached kernel configs (created once at start) for efficient operation.
                """
                return forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)

        else:
            # Fallback: Pure PyTorch loop-based implementation


            @torch.compiler.disable
            def forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                """
                Loop-based MoE forward pass. Loops over experts that have tokens routed to them.
                Uses @torch.compiler.disable because the loop is data-dependent.
                """


                final_hidden_states = torch.zeros_like(hidden_states)

                # Create expert mask and find which experts have tokens
                with torch.no_grad():
                    expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, n_tokens)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                # Only loop over experts that actually have tokens routed to them
                for expert_idx in expert_hit:
                    expert_idx = expert_idx[0]
                    if expert_idx == self.num_experts:
                        continue

                    # Find which tokens are routed to this expert
                    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

                    # Gather only the tokens for this expert
                    current_state = hidden_states[token_idx]

                    # Compute gate_up projection for this expert only
                    gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                    current_hidden_states = self.act_fn(gate) * up

                    # Compute down projection for this expert only
                    current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])

                    # Apply routing weights
                    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

                    # Scatter back to final output
                    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

                return final_hidden_states

        # SparseMoeBlock forward is disabled from compilation due to dynamic routing
        @torch.compiler.disable
        def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
            # _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
            gate_output = self.gate(hidden_states_reshaped)
            router_logits = None
            if isinstance(gate_output, tuple):
                if len(gate_output) == 3:
                     router_logits, routing_weights, selected_experts = gate_output
                elif len(gate_output) == 2:
                     routing_weights, selected_experts = gate_output
                else:
                     print(f"Unsloth: gate_output len {len(gate_output)}")
                     routing_weights, selected_experts = gate_output[1], gate_output[2]
                     if hasattr(gate_output[0], "shape"): router_logits = gate_output[0]
            else:
                 # It is just logits (Tensor). We need to compute routing manually.
                 # Matches standard Qwen2MoE logic.
                 router_logits = gate_output
                 routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

                 # Fast inference path might lose these attributes
                 top_k = getattr(self, "top_k", getattr(self, "num_experts_per_tok", 8))
                 norm_topk_prob = getattr(self, "norm_topk_prob", True)

                 routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                 if norm_topk_prob:
                     routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

                 routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states

        # Patch for new transformers (v5+)
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts, "forward", forward)
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock, "forward", sparse_moe_block_forward)


    # ====================================================================
    # Patch Qwen3MoeForCausalLM to return hidden states if requested
    # ====================================================================
    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

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
                return_dict=return_dict, # qwen3-moe passes router logits etc
                **kwargs
            )

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            hidden_states = outputs[0]

            # Unsloth Fast Inference Path
            if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
                if num_logits_to_keep != 0:
                    hidden_states = hidden_states[:, -num_logits_to_keep:, :]

                return CausalLMOutputWithPast(
                    loss = None,
                    logits = hidden_states, # Return hidden states in logits field!
                    past_key_values = outputs.past_key_values,
                    hidden_states = outputs.hidden_states,
                    attentions = outputs.attentions,
                )

            # Normal Path
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
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

        # Patch from_pretrained to force conversion
        OriginalFromPretrained = Qwen3MoeForCausalLM.from_pretrained

        @classmethod
        def from_pretrained_wrapper(cls, *args, **kwargs):
            model = OriginalFromPretrained(*args, **kwargs)
            # Perform conversion here
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                from unsloth.models.llama import logger
                logger.warning("Unsloth: Converting Qwen3 MoE experts to stacked format...")

                for i, layer in enumerate(model.model.layers):
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "_convert_to_stacked_experts"):
                        layer.mlp._convert_to_stacked_experts()

                # Final cleanup
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return model

        Qwen3MoeForCausalLM.from_pretrained = from_pretrained_wrapper

        patch_function(Qwen3MoeForCausalLM, "forward", forward_causal_lm)
        # Brute force patch in case verify_function fails or behaves safely
        Qwen3MoeForCausalLM.forward = forward_causal_lm


    except Exception as e:
        pass # If imports fail, just ignore

    transformers.models.qwen3_moe.modeling_qwen3_moe.__UNSLOTH_PATCHED__ = True
pass

# TEMPORARY_PATCHES.append(patch_qwen3_moe)
patch_qwen3_moe()
