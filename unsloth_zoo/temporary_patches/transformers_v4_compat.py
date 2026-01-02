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
Transformers v4.x Compatibility Layer for Qwen3 MoE

This module provides compatibility classes and functions to enable grouped GEMM
optimization for Qwen3 MoE models on older transformers versions (< 5.0).

In transformers v5+, MoE expert weights are stored as stacked 3D tensors
(Qwen3MoeExperts class). In v4.x, they are stored as nn.ModuleList of individual
Qwen3MoeMLP modules. This layer converts the v4 format to v5 format at load time,
enabling the use of efficient grouped GEMM kernels.

Classes:
    Qwen3MoeExperts: v5-compatible expert weights container for v4 transformers
    Qwen3MoeTopKRouter: v5-compatible TopK router for v4 transformers

Functions:
    apply_v4_compatibility_patches: Apply all v4 compatibility patches to transformers
"""

__all__ = [
    "Qwen3MoeExperts",
    "Qwen3MoeTopKRouter",
    "apply_v4_compatibility_patches",
]

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import logger


class Qwen3MoeExperts(nn.Module):
    """
    v5-compatible Experts class for use with v4 transformers.

    Stores expert weights as stacked 3D tensors for efficient grouped GEMM:
    - gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
    - down_proj: [num_experts, hidden_dim, intermediate_dim]

    This format enables batch matrix multiplication across all experts
    instead of looping through individual expert modules.
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

        # Get activation function from config
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    @classmethod
    def from_modulelist(cls, experts_list, config, dtype=None):
        """
        Convert nn.ModuleList[Qwen3MoeMLP] to stacked format.

        Memory-efficient: stacks directly without intermediate copies.

        Args:
            experts_list: nn.ModuleList of Qwen3MoeMLP modules
            config: Model config with num_experts, hidden_size, moe_intermediate_size
            dtype: Optional dtype override (defaults to first expert's dtype)

        Returns:
            Qwen3MoeExperts instance with stacked weights
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.num_experts = config.num_experts
        instance.hidden_dim = config.hidden_size
        instance.intermediate_dim = config.moe_intermediate_size

        from transformers.activations import ACT2FN
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

        for expert in experts_list:
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

    In v4 transformers, the gate is a simple nn.Linear. This class wraps
    the linear layer and adds TopK routing logic to match v5 behavior.

    Returns (router_logits, routing_weights, selected_experts) on forward.
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
        """
        Convert nn.Linear gate to TopKRouter.

        Args:
            gate_linear: nn.Linear with shape [num_experts, hidden_dim]
            config: Model config with num_experts_per_tok, num_experts, etc.

        Returns:
            Qwen3MoeTopKRouter instance
        """
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
        """
        Compute TopK routing.

        Args:
            hidden_states: [batch_size * seq_len, hidden_dim]

        Returns:
            Tuple of (router_logits, routing_weights, selected_experts)
        """
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)

        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)

        router_top_value = router_top_value.to(hidden_states.dtype)
        return router_logits, router_top_value, router_indices


def _create_lazy_convert_hook(Qwen3MoeExperts, Qwen3MoeTopKRouter, logger):
    """
    Create a pre-forward hook that converts nn.ModuleList to stacked format on first forward.

    This handles the case where accelerate/device_map loading bypasses _load_from_state_dict.
    """
    def _lazy_convert_pre_hook(module, args):
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Unsloth: Converted MoE experts to stacked format for grouped GEMM optimization")

    return _lazy_convert_pre_hook


def _create_new_init(original_init):
    """
    Create a new __init__ that stores config for potential future conversion.

    Conversion is optional - patched forward handles both ModuleList and stacked format.
    """
    def _new_sparse_moe_init(self, config):
        # Call original init - this creates nn.ModuleList
        original_init(self, config)

        # Store config for potential manual conversion later
        self._unsloth_config = config
        self._unsloth_needs_conversion = True  # Enabled - eager conversion during loading

        # NOTE: Lazy conversion disabled due to memory concerns (doubles expert memory).
        # The patched forward handles both ModuleList (loop-based) and stacked format (grouped GEMM).
        # Conversion can be manually triggered via `model.layers[i].mlp._convert_to_stacked_experts()` if needed.

    return _new_sparse_moe_init


def _create_convert_method(Qwen3MoeExperts, Qwen3MoeTopKRouter):
    """
    Create a method to convert nn.ModuleList to stacked Qwen3MoeExperts.

    Should be called once after model.load_state_dict() or from_pretrained().
    """
    def _convert_to_stacked_experts(self):
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _convert_to_stacked_experts


def _create_load_state_dict_hook(Qwen3MoeExperts, Qwen3MoeTopKRouter, original_load_state_dict):
    """
    Create a custom _load_from_state_dict that handles v4 checkpoint format.

    Detects v4 format (experts.0.gate_proj.weight) and converts to v5 format
    (experts.gate_up_proj) before loading.
    """
    def _load_from_state_dict_v4_compat(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
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
                dtype = gate_up_list[0].dtype
                state_dict[f"{prefix}experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0).to(dtype)
                state_dict[f"{prefix}experts.down_proj"] = torch.stack(down_list, dim=0).to(dtype)

                # Clear lists
                gate_up_list.clear()
                down_list.clear()

            # Now we need to replace self.experts with Qwen3MoeExperts BEFORE loading
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
        return original_load_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    return _load_from_state_dict_v4_compat


def apply_v4_compatibility_patches(transformers_module):
    """
    Apply v4 compatibility patches to the transformers Qwen3 MoE module.

    This function patches Qwen3MoeSparseMoeBlock to:
    1. Store config during __init__ for later conversion
    2. Convert v4 checkpoint format during state dict loading
    3. Register Qwen3MoeExperts and Qwen3MoeTopKRouter classes

    Args:
        transformers_module: The transformers.models.qwen3_moe.modeling_qwen3_moe module

    Returns:
        Tuple of (Qwen3MoeExperts, Qwen3MoeTopKRouter) classes for use in forward patches
    """
    OriginalSparseMoeBlock = transformers_module.Qwen3MoeSparseMoeBlock
    _original_sparse_moe_init = OriginalSparseMoeBlock.__init__

    _original_load_from_state_dict = (
        OriginalSparseMoeBlock._load_from_state_dict
        if hasattr(OriginalSparseMoeBlock, '_load_from_state_dict')
        else nn.Module._load_from_state_dict
    )

    # Apply patches
    OriginalSparseMoeBlock.__init__ = _create_new_init(_original_sparse_moe_init)
    OriginalSparseMoeBlock._load_from_state_dict = _create_load_state_dict_hook(
        Qwen3MoeExperts, Qwen3MoeTopKRouter, _original_load_from_state_dict
    )
    OriginalSparseMoeBlock._convert_to_stacked_experts = _create_convert_method(
        Qwen3MoeExperts, Qwen3MoeTopKRouter
    )

    # Register the new classes in the module for proper pickling/unpickling
    transformers_module.Qwen3MoeExperts = Qwen3MoeExperts
    transformers_module.Qwen3MoeTopKRouter = Qwen3MoeTopKRouter

    return Qwen3MoeExperts, Qwen3MoeTopKRouter
