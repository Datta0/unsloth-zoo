from __future__ import annotations

from types import SimpleNamespace

import torch

from unsloth_zoo import rl_replacements as rl


class _ToyModel:
    def __init__(self, weight: torch.Tensor, vocab_size: int):
        self._emb = SimpleNamespace(weight=weight)
        self.config = SimpleNamespace(vocab_size=vocab_size)

    def get_output_embeddings(self):
        return self._emb


class _ToyTrainer:
    def __init__(self, model):
        self.model = model
        self._step = 0


def test_resolve_grpo_lm_head_returns_full_weight_without_dist(monkeypatch):
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    trainer = _ToyTrainer(_ToyModel(weight, vocab_size=4))

    monkeypatch.setattr(rl.dist, "is_available", lambda: False)

    resolved = rl._resolve_grpo_lm_head_for_projection(trainer)

    assert torch.equal(resolved, weight)
    assert rl._resolve_grpo_lm_head_for_projection(trainer) is resolved


def test_resolve_grpo_lm_head_leaves_vocab_shard_without_initialized_dist(monkeypatch):
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    trainer = _ToyTrainer(_ToyModel(weight, vocab_size=4))

    monkeypatch.setattr(rl.dist, "is_available", lambda: True)
    monkeypatch.setattr(rl.dist, "is_initialized", lambda: False)

    resolved = rl._resolve_grpo_lm_head_for_projection(trainer)

    assert torch.equal(resolved, weight)
    assert resolved.shape == (2, 3)


def test_resolve_grpo_lm_head_all_gathers_vocab_shard_and_caches(monkeypatch):
    weight = torch.ones(2, 3, dtype=torch.float32)
    trainer = _ToyTrainer(_ToyModel(weight, vocab_size=4))
    calls = {"all_gather": 0}

    monkeypatch.setattr(rl.dist, "is_available", lambda: True)
    monkeypatch.setattr(rl.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rl.dist, "get_world_size", lambda: 2)

    def _fake_all_gather(gathered, tensor):
        calls["all_gather"] += 1
        gathered[0].copy_(tensor)
        gathered[1].copy_(tensor + 1)

    monkeypatch.setattr(rl.dist, "all_gather", _fake_all_gather)

    resolved = rl._resolve_grpo_lm_head_for_projection(trainer)
    resolved_again = rl._resolve_grpo_lm_head_for_projection(trainer)

    assert torch.equal(resolved, torch.cat([weight, weight + 1], dim=0))
    assert resolved_again is resolved
    assert calls["all_gather"] == 1


def test_resolve_grpo_lm_head_can_be_disabled(monkeypatch):
    weight = torch.ones(2, 3, dtype=torch.float32)
    trainer = _ToyTrainer(_ToyModel(weight, vocab_size=4))

    monkeypatch.setenv("UNSLOTH_GRPO_DISABLE_LM_HEAD_ALLGATHER", "1")

    assert rl._resolve_grpo_lm_head_for_projection(trainer) is weight


def test_rl_replacement_anchors_present():
    assert rl.RL_REPLACEMENTS["grpo_accumulated_loss"] is rl.grpo_accumulated_loss
    assert (
        rl.RL_REPLACEMENTS["grpo_selective_log_softmax"]
        is rl.chunked_hidden_states_selective_log_softmax
    )
