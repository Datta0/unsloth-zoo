# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import atexit
import functools
import os
import sys
import time
from collections import defaultdict
from contextvars import ContextVar
from typing import Optional

import torch

_MODULE_NAME = "unsloth_zoo.gradient_checkpointing"
_PROFILE_MODULE: ContextVar[str] = ContextVar("_unsloth_gc_profile_module", default="<unknown>")
_PATCHED = False
_ORIGINALS = {}
_TOKEN_MODULES = {}
_DUMPED = False
_HIT_DUMPED = False


_STATE = {}
_HIT_COUNTS = defaultdict(int)


def _env_enabled(name):
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _profile_enabled():
    return _env_enabled("UNSLOTH_GC_PROFILE")


def _hit_count_enabled():
    return _env_enabled("UNSLOTH_GC_CHECKPOINT_HIT_COUNT")


def _stat(path):
    node = _STATE
    for part in path[:-1]:
        node = node.setdefault(part, {})
    return node.setdefault(path[-1], defaultdict(float))


def _number(value, default=0.0):
    return value if isinstance(value, (int, float)) else default


def _shape_key(tensor):
    try:
        dims = "x".join(str(int(x)) for x in tensor.shape)
        return f"{dims}:{tensor.dtype}"
    except Exception:
        return "<unknown>"


def _num_bytes(tensor):
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _module_name_from_function(function) -> Optional[str]:
    module = getattr(function, "__self__", None)
    if isinstance(module, torch.nn.Module):
        return getattr(module, "_unsloth_gc_profile_module_name", None) or module.__class__.__name__

    closure = getattr(function, "__closure__", None)
    if closure:
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, torch.nn.Module):
                return getattr(value, "_unsloth_gc_profile_module_name", None) or value.__class__.__name__
            maybe_module = getattr(value, "__self__", None)
            if isinstance(maybe_module, torch.nn.Module):
                return getattr(maybe_module, "_unsloth_gc_profile_module_name", None) or maybe_module.__class__.__name__
    return None


def _add(kind, *, module_name=None, tensor=None, seconds=0.0, count=1, bytes_=None):
    module_name = module_name or _PROFILE_MODULE.get()
    bytes_ = _num_bytes(tensor) if bytes_ is None and torch.is_tensor(tensor) else (bytes_ or 0)
    shape = _shape_key(tensor) if torch.is_tensor(tensor) else "<none>"
    for path in (("totals",), ("modules", module_name), ("shapes", shape)):
        stats = _stat(path)
        stats[f"{kind}_count"] += count
        stats[f"{kind}_seconds"] += float(seconds)
        stats[f"{kind}_bytes"] += int(bytes_)


def _top_lines(parent, key, limit=10):
    rows = []
    for name, values in parent.items():
        if not isinstance(values, dict):
            continue
        value = _number(values.get(key), 0.0)
        if value <= 0:
            continue
        rows.append((name, values, value))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows[:limit]


def dump_unsloth_gc_profile():
    global _DUMPED
    if _DUMPED:
        return
    _DUMPED = True
    totals = _STATE.get("totals", {})
    if not totals:
        return
    print("Unsloth GC profile summary:")
    print(
        "  totals: "
        f"checkpoint_count={_number(totals.get('checkpoint_count'), 0):.0f}, "
        f"checkpoint_seconds={_number(totals.get('checkpoint_seconds'), 0.0):.6f}, "
        f"stream_pack_count={_number(totals.get('stream_pack_count'), 0):.0f}, "
        f"stream_pack_MiB={_number(totals.get('stream_pack_bytes'), 0) / 1024 / 1024:.1f}, "
        f"stream_unpack_count={_number(totals.get('stream_unpack_count'), 0):.0f}, "
        f"stream_unpack_seconds={_number(totals.get('stream_unpack_seconds'), 0.0):.6f}, "
        f"d2h_seconds={_number(totals.get('d2h_copy_seconds'), 0.0):.6f}, "
        f"h2d_seconds={_number(totals.get('h2d_restore_seconds'), 0.0):.6f}, "
        f"pinned_alloc_MiB={_number(totals.get('pinned_alloc_bytes'), 0) / 1024 / 1024:.1f}"
    )
    modules = _STATE.get("modules", {})
    for title, key in (
        ("top modules by checkpoint_seconds", "checkpoint_seconds"),
        ("top modules by stream_pack_bytes", "stream_pack_bytes"),
        ("top modules by stream_unpack_seconds", "stream_unpack_seconds"),
    ):
        rows = _top_lines(modules, key)
        if not rows:
            continue
        print(f"  {title}:")
        for name, values, value in rows:
            print(
                f"  - {name}: "
                f"{key}={value:.6f}" if "seconds" in key else f"  - {name}: {key}={value:.0f}"
            )
    shapes = _STATE.get("shapes", {})
    rows = _top_lines(shapes, "stream_pack_bytes")
    if rows:
        print("  top tensor shapes by stream_pack_bytes:")
        for name, values, value in rows:
            print(
                f"  - {name}: stream_pack_MiB={value / 1024 / 1024:.1f}, "
                f"stream_pack_count={_number(values.get('stream_pack_count'), 0):.0f}, "
                f"stream_unpack_count={_number(values.get('stream_unpack_count'), 0):.0f}"
            )


def dump_unsloth_gc_hit_count():
    global _HIT_DUMPED
    if _HIT_DUMPED:
        return
    _HIT_DUMPED = True
    if not _hit_count_enabled():
        return
    print(
        f"[gc_hit_count] unsloth_checkpoint entries={_HIT_COUNTS['checkpoint_entries']} "
        f"unsloth_stream_boundaries={_HIT_COUNTS['stream_boundaries']} "
        f"unsloth_stream_backward_boundaries={_HIT_COUNTS['stream_backward_boundaries']} "
        f"unsloth_stream_pushes={_HIT_COUNTS['stream_pushes']} "
        f"unsloth_stream_stashed_tensors={_HIT_COUNTS['stream_stashed_tensors']} "
        f"unsloth_stream_staged_windows={_HIT_COUNTS['stream_staged_windows']} "
        f"unsloth_stream_restored_windows={_HIT_COUNTS['stream_restored_windows']} "
        f"unsloth_stream_staged_tensors={_HIT_COUNTS['stream_staged_tensors']} "
        f"unsloth_stream_restored_tensors={_HIT_COUNTS['stream_restored_tensors']} "
        f"unsloth_stream_staged_MiB={_HIT_COUNTS['stream_staged_bytes']/1024/1024:.1f}",
        flush=True,
    )


def _wrap_bound_checkpoint(bound, module_name):
    if getattr(bound, "_unsloth_gc_profile_wrapped", False):
        return bound

    @functools.wraps(bound)
    def profiled_checkpoint_call(function, *args, **kwargs):
        token = _PROFILE_MODULE.set(module_name)
        try:
            return bound(function, *args, **kwargs)
        finally:
            _PROFILE_MODULE.reset(token)

    profiled_checkpoint_call._unsloth_gc_profile_wrapped = True
    return profiled_checkpoint_call


def _patch_binding(gc_mod):
    original = gc_mod._bind_gradient_checkpointing_func
    if getattr(original, "_unsloth_gc_profile_wrapped", False):
        return
    _ORIGINALS.setdefault("_bind_gradient_checkpointing_func", original)

    @functools.wraps(original)
    def profiled_bind(model, checkpoint_fn, use_reentrant, context_fn=None, offload_backend=None):
        result = original(model, checkpoint_fn, use_reentrant, context_fn, offload_backend)
        names = {
            id(module): (name or module.__class__.__name__)
            for name, module in model.named_modules()
        }
        for module in model.modules():
            if not hasattr(module, "_gradient_checkpointing_func"):
                continue
            module_name = names.get(id(module), module.__class__.__name__)
            module._unsloth_gc_profile_module_name = module_name
            module._gradient_checkpointing_func = _wrap_bound_checkpoint(
                module._gradient_checkpointing_func,
                module_name,
            )
        return result

    profiled_bind._unsloth_gc_profile_wrapped = True
    gc_mod._bind_gradient_checkpointing_func = profiled_bind
    training_utils = sys.modules.get("unsloth_zoo.training_utils")
    if training_utils is not None:
        training_utils._bind_gradient_checkpointing_func = profiled_bind


def _patch_checkpoint_dispatch(gc_mod):
    original = gc_mod.unsloth_checkpoint
    if getattr(original, "_unsloth_gc_profile_wrapped", False):
        return
    _ORIGINALS.setdefault("unsloth_checkpoint", original)

    @functools.wraps(original)
    def profiled_unsloth_checkpoint(function, *args, **kwargs):
        if _hit_count_enabled():
            _HIT_COUNTS["checkpoint_entries"] += 1
        if not _profile_enabled():
            return original(function, *args, **kwargs)
        token = None
        module_name = _module_name_from_function(function)
        if module_name is not None:
            token = _PROFILE_MODULE.set(module_name)
        start = time.perf_counter()
        try:
            return original(function, *args, **kwargs)
        finally:
            duration = time.perf_counter() - start
            _add("checkpoint", seconds=duration)
            if token is not None:
                _PROFILE_MODULE.reset(token)

    profiled_unsloth_checkpoint._unsloth_gc_profile_wrapped = True
    gc_mod.unsloth_checkpoint = profiled_unsloth_checkpoint
    try:
        if getattr(torch.utils.checkpoint.checkpoint, "__name__", "") == "unsloth_checkpoint":
            torch.utils.checkpoint.checkpoint = profiled_unsloth_checkpoint
    except Exception:
        pass


def _patch_pinned_alloc(gc_mod):
    original = gc_mod._track_pinned_alloc
    if getattr(original, "_unsloth_gc_profile_wrapped", False):
        return
    _ORIGINALS.setdefault("_track_pinned_alloc", original)

    @functools.wraps(original)
    def profiled_track_pinned_alloc(numel, dtype):
        if not _profile_enabled():
            return original(numel, dtype)
        start = time.perf_counter()
        result = original(numel, dtype)
        duration = time.perf_counter() - start
        try:
            item_size = torch.tensor([], dtype=dtype).element_size()
        except Exception:
            item_size = 0
        _add("pinned_alloc", seconds=duration, bytes_=int(numel) * int(item_size))
        return result

    profiled_track_pinned_alloc._unsloth_gc_profile_wrapped = True
    gc_mod._track_pinned_alloc = profiled_track_pinned_alloc


def _patch_stream_scheduler(gc_mod):
    scheduler_cls = gc_mod.UnslothStreamActivationScheduler
    ticket_cls = gc_mod._UnslothActivationTicket

    original_stash = scheduler_cls.stash_tensor
    if not getattr(original_stash, "_unsloth_gc_profile_wrapped", False):
        _ORIGINALS.setdefault("UnslothStreamActivationScheduler.stash_tensor", original_stash)

        @functools.wraps(original_stash)
        def profiled_stash(self, tensor):
            if _hit_count_enabled():
                _HIT_COUNTS["stream_pushes"] += 1
            profile = _profile_enabled()
            start = time.perf_counter() if profile else 0.0
            result = original_stash(self, tensor)
            duration = time.perf_counter() - start if profile else 0.0
            if isinstance(result, ticket_cls):
                if _hit_count_enabled():
                    _HIT_COUNTS["stream_stashed_tensors"] += 1
                    _HIT_COUNTS["stream_staged_bytes"] += _num_bytes(tensor)
                if profile:
                    module_name = _PROFILE_MODULE.get()
                    _TOKEN_MODULES[id(result)] = module_name
                    _add("stream_pack", module_name=module_name, tensor=tensor, seconds=duration)
            elif profile:
                _add("stream_skip", tensor=tensor, seconds=duration)
            return result

        profiled_stash._unsloth_gc_profile_wrapped = True
        scheduler_cls.stash_tensor = profiled_stash

    original_restore = scheduler_cls.restore_tensor
    if not getattr(original_restore, "_unsloth_gc_profile_wrapped", False):
        _ORIGINALS.setdefault("UnslothStreamActivationScheduler.restore_tensor", original_restore)

        @functools.wraps(original_restore)
        def profiled_restore(self, token):
            profile = _profile_enabled()
            start = time.perf_counter() if profile else 0.0
            result = original_restore(self, token)
            duration = time.perf_counter() - start if profile else 0.0
            if profile and isinstance(token, ticket_cls):
                module_name = _TOKEN_MODULES.pop(id(token), None)
                _add(
                    "stream_unpack",
                    module_name=module_name,
                    tensor=result if torch.is_tensor(result) else None,
                    seconds=duration,
                )
            return result

        profiled_restore._unsloth_gc_profile_wrapped = True
        scheduler_cls.restore_tensor = profiled_restore

    original_d2h = scheduler_cls._copy_to_host
    if not getattr(original_d2h, "_unsloth_gc_profile_wrapped", False):
        _ORIGINALS.setdefault("UnslothStreamActivationScheduler._copy_to_host", original_d2h)

        @functools.wraps(original_d2h)
        def profiled_copy_to_host(self, tensor):
            profile = _profile_enabled()
            start = time.perf_counter() if profile else 0.0
            result = original_d2h(self, tensor)
            if profile:
                _add("d2h_copy", tensor=tensor, seconds=time.perf_counter() - start)
            return result

        profiled_copy_to_host._unsloth_gc_profile_wrapped = True
        scheduler_cls._copy_to_host = profiled_copy_to_host

    original_h2d = scheduler_cls._restore_slot_to_device
    if not getattr(original_h2d, "_unsloth_gc_profile_wrapped", False):
        _ORIGINALS.setdefault("UnslothStreamActivationScheduler._restore_slot_to_device", original_h2d)

        @functools.wraps(original_h2d)
        def profiled_restore_slot(self, slot):
            profile = _profile_enabled()
            start = time.perf_counter() if profile else 0.0
            result = original_h2d(self, slot)
            if profile:
                tensor = getattr(slot, "host_tensor", None)
                _add("h2d_restore", tensor=tensor if torch.is_tensor(tensor) else None, seconds=time.perf_counter() - start)
            return result

        profiled_restore_slot._unsloth_gc_profile_wrapped = True
        scheduler_cls._restore_slot_to_device = profiled_restore_slot

    original_stage_window = scheduler_cls.stage_window_to_host
    if not getattr(original_stage_window, "_unsloth_gc_profile_wrapped", False):
        _ORIGINALS.setdefault("UnslothStreamActivationScheduler.stage_window_to_host", original_stage_window)

        @functools.wraps(original_stage_window)
        def profiled_stage_window_to_host(self, window_index):
            already_staged = window_index in self.host_windows
            result = original_stage_window(self, window_index)
            if _hit_count_enabled() and not already_staged:
                staged = self.host_windows.get(window_index)
                if staged:
                    _HIT_COUNTS["stream_staged_windows"] += 1
                    _HIT_COUNTS["stream_staged_tensors"] += len(staged)
            return result

        profiled_stage_window_to_host._unsloth_gc_profile_wrapped = True
        scheduler_cls.stage_window_to_host = profiled_stage_window_to_host

    original_restore_window = scheduler_cls.restore_window_to_device
    if not getattr(original_restore_window, "_unsloth_gc_profile_wrapped", False):
        _ORIGINALS.setdefault("UnslothStreamActivationScheduler.restore_window_to_device", original_restore_window)

        @functools.wraps(original_restore_window)
        def profiled_restore_window_to_device(self, window_index):
            mapping = self.host_windows.get(window_index)
            result = original_restore_window(self, window_index)
            if _hit_count_enabled() and mapping:
                _HIT_COUNTS["stream_restored_windows"] += 1
                _HIT_COUNTS["stream_restored_tensors"] += len(mapping)
            return result

        profiled_restore_window_to_device._unsloth_gc_profile_wrapped = True
        scheduler_cls.restore_window_to_device = profiled_restore_window_to_device

    original_seal = scheduler_cls.seal_forward_region
    if not getattr(original_seal, "_unsloth_gc_profile_wrapped", False):
        _ORIGINALS.setdefault("UnslothStreamActivationScheduler.seal_forward_region", original_seal)

        @functools.wraps(original_seal)
        def profiled_seal_forward_region(self):
            active = self.current_window in self.active_windows
            result = original_seal(self)
            if _hit_count_enabled() and active:
                _HIT_COUNTS["stream_boundaries"] += 1
            return result

        profiled_seal_forward_region._unsloth_gc_profile_wrapped = True
        scheduler_cls.seal_forward_region = profiled_seal_forward_region

    original_rewind = scheduler_cls.rewind_for_backward_region
    if not getattr(original_rewind, "_unsloth_gc_profile_wrapped", False):
        _ORIGINALS.setdefault("UnslothStreamActivationScheduler.rewind_for_backward_region", original_rewind)

        @functools.wraps(original_rewind)
        def profiled_rewind_for_backward_region(self):
            current_window = self.current_window
            result = original_rewind(self)
            if _hit_count_enabled() and current_window > 0:
                _HIT_COUNTS["stream_backward_boundaries"] += 1
            return result

        profiled_rewind_for_backward_region._unsloth_gc_profile_wrapped = True
        scheduler_cls.rewind_for_backward_region = profiled_rewind_for_backward_region


def patch_unsloth_gc_profile():
    global _PATCHED
    if _PATCHED:
        return
    gc_mod = sys.modules.get(_MODULE_NAME)
    if gc_mod is None:
        raise RuntimeError("Unsloth GC profiler must be installed after gradient_checkpointing is imported")
    _patch_binding(gc_mod)
    _patch_checkpoint_dispatch(gc_mod)
    _patch_pinned_alloc(gc_mod)
    _patch_stream_scheduler(gc_mod)
    if _profile_enabled():
        atexit.register(dump_unsloth_gc_profile)
    if _hit_count_enabled():
        atexit.register(dump_unsloth_gc_hit_count)
    _PATCHED = True
