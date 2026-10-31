"""Opt-in, bounded CPU/CUDA profiling of an otherwise unchanged PPO update."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import time

import torch

_active = ContextVar("ppo_update_profiler", default=None)
_profiled = False


def profile_scope(name: str):
    return torch.profiler.record_function(name) if _active.get() is not None else nullcontext()


def profile_step() -> None:
    profiler = _active.get()
    if profiler is not None:
        profiler.step()


@contextmanager
def _label_method(obj, name: str, label: str):
    original = getattr(obj, name)
    existed = name in vars(obj)
    saved = vars(obj).get(name)

    @wraps(original)
    def labelled(*args, **kwargs):
        with profile_scope(label):
            return original(*args, **kwargs)

    setattr(obj, name, labelled)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, saved)
        else:
            delattr(obj, name)


def profile_update(function):
    """Record only minibatch 2 of the first update when CR_PPO_PROFILE_DIR is set.

    Minibatch 1 warms the normal kernels. No tensor, loss, optimizer or sampling
    value is changed. Profiled iterations must not be quoted as throughput tests.
    """
    @wraps(function)
    def wrapped(model, optimizer, rollout, *args, **kwargs):
        global _profiled
        root = os.environ.get("CR_PPO_PROFILE_DIR")
        if not root or _profiled:
            return function(model, optimizer, rollout, *args, **kwargs)
        _profiled = True
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"update-rank{os.environ.get('RANK', '0')}-pid{os.getpid()}-{time.time_ns()}"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if next(model.parameters()).is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        metadata = {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads(),
            "autocast_dtype": str(kwargs.get("autocast_dtype")),
            "rollout_lanes": rollout.batch_size, "rollout_steps": rollout.time_steps,
            "packed_observations": rollout.packed_observation_rows is not None,
            "profiled_minibatch": 2,
        }
        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
            properties = torch.cuda.get_device_properties(device)
            metadata.update(gpu=properties.name, compute_capability=[properties.major, properties.minor],
                            memory_gib=properties.total_memory / 2**30,
                            allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                            reserved_gib=torch.cuda.memory_reserved(device) / 2**30)
        (destination / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2))

        def save(profiler):
            profiler.export_chrome_trace(str(destination / f"{stem}.trace.json"))
            rows = []
            for event in profiler.key_averages(group_by_input_shape=True):
                rows.append({
                    "name": event.key, "count": event.count, "shapes": event.input_shapes,
                    "cpu_total_us": event.cpu_time_total, "cpu_self_us": event.self_cpu_time_total,
                    "device_total_us": event.device_time_total,
                    "device_self_us": event.self_device_time_total,
                })
            rows.sort(key=lambda row: row["device_self_us"], reverse=True)
            (destination / f"{stem}.summary.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({"event": "ppo_update_profile_saved", "path": str(destination / stem)}), flush=True)

        with ExitStack() as stack:
            for role, actor in (("learner", model), ("reference", kwargs.get("reference_model"))):
                if actor is None:
                    continue
                for method, part in (("encode_observation", "encode"),
                                     ("evaluate_encoded_action_sequence", "recurrent_decode")):
                    stack.enter_context(_label_method(actor, method, f"ppo.{role}.{part}"))
                for part in ("relation_transformer", "spatial_encoder", "group_encoder",
                             "card_encoder", "tower_encoder", "candidate_encoder", "event_encoder"):
                    module = getattr(actor, part, None)
                    if module is not None:
                        stack.enter_context(_label_method(module, "forward", f"ppo.{role}.{part}"))
                for part in ("input", "blocks"):
                    stack.enter_context(_label_method(getattr(actor.spatial_encoder, part), "forward",
                                                     f"ppo.{role}.cnn_{part}"))
            profiler = stack.enter_context(torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
                on_trace_ready=save, record_shapes=False, with_stack=False, profile_memory=False,
            ))
            token = _active.set(profiler)
            try:
                return function(model, optimizer, rollout, *args, **kwargs)
            finally:
                _active.reset(token)
    return wrapped
