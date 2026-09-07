"""V4 checkpoint loading and recurrent inference for offline rendered matches."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
from pathlib import Path
import time
from typing import Mapping
import weakref

import torch

from ...battle_env import BattleEnvV1
from ...contracts import ObservationV1
from .checkpoint import load_actor_critic_checkpoint
from .decoding import decode_action_sequence_v4
from .expert import POLICY_DECISION_TICKS
from .factory import build_episode_tensorizer_v4, build_production_model_v4
from .model import UniversalCardPolicyV4
from .native_actions import DecodedActionSequenceV4
from .tensorizer import UniversalObservationTensorizerV4
from .tensors import (
    ActiveEffectSetV4,
    PolicyOutputV4,
    RelationEdgesV4,
    RecurrentPolicyStateV4,
    UniversalSemanticBatchV4,
)


_POLICY_CUDA_GRAPH_EFFECT_CAPACITY = 512
_POLICY_CUDA_GRAPH_RELATION_CAPACITY = 2048


@dataclass(frozen=True, slots=True)
class LoadedPolicyV4:
    model: UniversalCardPolicyV4
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_id: str
    update_step: int
    training_stage: str


@dataclass(frozen=True, slots=True)
class PolicyDecisionV4:
    decoded: DecodedActionSequenceV4
    inference_ms: float


def _copy_tensor_record_(target: object, source: object) -> None:
    """Copy matching nested tensor records without allocating new CUDA inputs."""

    if isinstance(target, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise TypeError("V4 tensor record source is not a tensor")
        if target.shape != source.shape:
            raise ValueError("V4 CUDA Graph input shape changed")
        target.copy_(source)
        return
    if is_dataclass(target):
        if type(source) is not type(target):
            raise TypeError("V4 CUDA Graph tensor record type changed")
        for field in fields(target):
            _copy_tensor_record_(getattr(target, field.name), getattr(source, field.name))
        return
    if target != source:
        raise ValueError("V4 CUDA Graph non-tensor input changed")


def _pad_axis_one(value: torch.Tensor, capacity: int, pad_value: int | float | bool) -> torch.Tensor:
    count = int(value.shape[1])
    if count > capacity:
        raise ValueError(f"V4 CUDA Graph capacity exceeded: {count} > {capacity}")
    if count == capacity:
        return value
    result = torch.full((value.shape[0], capacity, *value.shape[2:]), pad_value, dtype=value.dtype, device=value.device)
    if count:
        result[:, :count].copy_(value)
    return result


def _fixed_cuda_graph_batch_v4(
    batch: UniversalSemanticBatchV4,
    *,
    effect_capacity: int = _POLICY_CUDA_GRAPH_EFFECT_CAPACITY,
    relation_capacity: int = _POLICY_CUDA_GRAPH_RELATION_CAPACITY,
) -> UniversalSemanticBatchV4:
    """Pad the two intentionally variable V4 sets for one reusable graph."""

    if effect_capacity <= 0 or relation_capacity <= 0:
        raise ValueError("V4 CUDA Graph capacities must be positive")

    effects = batch.active_effects
    fixed_effects = ActiveEffectSetV4(
        effect_vocab_id=_pad_axis_one(effects.effect_vocab_id, effect_capacity, 0),
        parent_type=_pad_axis_one(effects.parent_type, effect_capacity, 0),
        parent_index=_pad_axis_one(effects.parent_index, effect_capacity, -1),
        source_owner_type=_pad_axis_one(effects.source_owner_type, effect_capacity, 0),
        runtime_features=_pad_axis_one(effects.runtime_features, effect_capacity, 0.0),
        mask=_pad_axis_one(effects.mask, effect_capacity, False),
    )
    relations = batch.relation_edges
    fixed_relations = RelationEdgesV4(
        source=_pad_axis_one(relations.source, relation_capacity, 0),
        target=_pad_axis_one(relations.target, relation_capacity, 0),
        relation_type=_pad_axis_one(relations.relation_type, relation_capacity, 0),
        mask=_pad_axis_one(relations.mask, relation_capacity, False),
    )
    return replace(batch, active_effects=fixed_effects, relation_edges=fixed_relations)


class _DeterministicCudaGraphV4:
    """One reusable batch-1 deterministic graph owned by a loaded model."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.batch: UniversalSemanticBatchV4 | None = None
        self.state: RecurrentPolicyStateV4 | None = None
        self.episode_start: torch.Tensor | None = None
        self.output: PolicyOutputV4 | None = None

    def _capture(
        self,
        model: UniversalCardPolicyV4,
        host_batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        episode_start: torch.Tensor,
    ) -> PolicyOutputV4:
        fixed_batch = _fixed_cuda_graph_batch_v4(host_batch)
        self.batch = fixed_batch.to_model_input(self.device)
        self.state = RecurrentPolicyStateV4(hidden=state.hidden.detach().clone(), cell=state.cell.detach().clone())
        self.episode_start = episode_start.detach().clone()

        # CUDA Graph capture requires allocations/kernels to be warmed on a
        # side stream. The captured graph then owns stable input/output storage
        # and can be reused across episodes by copying new values in-place.
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                model.act(self.batch, self.state, episode_start=self.episode_start, validate=False)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model.act(self.batch, self.state, episode_start=self.episode_start, validate=False)
        if self.output is None:
            raise RuntimeError("V4 CUDA Graph capture produced no output")
        # Replay once before exposing the static output tensors. Capture builds
        # the executable graph and reserves its pool; only replay is treated as
        # the authoritative inference execution across CUDA/PyTorch versions.
        self.graph.replay()
        return self.output

    def run(
        self,
        model: UniversalCardPolicyV4,
        host_batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        episode_start: torch.Tensor,
    ) -> PolicyOutputV4:
        if self.graph is None:
            return self._capture(model, host_batch, state, episode_start)
        if self.batch is None or self.state is None or self.episode_start is None or self.output is None:
            raise RuntimeError("V4 CUDA Graph storage is incomplete")
        _copy_tensor_record_(self.batch, _fixed_cuda_graph_batch_v4(host_batch))
        _copy_tensor_record_(self.state, state)
        self.episode_start.copy_(episode_start)
        self.graph.replay()
        return self.output


_POLICY_CUDA_GRAPHS_V4: weakref.WeakKeyDictionary[UniversalCardPolicyV4, dict[str, _DeterministicCudaGraphV4]] = (
    weakref.WeakKeyDictionary()
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_v4_checkpoint(path: str | Path) -> Path:
    """Resolve one exact V4 checkpoint, accepting a training directory."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"V4 checkpoint does not exist: {candidate}")
    matches = sorted(candidate.glob("checkpoint-step-*.pt"))
    if not matches:
        matches = sorted(candidate.glob("*.pt"))
    if not matches:
        raise FileNotFoundError(f"checkpoint directory contains no V4 .pt files: {candidate}")
    return matches[-1].resolve()


def load_policy_v4(checkpoint: str | Path, *, device: torch.device | str) -> LoadedPolicyV4:
    """Construct the production V4 model and load a strict V4 checkpoint."""

    checkpoint_path = resolve_v4_checkpoint(checkpoint)
    actual_device = torch.device(device)
    model = build_production_model_v4(device=actual_device)
    payload = load_actor_critic_checkpoint(checkpoint_path, model, map_location=actual_device, restore_rng=False)
    model.eval()
    return LoadedPolicyV4(
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_id=str(payload["checkpoint_id"]),
        update_step=int(payload["update_step"]),
        training_stage=str(payload["training_stage"]),
    )


class PolicySessionV4:
    """One recurrent V4 policy state spanning one complete offline battle."""

    def __init__(
        self,
        model: UniversalCardPolicyV4,
        tensorizer: UniversalObservationTensorizerV4,
        *,
        device: torch.device | str,
        sample: bool = False,
        validate_tensors: bool = False,
    ) -> None:
        self.model = model
        self.tensorizer = tensorizer
        self.device = torch.device(device)
        self.sample = bool(sample)
        self.validate_tensors = bool(validate_tensors)
        self.state = None
        self._episode_started = False
        self._first_decision = True
        self._cuda_graph: _DeterministicCudaGraphV4 | None = None
        if self.device.type == "cuda" and not self.sample and not self.validate_tensors:
            by_device = _POLICY_CUDA_GRAPHS_V4.setdefault(model, {})
            device_key = str(self.device)
            cached = by_device.get(device_key)
            if cached is None:
                cached = _DeterministicCudaGraphV4(self.device)
                by_device[device_key] = cached
            self._cuda_graph = cached

    @property
    def actor_owner(self) -> int:
        return int(self.tensorizer.actor_owner)

    def start_episode(self, observation: ObservationV1, *, initial_elixir: Mapping[int, float]) -> None:
        if observation.owner != self.actor_owner:
            raise ValueError("offline observation owner does not match V4 policy session")
        self.tensorizer.start_episode(observation, initial_elixir=initial_elixir)
        self.state = self.model.initial_state(1, device=self.device)
        self._episode_started = True
        self._first_decision = True

    def end_episode(self) -> None:
        if self._episode_started:
            self.tensorizer.end_episode()
        self.state = None
        self._episode_started = False
        self._first_decision = True

    @torch.inference_mode()
    def decide(self, observation: ObservationV1) -> PolicyDecisionV4:
        if not self._episode_started or self.state is None:
            raise RuntimeError("start_episode must run before offline V4 inference")
        if observation.owner != self.actor_owner:
            raise ValueError("offline observation owner changed during the battle")

        host_batch = self.tensorizer.tensorize(observation, validate=self.validate_tensors)
        episode_start = torch.tensor([self._first_decision], dtype=torch.bool, device=self.device)
        started_ns = time.perf_counter_ns()
        if self._cuda_graph is not None:
            output = self._cuda_graph.run(self.model, host_batch, self.state, episode_start)
        else:
            model_batch = host_batch.to_model_input(self.device)
            if self.sample:
                output = self.model.sample_for_ppo_rollout(
                    model_batch, self.state, episode_start=episode_start, validate=self.validate_tensors
                )
            else:
                output = self.model.act(
                    model_batch, self.state, episode_start=episode_start, validate=self.validate_tensors
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        output.actions.validate(self.tensorizer.config, candidate_count=int(host_batch.candidates.mask.shape[1]))

        base_latency_ticks = 1
        decoded = decode_action_sequence_v4(
            output.actions,
            host_batch.candidates,
            row=0,
            observation=observation,
            catalog=self.tensorizer.catalog,
            deck=self.tensorizer.deck,
            card_costs=self.tensorizer.card_costs,
            ability_id_by_vocab_id=self.tensorizer.ability_id_by_vocab_id,
            horizontal_mirror=self.tensorizer.perspective.horizontal_mirror,
            hand_slot_permutation=(self.tensorizer.perspective.hand_slot_permutation),
            config=self.tensorizer.config,
            base_latency_ticks=base_latency_ticks,
            base_latency_ms=0.0,
            validate=False,
        )
        # Training feeds the selected semantic action into the next decision.
        # Offline inference must do exactly the same on every five-tick turn.
        self.tensorizer.record_action(output.actions, host_batch, row=0, validate=False)
        self.state = output.next_state
        self._first_decision = False
        return PolicyDecisionV4(decoded=decoded, inference_ms=inference_ms)


def build_policy_session_v4(
    environment: BattleEnvV1,
    model: UniversalCardPolicyV4,
    *,
    actor_owner: int,
    device: torch.device | str,
    sample: bool = False,
    validate_tensors: bool = False,
) -> PolicySessionV4:
    episode = environment.episode_config
    if episode is None:
        raise RuntimeError("BattleEnv must be reset before building a V4 session")
    if episode.decision_hz != 20.0 / POLICY_DECISION_TICKS:
        raise RuntimeError("offline V4 BattleEnv is not configured for five-tick turns")
    return PolicySessionV4(
        model,
        build_episode_tensorizer_v4(episode, actor_owner=actor_owner),
        device=device,
        sample=sample,
        validate_tensors=validate_tensors,
    )
