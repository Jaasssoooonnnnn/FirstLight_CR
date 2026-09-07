"""Replay production collector protocol and PPO updates without external engines."""

from collections import defaultdict, deque
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
import hashlib
import torch
from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4 import async_cluster_self_play as collector


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
    if is_dataclass(value):
        return {item.name: fingerprint(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    if isinstance(value, dict):
        return {key: fingerprint(item) for key, item in value.items()}
    return value


class EagerGraph:
    def __init__(self, validate=False):
        self.validate = validate

    def run(self, model, batch, state, episode_start, packed):
        return model.act(batch.to_model_input("cpu"), state, episode_start=episode_start, validate=self.validate)


class RecordedConnection:
    def __init__(self, packets):
        self.packets = deque(packets)
        self.commands = []

    def recv(self):
        return self.packets.popleft()

    def send(self, packet):
        if packet["kind"] == "act_packed":
            self.commands.append(fingerprint(packet))

    def close(self):
        pass


def collect_recorded_segments(monkeypatch, *, validate=False, dynamic_effects=False):
    torch.manual_seed(923)
    model = _model().eval()

    def frame(actors, rewards, relations, boundary=False):
        batch = collector._pad_relation_edges(_batch(4), relations)
        if dynamic_effects:
            batch = collector._pad_active_effects(batch, relations * 8)
        batch = batch.to_storage("cpu", float_dtype=torch.float16)
        packed = collector._RowPackedTensorRecordV4(batch)
        result = dict(
            kind="frame",
            engine_index=0,
            actor_rows=actors,
            storage_actor_rows=actors,
            force_act_actor_rows=(),
            rewards=rewards,
            remote_storage_from_batch=True,
            batch=packed.record,
            row_packed_batch=packed,
        )
        if boundary:
            result.update(segment_boundary=True, timing=defaultdict(float))
        return result

    packets = [
        frame((0, 1, 2, 3), (), 10),
        frame(
            (0, 1), ((0, 0.1, False, False), (1, -0.1, False, False), (2, 1.0, True, False), (3, -1.0, True, False)), 12
        ),
        frame((0, 1), ((0, 0.2, False, False), (1, -0.2, False, False)), 12, True),
        frame((0, 1), (), 11),
        dict(
            kind="done",
            rewards=((0, 1.0, True, False), (1, -1.0, True, False)),
            reports=({"complete": True},),
            timing=defaultdict(float),
        ),
    ]
    connection = RecordedConnection(packets)
    monkeypatch.setattr(collector.RemoteWorkerConnection, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(collector, "_persistent_rollout_graph_v4", lambda *args, **kwargs: EagerGraph(validate))
    monkeypatch.setattr(collector.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    monkeypatch.setattr(collector.os, "sched_setaffinity", lambda pid, cpus: None, raising=False)
    spec = collector.AsyncResidentEngineSpecV4(engine_index=0, cpu=0, lanes=2, worker_host="test", worker_port=1)
    assignments = tuple(
        SimpleNamespace(matchup=SimpleNamespace(current_owners=(0, 1), policy_matchup="current-current"))
        for _ in range(2)
    )
    collected = []

    def callback(collection):
        collected.append(collection.rollout)
        return "segment-two" if len(collected) == 1 else None

    collector.collect_async_cluster_wave_v4(
        (spec,),
        assignments,
        model,
        model,
        {},
        gamma_per_decision=0.9997,
        shaping_beta=0.05,
        mutual_elixir_overflow_penalty=0.0,
        mutual_elixir_overflow_grace=4.0,
        unilateral_elixir_overflow_penalty=0.0,
        elixir_overflow_step_penalty_cap=0.1,
        policy_state_id="segment-one",
        max_decision_steps=100,
        segment_decision_steps=2,
        segment_callback=callback,
        validate_tensors=validate,
    )
    assert len(collected) == 2 and not connection.packets
    first, second = collected
    assert first.valid_mask.tolist() == [[True, True, True, True], [True, True, False, False]]
    assert first.terminated.tolist() == [[False, False, True, True], [False, False, False, False]]
    assert first.truncated.tolist() == [[False] * 4, [True, True, False, False]]
    assert first.episode_start.tolist() == [[True] * 4, [False] * 4]
    assert second.valid_mask.tolist() == [[True, True]]
    assert second.episode_start.tolist() == [[False, False]]
    assert second.terminated.tolist() == [[True, True]]
    assert torch.any(second.initial_state.hidden != 0)
    assert first.policy_state_id == "segment-one" and second.policy_state_id == "segment-two"
    assert len(connection.commands) == 3
    torch.testing.assert_close(first.rewards, torch.tensor([[0.1, -0.1, 1.0, -1.0], [0.2, -0.2, 0.0, 0.0]]))
    torch.testing.assert_close(second.rewards, torch.tensor([[1.0, -1.0]]))
    return model, (first, second)


def test_collector_preserves_partial_episodes_across_schema_changes_and_resume(monkeypatch):
    collect_recorded_segments(monkeypatch)
