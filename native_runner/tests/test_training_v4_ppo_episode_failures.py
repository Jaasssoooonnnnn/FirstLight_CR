"""A failed match (worker BattleEnvError / lost connection) is dropped, not the whole wave."""

from __future__ import annotations

from collections import defaultdict, deque
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4 import async_cluster_self_play as collector
from native_runner.training.v4 import train_ppo_self_play_cluster as trainer


class Connection:
    def __init__(self, packets: list[Any]) -> None:
        self.packets = deque(packets)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def recv(self) -> Any:
        packet = self.packets.popleft()
        if isinstance(packet, BaseException):
            raise packet
        return packet

    def send(self, packet: dict[str, Any]) -> None:
        self.sent.append(packet)

    def close(self) -> None:
        self.closed = True


def _frame(engine_index: int, rewards: tuple[Any, ...] = ()) -> dict[str, Any]:
    batch = collector._pad_relation_edges(_batch(2), 10).to_storage("cpu", float_dtype=torch.float16)
    packed = collector._RowPackedTensorRecordV4(batch)
    return dict(
        kind="frame",
        engine_index=engine_index,
        actor_rows=(0, 1),
        storage_actor_rows=(0, 1),
        force_act_actor_rows=(),
        rewards=rewards,
        remote_storage_from_batch=True,
        batch=packed.record,
        row_packed_batch=packed,
    )


def _done(engine_index: int) -> dict[str, Any]:
    return dict(
        kind="done",
        engine_index=engine_index,
        rewards=((0, 1.0, True, False), (1, -1.0, True, False)),
        reports=({"sequence": 1, "engine_index": engine_index},),
        timing=defaultdict(float),
    )


def _error(engine_index: int) -> dict[str, Any]:
    message = "active special Tower Troop lacks its native runtime snapshot " + "x" * 400
    return dict(
        kind="error",
        engine_index=engine_index,
        error=f"BattleEnvError({message!r})",
        error_type="BattleEnvError",
        error_message=message[:300],
        traceback="Traceback ...",
    )


def _assignment(sequence: int) -> SimpleNamespace:
    deck = lambda name, tower: SimpleNamespace(deck_id=name, tower_troop_id=tower)  # noqa: E731
    return SimpleNamespace(
        matchup=SimpleNamespace(
            current_owners=(0, 1),
            policy_matchup="current-current",
            sequence=sequence,
            category="focus-focus",
            deck0=deck(f"deck-{sequence}-a", 159_000_002),
            deck1=deck(f"deck-{sequence}-b", 159_000_000),
            battle_level=16,
            seed=1000 + sequence,
        ),
        opponent=None,
    )


def _collect(monkeypatch, connections: dict[int, Connection], *, tolerate: bool = True):
    torch.manual_seed(31)
    model = _model().eval()

    class EagerGraph:
        def run(self, model: Any, batch: Any, state: Any, episode_start: Any, packed: Any) -> Any:
            return model.act(batch.to_model_input("cpu"), state, episode_start=episode_start, validate=False)

    monkeypatch.setattr(collector.RemoteWorkerConnection, "connect", lambda host, port, timeout: connections[int(port)])
    monkeypatch.setattr(collector, "_persistent_rollout_graph_v4", lambda *args, **kwargs: EagerGraph())
    monkeypatch.setattr(collector.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    monkeypatch.setattr(collector.os, "sched_setaffinity", lambda pid, cpus: None, raising=False)
    specs = tuple(
        collector.AsyncResidentEngineSpecV4(engine_index=index, cpu=0, lanes=1, worker_host="test", worker_port=port)
        for index, port in ((0, 1), (1, 2))
    )
    segments: list[Any] = []

    def callback(collection: Any) -> None:
        segments.append(collection)
        return None

    collector.collect_async_cluster_wave_v4(
        specs,
        (_assignment(1), _assignment(2)),  # type: ignore[arg-type]
        model,
        model,
        {},
        gamma_per_decision=0.9997,
        shaping_beta=0.05,
        mutual_elixir_overflow_penalty=0.0,
        mutual_elixir_overflow_grace=4.0,
        unilateral_elixir_overflow_penalty=0.0,
        elixir_overflow_step_penalty_cap=0.1,
        policy_state_id="wave",
        max_decision_steps=100,
        segment_decision_steps=400,
        segment_callback=callback,
        tolerate_episode_failures=tolerate,
    )
    assert len(segments) == 1
    return segments[0]


def _healthy() -> Connection:
    return Connection([_frame(0), _frame(0, ((0, 0.25, False, False), (1, -0.25, False, False))), _done(0)])


def test_failed_match_is_dropped_and_the_other_lanes_are_complete(monkeypatch) -> None:
    healthy = _healthy()
    failing = Connection([_frame(1), _error(1)])
    collection = _collect(monkeypatch, {1: healthy, 2: failing})
    rollout = collection.rollout
    # Only engine 0's two learner actors remain; engine 1 left no frame behind.
    assert rollout.batch_size == 2
    assert rollout.valid_mask.tolist() == [[True, True], [True, True]]
    assert rollout.episode_start[0].tolist() == [True, True]  # whole episodes only (anchor constraint)
    assert rollout.terminated[-1].tolist() == [True, True]
    torch.testing.assert_close(rollout.rewards, torch.tensor([[0.25, -0.25], [1.0, -1.0]]))
    assert [report["engine_index"] for report in collection.reports] == [0]
    timing = collection.timing
    assert timing["episode_failure_count"] == 1.0 and timing["configured_matches"] == 2.0
    (record,) = timing["episode_failures"]
    assert record["engine_index"] == 1 and record["lane"] == 0
    assert record["error_type"] == "BattleEnvError"
    assert record["error_message"].startswith("active special Tower Troop") and len(record["error_message"]) == 300
    assert (record["deck0_id"], record["deck1_id"], record["battle_level"], record["seed"]) == (
        "deck-2-a",
        "deck-2-b",
        16,
        1002,
    )
    assert record["tower_troop_ids"] == (159_000_002, 159_000_000)
    assert failing.closed and not failing.packets and not healthy.packets


def test_failure_before_the_first_frame_and_lost_connections_are_dropped(monkeypatch) -> None:
    collection = _collect(monkeypatch, {1: _healthy(), 2: Connection([_error(1)])})
    assert collection.rollout.batch_size == 2 and collection.timing["episode_failure_count"] == 1.0
    collection = _collect(monkeypatch, {1: _healthy(), 2: Connection([_frame(1), EOFError("worker gone")])})
    (record,) = collection.timing["episode_failures"]
    assert record["error_type"] == "EOFError" and "worker connection lost" in record["error_message"]
    assert collection.rollout.batch_size == 2


def test_strict_collector_still_fails_the_wave(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="async engine 1 failed"):
        _collect(monkeypatch, {1: _healthy(), 2: Connection([_frame(1), _error(1)])}, tolerate=False)


def test_healthy_wave_reports_no_failures(monkeypatch) -> None:
    second = Connection([_frame(1), _frame(1), _done(1)])
    collection = _collect(monkeypatch, {1: _healthy(), 2: second})
    assert collection.rollout.batch_size == 4
    assert collection.timing["episode_failure_count"] == 0.0 and collection.timing["episode_failures"] == ()


# ---------------------------------------------------------------------------
# Trainer side: log, rate, threshold
# ---------------------------------------------------------------------------


def test_trainer_collects_failures_and_enforces_the_rate() -> None:
    shards = [
        {"timing": {"episode_failures": ({"engine_index": 3, "error_type": "BattleEnvError"},)}},
        {"timing": {"episode_failures": ()}},
        {"timing": {}},
    ]
    rows = trainer._episode_failures_from_shards(shards)
    assert rows == ({"engine_index": 3, "error_type": "BattleEnvError", "rank": 0},)
    assert trainer._episode_failure_rate(1, 100, max_rate=0.02) == (0.01, True)
    assert trainer._episode_failure_rate(2, 100, max_rate=0.02) == (0.02, True)
    rate, ok = trainer._episode_failure_rate(3, 100, max_rate=0.02)
    assert rate == pytest.approx(0.03) and not ok
    assert trainer._episode_failure_rate(1, 100, max_rate=0.0)[1] is False
    args = trainer.build_argument_parser().parse_args(
        ["--deck-pool", "p", "--checkpoint", "c", "--output-dir", "o", "--engine-ready", "e"]
    )
    assert args.max_episode_failure_rate == 0.02


def test_trainer_wires_the_tolerant_collector_and_the_audit() -> None:
    source = inspect.getsource(trainer.main)
    assert "tolerate_episode_failures=True" in source
    assert 'episode_failures_path = args.output_dir / "episode_failures.jsonl"' in source
    assert "and failure_rate_ok" in source  # above the threshold the wave audit raises
    assert '"episode_failure_rate": episode_failure_rate' in source
    assert "wave_episode_failures = 0" in source
    metrics = trainer._wandb_update_metrics(
        {
            "update": 1,
            "learner_owner_frame_fraction": 1.0,
            "native_rejected_actions": 0,
            "collection_truncated_count": 0,
            "collection_boundary_bootstrap_count": 0,
            "league_history_snapshots": 0,
            "episode_failure_rate": 0.01,
            "wave_episode_failures": 7,
        }
    )
    assert metrics["safety/episode_failure_rate"] == 0.01 and metrics["safety/wave_episode_failures"] == 7
