from __future__ import annotations

from dataclasses import dataclass, field
import threading
from types import SimpleNamespace

import native_runner.training.v4.producer as producer_module
import pytest
from native_runner.contracts import ActionKind, EnvironmentConfigV1, EpisodeConfigV1
from native_runner.semantic_subset import SEMANTIC_BASELINE_DECK


class _FakeCoordinator:
    def __init__(self) -> None:
        self.rounds: list[tuple[int, ...]] = []
        self.started = 0
        self.closed = 0
        self.aborted: BaseException | None = None
        self._condition = threading.Condition()
        self._expected: tuple[int, ...] = ()
        self._submitted: set[int] = set()

    def start_session(self) -> None:
        self.started += 1

    def close_session(self) -> None:
        self.closed += 1

    def begin_round(self, env_ids: tuple[int, ...]) -> None:
        with self._condition:
            assert env_ids
            assert not self._expected
            self.rounds.append(env_ids)
            self._expected = env_ids
            self._submitted = set()
            self.aborted = None

    def submit(self, env_id: int) -> tuple[int]:
        with self._condition:
            assert env_id in self._expected
            self._submitted.add(env_id)
            if self._submitted == set(self._expected):
                self._condition.notify_all()
            else:
                self._condition.wait_for(
                    lambda: self._submitted == set(self._expected)
                    or self.aborted is not None,
                    timeout=2.0,
                )
            if self.aborted is not None:
                raise self.aborted
            assert self._submitted == set(self._expected)
            return (env_id,)

    def abort_round(self, error: BaseException) -> None:
        with self._condition:
            if self._submitted != set(self._expected):
                self.aborted = error
                self._condition.notify_all()

    @property
    def round_succeeded(self) -> bool:
        with self._condition:
            return (
                bool(self._expected)
                and self._submitted == set(self._expected)
                and self.aborted is None
            )

    def finish_round(self) -> None:
        with self._condition:
            assert self._submitted == set(self._expected) or self.aborted is not None
            self._expected = ()
            self._submitted = set()


class _FakeEnvironment:
    def __init__(
        self,
        env_id: int,
        coordinator: _FakeCoordinator,
        *,
        fail_after_submit_call: int | None = None,
    ) -> None:
        self.native = SimpleNamespace(env_id=env_id, coordinator=coordinator)
        self.fail_after_submit_call = fail_after_submit_call
        self.step_calls = 0

    def step(self, _actions: object, *, record_trace: bool) -> tuple[int]:
        assert record_trace is False
        self.step_calls += 1
        result = self.native.coordinator.submit(self.native.env_id)
        if self.step_calls == self.fail_after_submit_call:
            raise ValueError("scripted lane-local post-processing failure")
        return result


class _FakeTensorizer:
    def __init__(self) -> None:
        self.end_calls = 0

    def end_episode(self) -> None:
        self.end_calls += 1


def test_il_lane_uses_true_headless_driver_without_raster() -> None:
    environment_config = EnvironmentConfigV1(max_battle_ticks=200)
    source_episode = EpisodeConfigV1(
        ruleset_id="source",
        deck0=SEMANTIC_BASELINE_DECK,
        deck1=SEMANTIC_BASELINE_DECK,
        seed=1,
        environment=environment_config,
    )
    captured: dict[str, object] = {}

    class _Environment:
        ruleset_id = "execution"
        native = object()

        def __init__(self) -> None:
            self.environment_config = environment_config

        def reset(
            self,
            episode: EpisodeConfigV1,
            *,
            options: object,
        ) -> tuple[dict[int, object], dict[str, object]]:
            captured["episode"] = episode
            captured["options"] = options
            players = (
                SimpleNamespace(owner=0, elixir_exact=6.0),
                SimpleNamespace(owner=1, elixir_exact=6.0),
            )
            return (
                {
                    0: SimpleNamespace(tick=producer_module.FIRST_POLICY_DECISION_TICK, players=players),
                    1: SimpleNamespace(tick=producer_module.FIRST_POLICY_DECISION_TICK, players=players),
                },
                {},
            )

    class _Tensorizer:
        tracker = None

        def __init__(self) -> None:
            self.started = False

        def start_episode(self, _observation: object, **_kwargs: object) -> None:
            self.started = True

    tensorizers = (_Tensorizer(), _Tensorizer())
    lane = SimpleNamespace(
        environment=_Environment(),
        prepared=SimpleNamespace(
            replay=SimpleNamespace(episode_config=source_episode),
        ),
        source_end_tick=200,
        replay_commands=(),
        replay_schedule_sequences=(),
        tensorizers=tensorizers,
        observations=None,
        current_tick=0,
    )

    producer_module._initialize_il_lane(lane)

    assert captured["options"] == {
        "render_mode": "headless",
        "defer_headless_warmup": True,
        "headless_initial_step_ticks": 1,
        "allow_initial_remaining_runtime_rejections": True,
        "allow_verified_standard_layout_alias": True,
        "decision_ticks": 5,
        "event_driven_decisions": False,
    }
    assert all(item.started for item in tensorizers)


@dataclass
class _FakeLane:
    environment: _FakeEnvironment
    prepared: object
    tensorizers: tuple[_FakeTensorizer, _FakeTensorizer]
    stop_after: int
    current_tick: int = 125
    frames: int = 0
    active: bool = True
    completed: bool = False
    failure_reason: str | None = None
    first_untrusted_tick: int | None = None
    validations: list[bool] = field(default_factory=list)

    @property
    def env_id(self) -> int:
        return int(self.environment.native.env_id)


def _patch_lane_pipeline(monkeypatch: object, *, fail_lane: int | None = None) -> None:
    def build_lane(
        environment: _FakeEnvironment,
        prepared: object,
        tensorizers: tuple[_FakeTensorizer, _FakeTensorizer],
    ) -> _FakeLane:
        return _FakeLane(
            environment=environment,
            prepared=prepared,
            tensorizers=tensorizers,
            stop_after=int(prepared.stop_after),
        )

    def prepare_lane(lane: _FakeLane, *, validate_tensors: bool) -> dict[int, tuple]:
        lane.validations.append(validate_tensors)
        if fail_lane == lane.env_id and lane.frames == 1:
            raise ValueError("scripted replay reconstruction failure")
        return producer_module._wait_group(producer_module.POLICY_DECISION_TICKS)

    def accept_step(
        lane: _FakeLane,
        _step_result: object,
        **_kwargs: object,
    ) -> None:
        lane.frames += 1
        lane.current_tick += 5
        if lane.frames >= lane.stop_after:
            lane.active = False
            lane.completed = True

    monkeypatch.setattr(producer_module, "_build_il_lane", build_lane)
    monkeypatch.setattr(producer_module, "_initialize_il_lane", lambda lane: None)
    monkeypatch.setattr(producer_module, "_prepare_il_lane", prepare_lane)
    monkeypatch.setattr(producer_module, "_accept_il_step", accept_step)
    monkeypatch.setattr(
        producer_module,
        "_reconcile_renderer_schedule",
        lambda lane: None,
    )
    monkeypatch.setattr(
        producer_module,
        "_finalize_il_lane",
        lambda lane, **_kwargs: lane,
    )


def test_vector_producer_batches_only_active_lanes(monkeypatch: object) -> None:
    coordinator = _FakeCoordinator()
    environments = tuple(_FakeEnvironment(index, coordinator) for index in range(3))
    tensorizers = tuple((_FakeTensorizer(), _FakeTensorizer()) for _ in range(3))
    prepared = tuple(
        SimpleNamespace(stop_after=stop_after) for stop_after in (1, 2, 3)
    )
    _patch_lane_pipeline(monkeypatch)

    results = producer_module.produce_il_replay_batch(
        environments,
        prepared,
        tensorizers,
        gamma_per_decision=0.99,
        batch_coordinator=coordinator,
        validate_tensors=True,
    )

    assert coordinator.rounds == [(0, 1, 2), (1, 2), (2,)]
    assert [lane.frames for lane in results] == [1, 2, 3]
    assert [lane.validations for lane in results] == [[True], [True, True], [True] * 3]
    assert coordinator.started == 3
    assert coordinator.closed == 3
    assert all(item.end_calls == 1 for pair in tensorizers for item in pair)


def test_vector_producer_isolates_prepare_failure(monkeypatch: object) -> None:
    coordinator = _FakeCoordinator()
    environments = tuple(_FakeEnvironment(index, coordinator) for index in range(2))
    tensorizers = tuple((_FakeTensorizer(), _FakeTensorizer()) for _ in range(2))
    prepared = tuple(SimpleNamespace(stop_after=3) for _ in range(2))
    _patch_lane_pipeline(monkeypatch, fail_lane=0)

    results = producer_module.produce_il_replay_batch(
        environments,
        prepared,
        tensorizers,
        gamma_per_decision=0.99,
        batch_coordinator=coordinator,
    )

    failed, survivor = results
    assert coordinator.rounds == [(0, 1), (0, 1), (1,)]
    assert failed.frames == 1
    assert failed.completed is False
    assert failed.failure_reason == "ValueError: scripted replay reconstruction failure"
    assert failed.first_untrusted_tick == 130
    assert survivor.frames == 3
    assert survivor.completed is True


def test_vector_producer_isolates_post_step_lane_failure(monkeypatch: object) -> None:
    coordinator = _FakeCoordinator()
    environments = (
        _FakeEnvironment(0, coordinator, fail_after_submit_call=2),
        _FakeEnvironment(1, coordinator),
    )
    tensorizers = tuple((_FakeTensorizer(), _FakeTensorizer()) for _ in range(2))
    prepared = tuple(SimpleNamespace(stop_after=3) for _ in range(2))
    _patch_lane_pipeline(monkeypatch)

    results = producer_module.produce_il_replay_batch(
        environments,
        prepared,
        tensorizers,
        gamma_per_decision=0.99,
        batch_coordinator=coordinator,
    )

    failed, survivor = results
    assert coordinator.rounds == [(0, 1), (0, 1), (1,)]
    assert failed.frames == 1
    assert failed.failure_reason == (
        "ValueError: scripted lane-local post-processing failure"
    )
    assert survivor.frames == 3
    assert survivor.completed is True


def test_observation_pass_submits_wait_only(monkeypatch: object) -> None:
    class _Tensorizer:
        def __init__(self, owner: int) -> None:
            self.perspective = SimpleNamespace(actor_owner=owner)
            self.config = SimpleNamespace()
            self.recorded: list[object] = []

        def tensorize(self, observation: object, *, validate: bool) -> object:
            assert observation is not None
            assert validate is True
            return SimpleNamespace(candidates=object())

        def record_action(self, sequence: object, *_args: object, **_kwargs: object) -> None:
            self.recorded.append(sequence)

    sequences = [object(), object()]

    def build_label(
        _batch: object,
        _windows: object,
        *,
        perspectives: tuple[object],
        **_kwargs: object,
    ) -> object:
        return sequences[int(perspectives[0].actor_owner)]

    monkeypatch.setattr(producer_module, "build_expert_action_batch", build_label)
    tensorizers = (_Tensorizer(0), _Tensorizer(1))
    lane = SimpleNamespace(
        observations={0: object(), 1: object()},
        tensorizers=tensorizers,
        windows={},
        current_tick=130,
        pending_frame_batches=None,
        pending_frame_sequences=None,
        pending_gate_loss=None,
    )

    actions = producer_module._prepare_il_lane(lane, validate_tensors=True)

    assert all(group[0].kind is ActionKind.WAIT for group in actions.values())
    assert all(group[0].next_decision_ticks == 5 for group in actions.values())
    assert lane.pending_frame_sequences == tuple(sequences)
    assert lane.pending_gate_loss == (True, True)
    assert tensorizers[0].recorded == [sequences[0]]
    assert tensorizers[1].recorded == [sequences[1]]


def test_renderer_schedule_rejection_masks_only_its_owner() -> None:
    receipts = {
        11: {
            "state": "succeeded",
            "error": "none",
            "queuedAtTick": 132,
        },
        12: {
            "state": "failed",
            "error": "card-unavailable",
            "queuedAtTick": 132,
        },
    }
    status_calls: list[int] = []

    def schedule_status(sequence: int) -> dict[str, object]:
        status_calls.append(sequence)
        return receipts[sequence]

    lane = SimpleNamespace(
        replay_schedule_index=0,
        replay_schedule_reconciled_index=0,
        replay_commands=(
            SimpleNamespace(
                kind="play_card", owner=0, request_tick=130, target_tick=133
            ),
            SimpleNamespace(
                kind="play_card", owner=1, request_tick=130, target_tick=133
            ),
        ),
        replay_schedule_sequences=(11, 12),
        executed_expert_action_count=0,
        pending_gate_loss=(True, True),
        decision_ticks=[130],
        gate_loss_rows=[[True], [True]],
        environment=SimpleNamespace(
            native=SimpleNamespace(replay_schedule_status=schedule_status)
        ),
    )

    producer_module._consume_renderer_schedule(lane, through_tick=135)

    assert lane.replay_schedule_index == 2
    assert status_calls == []
    producer_module._reconcile_renderer_schedule(lane)
    assert lane.executed_expert_action_count == 1
    assert lane.gate_loss_rows == [[True], [False]]


def test_renderer_schedule_pending_after_target_fails_closed() -> None:
    lane = SimpleNamespace(
        replay_schedule_index=0,
        replay_schedule_reconciled_index=0,
        replay_commands=(
            SimpleNamespace(
                kind="ability", owner=0, request_tick=130, target_tick=133
            ),
        ),
        replay_schedule_sequences=(11,),
        executed_expert_action_count=0,
        pending_gate_loss=(True, True),
        decision_ticks=[130],
        gate_loss_rows=[[True], [True]],
        environment=SimpleNamespace(
            native=SimpleNamespace(
                replay_schedule_status=lambda _sequence: {
                    "state": "pending",
                    "error": "none",
                    "queuedAtTick": -1,
                }
            )
        ),
    )

    producer_module._consume_renderer_schedule(lane, through_tick=135)
    with pytest.raises(RuntimeError, match="unresolved schedule state"):
        producer_module._reconcile_renderer_schedule(lane)


def test_source_timeline_completion_keeps_observation_without_fake_terminal() -> None:
    class _Frame:
        def to_storage(self, *_args: object, **_kwargs: object) -> "_Frame":
            return self

    lane = SimpleNamespace(
        current_tick=130,
        source_end_tick=135,
        replay_schedule_index=0,
        replay_commands=(),
        replay_schedule_sequences=(),
        executed_expert_action_count=0,
        pending_frame_batches=(_Frame(), _Frame()),
        pending_frame_sequences=(object(), object()),
        pending_gate_loss=(True, True),
        observations_by_owner=[[], []],
        actions_by_owner=[[], []],
        rewards_by_owner=[[], []],
        gate_loss_rows=[[], []],
        terminated_rows=[],
        truncated_rows=[],
        decision_ticks=[],
        observations=None,
        active=True,
        completed=False,
        terminal_observed=False,
        failure_reason=None,
        first_untrusted_tick=None,
        environment=SimpleNamespace(native=object()),
    )
    next_observations = {
        0: SimpleNamespace(tick=135),
        1: SimpleNamespace(tick=135),
    }

    producer_module._accept_il_step(
        lane,
        (
            next_observations,
            {0: 0.0, 1: 0.0},
            {0: False, 1: False},
            {0: True, 1: True},
            {},
        ),
        max_frames=None,
    )

    assert lane.completed is True
    assert lane.active is False
    assert lane.failure_reason is None
    assert lane.terminal_observed is False
    assert lane.decision_ticks == [130]
    assert lane.truncated_rows == [False]


def test_early_native_terminal_still_records_result_fidelity(
    monkeypatch: object,
) -> None:
    class _Frame:
        def to_storage(self, *_args: object, **_kwargs: object) -> "_Frame":
            return self

    def record_fidelity(lane: object) -> None:
        lane.winner_matches = True
        lane.crowns_match = False

    monkeypatch.setattr(producer_module, "_terminal_fidelity", record_fidelity)
    lane = SimpleNamespace(
        current_tick=130,
        source_end_tick=300,
        replay_schedule_index=0,
        replay_commands=(
            SimpleNamespace(kind="play_card", owner=0, target_tick=200),
        ),
        replay_schedule_sequences=(11,),
        executed_expert_action_count=0,
        pending_frame_batches=(_Frame(), _Frame()),
        pending_frame_sequences=(object(), object()),
        pending_gate_loss=(True, True),
        observations_by_owner=[[], []],
        actions_by_owner=[[], []],
        rewards_by_owner=[[], []],
        gate_loss_rows=[[], []],
        terminated_rows=[],
        truncated_rows=[],
        decision_ticks=[],
        observations=None,
        active=True,
        completed=False,
        terminal_observed=False,
        failure_reason=None,
        first_untrusted_tick=None,
        winner_matches=None,
        crowns_match=None,
        environment=SimpleNamespace(native=object()),
    )
    next_observations = {
        0: SimpleNamespace(tick=135),
        1: SimpleNamespace(tick=135),
    }

    producer_module._accept_il_step(
        lane,
        (
            next_observations,
            {0: 0.0, 1: 0.0},
            {0: True, 1: True},
            {0: False, 1: False},
            {},
        ),
        max_frames=None,
    )

    assert lane.completed is False
    assert lane.terminal_observed is True
    assert lane.failure_reason == "battle ended before all expert actions executed"
    assert lane.winner_matches is True
    assert lane.crowns_match is False


def test_partial_replay_gate_loss_is_limited_to_trusted_frames() -> None:
    lane = SimpleNamespace(
        prepared=SimpleNamespace(replay_tag="partial"),
        decision_ticks=[125, 195, 200],
        completed=False,
        first_untrusted_tick=400,
        terminated_rows=[False, False, False],
        truncated_rows=[False, False, False],
        rewards_by_owner=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        terminal_observed=False,
        observations_by_owner=[[object()] * 3, [object()] * 3],
        actions_by_owner=[[object()] * 3, [object()] * 3],
        gate_loss_rows=[[True, True, True], [True, False, True]],
        failure_reason="scripted partial replay",
        current_tick=205,
        source_end_tick=600,
        expert_actions=(),
        executed_expert_action_count=0,
        winner_matches=None,
        crowns_match=None,
    )

    result = producer_module._finalize_il_lane(lane, gamma_per_decision=0.99)

    assert result.sequences is not None
    assert result.sequences[0].valid_mask[:, 0].tolist() == [True, True, False]
    assert result.sequences[0].gate_loss_mask[:, 0].tolist() == [True, True, False]
    assert result.sequences[1].gate_loss_mask[:, 0].tolist() == [True, False, False]
