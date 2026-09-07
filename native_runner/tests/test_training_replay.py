from __future__ import annotations

import gzip
import json

import pytest

from native_runner.contracts import (
    ActionV1,
    EnvironmentConfigV1,
    EpisodeConfigV1,
)
from native_runner.snapshot import SnapshotOperationV1
from native_runner.training.replay_archive import (
    TrainingReplayError,
    TrainingReplayV1,
    list_training_replays,
    replay_filename,
    save_training_replay,
)
from native_runner.training.replay_viewer import (
    ReplayRewindControl,
    _direct_render_commands,
    _match_config_from_replay,
    _play_training_replay_direct,
    _replay_environment,
    _wait_while_paused,
    _native_deploy_delay,
)


DECK = (
    26_000_001,
    26_000_010,
    26_000_024,
    26_000_030,
    26_000_032,
    27_000_006,
    28_000_012,
    28_000_015,
)


def _replay(index: int = 512) -> TrainingReplayV1:
    episode = EpisodeConfigV1(
        ruleset_id="ruleset-test",
        deck0=DECK,
        deck1=DECK,
        seed=17,
        decision_hz=4.0,
        event_driven_decisions=False,
    )
    operations = (
        SnapshotOperationV1(
            actions=(
                ActionV1.wait(0, ticks=5),
                ActionV1.wait(1, ticks=5),
            ),
            advance_ticks=5,
            requested_advance_ticks=5,
            start_native_tick=130,
            end_native_tick=135,
        ),
        SnapshotOperationV1(
            actions=(
                ActionV1.wait(0, ticks=5),
                ActionV1.wait(1, ticks=5),
            ),
            advance_ticks=5,
            requested_advance_ticks=5,
            start_native_tick=135,
            end_native_tick=140,
        ),
    )
    return TrainingReplayV1(
        completed_match_index=index,
        ruleset_id=episode.ruleset_id,
        episode_config=episode,
        episode_id="12345678-1234-1234-1234-123456789abc",
        operations=operations,
        terminal={
            "ended": True,
            "winner": 0,
            "result_by_owner": [1.0, -1.0],
            "reason": "native_final_result",
            "terminal_tick": 140,
        },
        source={"worker_id": 2, "global_slot": 19},
        created_at_ns=123,
    )


def test_training_replay_roundtrip_is_gzip_and_checksummed(
    tmp_path,
) -> None:
    replay = _replay()
    destination = replay.save(tmp_path / replay_filename(replay))

    payload = destination.read_bytes()
    assert payload.startswith(b"\x1f\x8b")
    restored = TrainingReplayV1.load(destination)
    assert restored == replay
    assert restored.start_native_tick == 130
    assert restored.end_native_tick == 140

    decoded = json.loads(gzip.decompress(payload))
    decoded["terminal"]["winner"] = 1
    destination.write_bytes(
        gzip.compress(
            json.dumps(decoded).encode("utf-8"),
            mtime=0,
        )
    )
    with pytest.raises(
        TrainingReplayError,
        match="checksum mismatch",
    ):
        TrainingReplayV1.load(destination)


def test_training_replay_listing_is_newest_completion_first(
    tmp_path,
) -> None:
    older = _replay(512)
    newer = _replay(1024)
    older.save(tmp_path / replay_filename(older))
    newer.save(tmp_path / replay_filename(newer))

    assert [path.name for path in list_training_replays(tmp_path)] == [
        replay_filename(newer),
        replay_filename(older),
    ]


def test_training_replay_save_is_idempotent_after_checkpoint_lag(
    tmp_path,
) -> None:
    original = _replay(512)
    destination = save_training_replay(original, tmp_path)
    recaptured = TrainingReplayV1(
        completed_match_index=original.completed_match_index,
        ruleset_id=original.ruleset_id,
        episode_config=original.episode_config,
        episode_id=original.episode_id,
        operations=original.operations,
        terminal=original.terminal,
        source=original.source,
        created_at_ns=999,
    )

    assert save_training_replay(recaptured, tmp_path) == destination
    assert TrainingReplayV1.load(destination) == original


def test_native_render_deploy_delay_preserves_command_age_window() -> None:
    assert (
        _native_deploy_delay(
            current_tick=245,
            target_tick=286,
        )
        == 20
    )

    with pytest.raises(
        TrainingReplayError,
        match="missed its queueing window",
    ):
        _native_deploy_delay(
            current_tick=265,
            target_tick=286,
        )


def test_replay_pause_waits_at_operation_boundary_and_resumes() -> None:
    import threading
    import time

    class _Native:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def pause(self) -> None:
            self.calls.append("pause")

        def resume(self) -> None:
            self.calls.append("resume")

    native = _Native()
    pause_event = threading.Event()
    pause_event.set()

    def release() -> None:
        time.sleep(0.02)
        pause_event.clear()

    thread = threading.Thread(target=release)
    thread.start()
    try:
        assert not _wait_while_paused(
            native,
            pause_event=pause_event,
            stop_event=None,
        )
    finally:
        thread.join()

    assert native.calls == ["pause", "resume"]




def test_render_environment_uses_retained_environment_configuration() -> None:
    source = _replay()
    retained_environment = EnvironmentConfigV1(
        shaping_beta=0.05,
        shaping_tau_ticks=16_664.16654164975,
    )
    episode = EpisodeConfigV1(
        ruleset_id=source.ruleset_id,
        deck0=source.episode_config.deck0,
        deck1=source.episode_config.deck1,
        seed=source.episode_config.seed,
        decision_hz=source.episode_config.decision_hz,
        event_driven_decisions=source.episode_config.event_driven_decisions,
        environment=retained_environment,
    )
    replay = TrainingReplayV1(
        completed_match_index=source.completed_match_index,
        ruleset_id=source.ruleset_id,
        episode_config=episode,
        episode_id=source.episode_id,
        operations=source.operations,
        terminal=source.terminal,
        source=source.source,
        created_at_ns=source.created_at_ns,
    )

    environment = _replay_environment(replay, native=object())

    assert environment.environment_config == retained_environment


def test_direct_renderer_extracts_sparse_timed_commands() -> None:
    replay = _replay()
    play = ActionV1.play(
        0,
        2,
        (6, 10),
        card_id=26_000_010,
        execute_offset_ticks=5,
        subcell_offset=(0.25, -0.5),
    )
    ability = ActionV1(
        owner=1,
        kind="activate_ability",
        source_entity=123,
        execute_offset_ticks=3,
        metadata={
            "ability_runtime_hints": ["bowler"],
            "ability_source_names": ["Bowler"],
        },
    )
    operation = SnapshotOperationV1(
        actions=(play, ability),
        advance_ticks=5,
        requested_advance_ticks=5,
        start_native_tick=130,
        end_native_tick=135,
    )
    replay = TrainingReplayV1(
        completed_match_index=replay.completed_match_index,
        ruleset_id=replay.ruleset_id,
        episode_config=replay.episode_config,
        episode_id=replay.episode_id,
        operations=(operation,),
        terminal=replay.terminal,
        source=replay.source,
        created_at_ns=replay.created_at_ns,
    )

    commands = _direct_render_commands(replay)

    assert [item.kind for item in commands] == [
        "ability",
        "play_card",
    ]
    assert commands[0].target_tick == 133
    assert commands[0].ability_name_hints == ("bowler",)
    assert commands[0].ability_source_labels == ("Bowler",)
    assert commands[1].target_tick == 135
    assert (commands[1].x, commands[1].y) == (6_750, 10_000)








def test_direct_renderer_requires_unique_ready_for_ambiguous_ability_cards() -> None:
    replay = _replay()
    ability = ActionV1(
        owner=1,
        kind="activate_ability",
        source_entity=0,
        execute_offset_ticks=3,
        metadata={
            "ability_runtime_hints": ["minipekk", "deflect"],
            "ability_source_keys": ["mini-pekka-hero", "monk"],
            "ability_source_names": ["Mini P.E.K.K.A", "Monk"],
        },
    )
    operation = SnapshotOperationV1(
        actions=(ability,),
        advance_ticks=3,
        requested_advance_ticks=3,
        start_native_tick=130,
        end_native_tick=133,
    )
    replay = TrainingReplayV1(
        completed_match_index=replay.completed_match_index,
        ruleset_id=replay.ruleset_id,
        episode_config=replay.episode_config,
        episode_id=replay.episode_id,
        operations=(operation,),
        terminal=replay.terminal,
        source=replay.source,
        created_at_ns=replay.created_at_ns,
    )

    command = _direct_render_commands(replay)[0]

    assert command.ability_name_hints == ()
    assert command.ability_source_labels == ("Mini P.E.K.K.A", "Monk")


def test_direct_renderer_preserves_source_order_for_same_tick_actions() -> None:
    replay = _replay()
    operation = SnapshotOperationV1(
        actions=(
            ActionV1.play(
                1,
                0,
                (7, 10),
                card_id=DECK[0],
                execute_offset_ticks=5,
                metadata={"source_index": 8},
            ),
            ActionV1.play(
                0,
                0,
                (6, 10),
                card_id=DECK[1],
                execute_offset_ticks=5,
                metadata={"source_index": 9},
            ),
        ),
        advance_ticks=5,
        requested_advance_ticks=5,
        start_native_tick=130,
        end_native_tick=135,
    )
    replay = TrainingReplayV1(
        completed_match_index=replay.completed_match_index,
        ruleset_id=replay.ruleset_id,
        episode_config=replay.episode_config,
        episode_id=replay.episode_id,
        operations=(operation,),
        terminal=replay.terminal,
        source=replay.source,
        created_at_ns=replay.created_at_ns,
    )

    commands = _direct_render_commands(replay)

    assert [(command.owner, command.card_id) for command in commands] == [
        (1, DECK[0]),
        (0, DECK[1]),
    ]


def test_direct_renderer_registers_ability_without_per_action_braking() -> None:
    replay = _replay()
    ability = ActionV1(
        owner=1,
        kind="activate_ability",
        source_entity=123,
        execute_offset_ticks=7,
        metadata={
            "ability_runtime_hints": ["wizard"],
            "ability_source_names": ["Wizard"],
        },
    )
    operation = SnapshotOperationV1(
        actions=(ability,),
        advance_ticks=7,
        requested_advance_ticks=7,
        start_native_tick=100,
        end_native_tick=107,
    )
    replay = TrainingReplayV1(
        completed_match_index=replay.completed_match_index,
        ruleset_id=replay.ruleset_id,
        episode_config=replay.episode_config,
        episode_id=replay.episode_id,
        operations=(operation,),
        terminal=replay.terminal,
        source=replay.source,
        created_at_ns=replay.created_at_ns,
    )

    class _RichLatencyNative:
        def __init__(self) -> None:
            self.tick = 39
            self.running = False
            self.speed = 1.0
            self.queued: list[tuple[int, int]] = []
            self.pause_calls = 0
            self.speed_calls: list[float] = []

        def create_native_match(self, _match: object) -> None:
            return None

        def pause(self) -> None:
            self.running = False
            self.pause_calls += 1

        def set_speed(self, speed: float) -> None:
            self.speed = speed
            self.speed_calls.append(speed)

        def resume(self) -> None:
            self.running = True

        def status(self) -> dict[str, object]:
            if self.running and self.tick < 107:
                self.tick = min(107, self.tick + (2 if self.speed == 4.0 else 1))
            return {"tick": self.tick, "ended": False}

        def _request(self, command: str) -> dict[str, object]:
            raise AssertionError(f"unexpected telemetry request: {command}")

        def clear_replay_schedule(self) -> None:
            self.queued.clear()

        def schedule_replay_ability_at_tick(
            self,
            *,
            owner: int,
            ability_name_hints: tuple[str, ...],
            execute_tick: int,
        ) -> dict[str, object]:
            assert owner == 1
            assert ability_name_hints == ("wizard",)
            self.queued.append((self.tick, execute_tick))
            return {
                "ok": True,
                "sequence": 1,
                "registeredAtTick": self.tick,
                "executeTick": execute_tick,
            }

        def replay_schedule_status(self, sequence: int) -> dict[str, object]:
            assert sequence == 1
            assert self.tick >= 107
            return {
                "state": "succeeded",
                "error": "none",
                "queuedAtTick": 106,
                "executeTick": 107,
            }

    native = _RichLatencyNative()
    result = _play_training_replay_direct(
        replay,
        native=native,  # type: ignore[arg-type]
        speed=4.0,
        strict_ticks=True,
        stop_at_replay_end=True,
        stop_event=None,
        pause_event=None,
        on_native_ready=None,
        on_progress=None,
    )

    assert native.queued == [(39, 107)]
    assert native.speed == 4.0
    assert native.speed_calls == [4.0]
    assert native.pause_calls == 2
    assert result.completion == "source-timeline"


def test_direct_renderer_defers_late_ready_ability_resolution_to_native() -> None:
    replay = _replay()
    ability = ActionV1(
        owner=1,
        kind="activate_ability",
        source_entity=123,
        execute_offset_ticks=7,
        metadata={
            "ability_runtime_hints": ["mini-pekka"],
            "ability_source_names": ["Mini P.E.K.K.A"],
        },
    )
    operation = SnapshotOperationV1(
        actions=(ability,),
        advance_ticks=7,
        requested_advance_ticks=7,
        start_native_tick=100,
        end_native_tick=107,
    )
    replay = TrainingReplayV1(
        completed_match_index=replay.completed_match_index,
        ruleset_id=replay.ruleset_id,
        episode_config=replay.episode_config,
        episode_id=replay.episode_id,
        operations=(operation,),
        terminal=replay.terminal,
        source=replay.source,
        created_at_ns=replay.created_at_ns,
    )

    class _LateReadyNative:
        def __init__(self) -> None:
            self.tick = 39
            self.running = False
            self.queued: list[tuple[int, int]] = []

        def create_native_match(self, _match: object) -> None:
            return None

        def pause(self) -> None:
            self.running = False

        def set_speed(self, _speed: float) -> None:
            return None

        def resume(self) -> None:
            self.running = True

        def status(self) -> dict[str, object]:
            if self.running and self.tick < 107:
                self.tick += 1
            return {"tick": self.tick, "ended": False}

        def clear_replay_schedule(self) -> None:
            self.queued.clear()

        def schedule_replay_ability_at_tick(
            self,
            *,
            owner: int,
            ability_name_hints: tuple[str, ...],
            execute_tick: int,
        ) -> dict[str, object]:
            assert owner == 1
            assert ability_name_hints == ("mini-pekka",)
            self.queued.append((self.tick, execute_tick))
            return {
                "ok": True,
                "sequence": 1,
                "registeredAtTick": self.tick,
                "executeTick": execute_tick,
            }

        def replay_schedule_status(self, sequence: int) -> dict[str, object]:
            assert sequence == 1
            assert self.tick >= 107
            return {
                "state": "succeeded",
                "error": "none",
                "queuedAtTick": 106,
                "executeTick": 107,
            }

    native = _LateReadyNative()
    result = _play_training_replay_direct(
        replay,
        native=native,  # type: ignore[arg-type]
        speed=4.0,
        strict_ticks=True,
        stop_at_replay_end=True,
        stop_event=None,
        pause_event=None,
        on_native_ready=None,
        on_progress=None,
    )

    assert native.queued == [(39, 107)]
    assert result.completion == "source-timeline"


def test_direct_renderer_reports_expired_ability_at_source_tick() -> None:
    replay = _replay()
    ability = ActionV1(
        owner=1,
        kind="activate_ability",
        source_entity=123,
        execute_offset_ticks=7,
        metadata={"ability_runtime_hints": ["barblog"]},
    )
    operation = SnapshotOperationV1(
        actions=(ability,),
        advance_ticks=7,
        requested_advance_ticks=7,
        start_native_tick=100,
        end_native_tick=107,
    )
    replay = TrainingReplayV1(
        completed_match_index=replay.completed_match_index,
        ruleset_id=replay.ruleset_id,
        episode_config=replay.episode_config,
        episode_id=replay.episode_id,
        operations=(operation,),
        terminal=replay.terminal,
        source=replay.source,
        created_at_ns=replay.created_at_ns,
    )

    class _ExpiredAbilityNative:
        def __init__(self) -> None:
            self.tick = 39
            self.running = False

        def create_native_match(self, _match: object) -> None:
            return None

        def pause(self) -> None:
            self.running = False

        def set_speed(self, _speed: float) -> None:
            return None

        def resume(self) -> None:
            self.running = True

        def status(self) -> dict[str, object]:
            if self.running and self.tick < 107:
                self.tick += 1
            return {"tick": self.tick, "ended": False}

        def clear_replay_schedule(self) -> None:
            return None

        def schedule_replay_ability_at_tick(
            self,
            *,
            owner: int,
            ability_name_hints: tuple[str, ...],
            execute_tick: int,
        ) -> dict[str, object]:
            assert owner == 1
            assert ability_name_hints == ("barblog",)
            assert execute_tick == 107
            return {
                "ok": True,
                "sequence": 1,
                "registeredAtTick": self.tick,
                "executeTick": execute_tick,
            }

        def replay_schedule_status(self, sequence: int) -> dict[str, object]:
            assert sequence == 1
            assert self.tick >= 107
            return {
                "state": "failed",
                "error": "ability-unavailable",
                "queuedAtTick": 106,
                "executeTick": 107,
            }

    with pytest.raises(
        TrainingReplayError,
        match=r"retained ability for owner 1.*target tick 107",
    ):
        _play_training_replay_direct(
            replay,
            native=_ExpiredAbilityNative(),  # type: ignore[arg-type]
            speed=1.0,
            strict_ticks=True,
            stop_at_replay_end=True,
            stop_event=None,
            pause_event=None,
            on_native_ready=None,
            on_progress=None,
        )


class _SourceTimelineNative:
    def __init__(self, statuses: list[dict[str, object]]) -> None:
        self.statuses = iter(statuses)
        self.pause_calls = 0

    def create_native_match(self, _match: object) -> None:
        return None

    def pause(self) -> None:
        self.pause_calls += 1

    def set_speed(self, _speed: float) -> None:
        return None

    def resume(self) -> None:
        return None

    def status(self) -> dict[str, object]:
        return next(self.statuses)


def test_direct_renderer_stops_at_retained_source_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _SourceTimelineNative(
        [
            {"tick": 130, "ended": False},
            {"tick": 140, "ended": False},
        ]
    )
    monkeypatch.setattr(
        "native_runner.training.replay_viewer.time.sleep",
        lambda _seconds: None,
    )

    result = _play_training_replay_direct(
        _replay(),
        native=native,  # type: ignore[arg-type]
        speed=4.0,
        strict_ticks=True,
        stop_at_replay_end=True,
        stop_event=None,
        pause_event=None,
        on_native_ready=None,
        on_progress=None,
    )

    assert result.completion == "source-timeline"
    assert result.final_native_tick == 140
    assert result.source_end_tick == 140
    assert result.actual_winner is None
    assert native.pause_calls == 2


def test_direct_renderer_rewinds_native_state_and_replay_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = ReplayRewindControl()
    progress = []

    class _RewindNative:
        def __init__(self) -> None:
            self.tick = 0
            self.running = False
            self.requested = False
            self.next_handle = 1
            self.snapshots: dict[int, int] = {}
            self.restored_ticks: list[int] = []
            self.released_handles: list[int] = []

        def create_native_match(self, _match: object) -> None:
            return None

        def pause(self) -> None:
            self.running = False

        def set_speed(self, _speed: float) -> None:
            return None

        def resume(self) -> None:
            self.running = True

        def status(self) -> dict[str, object]:
            if self.running:
                self.tick += 20
            if self.tick >= 120 and not self.requested:
                assert control.request_rewind()
                self.requested = True
            return {"tick": self.tick, "ended": False}

        def create_snapshot(self) -> dict[str, object]:
            handle = self.next_handle
            self.next_handle += 1
            self.snapshots[handle] = self.tick
            return {"handle": handle, "tick": self.tick}

        def restore(self, snapshot: object) -> dict[str, object]:
            assert isinstance(snapshot, dict)
            self.tick = self.snapshots[int(snapshot["handle"])]
            self.restored_ticks.append(self.tick)
            return {"tick": self.tick}

        def release_snapshot(self, snapshot: object) -> None:
            assert isinstance(snapshot, dict)
            handle = int(snapshot["handle"])
            self.snapshots.pop(handle)
            self.released_handles.append(handle)

    native = _RewindNative()
    monkeypatch.setattr(
        "native_runner.training.replay_viewer.time.sleep",
        lambda _seconds: None,
    )

    result = _play_training_replay_direct(
        _replay(),
        native=native,  # type: ignore[arg-type]
        speed=4.0,
        strict_ticks=True,
        stop_at_replay_end=True,
        stop_event=None,
        pause_event=None,
        on_native_ready=None,
        on_progress=progress.append,
        rewind_control=control,
    )

    assert result.completion == "source-timeline"
    assert result.final_native_tick == 140
    assert native.restored_ticks == [20]
    assert not native.snapshots
    assert native.released_handles
    rewinds = [item for item in progress if item.phase == "rewound"]
    assert len(rewinds) == 1
    assert rewinds[0].native_tick == 20
    assert rewinds[0].expected_tick == 20


def test_direct_renderer_accepts_native_terminal_after_all_actions_queued() -> None:
    native = _SourceTimelineNative([{"tick": 135, "ended": True, "winner": 1}])

    result = _play_training_replay_direct(
        _replay(),
        native=native,  # type: ignore[arg-type]
        speed=4.0,
        strict_ticks=True,
        stop_at_replay_end=True,
        stop_event=None,
        pause_event=None,
        on_native_ready=None,
        on_progress=None,
    )

    assert result.completion == "native-terminal"
    assert result.final_native_tick == 135
    assert result.expected_winner == 0
    assert result.actual_winner == 1
    assert result.queued_action_count == result.action_count == 0


def test_direct_renderer_reports_native_terminal_with_unqueued_action() -> None:
    replay = _replay()
    operation = SnapshotOperationV1(
        actions=(
            ActionV1.play(
                0,
                0,
                (6, 10),
                card_id=DECK[0],
                execute_offset_ticks=5,
            ),
            ActionV1.wait(1, ticks=5),
        ),
        advance_ticks=5,
        requested_advance_ticks=5,
        start_native_tick=130,
        end_native_tick=135,
    )
    replay = TrainingReplayV1(
        completed_match_index=replay.completed_match_index,
        ruleset_id=replay.ruleset_id,
        episode_config=replay.episode_config,
        episode_id=replay.episode_id,
        operations=(operation,),
        terminal=replay.terminal,
        source=replay.source,
        created_at_ns=replay.created_at_ns,
    )
    native = _SourceTimelineNative([{"tick": 130, "ended": True, "winner": 1}])

    with pytest.raises(
        TrainingReplayError,
        match=r"ended early at tick 130.*queued 0/1 actions",
    ):
        _play_training_replay_direct(
            replay,
            native=native,  # type: ignore[arg-type]
            speed=4.0,
            strict_ticks=True,
            stop_at_replay_end=True,
            stop_event=None,
            pause_event=None,
            on_native_ready=None,
            on_progress=None,
        )


def test_replay_match_config_preserves_native_inputs() -> None:
    replay = _replay()
    match = _match_config_from_replay(replay)

    assert match.deck0 == replay.episode_config.deck0
    assert match.deck1 == replay.episode_config.deck1
    assert match.seed == replay.episode_config.seed
