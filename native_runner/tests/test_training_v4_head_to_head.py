"""Head-to-head evaluation tooling: opening hook, schedule, gate offset, reports, aggregation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch

from native_runner.contracts import ActionKind, TerminalV1
from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.tests.test_training_v4_ppo_matchmaking import _deck, _manifest
from native_runner.training.v4 import aggregate_head_to_head as aggregate
from native_runner.training.v4 import async_cluster_self_play as collector
from native_runner.training.v4 import evaluate_head_to_head as h2h
from native_runner.training.v4 import make_head_to_head_schedule as schedule_module
from native_runner.training.v4.cluster_self_play import ClusterMatchAssignmentV4, _MatchRuntimeV4
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.league import LeagueEntryV4
from native_runner.training.v4.matchmaking import MATCH_HARD, POLICY_CURRENT, POLICY_HISTORY, POLICY_IL, ScheduledMatchupV4
from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT


# ---------------------------------------------------------------------------
# Deck-rule opening force: pure function
# ---------------------------------------------------------------------------


def deck_rule(**overrides: Any) -> int | None:
    kwargs: dict[str, Any] = dict(
        active_actions=(0, 0),
        deck_average_elixir=(3.8, 2.9),
        deck_ids=("deck-a", "deck-b"),
        tick=300,
        force_tick=300,
        already_forced_owner=None,
    )
    kwargs.update(overrides)
    return collector._deck_rule_opening_force_owner_v4(**kwargs)


@pytest.mark.parametrize(("averages", "owner"), (((3.8, 2.9), 1), ((2.9, 3.8), 0)))
def test_deck_rule_forces_the_lower_elixir_deck_whatever_the_seat(averages, owner) -> None:
    assert deck_rule(deck_average_elixir=averages, tick=299) is None
    assert deck_rule(deck_average_elixir=averages, tick=300) == owner
    assert deck_rule(deck_average_elixir=averages, tick=3000) == owner


@pytest.mark.parametrize("acted", ((1, 0), (0, 1), (2, 5)))
def test_deck_rule_never_forces_once_either_side_acted(acted) -> None:
    for tick in (300, 600, 5990):
        assert deck_rule(active_actions=acted, tick=tick) is None


def test_deck_rule_is_disabled_by_default_and_fires_once() -> None:
    assert deck_rule(force_tick=None, tick=6000) is None
    assert deck_rule(already_forced_owner=1) is None


def test_deck_rule_tie_breaks_are_deck_properties() -> None:
    assert deck_rule(deck_average_elixir=(3.5, 3.5), deck_ids=("b", "a")) == 1
    assert deck_rule(deck_average_elixir=(3.5, 3.5), deck_ids=("a", "b")) == 0
    assert deck_rule(deck_average_elixir=(3.5, 3.5), deck_ids=("a", "a")) == 0


@pytest.mark.parametrize("bad", (dict(force_tick=0), dict(force_tick=-5), dict(deck_average_elixir=(0.0, 3.0))))
def test_deck_rule_rejects_invalid_inputs(bad) -> None:
    with pytest.raises(ValueError):
        deck_rule(**bad)


# ---------------------------------------------------------------------------
# Deck-rule opening force: driven through the real _engine_worker with fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeAction:
    kind: ActionKind
    card_id: int | None = None
    action_id: str | None = None


class FakeCoordinator:
    def start_session(self) -> None:
        pass

    def close_session(self) -> None:
        pass

    def begin_round(self, _ids: object) -> None:
        pass

    def finish_round(self) -> None:
        pass

    def abort_round(self, _error: object) -> None:
        pass

    def profile(self) -> dict[str, int]:
        return {}


class FakeEngine:
    engine_index = 0

    def __init__(self) -> None:
        self.coordinator = FakeCoordinator()

    def prepare(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeTensorizer:
    def __init__(self, batch: Any) -> None:
        self.batch = batch
        self.config = ModelConfigV4()
        self.catalog = None
        self.deck = ()
        self.card_costs: dict[int, float] = {}
        self.ability_id_by_vocab_id: dict[int, int] = {}
        self.perspective = SimpleNamespace(horizontal_mirror=False, hand_slot_permutation=None)

    def tensorize(self, _observation: object, *, validate: bool) -> Any:
        return self.batch

    def record_action(self, *_args: object, **_kwargs: object) -> None:
        pass


def _observation(tick: int, end_tick: int) -> SimpleNamespace:
    ended = tick >= end_tick
    towers = (
        SimpleNamespace(owner=1, tower_kind="king", hitpoints=4000.0, max_hitpoints=4824.0, active=True),
        SimpleNamespace(owner=0, tower_kind="princess_left", hitpoints=1200.0, max_hitpoints=3052.0, active=True),
        SimpleNamespace(owner=0, tower_kind="king", hitpoints=4824.0, max_hitpoints=4824.0, active=True),
    )
    return SimpleNamespace(
        tick=tick,
        players=(
            SimpleNamespace(owner=0, elixir_exact=5.0, crowns=1 if ended else 0),
            SimpleNamespace(owner=1, elixir_exact=5.0, crowns=0),
        ),
        events=(),
        towers=towers,
        truncated=False,
        terminal=(
            TerminalV1(ended=True, winner=0, result_by_owner=(1.0, -1.0), reason="crown_advantage", terminal_tick=tick)
            if ended
            else TerminalV1()
        ),
    )


class FakeEnvironment:
    def __init__(self, state: SimpleNamespace) -> None:
        self.state = state

    def step(self, decoded: dict[int, tuple[Any, ...]], *, record_trace: bool, policy_features_only: bool) -> Any:
        self.state.submitted.append({owner: tuple(item.kind for item in items) for owner, items in decoded.items()})
        self.state.tick += 5
        ended = self.state.tick >= self.state.end_tick
        observation = _observation(self.state.tick, self.state.end_tick)
        return ({0: observation, 1: observation}, {0: 0.0, 1: 0.0}, {0: ended, 1: ended}, {0: False, 1: False}, {})


class ScriptedConnection:
    """Plays the collector's side: WAIT unless scripted, honouring force requests like the GPU path."""

    def __init__(self, state: SimpleNamespace, acts: Callable[[int, int], bool]) -> None:
        self.state = state
        self.acts = acts
        self.sent: list[dict[str, Any]] = []
        self.frames: list[tuple[int, tuple[int, ...]]] = []

    def send(self, packet: dict[str, Any]) -> None:
        self.sent.append(packet)
        if packet["kind"] == "frame":
            self.frames.append((self.state.tick, tuple(packet["force_act_actor_rows"])))

    def recv(self) -> dict[str, Any]:
        packet = self.sent[-1]
        if packet.get("segment_boundary"):
            return {"kind": "resume", "policy_state_id": "resume"}
        rows = tuple(int(value) for value in packet["actor_rows"])
        requested = {int(value) for value in packet["force_act_actor_rows"]}
        packed = torch.zeros((len(rows), 10), dtype=torch.long)
        forced = []
        for position, local_actor in enumerate(rows):
            act = self.acts(self.state.tick, local_actor % 2)
            if local_actor in requested and not act:
                act = True
                forced.append(local_actor)
            if act:
                packed[position, 0] = GATE_ACT
                packed[position, 1] = 1
        return {
            "kind": "act_packed",
            "actor_rows": rows,
            "packed_actions": packed,
            "forced_act_actor_rows": tuple(forced),
        }

    def close(self) -> None:
        pass


def _fake_decode(action: Any, *_args: object, **_kwargs: object) -> SimpleNamespace:
    if int(action.gate[0]) == GATE_ACT:
        return SimpleNamespace(actions=(FakeAction(ActionKind.PLAY_CARD, card_id=26000000),))
    return SimpleNamespace(actions=(FakeAction(ActionKind.WAIT),))


def run_fake_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec_overrides: dict[str, Any] | None = None,
    acts: Callable[[int, int], bool] = lambda _tick, _owner: False,
    deck_average_elixir: tuple[float, float] = (3.8, 2.9),
    policy_matchup: str = POLICY_IL,
    end_tick: int = 420,
    reward_overrides: dict[str, Any] | None = None,
) -> tuple[ScriptedConnection, SimpleNamespace]:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(collector, "ctypes", SimpleNamespace(CDLL=lambda _name: SimpleNamespace(prctl=lambda *a: 0)))
    monkeypatch.setattr(collector.os, "sched_setaffinity", lambda *_args: None, raising=False)
    monkeypatch.setattr(collector.sys, "setswitchinterval", lambda _value: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _value: None)
    monkeypatch.setattr(collector, "decode_action_sequence_v4", _fake_decode)
    captured: dict[str, Any] = {}

    def build_engine(engine_index: int, *, environment_config: Any, **_kwargs: object) -> FakeEngine:
        captured["build_environment_config"] = environment_config
        return FakeEngine()

    state = SimpleNamespace(tick=0, end_tick=end_tick, submitted=[], captured=captured)
    batch = _batch(2).narrow_batch(0, 1)
    opponent = None
    if policy_matchup == POLICY_HISTORY:
        opponent = LeagueEntryV4(checkpoint_id="frozen", checkpoint_path="frozen.pt", update_step=0)
    matchup = ScheduledMatchupV4(
        sequence=7,
        category="focus-focus",
        policy_matchup=policy_matchup,  # type: ignore[arg-type]
        deck0=_deck(1, focus_rank=1),
        deck1=_deck(2, focus_rank=2),
        current_owners=(0, 1) if policy_matchup == POLICY_CURRENT else (0,),
        battle_level=11,
        seed=99,
    )
    assignment = ClusterMatchAssignmentV4(matchup, opponent=opponent)

    def reset_engine(engine: Any, assignments: Any, environment_config: Any) -> tuple[_MatchRuntimeV4, ...]:
        captured["reset_environment_config"] = environment_config
        state.tick = int(environment_config.warmup_ticks)
        observation = _observation(state.tick, end_tick)
        return (
            _MatchRuntimeV4(
                engine=engine,
                env_id=0,
                environment=FakeEnvironment(state),  # type: ignore[arg-type]
                assignment=assignments[0],
                observations={0: observation, 1: observation},
                tensorizers=(FakeTensorizer(batch), FakeTensorizer(batch)),
                deck_average_elixir=deck_average_elixir,
            ),
        )

    monkeypatch.setattr(collector, "build_resident_engine_v4", build_engine)
    monkeypatch.setattr(collector, "_reset_engine", reset_engine)
    connection = ScriptedConnection(state, acts)
    spec = replace(
        collector.AsyncResidentEngineSpecV4(engine_index=0, cpu=0, lanes=1, segment_decision_steps=10_000),
        **(spec_overrides or {}),
    )
    rewards = dict(
        hog_deploy_reward=0.0,
        hog_deploy_reward_episode_cap=0.1,
        hog_deploy_opening_reward=None,
        first_hog_timing_reward_max=0.0,
        first_hog_timing_start_seconds=10.0,
        first_hog_timing_deadline_seconds=30.0,
        first_hog_missed_deadline_penalty=0.0,
        first_hog_deferral_penalty=0.0,
        first_hog_deferral_penalty_episode_cap=0.0,
    )
    rewards.update(reward_overrides or {})
    collector._engine_worker(
        connection,  # type: ignore[arg-type]
        spec,
        (assignment,),
        0.9997,
        0.05,
        0.0,
        4.0,
        0.0,
        0.1,
        "fake-policy",
        False,
        2_000,
        **rewards,
    )
    return connection, state


def _done_report(connection: ScriptedConnection) -> dict[str, Any]:
    done = connection.sent[-1]
    assert done["kind"] == "done"
    (report,) = done["reports"]
    return report


@pytest.mark.parametrize(("averages", "forced_owner"), (((2.9, 3.8), 0), ((3.8, 2.9), 1)))
def test_worker_deck_rule_forces_lower_elixir_side_once_when_neither_acted(monkeypatch, averages, forced_owner):
    connection, _state = run_fake_worker(
        monkeypatch, spec_overrides=dict(force_deck_rule_opening_tick=300), deck_average_elixir=averages
    )
    requests = [(tick, rows) for tick, rows in connection.frames if rows]
    # The first decision at/after tick 300 asks for exactly one ACT from the lower-elixir deck, and never again.
    assert requests == [(300, (forced_owner,))]
    assert all(tick >= 300 or not rows for tick, rows in connection.frames)
    report = _done_report(connection)
    assert report["opening_forced_owner"] == forced_owner
    assert report["opening_forced_tick"] == 300
    assert report["active_actions"][forced_owner] == 1
    assert report["active_actions"][1 - forced_owner] == 0


@pytest.mark.parametrize("acting_owner", (0, 1))
def test_worker_deck_rule_never_forces_after_any_action(monkeypatch, acting_owner):
    connection, _state = run_fake_worker(
        monkeypatch,
        spec_overrides=dict(force_deck_rule_opening_tick=300),
        acts=lambda tick, owner: tick == 200 and owner == acting_owner,
    )
    assert all(not rows for _tick, rows in connection.frames)
    report = _done_report(connection)
    assert report["opening_forced_owner"] is None
    assert report["active_actions"][acting_owner] == 1


def test_worker_default_spec_never_forces_and_legacy_il_force_is_unchanged(monkeypatch):
    connection, _state = run_fake_worker(monkeypatch)
    assert all(not rows for _tick, rows in connection.frames)
    assert _done_report(connection)["opening_forced_owner"] is None
    # The pre-existing IL-only rule still forces only the frozen anchor (owner 1), even though owner 1 holds the
    # higher-elixir deck here, proving the new hook does not alter the legacy path.
    legacy, _state = run_fake_worker(
        monkeypatch, spec_overrides=dict(force_il_opponent_opening_after_seconds=15), deck_average_elixir=(2.9, 3.8)
    )
    assert [(tick, rows) for tick, rows in legacy.frames if rows] == [(300, (1,))]


def test_worker_rejects_mixing_deck_rule_with_role_based_force(monkeypatch):
    with pytest.raises(ValueError, match="deck-rule opening force cannot be combined"):
        run_fake_worker(
            monkeypatch,
            spec_overrides=dict(force_deck_rule_opening_tick=300, force_il_opponent_opening_after_seconds=15),
        )
    with pytest.raises(ValueError, match="deck-rule opening force tick"):
        run_fake_worker(monkeypatch, spec_overrides=dict(force_deck_rule_opening_tick=0))


def test_worker_deck_rule_applies_to_history_opponents_and_current_alike(monkeypatch):
    connection, _state = run_fake_worker(
        monkeypatch,
        spec_overrides=dict(force_deck_rule_opening_tick=300),
        deck_average_elixir=(2.9, 3.8),
        policy_matchup=POLICY_HISTORY,
    )
    assert [(tick, rows) for tick, rows in connection.frames if rows] == [(300, (0,))]


def test_worker_warmup_override_reaches_engine_and_reset(monkeypatch):
    _connection, state = run_fake_worker(monkeypatch, spec_overrides=dict(environment_warmup_ticks=90))
    assert state.captured["build_environment_config"].warmup_ticks == 90
    assert state.captured["reset_environment_config"].warmup_ticks == 90
    _connection, default_state = run_fake_worker(monkeypatch)
    assert default_state.captured["reset_environment_config"].warmup_ticks == 130


def test_worker_skip_shaping_telemetry_avoids_shadow_legality(monkeypatch):
    calls = []
    original = collector.ShadowCandidateLegality

    class CountingLegality(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(collector, "ShadowCandidateLegality", CountingLegality)
    run_fake_worker(monkeypatch, spec_overrides=dict(skip_shaping_telemetry=True))
    assert not calls
    run_fake_worker(monkeypatch)
    assert calls
    with pytest.raises(ValueError, match="shaping telemetry can only be skipped"):
        run_fake_worker(
            monkeypatch,
            spec_overrides=dict(skip_shaping_telemetry=True),
            reward_overrides=dict(first_hog_deferral_penalty=0.01, first_hog_deferral_penalty_episode_cap=0.1),
        )


def test_terminal_report_exposes_reason_winner_and_tower_hitpoints(monkeypatch):
    connection, _state = run_fake_worker(monkeypatch, end_tick=150)
    report = _done_report(connection)
    assert report["terminal_ended"] is True
    assert report["terminal_reason"] == "crown_advantage"
    assert report["terminal_winner"] == 0
    assert report["terminal_reported_tick"] == 150
    assert report["crowns"] == (1, 0)
    assert [(row["owner"], row["tower_kind"]) for row in report["final_tower_hitpoints"]] == [
        (0, "king"),
        (0, "princess_left"),
        (1, "king"),
    ]
    assert report["final_tower_hitpoints"][1]["hitpoints"] == 1200.0


def test_new_spec_fields_default_off_and_survive_remote_transport(monkeypatch):
    import pickle

    from native_runner.training.v4 import remote_ppo_worker_server as server

    default = collector.AsyncResidentEngineSpecV4(engine_index=0, cpu=0)
    assert default.force_deck_rule_opening_tick is None
    assert default.environment_warmup_ticks is None
    assert default.skip_shaping_telemetry is False
    spec = collector.AsyncResidentEngineSpecV4(
        engine_index=0,
        cpu=0,
        force_deck_rule_opening_tick=300,
        environment_warmup_ticks=90,
        skip_shaping_telemetry=True,
    )
    spec = pickle.loads(pickle.dumps(replace(spec, segment_decision_steps=160)))
    captured = []
    monkeypatch.setattr(server, "_prepare_worker_process", lambda _: None)
    monkeypatch.setattr(server, "_engine_worker", lambda *args, **kwargs: captured.append(args))
    request = dict(
        spec=spec,
        assignments=(),
        gamma_per_decision=0.9997,
        shaping_beta=0.05,
        mutual_elixir_overflow_penalty=0.0,
        mutual_elixir_overflow_grace=4.0,
        unilateral_elixir_overflow_penalty=0.0,
        elixir_overflow_step_penalty_cap=0.1,
        policy_state_id="test",
        validate_tensors=False,
        max_decision_steps=1444,
    )
    server._run_worker(SimpleNamespace(), request, -1)
    forwarded = captured[0][1]
    assert (forwarded.force_deck_rule_opening_tick, forwarded.environment_warmup_ticks) == (300, 90)
    assert forwarded.skip_shaping_telemetry is True


# ---------------------------------------------------------------------------
# Schedule generator
# ---------------------------------------------------------------------------


def _schedule(configurations: int = 40, seed: int = 11, **kwargs: Any) -> list[dict[str, Any]]:
    config = schedule_module.HeadToHeadScheduleConfigV4(configurations=configurations, seed=seed, **kwargs)
    rows, _statistics = schedule_module.generate_head_to_head_schedule_v4(_manifest(), config)
    return rows


def test_schedule_pairs_swap_models_and_keep_decks_with_seats() -> None:
    rows = _schedule()
    assert len(rows) == 80
    for config_id in range(40):
        first, second = rows[2 * config_id], rows[2 * config_id + 1]
        assert first["config_id"] == second["config_id"] == config_id
        for key in ("seed", "deck0_id", "deck1_id", "battle_level", "category", "deck0", "deck1"):
            assert first[key] == second[key]
        assert (first["owner0_slot"], first["owner1_slot"]) == ("A", "B")
        assert (second["owner0_slot"], second["owner1_slot"]) == ("B", "A")
        assert first["deck0_id"] != first["deck1_id"]
    assert MATCH_HARD not in {row["category"] for row in rows}
    assert len({row["seed"] for row in rows}) == 40


def test_schedule_levels_are_exact_half_split_or_pinned() -> None:
    mixed = _schedule(configurations=40)
    assert sum(row["battle_level"] == 11 for row in mixed) == 40
    assert sum(row["battle_level"] == 16 for row in mixed) == 40
    assert {row["battle_level"] for row in _schedule(battle_level="11")} == {11}
    assert {row["battle_level"] for row in _schedule(battle_level="16")} == {16}


def test_schedule_is_reproducible_by_seed() -> None:
    assert _schedule(seed=5) == _schedule(seed=5)
    assert _schedule(seed=5) != _schedule(seed=6)


def test_schedule_validator_rejects_broken_pairs() -> None:
    rows = _schedule(configurations=4)
    broken = [dict(row) for row in rows]
    broken[1]["seed"] = broken[1]["seed"] + 1
    with pytest.raises(ValueError, match="disagree"):
        schedule_module.validate_head_to_head_schedule_v4(broken)
    unswapped = [dict(row) for row in rows]
    unswapped[1]["owner0_slot"], unswapped[1]["owner1_slot"] = "A", "B"
    with pytest.raises(ValueError, match="swap"):
        schedule_module.validate_head_to_head_schedule_v4(unswapped)
    with pytest.raises(ValueError, match="complete"):
        schedule_module.validate_head_to_head_schedule_v4(rows[:-1])
    with pytest.raises(ValueError):
        schedule_module.HeadToHeadScheduleConfigV4(configurations=4, seed=1, deck_mix=(30, 30, 30, 20))


def test_schedule_cli_writes_rows_and_sidecar(monkeypatch, tmp_path) -> None:
    pool = tmp_path / "pool.json"
    pool.write_text("{}", encoding="utf-8")
    manifest = _manifest()
    monkeypatch.setattr(schedule_module, "DeckPoolManifestV4", SimpleNamespace(load=lambda _path: manifest))
    output = tmp_path / "h2h" / "schedule.jsonl"
    args = ["--deck-pool", str(pool), "--output", str(output), "--configurations", "6", "--seed", "3"]
    assert schedule_module.main(args) == 0
    rows = schedule_module.load_head_to_head_schedule_v4(output)
    assert len(rows) == 12
    meta = json.loads(output.with_name("schedule.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["parameters"]["seed"] == 3 and meta["parameters"]["deck_mix"]["focus_tail_hard"] == 0
    assert meta["statistics"]["rows"] == 12
    with pytest.raises(ValueError, match="refusing to overwrite"):
        schedule_module.main(args)


# ---------------------------------------------------------------------------
# Model spec parsing and gate logit offset
# ---------------------------------------------------------------------------


def test_model_spec_parsing_handles_options_and_drive_letters() -> None:
    spec = h2h.parse_model_spec_v4(r"C1=D:\ckpt\c1.pt:gate_logit_offset=-2.0794:mode=argmax")
    assert spec.name == "C1" and spec.path == Path(r"D:\ckpt\c1.pt")
    assert spec.gate_logit_offset == pytest.approx(-2.0794) and spec.mode == "argmax"
    plain = h2h.parse_model_spec_v4("IL=/gpfs/IL.pt")
    assert (plain.gate_logit_offset, plain.mode) == (0.0, None)
    reordered = h2h.parse_model_spec_v4("X=/a/b.pt:mode=sample:gate_logit_offset=1.5")
    assert (reordered.mode, reordered.gate_logit_offset, reordered.path) == ("sample", 1.5, Path("/a/b.pt"))
    for bad in ("noequals", "X=/a.pt:mode=greedy", "X=/a.pt:gate_logit_offset=abc", "bad name=/a.pt"):
        with pytest.raises(ValueError):
            h2h.parse_model_spec_v4(bad)


def test_pair_parsing_round_robin_and_validation() -> None:
    names = ("A", "B", "C")
    assert h2h.parse_pairs_v4([], names, round_robin=True) == (("A", "B"), ("A", "C"), ("B", "C"))
    assert h2h.parse_pairs_v4(["A:B,C:A"], names, round_robin=False) == (("A", "B"), ("C", "A"))
    for bad in (["A:A"], ["A:D"], ["A:B", "B:A"], ["A-B"]):
        with pytest.raises(ValueError):
            h2h.parse_pairs_v4(bad, names, round_robin=False)


def _gate_logits(model: Any, batch: Any) -> torch.Tensor:
    state = model.initial_state(batch.batch_size, device="cpu")
    with torch.no_grad():
        context = model.forward(
            batch, state, episode_start=torch.ones(batch.batch_size, dtype=torch.bool), validate=False
        )
        return model.gate_head(context.policy_context)


def test_gate_logit_offset_shifts_only_act_log_odds_once() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    batch = _batch(2)
    gate = model.gate_head[-1]
    bias_before = gate.bias.detach().clone()
    weight_before = gate.weight.detach().clone()
    logits_before = _gate_logits(model, batch)
    offset = -math.log(8.0)
    h2h.apply_gate_logit_offset_v4(model, offset)
    logits_after = _gate_logits(model, batch)
    torch.testing.assert_close(gate.weight, weight_before, rtol=0, atol=0)
    assert float(gate.bias[GATE_WAIT]) == float(bias_before[GATE_WAIT])
    assert float(gate.bias[GATE_ACT]) == pytest.approx(float(bias_before[GATE_ACT]) + offset, abs=1e-6)
    torch.testing.assert_close(logits_after[:, GATE_WAIT], logits_before[:, GATE_WAIT])
    torch.testing.assert_close(logits_after[:, GATE_ACT] - logits_before[:, GATE_ACT], torch.full((2,), offset))
    p_before = torch.softmax(logits_before, dim=-1)[:, GATE_ACT]
    p_after = torch.softmax(logits_after, dim=-1)[:, GATE_ACT]
    odds_ratio = (p_after / (1 - p_after)) / (p_before / (1 - p_before))
    torch.testing.assert_close(odds_ratio, torch.full((2,), 1.0 / 8.0), rtol=1e-5, atol=1e-6)
    with pytest.raises(RuntimeError, match="already applied"):
        h2h.apply_gate_logit_offset_v4(model, offset)
    with pytest.raises(ValueError):
        h2h.apply_gate_logit_offset_v4(_model(), float("nan"))


def test_decision_modes_set_sampling_and_temperatures() -> None:
    model = _model()
    model.set_ppo_gate_temperature(0.2)
    model.set_ppo_continue_temperature(5.0)
    h2h.configure_decision_mode_v4(model, "sample", keep_checkpoint_temperatures=True)
    assert model.ppo_rollout_sampling and model.ppo_gate_temperature == 0.2
    h2h.configure_decision_mode_v4(model, "sample", keep_checkpoint_temperatures=False)
    assert (model.ppo_gate_temperature, model.ppo_action_temperature, model.ppo_continue_temperature) == (1, 1, 1)
    h2h.configure_decision_mode_v4(model, "argmax", keep_checkpoint_temperatures=False)
    assert model.ppo_rollout_sampling is False
    with pytest.raises(ValueError):
        h2h.configure_decision_mode_v4(model, "greedy", keep_checkpoint_temperatures=False)


def test_model_config_round_trips_from_checkpoint_contract() -> None:
    from dataclasses import asdict

    config = ModelConfigV4()
    payload = asdict(config)
    payload["delay_offset_ms"] = list(payload["delay_offset_ms"])
    assert h2h.model_config_from_contract_v4(payload) == config
    with pytest.raises(ValueError, match="does not know"):
        h2h.model_config_from_contract_v4({**asdict(config), "future_field": 1})


# ---------------------------------------------------------------------------
# Game planning, waves, engine specs
# ---------------------------------------------------------------------------


def test_game_plan_uses_common_schedule_rows_both_seats_and_both_roles() -> None:
    rows = _schedule(configurations=8)
    jobs = h2h.plan_head_to_head_games_v4(rows, (("X", "Y"), ("X", "Z")), games_per_pair=8)
    assert len(jobs) == 16
    for pair in ("X:Y", "X:Z"):
        pair_jobs = [job for job in jobs if job.pair == pair]
        assert [job.schedule_row["schedule_row_id"] for job in pair_jobs] == list(range(8))
        model_a = pair_jobs[0].model_a
        assert sum(job.seat_models[0] == model_a for job in pair_jobs) == 4
        assert sum(job.current_model == model_a for job in pair_jobs) == 4
        assert {(job.seat_models.index(model_a), job.current_model == model_a) for job in pair_jobs} == {
            (0, True),
            (1, True),
            (0, False),
            (1, False),
        }
        for config_id in range(4):
            first, second = pair_jobs[2 * config_id], pair_jobs[2 * config_id + 1]
            assert first.seat_models == tuple(reversed(second.seat_models))
            assert first.current_model == second.current_model
    with pytest.raises(ValueError):
        h2h.plan_head_to_head_games_v4(rows, (("X", "Y"),), games_per_pair=7)
    with pytest.raises(ValueError):
        h2h.plan_head_to_head_games_v4(rows, (("X", "Y"),), games_per_pair=18)


def test_waves_hold_one_current_model_and_respect_capacity() -> None:
    rows = _schedule(configurations=10)
    jobs = h2h.plan_head_to_head_games_v4(rows, (("X", "Y"), ("Y", "Z"), ("X", "Z")), games_per_pair=20)
    waves = h2h.plan_waves_v4(jobs, capacity=7, model_order=("X", "Y", "Z"))
    assert sum(len(wave) for _name, wave in waves) == len(jobs)
    assert all(len(wave) <= 7 and {job.current_model for job in wave} == {name} for name, wave in waves)
    assert sorted(job.game_index for _name, wave in waves for job in wave) == list(range(len(jobs)))


def _topology(engines: int) -> dict[str, Any]:
    return {
        "engine_count": engines,
        "engine_hosts": [f"node{i // 2}" for i in range(engines)],
        "engine_ports": [28000 + i for i in range(engines)],
        "engine_worker_hosts": [f"node{i // 2}" for i in range(engines)],
        "engine_worker_ports": [29900] * engines,
        "worker_cpus": list(range(engines)),
    }


def test_engine_specs_pack_lanes_and_carry_evaluation_knobs() -> None:
    specs = h2h.engine_specs_for_wave_v4(
        _topology(4), (0, 1, 2, 3), games=7, lanes_per_engine=2, force_tick=300, warmup_ticks=90
    )
    assert [spec.lanes for spec in specs] == [2, 2, 2, 1]
    assert all(spec.base_port + spec.engine_index == 28000 + spec.engine_index for spec in specs)
    assert {
        (spec.force_deck_rule_opening_tick, spec.environment_warmup_ticks, spec.skip_shaping_telemetry)
        for spec in specs
    } == {(300, 90, True)}
    single = h2h.engine_specs_for_wave_v4(
        _topology(4), (2, 3), games=2, lanes_per_engine=1, force_tick=None, warmup_ticks=90
    )
    assert [spec.engine_index for spec in single] == [2, 3] and single[0].force_deck_rule_opening_tick is None
    with pytest.raises(ValueError):
        h2h.engine_specs_for_wave_v4(
            _topology(2), (0, 1), games=3, lanes_per_engine=1, force_tick=None, warmup_ticks=90
        )


def test_assignment_uses_history_role_for_the_frozen_model() -> None:
    manifest = _manifest()
    rows = _schedule(configurations=2)
    jobs = h2h.plan_head_to_head_games_v4(rows, (("X", "Y"),), games_per_pair=4)
    decks = {deck.deck_id: deck for deck in manifest.decks}
    for job in jobs:
        assignment = h2h.assignment_for_job_v4(job, decks, {"X": "x.pt", "Y": "y.pt"})
        assert assignment.matchup.policy_matchup == POLICY_HISTORY
        assert assignment.matchup.current_owners == (job.current_seat,)
        assert assignment.opponent is not None and assignment.opponent.checkpoint_id == job.frozen_model
        assert assignment.matchup.sequence == job.game_index + 1
        assert assignment.matchup.deck0.deck_id == job.schedule_row["deck0_id"]


# ---------------------------------------------------------------------------
# Report accumulation across collector segments
# ---------------------------------------------------------------------------


class RecordingConnection:
    def __init__(self, packets: list[dict[str, Any]]) -> None:
        self.packets = deque(packets)
        self.sent: list[dict[str, Any]] = []

    def recv(self) -> dict[str, Any]:
        return self.packets.popleft()

    def send(self, packet: dict[str, Any]) -> None:
        self.sent.append(packet)

    def close(self) -> None:
        pass


def _frame(actors: tuple[int, ...], rewards: tuple[Any, ...], *, boundary: bool = False) -> dict[str, Any]:
    batch = collector._pad_relation_edges(_batch(2), 10).to_storage("cpu", float_dtype=torch.float16)
    packed = collector._RowPackedTensorRecordV4(batch)
    packet = dict(
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
        packet.update(segment_boundary=True, timing=defaultdict(float))
    return packet


def _done(report: dict[str, Any]) -> dict[str, Any]:
    return dict(
        kind="done",
        rewards=((0, 1.0, True, False), (1, -1.0, True, False)),
        reports=(report,),
        timing=defaultdict(float),
    )


def test_segment_callback_accumulates_reports_cleared_by_each_segment(monkeypatch) -> None:
    torch.manual_seed(17)
    model = _model().eval()
    early = RecordingConnection([_frame((0, 1), ()), _done({"sequence": 1, "who": "early"})])
    late = RecordingConnection(
        [
            _frame((0, 1), ()),
            _frame((0, 1), ((0, 0.1, False, False), (1, -0.1, False, False)), boundary=True),
            _frame((0, 1), ()),
            _done({"sequence": 2, "who": "late"}),
        ]
    )
    connections = {1: early, 2: late}

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
    assignments = tuple(
        SimpleNamespace(matchup=SimpleNamespace(current_owners=(0, 1), policy_matchup="current-current"))
        for _ in range(2)
    )
    accumulator = h2h.SegmentReportAccumulatorV4("h2h00000")
    collector.collect_async_cluster_wave_v4(
        specs,
        assignments,  # type: ignore[arg-type]
        model,
        model,
        {},
        gamma_per_decision=0.9997,
        shaping_beta=0.05,
        mutual_elixir_overflow_penalty=0.0,
        mutual_elixir_overflow_grace=4.0,
        unilateral_elixir_overflow_penalty=0.0,
        elixir_overflow_step_penalty_cap=0.1,
        policy_state_id="h2h00000",
        max_decision_steps=100,
        segment_decision_steps=160,
        segment_callback=accumulator,
    )
    assert accumulator.segments == 2
    assert [report["who"] for report in accumulator.reports] == ["early", "late"]
    resumes = [packet for packet in late.sent if packet["kind"] == "resume"]
    assert resumes == [{"kind": "resume", "policy_state_id": "h2h00000"}]
    assert not early.packets and not late.packets
    assert accumulator.timing_summary()["matches"] == 2.0


def _fake_collect(segments: list[list[dict[str, Any]]]) -> Callable[..., Any]:
    def collect(*_args: Any, segment_callback: Callable[[Any], str], **_kwargs: Any) -> Any:
        result = None
        for reports in segments:
            result = SimpleNamespace(reports=tuple(reports), timing={"total_seconds": 1.0, "matches": len(reports)})
            assert segment_callback(result) == "h2h00003"
        return result

    return collect


def test_run_wave_orders_reports_and_fails_hard_on_missing_games() -> None:
    rows = _schedule(configurations=2)
    jobs = h2h.plan_head_to_head_games_v4(rows, (("X", "Y"),), games_per_pair=4)
    wave = [job for job in jobs if job.current_model == "X"]
    sequences = [job.game_index + 1 for job in wave]
    kwargs = dict(
        specs=(),
        jobs=wave,
        assignments=(),
        models={"X": None, "Y": None},
        current_model="X",
        policy_state_id="h2h00003",
        segment_decision_steps=160,
        max_decision_steps=1444,
    )
    reports, accumulator, _elapsed = h2h.run_wave_v4(
        _fake_collect([[{"sequence": sequences[1]}], [], [{"sequence": sequences[0]}]]), **kwargs
    )
    assert [report["sequence"] for report in reports] == sequences
    assert accumulator.segments == 3
    with pytest.raises(RuntimeError, match="missing"):
        h2h.run_wave_v4(_fake_collect([[{"sequence": sequences[0]}]]), **kwargs)
    with pytest.raises(RuntimeError, match="twice"):
        h2h.run_wave_v4(_fake_collect([[{"sequence": sequences[0]}], [{"sequence": sequences[0]}]]), **kwargs)


def _report_for(job: Any, *, seat0_result: float, truncated: bool = False, ended: bool = True) -> dict[str, Any]:
    row = job.schedule_row
    return {
        "sequence": job.game_index + 1,
        "deck0_id": row["deck0_id"],
        "deck1_id": row["deck1_id"],
        "battle_level": row["battle_level"],
        "current_owners": (job.current_seat,),
        "result": (seat0_result, -seat0_result),
        "crowns": (1, 0) if seat0_result > 0 else (0, 1),
        "terminal_ended": ended,
        "terminal_reason": "crown_advantage",
        "terminal_winner": 0 if seat0_result > 0 else 1,
        "terminal_tick": 6000,
        "final_tower_hitpoints": (),
        "active_actions": (3, 0),
        "rejected_actions": (1, 0),
        "rejection_details": (),
        "first_nonwait_action_tick": (150, None),
        "opening_forced_owner": 1,
        "opening_forced_tick": 300,
        "deck_average_elixir": (3.1, 3.4),
        "decision_steps": 1182,
        "engine_index": 0,
        "elapsed_seconds": 50.0,
        "truncated": truncated,
    }


def test_game_rows_attribute_results_forcing_and_hard_failures() -> None:
    rows = _schedule(configurations=1)
    jobs = h2h.plan_head_to_head_games_v4(rows, (("X", "Y"),), games_per_pair=2)
    behaviors = {"X": "ckpt|x", "Y": "ckpt|y"}
    first = h2h.game_row_from_report_v4(
        jobs[0], _report_for(jobs[0], seat0_result=1.0), wave_index=0, behaviors=behaviors
    )
    assert (first["seat0_model"], first["score_a"], first["forced_model"]) == ("X", 1.0, "Y")
    assert first["hard_failure"] is False
    second = h2h.game_row_from_report_v4(
        jobs[1], _report_for(jobs[1], seat0_result=1.0, truncated=True), wave_index=0, behaviors=behaviors
    )
    assert (second["seat0_model"], second["score_a"], second["forced_model"]) == ("Y", 0.0, "X")
    assert second["hard_failure"] is True
    not_ended = h2h.game_row_from_report_v4(
        jobs[1], _report_for(jobs[1], seat0_result=0.0, ended=False), wave_index=0, behaviors=behaviors
    )
    assert not_ended["hard_failure"] is True
    bad = _report_for(jobs[0], seat0_result=1.0)
    bad["deck0_id"] = "other"
    with pytest.raises(RuntimeError, match="differs"):
        h2h.game_row_from_report_v4(jobs[0], bad, wave_index=0, behaviors=behaviors)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _synthetic_rows(
    strengths: dict[str, float], *, configurations: int, seed: int, sanity: tuple[str, str] | None = None
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    names = list(strengths)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    if sanity is not None:
        pairs.append(sanity)
        strengths = {**strengths, sanity[1]: strengths[sanity[0]]}
    deck_effect = rng.normal(0.0, 0.8, size=configurations)
    rows: list[dict[str, Any]] = []
    for pair_index, (model_a, model_b) in enumerate(pairs):
        for config_id in range(configurations):
            for swap in (0, 1):
                seat0, seat1 = (model_a, model_b) if swap == 0 else (model_b, model_a)
                delta = strengths[seat0] - strengths[seat1] + deck_effect[config_id]
                win = rng.random() < 1.0 / (1.0 + math.exp(-delta))
                ticks = int(rng.integers(3000, 6001))
                rows.append(
                    {
                        "schema": "v4-head-to-head-game.v1",
                        "game_id": len(rows),
                        "pair": f"{model_a}:{model_b}",
                        "model_a": model_a,
                        "model_b": model_b,
                        "schedule_row_id": 2 * config_id + swap,
                        "config_id": config_id,
                        "seed": 1000 + config_id,
                        "deck0_id": f"d{config_id}a",
                        "deck1_id": f"d{config_id}b",
                        "battle_level": 11 if config_id % 2 else 16,
                        "category": "focus-focus",
                        "seat0_model": seat0,
                        "seat1_model": seat1,
                        "seat0_behavior": "same" if sanity and {seat0, seat1} == set(sanity) else seat0,
                        "seat1_behavior": "same" if sanity and {seat0, seat1} == set(sanity) else seat1,
                        "current_model": model_a if config_id % 2 == 0 else model_b,
                        "result_seat0": 1.0 if win else -1.0,
                        "crowns": [1, 0] if win else [0, 1],
                        "terminal_tick": ticks,
                        "terminal_reason": "crown_advantage",
                        "final_tower_hitpoints": [],
                        "active_actions": [0, 0] if ticks == 6000 else [5, 4],
                        "rejected_actions": [0, 1],
                        "forced_model": None,
                        "hard_failure": False,
                    }
                )
    return rows


def test_bradley_terry_recovers_planted_ordering_and_bootstrap_runs() -> None:
    strengths = {"weak": -0.6, "mid": 0.0, "strong": 0.6, "best": 1.2}
    rows = _synthetic_rows(strengths, configurations=300, seed=3, sanity=("mid", "mid_copy"))
    report = aggregate.aggregate_head_to_head_v4(rows, replicates=60, seed=4, reference="mid")
    ratings = report["ratings"]
    order = [item["model"] for item in ratings["table"]]
    assert order == ["best", "strong", "mid", "weak"]
    by_name = {item["model"]: item for item in ratings["table"]}
    for name, value in strengths.items():
        # The deck effect shrinks raw log-odds slightly; only direction and rough scale matter here.
        assert by_name[name]["theta"] == pytest.approx(value, abs=0.35)
        assert by_name[name]["theta_95"] is not None
    assert by_name["mid"]["theta"] == 0.0
    assert ratings["bootstrap_failed_replicates"] == 0
    best_vs_weak = next(
        item for item in ratings["differences"] if {item["model_i"], item["model_j"]} == {"best", "weak"}
    )
    assert best_vs_weak["excludes_zero"] is True
    assert "mid_copy" not in ratings["players"]
    assert set(report["sanity"]) == {"mid:mid_copy"}
    markdown = aggregate.render_markdown_v4(report)
    assert "A-vs-A sanity" in markdown and "Bradley-Terry ratings" in markdown


def test_bt_fit_matches_closed_form_for_two_players() -> None:
    score = np.array([[0.0, 30.0], [10.0, 0.0]])
    games = np.array([[0.0, 40.0], [40.0, 0.0]])
    theta = aggregate.fit_bradley_terry_v4(score, games, prior_draws=0.0)
    assert theta[0] - theta[1] == pytest.approx(math.log(3.0), abs=1e-8)
    with pytest.raises(ValueError, match="disconnected"):
        aggregate.fit_bradley_terry_v4(np.zeros((3, 3)), np.zeros((3, 3)))


def test_pair_summary_secondary_metrics() -> None:
    rows = _synthetic_rows({"x": 0.0, "y": 0.0}, configurations=20, seed=8)
    summary = aggregate.pair_summaries_v4(rows)["x:y"]
    assert summary["games"] == 40
    assert summary["tiebreak_games"] == sum(int(row["terminal_tick"]) >= 5995 for row in rows)
    assert summary["both_idle_games"] == sum(row["active_actions"] == [0, 0] for row in rows)
    assert summary["splits"]["a_seat"]["0"]["games"] == 20 and summary["splits"]["a_role"]["current"]["games"] == 20
    assert summary["crown_diff_mean"] == pytest.approx(2.0 * summary["score_rate"] - 1.0)
    low, high = summary["score_rate_wilson_95"]
    assert low < summary["score_rate"] < high
    assert aggregate.wilson_interval(0, 0) == (0.0, 1.0)


def test_aggregator_cli_writes_json_and_markdown(tmp_path) -> None:
    rows = _synthetic_rows({"a": 0.0, "b": 0.5, "c": 1.0}, configurations=30, seed=9)
    games = tmp_path / "games.jsonl"
    games.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert aggregate.main(["--games", str(games), "--output-dir", str(tmp_path / "agg"), "--bootstrap", "20"]) == 0
    report = json.loads((tmp_path / "agg" / "aggregate.json").read_text(encoding="utf-8"))
    assert report["games"] == len(rows) and report["ratings"]["players"] == ["a", "b", "c"]
    assert (tmp_path / "agg" / "aggregate.md").read_text(encoding="utf-8").startswith("# Head-to-head aggregate")
    duplicated = tmp_path / "dup.jsonl"
    duplicated.write_text(games.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        aggregate.load_game_rows([duplicated])


# ---------------------------------------------------------------------------
# Checkpoint loading and the driver's main() with a fake collector
# ---------------------------------------------------------------------------


def test_build_model_from_checkpoint_round_trips_a_production_checkpoint(tmp_path) -> None:
    from native_runner.training.v4.checkpoint import save_actor_critic_checkpoint
    from native_runner.training.v4.factory import build_production_model_v4

    torch.manual_seed(21)
    source = build_production_model_v4()
    source.set_ppo_gate_temperature(0.2)
    path = tmp_path / "source.pt"
    checkpoint_id = save_actor_critic_checkpoint(path, source, training_stage="test", gamma_per_decision=0.9997)
    loaded, payload = h2h.build_model_from_checkpoint_v4(path, torch.device("cpu"))
    assert payload["checkpoint_id"] == checkpoint_id
    assert loaded.config == source.config and loaded.ppo_gate_temperature == 0.2
    for (name, expected), (_other, actual) in zip(
        source.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert torch.equal(expected, actual), name


def _fake_report(assignment: Any, *, seat0_wins: bool) -> dict[str, Any]:
    matchup = assignment.matchup
    result = 1.0 if seat0_wins else -1.0
    return {
        "sequence": matchup.sequence,
        "deck0_id": matchup.deck0.deck_id,
        "deck1_id": matchup.deck1.deck_id,
        "battle_level": matchup.battle_level,
        "current_owners": matchup.current_owners,
        "result": (result, -result),
        "crowns": (1, 0) if seat0_wins else (0, 1),
        "terminal_ended": True,
        "terminal_reason": "crown_advantage",
        "terminal_winner": 0 if seat0_wins else 1,
        "terminal_tick": 6000 if matchup.sequence % 3 == 0 else 4000,
        "final_tower_hitpoints": (),
        "active_actions": (4, 5),
        "rejected_actions": (0, 0),
        "rejection_details": (),
        "first_nonwait_action_tick": (140, 150),
        "opening_forced_owner": None,
        "opening_forced_tick": None,
        "deck_average_elixir": (3.0, 3.5),
        "decision_steps": 1182,
        "engine_index": 0,
        "elapsed_seconds": 1.0,
        "truncated": matchup.sequence == 1,
    }


def test_driver_main_runs_waves_resumes_and_flags_truncation(monkeypatch, tmp_path) -> None:
    manifest = _manifest()
    pool = tmp_path / "pool.json"
    pool.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(schedule_module, "DeckPoolManifestV4", SimpleNamespace(load=lambda _path: manifest))
    monkeypatch.setattr(h2h, "DeckPoolManifestV4", SimpleNamespace(load=lambda _path: manifest))
    schedule = tmp_path / "schedule.jsonl"
    schedule_module.main(["--deck-pool", str(pool), "--output", str(schedule), "--configurations", "4", "--seed", "2"])
    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps(_topology(3)), encoding="utf-8")
    checkpoints = {}
    for name in ("X", "Y"):
        checkpoints[name] = tmp_path / f"{name}.pt"
        checkpoints[name].write_bytes(name.encode())

    def fake_build(path: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
        return _model(), {"checkpoint_id": f"id-{Path(path).stem}", "update_step": 0, "training_stage": "test"}

    calls: list[dict[str, Any]] = []

    def fake_collect(specs, assignments, current_model, anchor_model, history_models, **kwargs):
        assert sum(spec.lanes for spec in specs) == len(assignments)
        assert all(spec.force_deck_rule_opening_tick == 300 and spec.environment_warmup_ticks == 90 for spec in specs)
        assert kwargs["segment_decision_steps"] == 160
        calls.append({"history": sorted(history_models), "games": len(assignments)})
        half = len(assignments) // 2
        callback = kwargs["segment_callback"]
        reports = [_fake_report(item, seat0_wins=item.matchup.sequence % 2 == 1) for item in assignments]
        assert (
            callback(SimpleNamespace(reports=tuple(reports[:half]), timing={"total_seconds": 1.0}))
            == (kwargs["policy_state_id"])
        )
        return callback(SimpleNamespace(reports=tuple(reports[half:]), timing={"total_seconds": 2.0}))

    monkeypatch.setattr(h2h, "build_model_from_checkpoint_v4", fake_build)
    monkeypatch.setattr(h2h, "collect_async_cluster_wave_v4", fake_collect)
    monkeypatch.setattr(torch, "set_num_threads", lambda _value: None)
    output = tmp_path / "run"
    args = [
        "--topology", str(topology), "--deck-pool", str(pool), "--schedule", str(schedule),
        "--model", f"X={checkpoints['X']}:gate_logit_offset=-2.0794", "--model", f"Y={checkpoints['Y']}",
        "--pairs", "X:Y", "--games-per-pair", "8", "--mode", "sample", "--forced-opening", "deck-rule",
        "--output-dir", str(output), "--device", "cpu", "--max-waves", "2",
    ]  # fmt: skip
    # Game 1 is truncated: every wave still runs, but the exit status is non-zero.
    assert h2h.main(args) == 1
    assert [call["history"] for call in calls] == [["Y"], ["Y"]]
    first_rows = [json.loads(line) for line in (output / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(first_rows) == 4 and {row["current_model"] for row in first_rows} == {"X"}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["hard_failures"] == 1 and summary["partial_by_max_waves"] is True and summary["ok"] is False
    assert summary["collector_timing"]["waves"][0]["segments"] == 2
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    assert identity["run_key"]["models"]["X"]["gate_logit_offset"] == pytest.approx(-2.0794)
    with pytest.raises(ValueError, match="--resume"):
        h2h.main(args)
    calls.clear()
    assert h2h.main([*args[:-2], "--resume"]) == 1
    rows = [json.loads(line) for line in (output / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and [call["games"] for call in calls] == [3, 1]
    assert sorted(row["schedule_row_id"] for row in rows) == list(range(8))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True and summary["pairs"]["X:Y"]["games"] == 8
    with pytest.raises(ValueError, match="resume identity differs"):
        h2h.main([*args[:-2], "--resume", "--warmup-ticks", "130"])


# ---------------------------------------------------------------------------
# Pinned specialist deck (``:deck=<deck_id>``)
# ---------------------------------------------------------------------------


def test_model_spec_parses_a_pinned_deck() -> None:
    spec = h2h.parse_model_spec_v4(r"HOG=D:\ckpt\hog.pt:deck=deck-007:gate_logit_offset=0")
    assert spec.path == Path(r"D:\ckpt\hog.pt") and spec.fixed_deck_id == "deck-007"
    assert h2h.parse_model_spec_v4("X=/a.pt").fixed_deck_id is None
    with pytest.raises(ValueError, match="repeats"):
        h2h.parse_model_spec_v4("X=/a.pt:deck=a:deck=b")


def test_pinned_deck_follows_the_specialist_and_opponent_keeps_scheduled_decks() -> None:
    manifest = _manifest()
    decks = {deck.deck_id: deck for deck in manifest.decks}
    rows = _schedule(configurations=6)
    pinned_id = "deck-059"
    assert all(pinned_id not in (row["deck0_id"], row["deck1_id"]) for row in rows)
    jobs = h2h.plan_head_to_head_games_v4(rows, (("X", "H"),), games_per_pair=12)
    fixed = {"H": pinned_id}
    opponent_decks = []
    for job in jobs:
        row = job.schedule_row
        seat_h = job.seat_models.index("H")
        deck_ids = h2h.seat_deck_ids_v4(job, fixed)
        assert deck_ids[seat_h] == pinned_id
        assert deck_ids[1 - seat_h] == row[f"deck{1 - seat_h}_id"]
        opponent_decks.append((int(row["config_id"]), deck_ids[1 - seat_h]))
        assignment = h2h.assignment_for_job_v4(job, decks, {"X": "x.pt", "H": "h.pt"}, fixed)
        played = (assignment.matchup.deck0.deck_id, assignment.matchup.deck1.deck_id)
        assert played == deck_ids
        # The opponent's runtime layout is still the scheduled one.
        layouts = assignment.matchup.runtime_layouts()
        assert list(layouts[1 - seat_h].card_ids) == list(row[f"deck{1 - seat_h}"])
        assert list(layouts[seat_h].card_ids) != list(row[f"deck{seat_h}"])
        assert h2h.seat_deck_ids_v4(job) == (row["deck0_id"], row["deck1_id"])
    # Over a configuration's two seat-swapped rows the opponent plays both scheduled decks.
    for config_id in range(6):
        row = rows[2 * config_id]
        assert {deck for cid, deck in opponent_decks if cid == config_id} == {row["deck0_id"], row["deck1_id"]}


def test_game_rows_record_pinned_and_scheduled_decks() -> None:
    rows = _schedule(configurations=1)
    jobs = h2h.plan_head_to_head_games_v4(rows, (("X", "H"),), games_per_pair=2)
    behaviors = {"X": "ckpt|x", "H": "ckpt|h|deck=deck-059"}
    fixed = {"H": "deck-059"}
    for job in jobs:
        report = _report_for(job, seat0_result=1.0)
        deck_ids = h2h.seat_deck_ids_v4(job, fixed)
        report["deck0_id"], report["deck1_id"] = deck_ids
        game = h2h.game_row_from_report_v4(job, report, wave_index=0, behaviors=behaviors, fixed_decks=fixed)
        assert (game["deck0_id"], game["deck1_id"]) == deck_ids
        assert (game["schedule_deck0_id"], game["schedule_deck1_id"]) == (
            job.schedule_row["deck0_id"],
            job.schedule_row["deck1_id"],
        )
        with pytest.raises(RuntimeError, match="differs"):
            h2h.game_row_from_report_v4(job, _report_for(job, seat0_result=1.0), wave_index=0, behaviors=behaviors,
                                        fixed_decks=fixed)
    plain = h2h.game_row_from_report_v4(
        jobs[0], _report_for(jobs[0], seat0_result=1.0), wave_index=0, behaviors={"X": "a", "H": "b"}
    )
    assert "schedule_deck0_id" not in plain and "schedule_deck1_id" not in plain


def test_behavior_key_appends_pinned_deck_only_when_set() -> None:
    model = _model()
    plain = h2h.behavior_key_v4("ckpt", 0.0, "sample", model)
    pinned = h2h.behavior_key_v4("ckpt", 0.0, "sample", model, fixed_deck_id="deck-059")
    assert pinned == plain + "|deck=deck-059"
