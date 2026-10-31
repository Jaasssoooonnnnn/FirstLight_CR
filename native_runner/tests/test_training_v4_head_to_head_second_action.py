"""Head-to-head records count ACT decision steps and steps that issued a second micro-action."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from native_runner.contracts import ActionKind
from native_runner.tests import test_training_v4_head_to_head as harness
from native_runner.training.v4 import aggregate_head_to_head as aggregate
from native_runner.training.v4 import evaluate_head_to_head as h2h
from native_runner.training.v4.cluster_self_play import _MatchRuntimeV4
from native_runner.training.v4.tensors import GATE_ACT

# (tick, owner) -> micro-action count; every other decision is WAIT.
SCRIPT = {(200, 0): 1, (250, 0): 2, (300, 0): 2, (210, 1): 1, (400, 1): 2}


class TwoActionConnection(harness.ScriptedConnection):
    """Like the harness connection, but scripted steps may carry two micro-actions."""

    def recv(self) -> dict[str, Any]:
        packet = self.sent[-1]
        if packet.get("segment_boundary"):
            return {"kind": "resume", "policy_state_id": "resume"}
        rows = tuple(int(value) for value in packet["actor_rows"])
        packed = torch.zeros((len(rows), 10), dtype=torch.long)
        for position, local_actor in enumerate(rows):
            count = SCRIPT.get((self.state.tick, local_actor % 2), 0)
            if count:
                packed[position, 0] = GATE_ACT
                packed[position, 1] = count
        return {"kind": "act_packed", "actor_rows": rows, "packed_actions": packed, "forced_act_actor_rows": ()}


def _two_action_decode(action: Any, *_args: object, **_kwargs: object) -> Any:
    from types import SimpleNamespace

    if int(action.gate[0]) != GATE_ACT:
        return SimpleNamespace(actions=(harness.FakeAction(ActionKind.WAIT),))
    count = int(action.micro_action_count[0])
    return SimpleNamespace(
        actions=tuple(harness.FakeAction(ActionKind.PLAY_CARD, card_id=26000000) for _ in range(count))
    )


def test_runtime_counters_default_to_zero() -> None:
    fields = _MatchRuntimeV4.__dataclass_fields__
    assert fields["act_steps"].default == (0, 0)
    assert fields["second_action_steps"].default == (0, 0)


def test_worker_counts_act_and_second_action_steps_per_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "ScriptedConnection", TwoActionConnection)
    monkeypatch.setattr(harness, "_fake_decode", _two_action_decode)
    connection, state = harness.run_fake_worker(monkeypatch, end_tick=420)
    report = harness._done_report(connection)
    assert report["act_steps"] == (3, 2)
    assert report["second_action_steps"] == (2, 1)
    assert report["active_actions"] == (5, 3)
    # Every scripted step was played, nothing else.
    submitted = [
        (owner, len(kinds))
        for frame in state.submitted
        for owner, kinds in frame.items()
        if ActionKind.WAIT not in kinds
    ]
    assert sorted(submitted) == [(0, 1), (0, 2), (0, 2), (1, 1), (1, 2)]


def test_forced_opening_step_counts_as_an_act_step(monkeypatch: pytest.MonkeyPatch) -> None:
    connection, _state = harness.run_fake_worker(
        monkeypatch, spec_overrides=dict(force_deck_rule_opening_tick=300), deck_average_elixir=(2.9, 3.8)
    )
    report = harness._done_report(connection)
    assert report["act_steps"] == (1, 0) and report["second_action_steps"] == (0, 0)


def test_game_rows_carry_the_counters_and_old_reports_default_to_zero() -> None:
    rows = harness._schedule(configurations=1)
    jobs = h2h.plan_head_to_head_games_v4(rows, (("X", "Y"),), games_per_pair=2)
    behaviors = {"X": "ckpt|x", "Y": "ckpt|y"}
    old = harness._report_for(jobs[0], seat0_result=1.0)
    old.pop("act_steps", None)
    old.pop("second_action_steps", None)
    row = h2h.game_row_from_report_v4(jobs[0], old, wave_index=0, behaviors=behaviors)
    assert row["act_steps"] == [0, 0] and row["second_action_steps"] == [0, 0]
    new = {**old, "act_steps": (7, 4), "second_action_steps": (3, 0)}
    row = h2h.game_row_from_report_v4(jobs[0], new, wave_index=0, behaviors=behaviors)
    assert row["act_steps"] == [7, 4] and row["second_action_steps"] == [3, 0]
    json.dumps(row)
    with pytest.raises(ValueError, match="two owners"):
        h2h.game_row_from_report_v4(jobs[0], {**old, "act_steps": (1,)}, wave_index=0, behaviors=behaviors)


def test_aggregate_sums_second_action_rate_per_model_and_pair() -> None:
    rows = harness._synthetic_rows({"x": 0.0, "y": 0.3}, configurations=10, seed=5)
    # Legacy rows (no counters) aggregate as zeros.
    legacy = aggregate.aggregate_head_to_head_v4(rows, replicates=10, seed=1)
    assert legacy["models"]["x"] == {
        "seat_games": 20,
        "active_actions": legacy["models"]["x"]["active_actions"],
        "act_steps": 0,
        "second_action_steps": 0,
        "second_action_rate": 0.0,
    }
    assert legacy["pairs"]["x:y"]["second_action_rate"] == {"x": 0.0, "y": 0.0}
    for row in rows:
        seat_of_x = 0 if row["seat0_model"] == "x" else 1
        act = [0, 0]
        second = [0, 0]
        act[seat_of_x], second[seat_of_x] = 4, 1
        act[1 - seat_of_x], second[1 - seat_of_x] = 5, 3
        row["act_steps"], row["second_action_steps"] = act, second
    report = aggregate.aggregate_head_to_head_v4(rows, replicates=10, seed=1)
    x, y = report["models"]["x"], report["models"]["y"]
    assert (x["act_steps"], x["second_action_steps"], x["second_action_rate"]) == (80, 20, 0.25)
    assert (y["act_steps"], y["second_action_steps"], y["second_action_rate"]) == (100, 60, 0.6)
    pair = report["pairs"]["x:y"]
    assert pair["act_steps"] == {"x": 80, "y": 100} and pair["second_action_steps"] == {"x": 20, "y": 60}
    assert pair["second_action_rate"] == {"x": 0.25, "y": 0.6}
    markdown = aggregate.render_markdown_v4(report)
    assert "## Per-model action counts (all games)" in markdown
    assert "| x | 20 |" in markdown and "| 80 | 20 | 0.250 |" in markdown
    json.dumps(report)
