from __future__ import annotations

import math

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.train_ppo_self_play_cluster import (
    _gate_temperature_for_update,
    _rollout_behavior_metrics,
)


def test_gate_temperature_anneals_linearly_and_then_holds() -> None:
    kwargs = {"start": 0.30, "end": 0.20, "anneal_updates": 400}

    assert _gate_temperature_for_update(1, **kwargs) == 0.30
    assert math.isclose(
        _gate_temperature_for_update(200, **kwargs),
        0.30 - 0.10 * 199 / 399,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert _gate_temperature_for_update(400, **kwargs) == 0.20
    assert _gate_temperature_for_update(4_000, **kwargs) == 0.20


def test_frozen_rollout_mode_uses_argmax_for_all_heads() -> None:
    torch.manual_seed(73)
    model = _model().eval()
    batch = _batch(batch_size=3)
    state = model.initial_state(3)
    starts = torch.ones(3, dtype=torch.bool)
    model.set_ppo_gate_temperature(0.2)
    model.set_ppo_action_temperature(1.0)
    model.set_ppo_continue_temperature(2.0)
    model.set_ppo_rollout_sampling(False)

    expected = model.act(batch, state, episode_start=starts)
    actual = model.sample_for_ppo_rollout(
        batch,
        state,
        episode_start=starts,
    )

    assert model.ppo_rollout_sampling is False
    assert torch.equal(actual.actions.gate, expected.actions.gate)
    assert torch.equal(
        actual.actions.micro_action_count,
        expected.actions.micro_action_count,
    )
    assert torch.equal(
        actual.actions.candidate_index,
        expected.actions.candidate_index,
    )
    assert torch.equal(actual.actions.target_cell, expected.actions.target_cell)
    assert torch.equal(
        actual.actions.delay_offset_bin,
        expected.actions.delay_offset_bin,
    )


def test_frozen_forced_act_mode_keeps_downstream_heads_argmax() -> None:
    model = _model().eval()
    batch = _batch(batch_size=1)
    state = model.initial_state(1)
    context = model.forward(
        batch,
        state,
        episode_start=torch.ones(1, dtype=torch.bool),
    )
    model.set_ppo_rollout_sampling(False)

    first = model.sample_after_preselected_act(batch, context)
    torch.manual_seed(999)
    second = model.sample_after_preselected_act(batch, context)

    assert torch.equal(first.actions.gate, torch.ones(1, dtype=torch.long))
    assert torch.equal(first.actions.candidate_index, second.actions.candidate_index)
    assert torch.equal(first.actions.target_cell, second.actions.target_cell)
    assert torch.equal(
        first.actions.delay_offset_bin,
        second.actions.delay_offset_bin,
    )
    assert torch.equal(
        first.actions.micro_action_count,
        second.actions.micro_action_count,
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"start": 0.0, "end": 0.20, "anneal_updates": 400},
        {"start": 0.20, "end": 0.30, "anneal_updates": 400},
        {"start": 0.30, "end": 0.20, "anneal_updates": 0},
    ),
)
def test_gate_temperature_schedule_rejects_invalid_values(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        _gate_temperature_for_update(1, **kwargs)


def test_rollout_behavior_metrics_report_gate_continue_and_delay_distributions() -> None:
    counts = torch.tensor(
        [100, 80, 60, 2, 30, 20, 10, 25, 15, 40, 10, 8, 7, 6, 9],
        dtype=torch.long,
    )

    metrics = _rollout_behavior_metrics(
        counts,
        delay_offset_ms=(0, 50, 100, 150, 200),
    )

    assert metrics["eligible_wait_rate"] == pytest.approx(0.75)
    assert metrics["eligible_act_rate"] == pytest.approx(0.25)
    assert metrics["forced_opening_act_ticks"] == 2
    assert metrics["continue_stop_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["continue_second_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["continue_eligible_stop_rate"] == pytest.approx(0.6)
    assert metrics["continue_eligible_second_rate"] == pytest.approx(0.4)
    assert metrics["delay_0ms_fraction"] == pytest.approx(0.25)
    assert metrics["delay_200ms_fraction"] == pytest.approx(0.225)
