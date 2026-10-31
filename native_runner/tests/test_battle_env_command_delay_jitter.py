"""Simulated card execution jitter, without changing live server timing."""
import random
from unittest.mock import Mock

import pytest

from native_runner.contracts import ActionKind, ActionV1
from native_runner.cr_native_env import COMMAND_EXECUTION_DELAY_TICKS, FIRST_PLAYABLE_TICK
from native_runner.tests.test_battle_env import FakeNativeEnv, _make_env, _reset


def _env(seed=7):
    native = FakeNativeEnv()
    native.simulated_command_delay_ticks = COMMAND_EXECUTION_DELAY_TICKS
    env, native = _make_env(native)
    env.simulated_card_delay_jitter_ticks = 3
    _reset(env, seed=seed)
    return env, native


def _play(env, owner=0, *, compound=False):
    grid = (8, 10) if owner == 0 else (8, 25)
    actions = [ActionV1.play(owner=owner, hand_slot=0, grid=grid,
                            execute_offset_ticks=1, next_decision_ticks=5)]
    if compound:
        actions.append(ActionV1.play(
            owner=owner, hand_slot=1, grid=(9, 25), execute_offset_ticks=2, next_decision_ticks=5))
    env.step({owner: tuple(actions), 1 - owner: ActionV1.wait(1 - owner, ticks=5)})
    return [action.execute_offset_ticks for action in env.trace[-1].actions
            if action.kind == ActionKind.PLAY_CARD]


@pytest.mark.parametrize('jitter', range(4))
def test_earliest_card_is_queued_at_each_real_execution_boundary(jitter):
    env, native = _env()
    env._command_delay_rng[0].randrange = Mock(return_value=jitter)
    assert _play(env) == [21 + jitter]
    assert [command['execute_tick'] for command in native.queued] == [FIRST_PLAYABLE_TICK + 21 + jitter]
    assert env.observe(owner=0).action_mask.reasons['slots']['0'] == 'pending_execution'


def test_jitter_repeats_by_seed_and_waits_do_not_consume_card_randomness():
    global_state = random.getstate()
    direct, _ = _env(seed=23)
    waited, _ = _env(seed=23)
    for _ in range(3):
        waited.step({0: ActionV1.wait(0, ticks=5), 1: ActionV1.wait(1, ticks=5)})
    assert _play(direct) == _play(waited)
    assert random.getstate() == global_state
    # Resetting the same environment also resets the dedicated actor stream.
    _reset(direct, seed=23)
    fresh, _ = _env(seed=23)
    assert _play(direct) == _play(fresh)


def test_compound_cards_share_one_jitter_and_keep_relative_order():
    env, native = _env()
    draw = Mock(return_value=3)
    env._command_delay_rng[1].randrange = draw
    assert _play(env, owner=1, compound=True) == [24, 25]
    assert draw.call_count == 1
    assert [command['execute_tick'] for command in native.queued] == [
        FIRST_PLAYABLE_TICK + 24, FIRST_PLAYABLE_TICK + 25]


def test_zero_delay_endpoints_do_not_draw_extra_jitter():
    native = FakeNativeEnv()
    native.simulated_command_delay_ticks = 0
    env, native = _make_env(native)
    _reset(env)
    env.simulated_card_delay_jitter_ticks = 3  # a zero-delay endpoint must reject artificial latency even under opt-in
    env._command_delay_rng[0].randrange = Mock(side_effect=AssertionError('must not add jitter'))
    assert _play(env) == [1]


def test_different_match_seeds_cover_all_four_execution_delays():
    delays = set()
    for seed in range(32):
        env, _ = _env(seed)
        delays.update(_play(env))
    assert delays == {21, 22, 23, 24}


def test_non_training_simulation_keeps_fixed_server_delay_without_network_jitter():
    env, native = _env()
    env.simulated_card_delay_jitter_ticks = None
    env._command_delay_rng[0].randrange = Mock(side_effect=AssertionError('not PPO training'))
    assert _play(env) == [21]
    assert [command['execute_tick'] for command in native.queued] == [FIRST_PLAYABLE_TICK + 21]


def test_worker_and_h2h_specs_require_explicit_training_opt_in():
    from types import SimpleNamespace
    from native_runner.training.v4.async_cluster_self_play import (
        AsyncResidentEngineSpecV4, _configure_ppo_training_command_delay_v4,
    )
    spec = AsyncResidentEngineSpecV4(engine_index=0, cpu=0)
    assert spec.training_command_delay_jitter_ticks is None
    env, _ = _make_env()
    engine = SimpleNamespace(environments=(env,))
    _configure_ppo_training_command_delay_v4(engine, spec.training_command_delay_jitter_ticks)
    assert env.simulated_card_delay_jitter_ticks is None
    _configure_ppo_training_command_delay_v4(engine, 3)
    assert env.simulated_card_delay_jitter_ticks == 3
    for invalid in (-1, 4, True, 1.5):
        with pytest.raises(ValueError):
            _configure_ppo_training_command_delay_v4(engine, invalid)
