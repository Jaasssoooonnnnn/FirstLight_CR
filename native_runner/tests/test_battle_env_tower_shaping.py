"""Tower-HP potential-based shaping (PBRS) and the King gate.

``test_battle_env.py`` builds every environment with ``shaping_beta=0``; these
tests pin the shaping itself with ``beta > 0`` on the in-memory fake engine.
Fake towers all have 1000 max HP, so loss fractions are ``(1000 - hp) / 1000``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from native_runner.battle_env import BattleEnvV1, DEFAULT_KING_POTENTIAL_WEIGHT, KING_SHAPING_MODES
from native_runner.contracts import ActionV1
from native_runner.tests.test_battle_env import FakeNativeEnv, _reset, _test_catalog

BETA = 0.05
TAU = 1200.0
OWNER0_KING = (9000, 3000)
OWNER0_LEFT = (3500, 6500)
OWNER1_KING = (9000, 29000)
OWNER1_LEFT = (3500, 25500)
OWNER1_RIGHT = (14500, 25500)


def _env(
    *, beta: float = BETA, mode: str | None = None, weight: float | None = None,
    native: FakeNativeEnv | None = None,
) -> tuple[BattleEnvV1, FakeNativeEnv]:
    native = native or FakeNativeEnv()
    env = BattleEnvV1(
        native=native,  # type: ignore[arg-type]
        card_catalog=_test_catalog(),
        ruleset_id="a" * 64,
        warmup_ticks=0,
        shaping_beta=beta,
        shaping_tau_ticks=TAU,
    )
    if mode is not None or weight is not None:
        env.configure_tower_shaping(
            king_shaping_mode=mode or "legacy",
            king_potential_weight=DEFAULT_KING_POTENTIAL_WEIGHT if weight is None else weight,
        )
    _reset(env)
    return env, native


def _set_hp(*changes: tuple[tuple[int, int], int], end: bool = False):
    def mutate(state: dict[str, Any], _old: int, _new: int) -> None:
        for (x, y), hp in changes:
            tower = next(item for item in state["objects"] if (item["x"], item["y"]) == (x, y))
            tower["hp"] = hp
        if end:
            state["ended"] = True

    return mutate


def _step(env: BattleEnvV1, native: FakeNativeEnv, mutate=None, *, ticks: int = 1):
    """Advance one env step; return (rewards, gamma_env, terminated)."""
    before = int(env.raw_observation["tick"])
    if mutate is not None:
        native.step_mutators.append(mutate)
    _obs, rewards, terminations, _trunc, _info = env.step(
        {0: ActionV1.wait(0, ticks=ticks), 1: ActionV1.wait(1, ticks=ticks)}
    )
    after = int(env.raw_observation["tick"])
    return rewards, math.exp(-max(1, after - before) / TAU), bool(terminations["__all__"]), after - before


# --------------------------------------------------------------------------- legacy PBRS characterization


def test_default_shaping_mode_is_legacy_with_king_weight_three() -> None:
    env, _ = _env()
    assert env.king_shaping_mode == "legacy" and env.king_potential_weight == 3.0
    assert KING_SHAPING_MODES == ("legacy", "princess_gated", "hp_point_weighted")


def test_king_tower_loss_has_weight_three_and_princess_weight_one() -> None:
    env, native = _env()
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_KING, 700)))
    assert rewards[0] == pytest.approx(BETA * gamma * 3.0 * 0.3)
    env, native = _env()
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_LEFT, 700)))
    assert rewards[0] == pytest.approx(BETA * gamma * 1.0 * 0.3)


def test_hp_point_half_princess_is_point_four_with_beta_point_eight() -> None:
    env, native = _env(beta=.8, mode="hp_point_weighted", weight=1.)
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_LEFT, 500)))
    assert rewards[0] == pytest.approx(.4 * gamma)
    assert rewards[1] == pytest.approx(-rewards[0])


def test_hp_point_king_weight_uses_actual_hp_not_king_loss_fraction() -> None:
    class BiggerKingNative(FakeNativeEnv):
        def _initial_state(self, match):
            state = super()._initial_state(match)
            for tower in state["objects"]:
                if tower.get("kind") == "king":
                    tower["hp"] = tower["maxHp"] = 2000
            return state

    env, native = _env(beta=.8, mode="hp_point_weighted", weight=1., native=BiggerKingNative())
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_KING, 1900)))
    # Same 100 HP as 10% of a Princess, discounted to 0.2 before its side tower falls.
    assert rewards[0] == pytest.approx(.8 * gamma * .1 * .2)


def test_hp_point_gate_does_not_reprice_old_damage_and_later_hp_is_one_to_one() -> None:
    env, native = _env(beta=.8, mode="hp_point_weighted", weight=1.)
    _step(env, native, _set_hp((OWNER1_KING, 500)))
    assert env._last_potential[0] == pytest.approx(.5 * .2)
    before = env._last_potential[0]
    _step(env, native, _set_hp((OWNER1_LEFT, 0)))
    assert env._last_potential[0] - before == pytest.approx(2.)  # one crown + Princess HP
    before = env._last_potential[0]
    _step(env, native, _set_hp((OWNER1_KING, 400)))
    assert env._last_potential[0] - before == pytest.approx(.1)
    # The victim side controls this gate; damaging our King is still worth only 0.2.
    before = env._last_potential[0]
    _step(env, native, _set_hp((OWNER0_KING, 900)))
    assert env._last_potential[0] - before == pytest.approx(-.1 * .2)


def test_hp_point_shaping_preserves_discounted_win_reward_at_large_beta() -> None:
    env, native = _env(beta=.8, mode="hp_point_weighted", weight=1.)
    total, discount = 0., 1.
    for changes in [((OWNER1_KING, 700),), ((OWNER1_LEFT, 0),), ((OWNER1_KING, 500),)]:
        rewards, gamma, _, _ = _step(env, native, _set_hp(*changes))
        total += discount * rewards[0]
        discount *= gamma
    rewards, _, ended, _ = _step(env, native, _set_hp((OWNER1_KING, 0), end=True))
    assert ended and env._last_potential[0] == 0.
    total += discount * rewards[0]
    assert total == pytest.approx(discount)  # Only the discounted terminal +1 survives.


def test_legacy_splash_on_sleeping_king_is_rewarded_positively() -> None:
    """Pins the audited exploit: King splash multiplies the princess-only shaping."""
    env, native = _env()
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_LEFT, 700), (OWNER1_KING, 700)))
    princess_only = BETA * gamma * 0.3
    assert rewards[0] == pytest.approx(princess_only + BETA * gamma * 0.9)
    assert rewards[0] > princess_only > 0.0


def test_destroyed_princess_tower_adds_one_crown_to_the_potential() -> None:
    env, native = _env()
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_LEFT, 0)))
    # +1 crown and a full princess loss.
    assert rewards[0] == pytest.approx(BETA * gamma * (1.0 + 1.0))


def test_terminal_potential_is_zero() -> None:
    env, native = _env()
    _step(env, native, _set_hp((OWNER1_KING, 700)))
    rewards, _gamma, terminated, _ = _step(env, native, _set_hp((OWNER1_KING, 0), end=True))
    assert terminated
    # Win +1 and the potential accumulated so far (3 * 0.3) is handed back.
    assert rewards[0] == pytest.approx(1.0 - BETA * 0.9)
    assert rewards[1] == pytest.approx(-rewards[0])


def test_gamma_follows_actual_tick_delta_not_a_fixed_five() -> None:
    env, native = _env()
    _step(env, native, _set_hp((OWNER1_KING, 700)))
    rewards, gamma, _, delta = _step(env, native, ticks=7)
    assert delta == 7
    assert gamma == pytest.approx(math.exp(-7 / TAU))
    # Unchanged towers leak (gamma - 1) * beta * Phi.
    assert rewards[0] == pytest.approx(BETA * (gamma - 1.0) * 0.9)


@pytest.mark.parametrize("mode", ["legacy", "princess_gated"])
def test_shaping_is_zero_sum_and_discounted_sum_telescopes_to_zero(mode: str) -> None:
    env, native = _env(mode=mode)
    script = [
        (_set_hp((OWNER1_KING, 900)), 1),
        (_set_hp((OWNER0_KING, 850), (OWNER1_LEFT, 600)), 3),
        (None, 7),
        (_set_hp((OWNER1_LEFT, 0)), 2),
        (_set_hp((OWNER1_KING, 500), (OWNER0_LEFT, 300)), 5),
        (_set_hp((OWNER1_KING, 0), end=True), 1),
    ]
    discounted = 0.0
    discount = 1.0
    terminated = False
    for mutate, ticks in script:
        rewards, gamma, terminated, _ = _step(env, native, mutate, ticks=ticks)
        assert rewards[0] + rewards[1] == pytest.approx(0.0, abs=1e-12)
        discounted += discount * rewards[0]
        discount *= gamma
    assert terminated
    # Phi_0 = 0 and Phi_T = 0: sum_t prod(gamma_<t) * shaping_t = gamma_T*Phi_T - Phi_0 = 0,
    # so only the discounted terminal win survives.
    assert discounted == pytest.approx(discount / gamma * 1.0, abs=1e-12)


# --------------------------------------------------------------------------- King gate (scheme C)


def test_gated_king_damage_is_excluded_while_all_princess_towers_stand() -> None:
    env, native = _env(mode="princess_gated")
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_LEFT, 700), (OWNER1_KING, 700)))
    # Only the princess part of the legacy reward survives.
    assert rewards[0] == pytest.approx(BETA * gamma * 0.3)
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_KING, 400)))
    assert rewards[0] == pytest.approx(BETA * (gamma - 1.0) * 0.3)
    assert env._king_loss_baseline == {0: 0.0, 1: pytest.approx(0.6)}


def test_gated_own_king_damage_is_also_excluded_symmetrically() -> None:
    env, native = _env(mode="princess_gated")
    rewards, _gamma, _, _ = _step(env, native, _set_hp((OWNER0_KING, 500)))
    assert rewards == {0: pytest.approx(0.0), 1: pytest.approx(0.0)}
    env, native = _env(mode="princess_gated")
    mirrored, _gamma, _, _ = _step(env, native, _set_hp((OWNER1_KING, 500)))
    assert mirrored == {0: pytest.approx(0.0), 1: pytest.approx(0.0)}


@pytest.mark.parametrize("fallen", [OWNER1_LEFT, OWNER1_RIGHT])
def test_gated_baseline_freezes_once_a_victim_side_princess_tower_falls(fallen: tuple[int, int]) -> None:
    env, native = _env(mode="princess_gated")
    _step(env, native, _set_hp((OWNER1_KING, 700)))
    _step(env, native, _set_hp((fallen, 0)))
    baseline = dict(env._king_loss_baseline)
    assert baseline == {0: 0.0, 1: pytest.approx(0.3)}
    phi_before = env._last_potential[0]
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_KING, 400)))
    # The later 0.3 loss counts at full weight 3; the earlier 0.3 never does.
    assert env._last_potential[0] - phi_before == pytest.approx(3.0 * 0.3)
    assert rewards[0] == pytest.approx(BETA * (gamma * (phi_before + 0.9) - phi_before))
    assert env._king_loss_baseline == baseline


def test_gated_enemy_king_stays_excluded_after_my_princess_tower_falls() -> None:
    """Losing MY tower does not wake the enemy King: chipping it is still the same exploit."""
    env, native = _env(mode="princess_gated")
    _step(env, native, _set_hp((OWNER1_KING, 700)))
    # Owner 0 loses a Princess Tower; both of owner 1's still stand.
    _step(env, native, _set_hp((OWNER0_LEFT, 0)))
    phi_before = env._last_potential[0]
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_KING, 400)))
    assert env._last_potential[0] == pytest.approx(phi_before)
    assert rewards[0] == pytest.approx(BETA * (gamma - 1.0) * phi_before)
    assert env._king_loss_baseline == {0: 0.0, 1: pytest.approx(0.6)}
    # Owner 0's own King is now ungated: damage to it counts at weight 3 against owner 0.
    phi_before = env._last_potential[0]
    _step(env, native, _set_hp((OWNER0_KING, 800)))
    assert env._last_potential[0] - phi_before == pytest.approx(-3.0 * 0.2)
    assert env._king_loss_baseline == {0: 0.0, 1: pytest.approx(0.6)}


def test_gated_terminal_clears_potential_and_early_king_chip_is_never_refunded() -> None:
    env, native = _env(mode="princess_gated")
    _step(env, native, _set_hp((OWNER1_KING, 700)))
    assert env._last_potential[0] == pytest.approx(0.0)
    rewards, _gamma, terminated, _ = _step(env, native, _set_hp((OWNER1_KING, 0), end=True))
    assert terminated
    assert rewards[0] == pytest.approx(1.0)


def test_gated_baseline_resets_between_episodes() -> None:
    env, native = _env(mode="princess_gated")
    _step(env, native, _set_hp((OWNER1_KING, 700)))
    _step(env, native, _set_hp((OWNER1_KING, 0), end=True))
    _reset(env)
    assert env._king_loss_baseline == {0: 0.0, 1: 0.0}
    assert env.king_shaping_mode == "princess_gated"
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_LEFT, 700), (OWNER1_KING, 700)))
    assert rewards[0] == pytest.approx(BETA * gamma * 0.3)


def test_beta_zero_has_no_shaping_in_any_mode() -> None:
    for mode in KING_SHAPING_MODES:
        env, native = _env(beta=0.0, mode=mode)
        rewards, _gamma, _, _ = _step(env, native, _set_hp((OWNER1_KING, 700), (OWNER1_LEFT, 0)))
        assert rewards == {0: 0.0, 1: 0.0}


def test_legacy_mode_matches_unconfigured_environment_exactly() -> None:
    script = [_set_hp((OWNER1_KING, 700)), _set_hp((OWNER0_LEFT, 0)), _set_hp((OWNER1_KING, 300))]
    plain, plain_native = _env()
    legacy, legacy_native = _env(mode="legacy", weight=3.0)
    for mutate in script:
        assert _step(plain, plain_native, mutate)[0] == _step(legacy, legacy_native, mutate)[0]


def test_king_potential_weight_is_configurable() -> None:
    env, native = _env(weight=1.6)
    rewards, gamma, _, _ = _step(env, native, _set_hp((OWNER1_KING, 700)))
    assert rewards[0] == pytest.approx(BETA * gamma * 1.6 * 0.3)


def test_tower_shaping_knobs_stay_out_of_environment_config_and_ruleset() -> None:
    plain, _ = _env()
    gated, _ = _env(mode="princess_gated", weight=1.6)
    assert gated.environment_config == plain.environment_config
    assert "king" not in " ".join(gated.environment_config.to_dict())
    assert gated.episode_config.environment == plain.episode_config.environment
    assert gated.ruleset_id == plain.ruleset_id


def test_configure_tower_shaping_validates_and_refuses_mid_episode_changes() -> None:
    env, native = _env()
    with pytest.raises(ValueError):
        env.configure_tower_shaping(king_shaping_mode="awake_only")
    with pytest.raises(ValueError):
        env.configure_tower_shaping(king_potential_weight=float("nan"))
    with pytest.raises(ValueError):
        env.configure_tower_shaping(king_potential_weight=-1.0)
    with pytest.raises(RuntimeError):
        env.configure_tower_shaping(king_shaping_mode="princess_gated")
    env.configure_tower_shaping()  # unchanged knobs are a no-op mid-episode
    _step(env, native, _set_hp((OWNER1_KING, 0), end=True))
    env.configure_tower_shaping(king_shaping_mode="princess_gated")
    assert env.king_shaping_mode == "princess_gated"
