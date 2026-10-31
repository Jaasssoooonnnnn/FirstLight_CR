"""Tower Troop runtime records evicted from the probe ring are carried forward.

The probe publishes Dagger Duchess / Royal Chef runtime only when the native
Action ticks, into one 256-slot ring shared by every special tower.  A record
that stops being refreshed is overwritten by an opposing Dagger Duchess (one
record per tower per tick), and the tower then arrives without
``towerTroopRuntime``.  BattleEnv keeps the last record of the same native
object in the same telemetry epoch instead of failing the episode.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from native_runner.battle_env import BattleEnvError
from native_runner.card_specs import build_card_catalog
from native_runner.contracts import ActionV1, DaggerDuchessRuntimeStateV1, RoyalChefRuntimeStateV1
from native_runner.match_factory import MatchConfig
from native_runner.semantic_subset import SEMANTIC_BASELINE_DECK
from native_runner.tests.test_battle_env_rich_telemetry import (
    _RichNativeStub,
    _TransitionNativeStub,
    _environment,
    _ordinary,
    _reset as _reset_rich,
    _rich,
    _set_tower_troop_runtime,
)

DAGGER = {
    "schema": "native-tower-troop-runtime.v1",
    "kind": "dagger_duchess",
    "observedTick": 20,
    "chargeCount": 2,
    "maxChargeCount": 8,
    "rechargeElapsedMs": 600,
    "rechargeDurationMs": 900,
}
CHEF = {
    "schema": "native-tower-troop-runtime.v1",
    "kind": "royal_chef",
    "observedTick": 20,
    "startDelayRemainingMs": 0,
    "startDelayDurationMs": 7_000,
    "cookingContribution": 345_000,
    "contributionNeeded": 460_000,
    "throwDelayRemainingMs": None,
    "targetNativeObjectId": None,
}
MATCH = MatchConfig(
    deck0=SEMANTIC_BASELINE_DECK, deck1=SEMANTIC_BASELINE_DECK, tower_troop0_id=159_000_002, tower_troop1_id=159_000_004
)


def _rich_with(ordinary, *, chef: bool, dagger_tick: int = 20):
    rich = _rich(ordinary)
    by_slot = {1: {**DAGGER, "observedTick": dagger_tick}, 2: {**DAGGER, "observedTick": dagger_tick}}
    if chef:
        by_slot[3] = dict(CHEF)
    _set_tower_troop_runtime(rich, by_slot=by_slot)
    return rich


def _chef_king(observation):
    return next(t for t in observation.towers if t.owner == 1 and t.tower_kind == "king")


def test_evicted_royal_chef_record_is_carried_forward_within_the_epoch() -> None:
    before = _ordinary()
    after = deepcopy(before)
    after["tick"] = 21
    # The Dagger Duchess towers keep publishing; the frozen Chef record was overwritten.
    native = _TransitionNativeStub(
        before, _rich_with(before, chef=True), after, _rich_with(after, chef=False, dagger_tick=21)
    )
    env = _environment(native, build_card_catalog())
    initial, _ = _reset_rich(env, match_config=MATCH)
    assert _chef_king(initial[0]).tower_troop_runtime.cooking_contribution == 345_000  # type: ignore[union-attr]

    observations, *_ = env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})
    for observation in observations.values():
        chef = _chef_king(observation).tower_troop_runtime
        assert isinstance(chef, RoyalChefRuntimeStateV1)
        assert (chef.cooking_contribution, chef.contribution_needed, chef.target_entity) == (345_000, 460_000, None)
        sides = [t for t in observation.towers if t.owner == 0 and t.tower_kind != "king"]
        assert all(isinstance(t.tower_troop_runtime, DaggerDuchessRuntimeStateV1) for t in sides)


def test_never_observed_special_tower_runtime_still_fails_closed() -> None:
    ordinary = _ordinary()
    env = _environment(_RichNativeStub(ordinary, _rich_with(ordinary, chef=False)), build_card_catalog())
    with pytest.raises(BattleEnvError, match="lacks its native runtime snapshot"):
        _reset_rich(env, match_config=MATCH)


def _item(native_object_id: int, record):
    return SimpleNamespace(native_object_id=native_object_id, tower_troop_runtime=record)


def _snapshot(generation: int, state_epoch: int):
    return SimpleNamespace(generation=generation, state_epoch=state_epoch)


def test_carry_is_scoped_to_generation_epoch_and_native_object() -> None:
    ordinary = _ordinary()
    env = _environment(_RichNativeStub(ordinary, _rich_with(ordinary, chef=True)), build_card_catalog())
    _reset_rich(env, match_config=MATCH)
    record = object()
    carry = env._carried_tower_troop_runtime
    assert carry(3, _item(77, record), _snapshot(5, 9), relevant=True) is record
    assert carry(3, _item(77, None), _snapshot(5, 9), relevant=True) is record
    # Irrelevant towers (dead / not special) never resurrect a record.
    assert carry(3, _item(77, None), _snapshot(5, 9), relevant=False) is None
    for item, snapshot in (
        (_item(78, None), _snapshot(5, 9)),  # different native object
        (_item(77, None), _snapshot(5, 10)),  # snapshot restore / new state epoch
        (_item(77, None), _snapshot(6, 9)),  # new match generation
    ):
        assert carry(3, item, snapshot, relevant=True) is None
    assert carry(4, _item(77, None), _snapshot(5, 9), relevant=True) is None  # per tower
    newer = object()
    assert carry(3, _item(77, newer), _snapshot(5, 9), relevant=True) is newer
    assert carry(3, _item(77, None), _snapshot(5, 9), relevant=True) is newer


def test_reset_clears_carried_runtime() -> None:
    ordinary = _ordinary()
    env = _environment(_RichNativeStub(ordinary, _rich_with(ordinary, chef=True)), build_card_catalog())
    _reset_rich(env, match_config=MATCH)
    assert env._tower_troop_runtime_carry
    env._tower_troop_runtime_carry[99] = (0, 0, 0, object())
    _reset_rich(env, match_config=MATCH)
    assert 99 not in env._tower_troop_runtime_carry
