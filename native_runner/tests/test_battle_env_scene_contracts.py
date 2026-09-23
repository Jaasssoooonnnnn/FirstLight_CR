from __future__ import annotations

from copy import deepcopy

import pytest

from native_runner.battle_env import ENTITY_LIMIT, BattleEnvError
from native_runner.card_specs import build_card_catalog
from native_runner.contracts import (
    ActionV1,
    CombatEventKind,
    ContractError,
    DaggerDuchessRuntimeStateV1,
    EnvironmentConfigV1,
    SemanticEvidenceLevel,
    RoyalChefRuntimeStateV1,
    TowerStateV1,
)
from native_runner.match_factory import MatchConfig
from native_runner.semantic_subset import SEMANTIC_BASELINE_DECK
from native_runner.tests.test_battle_env_rich_telemetry import (
    _RichNativeStub,
    _TransitionNativeStub,
    _combat_fact,
    _environment,
    _null_combat_fact,
    _ordinary,
    _reset as _reset_rich,
    _rich,
    _set_remaining_runtime,
    _set_tower_troop_runtime,
)


def _wait_actions() -> dict[int, ActionV1]:
    return {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}


def _tower_activation_event(
    ordinary: dict[str, object],
    *,
    sequence: int,
    tick: int,
    tower_index: int,
) -> dict[str, object]:
    absent = _null_combat_fact()
    return {
        "sequence": sequence,
        "tick": tick,
        "kind": "tower_activate",
        "hookOffset": 0xF23110,
        "callerOffset": 0xF58820,
        "entity": _combat_fact(ordinary["objects"][tower_index]),  # type: ignore[index]
        "source": absent,
        "target": absent,
        "targetBefore": absent,
        "targetAfter": absent,
        "objectKind": None,
        "runtimeVtableOffset": None,
        "dataBeforeGlobalId": None,
        "dataAfterGlobalId": None,
        "expectedCharacterDataGlobalId": None,
        "expectedProjectileDataGlobalId": None,
        "configuredDataGlobalId": None,
        "resourcePreFixed": None,
        "resourcePostFixed": None,
        "resourceActualDeltaFixed": None,
        "amountArgument": None,
        "configuredAmountArgument": None,
        "resourceOwner": None,
        "areaRemainingLifeMs": None,
        "resourceCause": "none",
        "transformKind": "none",
        "result": True,
        "option": False,
        "committed": False,
        "resetTarget": False,
        "completeContext": True,
    }


def test_default_live_entity_contract_is_wider_than_v4_child_memory() -> None:
    assert ENTITY_LIMIT == 256
    assert EnvironmentConfigV1().entity_limit == 256
    before = _ordinary()
    troop = before["objects"][6]
    before["objects"] = before["objects"][:6]
    before["count"] = before["returned"] = 6
    after = deepcopy(before)
    after["tick"] = 21
    after["objects"].extend(
        {
            **troop,
            "slot": 6 + index,
            "nativeObjectId": 5_000_006 + index,
            "objectIndex": 1_000 + index,
            "owner": index % 2,
            "cardId": SEMANTIC_BASELINE_DECK[index % 8],
            "x": 500 + (index % 18) * 1_000,
            "y": 500 + (index % 32) * 1_000,
        }
        for index in range(265)
    )
    after["count"] = after["returned"] = len(after["objects"])
    env = _environment(
        _TransitionNativeStub(before, _rich(before), after, _rich(after)),
        build_card_catalog(),
    )
    _reset_rich(env)
    observations, *_ = env.step(_wait_actions())
    for observation in observations.values():
        assert len(observation.entities) == 256
        assert observation.metadata["entity_limit"] == 256
        assert observation.metadata["entity_count_before_limit"] == 265
        assert observation.metadata["entity_overflow_count"] == 9
        assert observation.truncated is True


def test_fair_hp_delta_events_name_the_victim_without_inventing_a_source() -> None:
    before = _ordinary()
    after = deepcopy(before)
    after["tick"] = 21
    after["objects"][6]["hp"] = 70
    next(
        item
        for item in after["objects"]
        if item["x"] == 3500 and item["y"] == 25500
    )["hp"] = 800
    env = _environment(
        _TransitionNativeStub(before, _rich(before), after, _rich(after)),
        build_card_catalog(),
    )
    _reset_rich(env)
    observations, *_ = env.step(_wait_actions())
    payloads: list[list[dict[str, object]]] = []
    for observation in observations.values():
        damage_events = [
            event
            for event in observation.events
            if event.event_type in {"damage", "tower_damage"}
        ]
        assert {event.event_type for event in damage_events} == {
            "damage",
            "tower_damage",
        }
        payloads.append([event.to_dict() for event in damage_events])
        for event in damage_events:
            assert event.combat is not None
            assert event.combat.kind == CombatEventKind.DAMAGE
            assert event.combat.target_entity == event.entity_id
            assert event.combat.amount == float(event.data["amount"])
            assert event.combat.source_entity is None
            assert event.combat.source_card_id is None
            assert event.combat.visible_by_owner == {}
            evidence = event.combat.provenance.field_evidence
            assert evidence["source_entity"] == SemanticEvidenceLevel.UNKNOWN
            assert evidence["source_card_id"] == SemanticEvidenceLevel.UNKNOWN
            assert evidence["target_entity"] == SemanticEvidenceLevel.NATIVE_DERIVED
            assert evidence["amount"] == SemanticEvidenceLevel.NATIVE_DERIVED
            assert (
                event.runtime_provenance.field_evidence["combat"]
                == SemanticEvidenceLevel.NATIVE_DERIVED
            )
            assert "oracle_only" not in event.data
            assert "private_to" not in event.data
        troop_damage = next(
            event for event in damage_events if event.event_type == "damage"
        )
        assert troop_damage.combat is not None
        assert troop_damage.combat.target_card_id == troop_damage.card_id
        tower_damage = next(
            event for event in damage_events if event.event_type == "tower_damage"
        )
        assert tower_damage.combat is not None
        assert tower_damage.combat.target_card_id is None
    assert payloads[0] == payloads[1]


def test_tower_troop_identity_is_owner_exact_and_competitive_only() -> None:
    ordinary = _ordinary()
    env = _environment(
        _RichNativeStub(ordinary, _special_tower_rich(ordinary)), build_card_catalog()
    )
    observations, _ = env.reset(
        match_config=MatchConfig(
            deck0=SEMANTIC_BASELINE_DECK,
            deck1=SEMANTIC_BASELINE_DECK,
            tower_troop0_id=159_000_002,
            tower_troop1_id=159_000_004,
        )
    )
    for observation in observations.values():
        for tower in observation.towers:
            if tower.owner == 0:
                expected = None if tower.tower_kind == "king" else 159_000_002
            else:
                expected = (
                    159_000_004
                    if tower.tower_kind == "king"
                    else 159_000_000
                )
            assert tower.tower_troop_id == expected

    with pytest.raises(ContractError, match="competitive pool"):
        TowerStateV1(
            entity_id=1,
            owner=0,
            tower_kind="princess_left",
            position=(3500.0, 6500.0),
            hitpoints=1.0,
            max_hitpoints=1.0,
            tower_troop_id=159_000_003,
        )
    with pytest.raises(ContractError, match="requires an exact tower troop"):
        TowerStateV1(
            entity_id=2,
            owner=0,
            tower_kind="princess_right",
            position=(14_500.0, 6_500.0),
            hitpoints=1.0,
            max_hitpoints=1.0,
        )
    with pytest.raises(ContractError, match="belongs to the king tower"):
        TowerStateV1(
            entity_id=2,
            owner=0,
            tower_kind="princess_right",
            position=(14_500.0, 6_500.0),
            hitpoints=1.0,
            max_hitpoints=1.0,
            tower_troop_id=159_000_004,
        )
    with pytest.raises(BattleEnvError, match="current competitive tower troop"):
        invalid_env = _environment(
            _RichNativeStub(ordinary, _rich(ordinary)), build_card_catalog()
        )
        invalid_env.reset(
            match_config=MatchConfig(
                deck0=SEMANTIC_BASELINE_DECK,
                deck1=SEMANTIC_BASELINE_DECK,
                tower_troop0_id=159_000_003,
            )
        )


def _special_tower_rich(ordinary: dict[str, object]) -> dict[str, object]:
    rich = _rich(ordinary)
    dagger = {
        "schema": "native-tower-troop-runtime.v1",
        "kind": "dagger_duchess",
        "observedTick": 20,
        "chargeCount": 2,
        "maxChargeCount": 8,
        "rechargeElapsedMs": 600,
        "rechargeDurationMs": 900,
    }
    _set_tower_troop_runtime(
        rich,
        by_slot={
            1: dict(dagger),
            2: {**dagger, "chargeCount": 7, "rechargeElapsedMs": 100},
            3: {
                "schema": "native-tower-troop-runtime.v1",
                "kind": "royal_chef",
                "observedTick": 20,
                "startDelayRemainingMs": 0,
                "startDelayDurationMs": 7_000,
                "cookingContribution": 345_000,
                "contributionNeeded": 460_000,
                "throwDelayRemainingMs": 100,
                "targetNativeObjectId": 5_000_006,
            },
        },
    )
    return rich


def test_special_tower_troop_runtime_reaches_fair_tower_state() -> None:
    ordinary = _ordinary()
    rich = _special_tower_rich(ordinary)
    native = _TransitionNativeStub(ordinary, rich, ordinary, rich)
    env = _environment(native, build_card_catalog())

    observations, _ = _reset_rich(
        env,
        match_config=MatchConfig(
            deck0=SEMANTIC_BASELINE_DECK,
            deck1=SEMANTIC_BASELINE_DECK,
            tower_troop0_id=159_000_002,
            tower_troop1_id=159_000_004,
        ),
    )

    for observation in observations.values():
        owner_zero_sides = sorted(
            (
                tower
                for tower in observation.towers
                if tower.owner == 0 and tower.tower_kind != "king"
            ),
            key=lambda tower: tower.entity_id,
        )
        assert [
            tower.tower_troop_runtime.charge_count  # type: ignore[union-attr]
            for tower in owner_zero_sides
        ] == [2, 7]
        assert all(
            isinstance(
                tower.tower_troop_runtime,
                DaggerDuchessRuntimeStateV1,
            )
            for tower in owner_zero_sides
        )
        chef_king = next(
            tower
            for tower in observation.towers
            if tower.owner == 1 and tower.tower_kind == "king"
        )
        assert isinstance(
            chef_king.tower_troop_runtime,
            RoyalChefRuntimeStateV1,
        )
        assert chef_king.tower_troop_runtime.cooking_contribution == 345_000
        assert chef_king.tower_troop_runtime.surviving_side_towers == 2
        assert chef_king.tower_troop_runtime.target_entity is not None
        assert (
            chef_king.runtime_provenance.field_evidence["tower_troop_runtime"]
            == SemanticEvidenceLevel.NATIVE_DERIVED
        )


def test_royal_chef_runtime_can_end_after_both_side_towers_fall() -> None:
    ordinary = _ordinary()
    for slot in (4, 5):
        ordinary["objects"][slot]["hp"] = 0  # type: ignore[index]
    rich = _rich(ordinary)
    _set_tower_troop_runtime(rich, by_slot={})
    env = _environment(_RichNativeStub(ordinary, rich), build_card_catalog())

    observations, _ = _reset_rich(
        env,
        match_config=MatchConfig(
            deck0=SEMANTIC_BASELINE_DECK,
            deck1=SEMANTIC_BASELINE_DECK,
            tower_troop1_id=159_000_004,
        ),
    )

    for observation in observations.values():
        chef_king = next(
            tower for tower in observation.towers
            if tower.owner == 1 and tower.tower_kind == "king"
        )
        assert chef_king.active
        assert chef_king.tower_troop_runtime is None
        assert (
            chef_king.runtime_provenance.field_evidence["tower_troop_runtime"]
            == SemanticEvidenceLevel.NOT_APPLICABLE
        )

    ordinary["objects"][5]["hp"] = 1  # type: ignore[index]
    rich = _rich(ordinary)
    _set_tower_troop_runtime(rich, by_slot={})
    env = _environment(_RichNativeStub(ordinary, rich), build_card_catalog())
    with pytest.raises(BattleEnvError, match="active special Tower Troop lacks"):
        _reset_rich(
            env,
            match_config=MatchConfig(
                deck0=SEMANTIC_BASELINE_DECK,
                deck1=SEMANTIC_BASELINE_DECK,
                tower_troop1_id=159_000_004,
            ),
        )


def test_exact_tower_activation_sets_public_status_from_native_event() -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_remaining_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_remaining_runtime(
        after_rich,
        events=[
            _tower_activation_event(
                after,
                sequence=200,
                tick=21,
                tower_index=0,
            )
        ],
    )
    native = _TransitionNativeStub(before, before_rich, after, after_rich)
    env = _environment(native, build_card_catalog())
    initial, _ = _reset_rich(env)
    assert all("activated" not in tower.status for tower in initial[0].towers)

    observations, *_ = env.step(_wait_actions())
    for observation in observations.values():
        activated = [tower for tower in observation.towers if "activated" in tower.status]
        assert [(tower.entity_id, tower.owner, tower.tower_kind) for tower in activated] == [
            (0, 0, "king")
        ]
    assert all(
        event.event_type != "runtime_tower_activate"
        for event in observations[0].events
    )
    native_event = next(
        event
        for event in env._events
        if event.event_type == "runtime_tower_activate"
    )
    assert native_event.entity_id == 0
    assert native_event.data["oracle_only"] is True


def test_tower_activation_rejects_a_non_king_subject() -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_remaining_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_remaining_runtime(
        after_rich,
        events=[
            _tower_activation_event(
                after,
                sequence=200,
                tick=21,
                tower_index=1,
            )
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        build_card_catalog(),
    )
    _reset_rich(env)
    with pytest.raises(BattleEnvError, match="fixed king tower"):
        env.step(_wait_actions())
