from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import pytest

from native_runner.battle_env import BattleEnvError, BattleEnvV1, TOWER_LAYOUT
from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.card_specs import build_ability_specs, build_card_catalog
from native_runner.contracts import (
    ActionKind,
    ActionV1,
    AbilityPhase,
    AttackPhase,
    CaptureActionPhase,
    CaptureTargetPhase,
    CausalGroupKind,
    CombatEventKind,
    EffectKind,
    EvolutionPhase,
    PeriodicAttackModifierPhase,
    ProjectileDragStage,
    ProjectilePhase,
    SemanticEvidenceLevel,
    TargetKind,
    ThresholdRelocationPhase,
    VisibilityPhase,
)
from native_runner.cr_native_env import AbilityAction, HandAction, RunnerError
from native_runner.effect_catalog import build_effect_catalog
from native_runner.match_factory import MatchConfig
from native_runner.phase_runtime import DamageRampIdentity
from native_runner.rich_telemetry_adapter import (
    RichTelemetryMergeError,
    RichDaggerDuchessRuntimeTelemetry,
    RichRoyalChefRuntimeTelemetry,
    RuntimeEffectCatalog,
    _damage_ramp_spec,
    bind_rich_telemetry,
    normalized_ability_cooldown_ms,
)
from native_runner.semantic_subset import SEMANTIC_BASELINE_DECK
from native_runner.training.v4.grouping import build_causal_groups
from native_runner.training.tracking import DeterministicPublicTracker


def test_august86_native_ability_cooldowns_use_exact_zero_normalization() -> None:
    abilities = {item.ability_id: item for item in build_ability_specs()}
    assert normalized_ability_cooldown_ms(abilities["ArcherQueenRapid"]) == 0
    assert normalized_ability_cooldown_ms(abilities["ChampGuardianAbility"]) == 0
    assert normalized_ability_cooldown_ms(abilities["BossBandit_ability"]) == 3_000
    assert (
        normalized_ability_cooldown_ms(
            abilities["MegaMinion_Teleport_Ability"]
        )
        == 1_000
    )


@pytest.mark.parametrize(
    ("card_name", "identity", "damage"),
    (
        ("InfernoTower", DamageRampIdentity.INFERNO_TOWER, (17, 62, 331)),
        ("InfernoDragon", DamageRampIdentity.INFERNO_DRAGON, (14, 47, 165)),
        ("MightyMiner", DamageRampIdentity.MIGHTY_MINER, (17, 80, 160)),
    ),
)
def test_competitive_continuous_damage_ramps_join_exact_static_stages(
    card_name: str,
    identity: DamageRampIdentity,
    damage: tuple[int, int, int],
) -> None:
    card_spec = next(
        spec for spec in build_card_catalog().by_id.values() if spec.name == card_name
    )

    ramp = _damage_ramp_spec(card_spec, observed_stage=2)

    assert ramp is not None
    assert ramp.identity == identity
    assert ramp.stage_durations_ms == (2_000, 2_000, 2_000)
    assert ramp.stage_damage == damage
    assert _damage_ramp_spec(card_spec, observed_stage=3) is None


def test_evolution_inferno_dragon_is_not_projected_as_a_timed_ramp() -> None:
    card_spec = next(
        spec
        for spec in build_card_catalog().by_id.values()
        if spec.name == "InfernoDragon"
    )

    assert (
        _damage_ramp_spec(
            card_spec,
            observed_stage=1,
            evolution_active=True,
        )
        is None
    )


def _players() -> list[dict[str, object]]:
    card_specs = build_card_catalog().by_id
    return [
        {
            "owner": owner,
            "accountId": 100 + owner,
            "elixirRaw": 60_000,
            "crownsRaw": 0,
            "hand": [
                {
                    "handIndex": index,
                    "deckSlot": index,
                    "cardId": card_id,
                    "commandCardId": card_id,
                    "cost": int(card_specs[card_id].elixir_cost or 0),
                    "cardParameter": (
                        int(card_specs[card_id].elixir_cost or 0) << 28
                    )
                    | ((index + 1) << 22),
                }
                for index, card_id in enumerate(SEMANTIC_BASELINE_DECK[:4])
            ],
            "nextCard": {"cardId": SEMANTIC_BASELINE_DECK[4]},
            "deck": [
                {"deckSlot": index, "cardId": card_id}
                for index, card_id in enumerate(SEMANTIC_BASELINE_DECK)
            ],
            "cycle": [
                {"cardId": card_id} for card_id in SEMANTIC_BASELINE_DECK[4:]
            ],
        }
        for owner in (0, 1)
    ]


def _ordinary() -> dict[str, object]:
    objects: list[dict[str, object]] = [
        {
            "slot": slot,
            "nativeObjectId": 5_000_000 + slot,
            "objectIndex": 100 + slot,
            "secondaryIndex": 0,
            "owner": owner,
            "cardId": -1,
            "x": x,
            "y": y,
            "targetX": x,
            "targetY": y,
            "hp": 3_000,
            "maxHp": 3_000,
        }
        for slot, (_, owner, _, x, y) in enumerate(TOWER_LAYOUT)
    ]
    objects.extend(
        (
            {
                "slot": 6,
                "nativeObjectId": 5_000_006,
                "objectIndex": 10,
                "secondaryIndex": 0,
                "owner": 0,
                "cardId": SEMANTIC_BASELINE_DECK[0],
                "x": 8_000,
                "y": 12_000,
                "targetX": 8_500,
                "targetY": 20_000,
                "hp": 800,
                "maxHp": 1_000,
            },
            {
                "slot": 7,
                "nativeObjectId": 5_000_007,
                "objectIndex": 11,
                "secondaryIndex": 0,
                "owner": 1,
                "cardId": SEMANTIC_BASELINE_DECK[1],
                "x": 8_500,
                "y": 20_000,
                "targetX": 8_000,
                "targetY": 12_000,
                "hp": 700,
                "maxHp": 900,
            },
        )
    )
    return {
        "tick": 20,
        "generation": 3,
        "stateEpoch": 9,
        "count": len(objects),
        "returned": len(objects),
        "truncated": False,
        "objects": objects,
        "players": _players(),
        "queuedCommands": 0,
        "ended": False,
        "finalized": False,
        "crownsRaw": [0, 0],
    }


def _empty_combat_events(
    ordinary: dict[str, object],
) -> dict[str, object]:
    return {
        "ok": True,
        "schema": "native-combat-events.v1",
        "generation": ordinary["generation"],
        "stateEpoch": ordinary["stateEpoch"],
        "observationTick": ordinary["tick"],
        "capacity": 1024,
        "hookSetAttested": False,
        "hookSetInstalled": False,
        "capability": {
            "status": "unavailable",
            "source": "exact-build-native-hook-ring",
            "validation": "test-exact-build-guards",
            "confidence": "high",
            "failClosed": True,
        },
        "epochFirstSequence": 1,
        "oldestRetainedSequence": 1,
        "nextSequence": 1,
        "overflowCount": 0,
        "sequenceGapBeforeOldest": False,
        "rejectedCaptureCount": 0,
        "complete": True,
        "events": [],
    }


def _combat_fact(
    ordinary_object: dict[str, object],
    *,
    invisible_count: int = 0,
) -> dict[str, object]:
    native_object_id = int(ordinary_object["nativeObjectId"])
    return {
        "validated": True,
        "present": True,
        "nativeObjectId": native_object_id,
        "entityKey": [
            ordinary_object["owner"],
            -2,
            native_object_id,
        ],
        "owner": ordinary_object["owner"],
        "objectIndex": ordinary_object["objectIndex"],
        "secondaryIndex": ordinary_object["secondaryIndex"],
        "cardId": ordinary_object["cardId"],
        "objectKind": 1,
        "position": [ordinary_object["x"], ordinary_object["y"]],
        "visibilityValidated": True,
        "invisibleCount": invisible_count,
    }


def _null_combat_fact(*, validated: bool = True) -> dict[str, object]:
    return {
        "validated": validated,
        "present": False,
        "nativeObjectId": None,
        "entityKey": None,
        "owner": None,
        "objectIndex": None,
        "secondaryIndex": None,
        "cardId": None,
        "objectKind": None,
        "position": None,
        "visibilityValidated": False,
        "invisibleCount": None,
    }


def _damage_combat_event(
    ordinary: dict[str, object],
    *,
    sequence: int,
    tick: int,
    hidden_target: bool = False,
) -> dict[str, object]:
    source_object = ordinary["objects"][6]  # type: ignore[index]
    target_object = ordinary["objects"][7]  # type: ignore[index]
    source = _combat_fact(source_object)
    target = _combat_fact(
        target_object,
        invisible_count=1 if hidden_target else 0,
    )
    absent = _null_combat_fact()
    return {
        "sequence": sequence,
        "tick": tick,
        "generation": ordinary["generation"],
        "stateEpoch": ordinary["stateEpoch"],
        "kind": "damage",
        "hookOffset": 0xF642C4,
        "callerOffset": None,
        "causeSequence": None,
        "pool": "hitpoints",
        "terminalReason": "none",
        "lethal": False,
        "deploymentContext": None,
        "target": target,
        "immediateSource": source,
        "source": source,
        "projectile": absent,
        "related": absent,
        "routeSourceBefore": absent,
        "routeTargetBefore": absent,
        "routeSourceAfter": absent,
        "routeTargetAfter": absent,
        "requestedAmount": 50,
        "actualAmount": 50,
        "preHp": 700,
        "postHp": 650,
        "preBuiltInShield": 0,
        "postBuiltInShield": 0,
        "preBuffShield": None,
        "postBuffShield": None,
        "destinationBefore": None,
        "destinationAfter": None,
    }


def _evolved_spawn_combat_event(
    ordinary: dict[str, object],
    *,
    sequence: int,
    tick: int,
) -> dict[str, object]:
    target_object = ordinary["objects"][6]  # type: ignore[index]
    target = _combat_fact(target_object)
    absent = _null_combat_fact()
    deck_slot = 1
    cost = 3
    card_parameter = (cost << 28) | ((deck_slot + 1) << 22) | 1
    return {
        "sequence": sequence,
        "tick": tick,
        "generation": ordinary["generation"],
        "stateEpoch": ordinary["stateEpoch"],
        "kind": "spawn",
        "hookOffset": 0xF25B8C,
        "callerOffset": None,
        "causeSequence": None,
        "pool": "none",
        "terminalReason": "none",
        "lethal": False,
        "deploymentContext": {
            "deploymentSequence": 11,
            "owner": target_object["owner"],
            "playedCardGlobalId": target_object["cardId"],
            "effectiveCardGlobalId": 26_000_101,
            "cardParameter": card_parameter,
            "deckSlot": deck_slot,
            "cost": cost,
            "formCode": 1,
            "formName": "EvoForm",
            "consumeHookOffset": 0xF38A68,
        },
        "target": target,
        "immediateSource": absent,
        "source": absent,
        "projectile": absent,
        "related": absent,
        "routeSourceBefore": absent,
        "routeTargetBefore": absent,
        "routeSourceAfter": absent,
        "routeTargetAfter": absent,
        "requestedAmount": None,
        "actualAmount": None,
        "preHp": None,
        "postHp": None,
        "preBuiltInShield": None,
        "postBuiltInShield": None,
        "preBuffShield": None,
        "postBuffShield": None,
        "destinationBefore": None,
        "destinationAfter": None,
    }


def _causal_spawn_combat_event(
    ordinary: dict[str, object],
    *,
    target_index: int,
    sequence: int,
    tick: int,
    source_index: int | None = None,
    deployment_card_id: int | None = None,
    deployment_sequence: int = 1,
    projectile: bool = False,
    object_kind: int | None = None,
    cause_sequence: int | None = None,
) -> dict[str, object]:
    target_object = ordinary["objects"][target_index]  # type: ignore[index]
    target = _combat_fact(target_object)
    target["objectKind"] = (
        int(object_kind)
        if object_kind is not None
        else 4
        if projectile
        else 1
    )
    absent = _null_combat_fact()
    source = (
        _combat_fact(ordinary["objects"][source_index])  # type: ignore[index]
        if source_index is not None
        else absent
    )
    deployment = None
    if deployment_card_id is not None:
        deck_slot = SEMANTIC_BASELINE_DECK.index(deployment_card_id)
        cost = int(
            build_card_catalog().by_id[deployment_card_id].elixir_cost or 0
        )
        deployment = {
            "deploymentSequence": deployment_sequence,
            "owner": target_object["owner"],
            "playedCardGlobalId": deployment_card_id,
            "effectiveCardGlobalId": deployment_card_id,
            "cardParameter": (cost << 28) | ((deck_slot + 1) << 22),
            "deckSlot": deck_slot,
            "cost": cost,
            "formCode": 0,
            "formName": "BasicForm",
            "consumeHookOffset": 0xF38A68,
        }
    return {
        "sequence": sequence,
        "tick": tick,
        "generation": ordinary["generation"],
        "stateEpoch": ordinary["stateEpoch"],
        "kind": "projectile_spawn" if projectile else "spawn",
        "hookOffset": 0xF25B8C,
        "callerOffset": None,
        "causeSequence": cause_sequence,
        "pool": "none",
        "terminalReason": "none",
        "lethal": False,
        "deploymentContext": deployment,
        "target": target,
        "immediateSource": source,
        "source": source,
        "projectile": target if projectile else absent,
        "related": absent,
        "routeSourceBefore": absent,
        "routeTargetBefore": absent,
        "routeSourceAfter": absent,
        "routeTargetAfter": absent,
        "requestedAmount": None,
        "actualAmount": None,
        "preHp": None,
        "postHp": None,
        "preBuiltInShield": None,
        "postBuiltInShield": None,
        "preBuffShield": None,
        "postBuffShield": None,
        "destinationBefore": None,
        "destinationAfter": (
            [target_object["x"], target_object["y"]]
            if projectile
            else None
        ),
    }


def _causal_test_object(
    *,
    slot: int,
    owner: int,
    card_id: int,
    x: int,
    y: int,
) -> dict[str, object]:
    return {
        "slot": slot,
        "nativeObjectId": 6_000_000 + slot,
        "objectIndex": 200 + slot,
        "secondaryIndex": 0,
        "owner": owner,
        "cardId": card_id,
        "x": x,
        "y": y,
        "targetX": x,
        "targetY": y,
        "hp": 100,
        "maxHp": 100,
    }


def _card_play_combat_event(
    ordinary: dict[str, object],
    *,
    sequence: int,
    tick: int,
    owner: int,
    card_id: int,
    deployment_sequence: int,
) -> dict[str, object]:
    absent = _null_combat_fact()
    deck_slot = SEMANTIC_BASELINE_DECK.index(card_id)
    cost = 3
    card_parameter = (cost << 28) | ((deck_slot + 1) << 22)
    return {
        "sequence": sequence,
        "tick": tick,
        "generation": ordinary["generation"],
        "stateEpoch": ordinary["stateEpoch"],
        "kind": "card_play",
        "hookOffset": 0xF38A68,
        "callerOffset": None,
        "causeSequence": None,
        "pool": "none",
        "terminalReason": "none",
        "lethal": False,
        "deploymentContext": {
            "deploymentSequence": deployment_sequence,
            "owner": owner,
            "playedCardGlobalId": card_id,
            "effectiveCardGlobalId": card_id,
            "cardParameter": card_parameter,
            "deckSlot": deck_slot,
            "cost": cost,
            "formCode": 0,
            "formName": "BasicForm",
            "consumeHookOffset": 0xF38A68,
        },
        "target": absent,
        "immediateSource": absent,
        "source": absent,
        "projectile": absent,
        "related": absent,
        "routeSourceBefore": absent,
        "routeTargetBefore": absent,
        "routeSourceAfter": absent,
        "routeTargetAfter": absent,
        "requestedAmount": None,
        "actualAmount": None,
        "preHp": None,
        "postHp": None,
        "preBuiltInShield": None,
        "postBuiltInShield": None,
        "preBuffShield": None,
        "postBuffShield": None,
        "destinationBefore": None,
        "destinationAfter": None,
    }


def _set_combat_events(
    rich: dict[str, object],
    *,
    epoch_first_sequence: int,
    events: list[dict[str, object]],
) -> None:
    rich["combatEvents"] = {
        **_empty_combat_events(
            {
                "generation": rich["generation"],
                "stateEpoch": rich["stateEpoch"],
                "tick": rich["tick"],
            }
        ),
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": {
            "status": "derived",
            "source": "exact-build-native-hook-ring",
            "validation": "test-exact-build-guards",
            "confidence": "high",
            "failClosed": True,
        },
        "epochFirstSequence": epoch_first_sequence,
        "oldestRetainedSequence": epoch_first_sequence,
        "nextSequence": epoch_first_sequence + len(events),
        "events": events,
    }


def _phase_object(
    *,
    tick: int,
    attack_timeline_ms: int = 0,
    load_remaining_ms: int = 0,
    deploy_remaining_ms: int = 0,
    deploy_previous_ms: int = 0,
    attack_step: tuple[int, int] | None = None,
    movement_step: tuple[int, int] | None = None,
    deploy_step: tuple[int, int] | None = None,
    effective_speed: int | None = None,
    movement_delta: int | None = None,
    attack_sequence_stage: int = 0,
    attack_sequence_progress: int | None = None,
    attack_sequence_decay_remaining_ms: int | None = None,
) -> dict[str, object]:
    result = {
        "schema": "native-phase-object.v1",
        "attackValidated": True,
        "movementValidated": True,
        "buffsValidated": True,
        "attackSequenceStage": attack_sequence_stage,
        "attackTimelineMs": attack_timeline_ms,
        "loadRemainingMs": load_remaining_ms,
        "deployRemainingMs": deploy_remaining_ms,
        "deployPreviousMs": deploy_previous_ms,
        "configuredDeployTimeMs": 1_000,
        "hitSpeedMs": 400,
        "attackDashTimeMs": 0,
        "baseMovementSpeed": 100,
        "chargeSpeedMultiplier": 200,
        "classicChargeProgress": 500,
        "speedPositivePercent": 130,
        "speedNegativeMagnitude": 30,
        "hitSpeedPositivePercent": 130,
        "hitSpeedNegativeMagnitude": 30,
        "attackStepInput": None if attack_step is None else attack_step[0],
        "attackStepOutput": None if attack_step is None else attack_step[1],
        "attackStepTick": -1 if attack_step is None else tick,
        "movementStepInput": None if movement_step is None else movement_step[0],
        "movementStepOutput": None if movement_step is None else movement_step[1],
        "movementStepTick": -1 if movement_step is None else tick,
        "deployStepInput": None if deploy_step is None else deploy_step[0],
        "deployStepOutput": None if deploy_step is None else deploy_step[1],
        "deployStepTick": -1 if deploy_step is None else tick,
        "effectiveMovementSpeed": effective_speed,
        "effectiveMovementSpeedTick": -1 if effective_speed is None else tick,
        "movementDelta": movement_delta,
        "movementDeltaTick": -1 if movement_delta is None else tick,
    }
    if attack_sequence_progress is not None:
        result.update(
            {
                "attackSequenceProgressRaw": attack_sequence_progress,
                "attackSequenceProgressLimit": 50,
                "attackSequenceDecayRemainingMs": (
                    attack_sequence_decay_remaining_ms
                ),
                "attackSequenceDecayDurationMs": (
                    7_000
                    if attack_sequence_decay_remaining_ms is not None
                    else None
                ),
            }
        )
    return result


def _phase_event(
    ordinary: dict[str, object],
    *,
    sequence: int,
    tick: int,
    kind: str = "attack_start",
) -> dict[str, object]:
    source = ordinary["objects"][6]  # type: ignore[index]
    absent = _null_combat_fact()
    hook = {
        "attack_start": 0xF23110,
        "attack_release": 0xF5F100,
    }[kind]
    return {
        "sequence": sequence,
        "tick": tick,
        "kind": kind,
        "hookOffset": hook,
        "callerOffset": 0xF61A7C,
        "entity": _combat_fact(source),
        "targetBefore": absent,
        "targetAfter": absent,
        "buffGlobalId": None,
        "buffRemainingMs": None,
        "speedMultiplier": None,
        "hitSpeedMultiplier": None,
        "inputStep": None,
        "outputStep": None,
        "timelineBefore": None,
        "timelineAfter": None,
        "classicChargeBefore": None,
        "classicChargeAfter": None,
        "success": kind == "attack_release",
    }


def _set_phase_runtime(
    rich: dict[str, object],
    *,
    events: list[dict[str, object]],
    epoch_first_sequence: int = 100,
    object_phase: dict[str, object] | None = None,
    rejected_count: int = 0,
    complete: bool = True,
) -> None:
    for item in rich["objects"]:  # type: ignore[union-attr]
        item["phaseRuntime"] = None
    rich["objects"][6]["phaseRuntime"] = object_phase  # type: ignore[index]
    for name in (
        "slow",
        "rage",
        "stun",
        "freeze",
        "attackPhase",
        "deployPhase",
        "chargeStage",
    ):
        rich["capabilities"][name] = {"status": "derived"}  # type: ignore[index]
    rich["phaseRuntime"] = {
        "ok": True,
        "schema": "native-phase-runtime.v1",
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "observationTick": rich["tick"],
        "capacity": 4_096,
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": {
            "status": "derived",
            "source": "exact-build-native-phase-hook-ring",
            "validation": "test-exact-build-guards",
            "confidence": "high",
            "failClosed": True,
        },
        "epochFirstSequence": epoch_first_sequence,
        "oldestRetainedSequence": epoch_first_sequence,
        "nextSequence": epoch_first_sequence + len(events),
        "overflowCount": 0,
        "rejectedCount": rejected_count,
        "sequenceGapBeforeOldest": False,
        "complete": complete,
        "events": events,
    }


def _set_special_movement_runtime(
    ordinary: dict[str, object],
    rich: dict[str, object],
) -> None:
    source_before = _combat_fact(ordinary["objects"][6])  # type: ignore[index]
    source_after = deepcopy(source_before)
    source_after["position"] = [8_300, 12_300]
    target = _combat_fact(ordinary["objects"][7])  # type: ignore[index]
    rich["specialMovementRuntime"] = {
        "ok": True,
        "schema": "native-special-movement-runtime.v1",
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "observationTick": rich["tick"],
        "capacity": 1_024,
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": {
            "status": "derived",
            "source": "caller-filtered-native-special-movement-hook-ring",
            "validation": (
                "sha-build-id-prologue-abi-type0-owner-target-identity"
            ),
            "confidence": "high",
            "failClosed": True,
        },
        "epochFirstSequence": 500,
        "oldestRetainedSequence": 500,
        "nextSequence": 501,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            {
                "sequence": 500,
                "tick": rich["tick"],
                "kind": "ordinary_dash_execute",
                "mode": "dash",
                "stage": "execute",
                "hookOffset": 0xF609DC,
                "callerOffset": 0xF61BF0,
                "sourceBefore": source_before,
                "sourceAfter": source_after,
                "target": target,
                "requestedX": 8_200,
                "requestedY": 19_700,
                "targetRadius": 500,
                "operationFlag": 1,
                "cooldownBeforeMs": 1_000,
                "cooldownAfterMs": 1_000,
                "positionChanged": True,
                "completeContext": True,
            }
        ],
    }


def _set_action_movement_runtime(
    ordinary: dict[str, object],
    rich: dict[str, object],
) -> None:
    source = _combat_fact(ordinary["objects"][6])  # type: ignore[index]
    target = _combat_fact(ordinary["objects"][7])  # type: ignore[index]
    rich["actionMovementRuntime"] = {
        "ok": True,
        "schema": "native-action-movement-runtime.v1",
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "observationTick": rich["tick"],
        "capacity": 1_024,
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": {
            "status": "derived",
            "source": "allowlisted-native-action-movement-hook-ring",
            "validation": (
                "sha-build-id-prologue-abi-action-name-global-id-"
                "class-vtable-source-identity"
            ),
            "confidence": "high",
            "failClosed": True,
        },
        "epochFirstSequence": 700,
        "oldestRetainedSequence": 700,
        "nextSequence": 701,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            {
                "sequence": 700,
                "tick": rich["tick"],
                "kind": "golden_knight_chain_hop_launch",
                "mode": "dash_chain",
                "stage": "hop_launch",
                "hookOffset": 0xF49D24,
                "callerOffset": 0xF49F10,
                "actionDataGlobalId": 4_104_010_032,
                "actionName": "GoldenKnight_Execute_Charge",
                "actionClassVtableOffset": 0x188DAF8,
                "runtimeClassVtableOffset": 0x189ECA0,
                "sourceBefore": source,
                "sourceAfter": deepcopy(source),
                "target": target,
                "chainIndex": 0,
                "positionChanged": False,
                "completeContext": True,
            }
        ],
    }


def _set_character_state_runtime(
    ordinary: dict[str, object],
    rich: dict[str, object],
) -> None:
    source = _combat_fact(ordinary["objects"][6])  # type: ignore[index]
    source["objectKind"] = 5
    rich["characterStateRuntime"] = {
        "ok": True,
        "schema": "native-character-state-runtime.v1",
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "observationTick": rich["tick"],
        "capacity": 2_048,
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": {
            "status": "derived",
            "source": "allowlisted-native-character-state-hook-ring",
            "validation": (
                "sha-build-id-prologue-abi-kind5-card-identity-"
                "committed-state"
            ),
            "confidence": "high",
            "failClosed": True,
        },
        "epochFirstSequence": 800,
        "oldestRetainedSequence": 800,
        "nextSequence": 801,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            {
                "sequence": 800,
                "tick": rich["tick"],
                "kind": "native_character_state_transition",
                "hookOffset": 0xF18654,
                "callerOffset": 0xF60CC4,
                "previousState": 1,
                "requestedState": 3,
                "committedState": 3,
                "sourceBefore": source,
                "sourceAfter": deepcopy(source),
                "positionChanged": False,
                "completeContext": True,
            }
        ],
    }


def _visibility_event(
    ordinary: dict[str, object],
    *,
    sequence: int,
    tick: int,
    kind: str,
) -> dict[str, object]:
    source = ordinary["objects"][6]  # type: ignore[index]
    became_invisible = kind == "became_invisible"
    before, after = ((0, 1) if became_invisible else (1, 0))
    return {
        "sequence": sequence,
        "tick": tick,
        "kind": kind,
        "hookOffset": 0xF5A2D4 if became_invisible else 0xF5A578,
        "callerOffset": 0xF59EAC if became_invisible else 0xF59308,
        "subject": _combat_fact(
            source,
            invisible_count=after,
        ),
        "buffGlobalId": 9_000_001,
        "invisibleCountBefore": before,
        "invisibleCountAfter": after,
        "scope": "native_invisibility_phase",
        "completeContext": True,
    }


def _set_visibility_runtime(
    rich: dict[str, object],
    *,
    events: list[dict[str, object]],
    epoch_first_sequence: int = 300,
    rejected_count: int = 0,
    complete: bool = True,
) -> None:
    rich["visibilityRuntime"] = {
        "ok": True,
        "schema": "native-visibility-runtime.v1",
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "observationTick": rich["tick"],
        "capacity": 4_096,
        "hookSetAttested": True,
        "transitionHookSetInstalled": True,
        "contextualGateHookSetInstalled": False,
        "capability": {
            "status": "derived",
            "source": "exact-build-native-invisibility-transition-ring",
            "validation": "test-exact-build-guards",
            "confidence": "high",
            "failClosed": True,
        },
        "contextualGateCapability": {
            "status": "unavailable",
            "source": "no-safe-pc-relative-gate-bridge",
            "validation": "fail-closed-no-generic-trampoline",
            "confidence": "none",
            "failClosed": True,
        },
        "ownerRelativeVisibility": {
            name: {
                "status": "unavailable",
                "reason": "no-exact-owner-conditioned-native-producer",
            }
            for name in ("publicByOwner", "targetableByOwner")
        },
        "epochFirstSequence": epoch_first_sequence,
        "oldestRetainedSequence": epoch_first_sequence,
        "nextSequence": epoch_first_sequence + len(events),
        "overflowCount": 0,
        "rejectedCount": rejected_count,
        "sequenceGapBeforeOldest": False,
        "complete": complete,
        "events": events,
    }


def _remaining_resource_event(
    ordinary: dict[str, object],
    *,
    sequence: int,
    tick: int,
) -> dict[str, object]:
    source = _combat_fact(
        ordinary["objects"][6],  # type: ignore[index]
    )
    absent = _null_combat_fact()
    return {
        "sequence": sequence,
        "tick": tick,
        "kind": "resource_delta",
        "hookOffset": 0xF3BA4C,
        "callerOffset": 0xF1ACD4,
        "entity": absent,
        "source": source,
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
        "resourcePreFixed": 20_000,
        "resourcePostFixed": 30_000,
        "resourceActualDeltaFixed": 10_000,
        "amountArgument": 1,
        "configuredAmountArgument": 1,
        "resourceOwner": 0,
        "areaRemainingLifeMs": None,
        "resourceCause": "periodic",
        "transformKind": "none",
        "result": False,
        "option": False,
        "committed": False,
        "resetTarget": False,
        "completeContext": True,
    }


def _set_remaining_runtime(
    rich: dict[str, object],
    *,
    events: list[dict[str, object]],
    epoch_first_sequence: int = 200,
    rejected_count: int = 0,
    complete: bool = True,
) -> None:
    rich["remainingRuntime"] = {
        "ok": True,
        "schema": "native-remaining-runtime.v1",
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "observationTick": rich["tick"],
        "capacity": 4_096,
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": {
            "status": "derived",
            "source": "exact-build-native-remaining-hook-ring",
            "validation": "test-exact-build-guards",
            "confidence": "high",
            "failClosed": True,
        },
        "ownerRelativeVisibility": {
            name: {
                "status": "unavailable",
                "reason": "no-exact-owner-conditioned-native-producer",
            }
            for name in ("publicByOwner", "targetableByOwner")
        },
        "epochFirstSequence": epoch_first_sequence,
        "oldestRetainedSequence": epoch_first_sequence,
        "nextSequence": epoch_first_sequence + len(events),
        "overflowCount": 0,
        "rejectedCount": rejected_count,
        "sequenceGapBeforeOldest": False,
        "complete": complete,
        "events": events,
    }


def _set_tower_troop_runtime(
    rich: dict[str, object],
    *,
    by_slot: dict[int, dict[str, object]],
) -> None:
    rich["provenance"]["towerTroopRuntime"] = {  # type: ignore[index]
        "status": "derived"
    }
    rich["capabilities"]["towerTroopRuntime"] = {  # type: ignore[index]
        "status": "derived"
    }
    for item in rich["objects"]:  # type: ignore[union-attr]
        item["towerTroopRuntime"] = by_slot.get(int(item["slot"]))
    rich["towerTroopRuntime"] = {
        "ok": True,
        "schema": "native-tower-troop-runtime.v1",
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "observationTick": rich["tick"],
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "rejectedCount": 0,
        "complete": True,
    }


def _rich(ordinary: dict[str, object]) -> dict[str, object]:
    rich_objects: list[dict[str, object]] = []
    for raw in ordinary["objects"]:  # type: ignore[union-attr]
        item = dict(raw)
        entity_key = [item["owner"], -2, item["nativeObjectId"]]
        rich_objects.append(
            {
                "slot": item["slot"],
                "nativeObjectId": item["nativeObjectId"],
                "entityKey": entity_key,
                "owner": item["owner"],
                "cardId": item["cardId"],
                "dataGlobalId": 34_000_000 + int(item["slot"]),
                "x": item["x"],
                "y": item["y"],
                "targetX": item["targetX"],
                "targetY": item["targetY"],
                "objectIndex": item["objectIndex"],
                "secondaryIndex": item["secondaryIndex"],
                "hp": item["hp"],
                "maxHp": item["maxHp"],
                "shield": None,
                "targetEntityKey": None,
                "targetEntityValidated": False,
                "attackSequenceStage": None,
                "activeEffects": [],
                "invisibleCount": 0,
                "visibilityState": "visible",
                "projectile": None,
                "entityResourceRuntime": None,
                "periodicAttackModifierRuntime": None,
                "captureRuntime": None,
                "thresholdRelocationRuntime": None,
                "components": {
                    "valid": False,
                    "count": 0,
                    "capacity": 0,
                    "slots": [],
                },
            }
        )
    authoritative_fields = (
        "nativeObjectId",
        "dataGlobalId",
        "targetEntityKey",
        "shield.current",
        "shield.max",
        "attackSequenceStage",
        "activeEffects",
        "activeEffects.buffGlobalId",
        "activeEffects.name",
        "activeEffects.remainingMs",
        "activeEffects.sourceEntityKey",
        "invisibleCount",
        "projectile.dragStage",
        "entityResourceRuntime",
        "periodicAttackModifierRuntime",
        "captureRuntime",
        "thresholdRelocationRuntime",
    )
    authoritative_capabilities = (
        "targetEntity",
        "shield",
        "attackSequenceStage",
        "buffs",
        "invisibility",
    )
    return {
        "ok": True,
        "schema": "native-rich-telemetry.v3",
        "tick": ordinary["tick"],
        "generation": ordinary["generation"],
        "stateEpoch": ordinary["stateEpoch"],
        "count": len(rich_objects),
        "returned": len(rich_objects),
        "truncated": False,
        "objects": rich_objects,
        "provenance": {
            **{name: {"status": "authoritative"} for name in authoritative_fields},
            "entityKey": {"status": "derived"},
            "activeEffects.sourceEntityValidated": {"status": "derived"},
            "visibilityState": {"status": "derived"},
            **{
                name: {"status": "derived"}
                for name in (
                    "projectile",
                    "projectile.projectileDataGlobalId",
                    "projectile.sourceEntityKey",
                    "projectile.targetEntityKey",
                    "projectile.homingTargetEntityKey",
                    "projectile.destination",
                    "projectile.terminal",
                    "projectile.nativePhase",
                    "players.ownerRoot",
                    "players.abilityRuntime",
                    "players.abilityRuntime.buttonState",
                    "players.abilityRuntime.cooldown",
                    "players.abilityRuntime.charges",
                    "players.evolutionRuntime",
                    "players.evolutionRuntime.cycle",
                )
            },
            "players.evolutionRuntime.playedForm": {"status": "unavailable"},
        },
        "capabilities": {
            **{
                name: {"status": "authoritative"}
                for name in authoritative_capabilities
            },
            "projectile": {"status": "derived"},
            "abilityRuntime": {"status": "derived"},
            "evolutionRuntime": {"status": "derived"},
            "impact": {"status": "unavailable"},
        },
        "players": [
            {
                "owner": owner,
                "ownerRootValidated": False,
                "ownerEntityKey": None,
                "abilityRuntime": None,
                "evolutionRuntime": None,
            }
            for owner in (0, 1)
        ],
        "combatEvents": _empty_combat_events(ordinary),
    }


def _add_player_runtime(
    ordinary: dict[str, object], rich: dict[str, object]
) -> None:
    archer_queen = 26_000_072
    archer = 26_000_001
    ordinary_players = ordinary["players"]
    owner_zero = ordinary_players[0]  # type: ignore[index]
    owner_zero["deck"][0]["cardId"] = archer_queen  # type: ignore[index]
    owner_zero["deck"][1]["cardId"] = archer  # type: ignore[index]
    archer_queen_cost = int(
        build_card_catalog().by_id[archer_queen].elixir_cost or 0
    )
    owner_zero["hand"][0].update(  # type: ignore[index]
        {
            "cardId": archer_queen,
            "commandCardId": archer_queen,
            "deckSlot": 0,
            "cost": archer_queen_cost,
            "cardParameter": (archer_queen_cost << 28) | (1 << 22),
        }
    )
    archer_cost = int(build_card_catalog().by_id[archer].elixir_cost or 0)
    owner_zero["hand"][1].update(  # type: ignore[index]
        {
            "cardId": archer,
            "commandCardId": archer,
            "deckSlot": 1,
            "cost": archer_cost,
            "cardParameter": (archer_cost << 28) | (2 << 22),
        }
    )

    ordinary["objects"][6]["cardId"] = archer_queen  # type: ignore[index]
    rich["objects"][6]["cardId"] = archer_queen  # type: ignore[index]
    root = {
        "slot": 8,
        "nativeObjectId": 5_000_008,
        "objectIndex": 500,
        "secondaryIndex": 0,
        "owner": 0,
        "cardId": -1,
        "x": 0,
        "y": 0,
        "targetX": 0,
        "targetY": 0,
        "hp": None,
        "maxHp": None,
    }
    ordinary["objects"].append(root)  # type: ignore[union-attr]
    rich["objects"].append(  # type: ignore[union-attr]
        {
            "slot": 8,
            "nativeObjectId": 5_000_008,
            "entityKey": [0, -2, 5_000_008],
            "owner": 0,
            "cardId": -1,
            "dataGlobalId": 5_000_008,
            "x": 0,
            "y": 0,
            "targetX": 0,
            "targetY": 0,
            "objectIndex": 500,
            "secondaryIndex": 0,
            "hp": None,
            "maxHp": None,
            "shield": None,
            "targetEntityKey": None,
            "targetEntityValidated": False,
            "attackSequenceStage": None,
            "activeEffects": [],
            "invisibleCount": 0,
            "visibilityState": "visible",
            "projectile": None,
            "entityResourceRuntime": None,
            "periodicAttackModifierRuntime": None,
            "captureRuntime": None,
            "thresholdRelocationRuntime": None,
            "components": {
                "valid": False,
                "count": 0,
                "capacity": 0,
                "slots": [],
            },
        }
    )
    ordinary["count"] = ordinary["returned"] = 9
    rich["count"] = rich["returned"] = 9

    evolution_runtime = []
    for deck_slot, deck_card in enumerate(owner_zero["deck"]):  # type: ignore[union-attr]
        card_id = int(deck_card["cardId"])
        evolvable = deck_slot == 1
        evolution_runtime.append(
            {
                "deckSlot": deck_slot,
                "cardId": card_id,
                "baseSpellGlobalId": card_id,
                "evolvable": evolvable,
                "evolutionFormGlobalId": 26_100_001 if evolvable else None,
                "progress": 1 if evolvable else 0,
                "cycleRequired": 2 if evolvable else None,
                "cycleRemaining": 1 if evolvable else None,
                "ready": False if evolvable else None,
            }
        )
    rich["players"][0] = {  # type: ignore[index]
        "owner": 0,
        "ownerRootValidated": True,
        "ownerEntityKey": [0, -2, 5_000_008],
        "abilityRuntime": [
            {
                "controllerSlot": 1,
                "actionDataGlobalId": 88_000_001,
                "actionDataName": "ArcherQueenRapid",
                "selectedCharacterDataGlobalId": 26_000_072,
                "remainingCooldownMs": 0,
                "configuredCooldownMs": 0,
                "remainingChargesRaw": 1,
                "maxCharges": 1,
                "buttonState": 2,
                "buttonStateLabel": "Ready",
                "available": True,
                "championEntityKeys": [[0, -2, 5_000_006]],
            }
        ],
        "evolutionRuntime": evolution_runtime,
    }


class _RichNativeStub:
    def __init__(
        self,
        ordinary: dict[str, object],
        rich: dict[str, object],
    ) -> None:
        self.ordinary = ordinary
        self.rich = rich
        self.observe_rich_calls = 0

    def create_match(self, _config: MatchConfig) -> dict[str, object]:
        return deepcopy(self.ordinary)

    def create_native_match(self, _config: MatchConfig) -> dict[str, object]:
        return deepcopy(self.ordinary)

    def observe_atomic(self) -> dict[str, object]:
        self.observe_rich_calls += 1
        return {"ordinary": deepcopy(self.ordinary), "rich": deepcopy(self.rich)}

    def digest(self) -> dict[str, str]:
        return {"digest": "rich-telemetry-unit-test"}


class _TransitionNativeStub(_RichNativeStub):
    def __init__(
        self,
        before: dict[str, object],
        before_rich: dict[str, object],
        after: dict[str, object],
        after_rich: dict[str, object],
    ) -> None:
        super().__init__(before, before_rich)
        self.after = after
        self.after_rich = after_rich
        self.advanced = False

    def step(self, _ticks: int) -> dict[str, object]:
        self.advanced = True
        return {"ok": True}

    def observe(self) -> dict[str, object]:
        return deepcopy(self.after if self.advanced else self.ordinary)

    def observe_atomic(self) -> dict[str, object]:
        self.observe_rich_calls += 1
        return {
            "ordinary": deepcopy(self.after if self.advanced else self.ordinary),
            "rich": deepcopy(self.after_rich if self.advanced else self.rich),
        }


class _AbilityTransitionNativeStub(_TransitionNativeStub):
    def __init__(
        self,
        before: dict[str, object],
        before_rich: dict[str, object],
        after: dict[str, object],
        after_rich: dict[str, object],
    ) -> None:
        super().__init__(before, before_rich, after, after_rich)
        self.ability_actions: list[tuple[AbilityAction, int]] = []

    def queue_ability_action_at(
        self,
        action: AbilityAction,
        *,
        execute_in_ticks: int = 1,
    ) -> dict[str, object]:
        self.ability_actions.append((action, execute_in_ticks))
        tick = int(self.ordinary["tick"])
        return {
            "ok": True,
            "tick": tick,
            "queuedAtTick": tick,
            "executeTick": tick + execute_in_ticks,
        }

    def activate_ability(self, action: AbilityAction) -> dict[str, object]:
        return self.queue_ability_action_at(action)


class _HandTransitionNativeStub(_TransitionNativeStub):
    def queue_hand_action_at(
        self,
        action: HandAction,
        *,
        execute_in_ticks: int = 1,
    ) -> dict[str, object]:
        tick = int(self.ordinary["tick"])
        player = next(
            item
            for item in self.ordinary["players"]  # type: ignore[union-attr]
            if int(item["owner"]) == action.owner
        )
        card = next(
            item
            for item in player["hand"]
            if int(item["handIndex"]) == action.hand_index
        )
        card_parameter = int(card["cardParameter"])
        form_code = card_parameter & 0xF
        return {
            "ok": True,
            "owner": action.owner,
            "handIndex": action.hand_index,
            "cardId": int(card["cardId"]),
            "commandCardId": int(card.get("commandCardId", card["cardId"])),
            "cardParameter": card_parameter,
            "deckSlot": int(card.get("deckSlot", action.hand_index)),
            "cost": int(card["cost"]),
            "formCode": form_code,
            "formName": ("EvoForm" if form_code == 1 else "BasicForm"),
            "queuedAtTick": tick,
            "executeTick": tick + execute_in_ticks,
        }


class _AtomicNativeStub(_RichNativeStub):
    def __init__(
        self,
        stale_ordinary: dict[str, object],
        atomic_ordinary: dict[str, object],
        atomic_rich: dict[str, object],
    ) -> None:
        super().__init__(stale_ordinary, atomic_rich)
        self.atomic_ordinary = atomic_ordinary
        self.atomic_rich = atomic_rich
        self.observe_atomic_calls = 0

    def observe_atomic(self) -> dict[str, object]:
        self.observe_atomic_calls += 1
        return {
            "ordinary": deepcopy(self.atomic_ordinary),
            "rich": deepcopy(self.atomic_rich),
        }

    def observe_rich(self) -> dict[str, object]:
        raise AssertionError("atomic-capable BattleEnv used observe_rich fallback")


class _OldProbeNativeStub(_RichNativeStub):
    def observe_atomic(self) -> dict[str, object]:
        raise RunnerError("commands: observe, observe-rich, step")


class _FailedAtomicNativeStub(_RichNativeStub):
    def observe_atomic(self) -> dict[str, object]:
        raise RunnerError("atomic rich observation is incomplete")


class _AtomicTransitionNativeStub(_TransitionNativeStub):
    def __init__(
        self,
        before: dict[str, object],
        before_rich: dict[str, object],
        after: dict[str, object],
        after_rich: dict[str, object],
    ) -> None:
        super().__init__(before, before_rich, after, after_rich)
        self.observe_atomic_calls = 0

    def observe_atomic(self) -> dict[str, object]:
        self.observe_atomic_calls += 1
        return {
            "ordinary": deepcopy(self.after if self.advanced else self.ordinary),
            "rich": deepcopy(self.after_rich if self.advanced else self.rich),
        }

    def observe_rich(self) -> dict[str, object]:
        raise AssertionError("atomic transition used observe_rich fallback")




class _SynchronizedAtomicTransitionNativeStub(_AtomicTransitionNativeStub):
    def __init__(
        self,
        before: dict[str, object],
        before_rich: dict[str, object],
        after: dict[str, object],
        after_rich: dict[str, object],
    ) -> None:
        super().__init__(before, before_rich, after, after_rich)
        self.pause_calls = 0
        self.speed_calls: list[float] = []
        self.advance_calls: list[int] = []

    def pause(self) -> dict[str, object]:
        self.pause_calls += 1
        tick = int((self.after if self.advanced else self.ordinary)["tick"])
        return {"ok": True, "mode": "native-render", "paused": True, "tick": tick}

    def set_speed(self, speed: float) -> dict[str, object]:
        self.speed_calls.append(speed)
        return {"ok": True, "mode": "native-render", "speed": speed}

    def advance_native_render(self, ticks: int) -> dict[str, object]:
        self.advance_calls.append(ticks)
        self.advanced = True
        return {
            "ok": True,
            "mode": "native-render",
            "requestedTicks": ticks,
            "targetTick": int(self.after["tick"]),
            "tick": int(self.after["tick"]),
            "ended": False,
            "paused": True,
        }


@pytest.fixture(scope="module")
def catalog():
    return build_card_catalog()


@lru_cache(maxsize=1)
def _fresh_effect_catalog() -> RuntimeEffectCatalog:
    return RuntimeEffectCatalog.from_catalog(
        build_effect_catalog(build_static_card_logic_catalog())
    )


def _environment(native: _RichNativeStub, catalog) -> BattleEnvV1:
    return BattleEnvV1(
        native=native,  # type: ignore[arg-type]
        card_catalog=catalog,
        effect_catalog=_fresh_effect_catalog(),
        ruleset_id="b" * 64,
        warmup_ticks=0,
    )


def _reset(
    env: BattleEnvV1,
    *,
    match_config: MatchConfig | None = None,
):
    match = match_config or MatchConfig(
        deck0=SEMANTIC_BASELINE_DECK,
        deck1=SEMANTIC_BASELINE_DECK,
    )
    return env.reset(match_config=match)


def _match_from_raw(
    raw: dict[str, object],
    *,
    deck0_form_availability: tuple[int, ...] = (0,) * 8,
    deck1_form_availability: tuple[int, ...] = (0,) * 8,
) -> MatchConfig:
    decks: dict[int, tuple[int, ...]] = {}
    for player in raw["players"]:  # type: ignore[union-attr]
        owner = int(player["owner"])
        decks[owner] = tuple(
            int(item["cardId"])
            for item in sorted(
                player["deck"],
                key=lambda item: int(item["deckSlot"]),
            )
        )
    return MatchConfig(
        deck0=decks[0],
        deck1=decks[1],
        deck0_form_availability=deck0_form_availability,
        deck1_form_availability=deck1_form_availability,
    )


def test_adapter_joins_negative_indices_by_tagged_native_object_id() -> None:
    ordinary = _ordinary()
    tower = ordinary["objects"][0]  # type: ignore[index]
    tower["objectIndex"] = -1
    tower["secondaryIndex"] = -1
    rich = _rich(ordinary)

    snapshot = bind_rich_telemetry(ordinary, rich)

    assert snapshot.objects[0] is not None
    assert snapshot.objects[0].entity_key == (  # type: ignore[union-attr]
        tower["owner"],
        -2,
        tower["nativeObjectId"],
    )
    assert snapshot.objects[0].data_global_id == 34_000_000  # type: ignore[union-attr]
    assert snapshot.special_movement_runtime is None
    assert snapshot.action_movement_runtime is None
    assert snapshot.character_state_runtime is None


def test_adapter_binds_exact_tower_troop_action_runtime() -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _set_tower_troop_runtime(
        rich,
        by_slot={
            1: {
                "schema": "native-tower-troop-runtime.v1",
                "kind": "dagger_duchess",
                "observedTick": 20,
                "chargeCount": 3,
                "maxChargeCount": 8,
                "rechargeElapsedMs": 450,
                "rechargeDurationMs": 900,
            },
            3: {
                "schema": "native-tower-troop-runtime.v1",
                "kind": "royal_chef",
                "observedTick": 20,
                "startDelayRemainingMs": 0,
                "startDelayDurationMs": 7_000,
                "cookingContribution": 230_000,
                "contributionNeeded": 460_000,
                "throwDelayRemainingMs": 150,
                "targetNativeObjectId": 5_000_006,
            },
        },
    )

    snapshot = bind_rich_telemetry(ordinary, rich)

    assert snapshot.tower_troop_runtime is not None
    assert snapshot.tower_troop_runtime.complete is True
    dagger = snapshot.objects[1]
    chef = snapshot.objects[3]
    assert dagger is not None and isinstance(
        dagger.tower_troop_runtime,
        RichDaggerDuchessRuntimeTelemetry,
    )
    assert dagger.tower_troop_runtime.charge_count == 3
    assert chef is not None and isinstance(
        chef.tower_troop_runtime,
        RichRoyalChefRuntimeTelemetry,
    )
    assert chef.tower_troop_runtime.cooking_contribution == 230_000
    assert chef.tower_troop_runtime.target_native_object_id == 5_000_006


def test_adapter_preserves_royal_chef_wasted_throw() -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _set_tower_troop_runtime(
        rich,
        by_slot={
            3: {
                "schema": "native-tower-troop-runtime.v1",
                "kind": "royal_chef",
                "observedTick": 20,
                "startDelayRemainingMs": 0,
                "startDelayDurationMs": 7_000,
                "cookingContribution": 460_460,
                "contributionNeeded": 460_000,
                "throwDelayRemainingMs": -50,
                "targetNativeObjectId": None,
            },
        },
    )

    snapshot = bind_rich_telemetry(ordinary, rich)

    chef = snapshot.objects[3]
    assert chef is not None and isinstance(
        chef.tower_troop_runtime,
        RichRoyalChefRuntimeTelemetry,
    )
    assert chef.tower_troop_runtime.throw_delay_remaining_ms == -50
    assert chef.tower_troop_runtime.target_native_object_id is None


def test_adapter_binds_typed_special_movement_runtime() -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _set_special_movement_runtime(ordinary, rich)

    snapshot = bind_rich_telemetry(ordinary, rich)

    envelope = snapshot.special_movement_runtime
    assert envelope is not None
    assert envelope.complete is True
    event = envelope.events[0]
    assert event.sequence == 500
    assert event.source_before.entity_key == event.source_after.entity_key
    assert event.source_before.position != event.source_after.position
    assert (event.requested_x, event.requested_y) != event.target.position


def test_adapter_rejects_special_movement_source_identity_change() -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _set_special_movement_runtime(ordinary, rich)
    event = rich["specialMovementRuntime"]["events"][0]  # type: ignore[index]
    event["sourceAfter"]["cardId"] = 26_000_055

    with pytest.raises(RichTelemetryMergeError, match="source identity changed"):
        bind_rich_telemetry(ordinary, rich)


def test_adapter_binds_typed_action_movement_runtime() -> None:
    ordinary = _ordinary()
    ordinary["objects"][6]["cardId"] = 26_000_074  # type: ignore[index]
    rich = _rich(ordinary)
    _set_action_movement_runtime(ordinary, rich)

    snapshot = bind_rich_telemetry(ordinary, rich)

    envelope = snapshot.action_movement_runtime
    assert envelope is not None
    assert envelope.complete is True
    event = envelope.events[0]
    assert event.sequence == 700
    assert event.action_name == "GoldenKnight_Execute_Charge"
    assert event.stage == "hop_launch"
    assert event.target is not None


def test_adapter_rejects_action_movement_source_card_drift() -> None:
    ordinary = _ordinary()
    ordinary["objects"][6]["cardId"] = 26_000_074  # type: ignore[index]
    rich = _rich(ordinary)
    _set_action_movement_runtime(ordinary, rich)
    event = rich["actionMovementRuntime"]["events"][0]  # type: ignore[index]
    event["sourceAfter"]["cardId"] = 26_000_055

    with pytest.raises(RichTelemetryMergeError, match="source identity changed"):
        bind_rich_telemetry(ordinary, rich)


def test_adapter_binds_typed_character_state_runtime() -> None:
    ordinary = _ordinary()
    ordinary["objects"][6]["cardId"] = 26_000_046  # type: ignore[index]
    rich = _rich(ordinary)
    _set_character_state_runtime(ordinary, rich)

    snapshot = bind_rich_telemetry(ordinary, rich)

    envelope = snapshot.character_state_runtime
    assert envelope is not None
    assert envelope.complete is True
    event = envelope.events[0]
    assert event.sequence == 800
    assert event.previous_state == 1
    assert event.committed_state == 3
    assert event.source_before.object_kind == 5


def test_adapter_rejects_character_state_identity_change() -> None:
    ordinary = _ordinary()
    ordinary["objects"][6]["cardId"] = 26_000_046  # type: ignore[index]
    rich = _rich(ordinary)
    _set_character_state_runtime(ordinary, rich)
    event = rich["characterStateRuntime"]["events"][0]  # type: ignore[index]
    event["sourceAfter"]["cardId"] = 26_000_055

    with pytest.raises(RichTelemetryMergeError, match="source identity changed"):
        bind_rich_telemetry(ordinary, rich)


def test_fixed_tower_remains_public_when_it_is_the_validated_owner_root(
    catalog,
) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    tower_key = rich["objects"][0]["entityKey"]  # type: ignore[index]
    rich["players"][0].update(  # type: ignore[index,union-attr]
        {
            "ownerRootValidated": True,
            "ownerEntityKey": tower_key,
        }
    )
    env = _environment(_RichNativeStub(ordinary, rich), catalog)

    observations, _ = _reset(env)

    for owner in (0, 1):
        assert len(observations[owner].towers) == len(TOWER_LAYOUT)


def test_adapter_rejects_duplicate_native_object_id_even_with_distinct_raw_keys() -> None:
    ordinary = _ordinary()
    ordinary["objects"][1]["nativeObjectId"] = ordinary["objects"][0][  # type: ignore[index]
        "nativeObjectId"
    ]
    rich = _rich(ordinary)

    with pytest.raises(RichTelemetryMergeError, match="duplicate rich native object"):
        bind_rich_telemetry(ordinary, rich)


def test_rich_runtime_projects_typed_state_and_static_effect_semantics(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    objects = rich["objects"]
    source_key = [0, -2, 5_000_006]
    target_key = [1, -2, 5_000_004]
    objects[6].update(  # type: ignore[index,union-attr]
        {
            "shield": {"current": 150, "max": 200, "status": "authoritative"},
            "targetEntityKey": target_key,
            "targetEntityValidated": True,
            "attackSequenceStage": 7,
            "activeEffects": [
                {
                    "buffGlobalId": 9_000_000,
                    "name": "Rage",
                    "remainingMs": 750,
                    "sourceEntityKey": source_key,
                    "sourceEntityValidated": True,
                },
                {
                    "buffGlobalId": 9_000_001,
                    "name": "Freeze",
                    "remainingMs": -1,
                    "sourceEntityKey": None,
                    "sourceEntityValidated": False,
                },
            ],
        }
    )
    native = _RichNativeStub(ordinary, rich)
    observations, _ = _reset(_environment(native, catalog))

    assert all(
        left is right
        for left, right in zip(
            observations[0].towers,
            observations[1].towers,
            strict=True,
        )
    )
    assert all(
        left is right
        for left, right in zip(
            observations[0].entities,
            observations[1].entities,
            strict=True,
        )
    )

    entity = next(item for item in observations[0].entities if item.owner == 0)
    assert entity.native_data_global_id == 34_000_006
    assert entity.shield == 150
    assert entity.shield_state is not None
    assert entity.shield_state.max_hitpoints == 200
    assert entity.visible_target == 4
    assert entity.attack_state is not None
    assert entity.attack_state.sequence_index == 7
    assert entity.attack_state.charge_stage is None
    assert (
        entity.attack_state.provenance.field_evidence["sequence_index"]
        == SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
    )
    assert (
        entity.attack_state.provenance.field_evidence["charge_stage"]
        == SemanticEvidenceLevel.UNKNOWN
    )

    rage, freeze = entity.effect_states
    assert rage.kind == EffectKind.RAGE
    assert rage.remaining_ms == 750
    assert rage.attributes["classification"] == "static_catalog_resolved"
    assert "rage" in rage.attributes["mechanic_tags"]
    assert rage.attributes["modifiers"]["HitSpeedMultiplier"] == 130
    assert (
        rage.provenance.field_evidence["kind"]
        == SemanticEvidenceLevel.STATIC_DECLARED
    )
    assert freeze.kind == EffectKind.FREEZE
    assert freeze.remaining_ms is None
    assert freeze.attributes["native_remaining_ms"] == -1
    assert freeze.attributes["non_expiring"] is True
    assert (
        freeze.provenance.field_evidence["remaining_ms"]
        == SemanticEvidenceLevel.NOT_APPLICABLE
    )
    assert freeze.source_entity is None
    assert (
        freeze.provenance.field_evidence["source_entity"]
        == SemanticEvidenceLevel.UNKNOWN
    )
    assert observations[0].metadata["native_effect_catalog_id"] is not None
    assert native.observe_rich_calls == 1


def test_rich_runtime_projects_entity_owned_extra_spawn_resource(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    rich["objects"][6]["entityResourceRuntime"] = {  # type: ignore[index]
        "kind": "extra_spawn_accumulator",
        "currentRaw": 3,
        "capacityRaw": 10,
        "baseRaw": 6,
        "limitRaw": 16,
        "normalized": 0.3,
        "status": "authoritative",
    }

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog)
    )

    entity = next(item for item in observations[0].entities if item.owner == 0)
    assert len(entity.resource_states) == 1
    resource = entity.resource_states[0]
    assert resource.kind == "extra_spawn_accumulator"
    assert resource.current_raw == 3
    assert resource.capacity_raw == 10
    assert resource.normalized == pytest.approx(0.3)
    assert (
        entity.runtime_provenance.field_evidence["resource_states"]
        == SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
    )


def test_rich_runtime_projects_periodic_attack_modifier_linger(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    rich["objects"][6]["periodicAttackModifierRuntime"] = {  # type: ignore[index]
        "schema": "native-periodic-attack-modifier-runtime.v1",
        "actionDataGlobalId": 3_398_518_713,
        "phase": "source_death_linger",
        "periodAttacks": 3,
        "completedAttacks": 2,
        "addedDamageRaw": 86,
        "lingerDurationMs": 5_000,
        "lingerRemainingMs": 4_950,
        "sourceNativeObjectId": 5_999_999,
        "sourceEntityKey": None,
        "sourceResolved": False,
        "status": "authoritative",
    }

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog)
    )

    entity = next(item for item in observations[0].entities if item.owner == 0)
    modifier = entity.periodic_attack_modifier
    assert modifier is not None
    assert modifier.phase == PeriodicAttackModifierPhase.SOURCE_DEATH_LINGER
    assert modifier.period_attacks == 3
    assert modifier.completed_attacks == 2
    assert modifier.linger_duration_ms == 5_000
    assert modifier.linger_remaining_ms == 4_950
    assert modifier.source_entity is None
    assert (
        modifier.provenance.field_evidence["linger_remaining_ms"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )
    assert (
        entity.runtime_provenance.field_evidence["periodic_attack_modifier"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )


def test_rich_runtime_projects_exact_capture_phase_and_public_target(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    target = rich["objects"][7]  # type: ignore[index]
    rich["objects"][6]["captureRuntime"] = {  # type: ignore[index]
        "schema": "native-capture-runtime.v1",
        "actionDataGlobalId": 1_941_152_550,
        "phase": "active",
        "dragDelayMs": 100,
        "grabPauseMs": 500,
        "captureDragTimeMs": 300,
        "configuredCooldownMs": 300,
        "cooldownRemainingMs": 0,
        "hitFrequencyMs": 1_000,
        "hitAccumulatorMs": 850,
        "completionResultCurrentUpdate": True,
        "firstCaptureHandled": True,
        "targets": [
            {
                "targetNativeObjectId": target["nativeObjectId"],
                "targetEntityKey": target["entityKey"],
                "targetResolved": True,
                "phase": "contained",
                "elapsedMs": 950,
                "phaseBudgetRemainingMs": None,
            }
        ],
        "status": "authoritative",
    }

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog)
    )

    source = next(item for item in observations[0].entities if item.owner == 0)
    victim = next(item for item in observations[0].entities if item.owner == 1)
    capture = source.capture_runtime
    assert capture is not None
    assert capture.phase == CaptureActionPhase.ACTIVE
    assert capture.targets[0].phase == CaptureTargetPhase.CONTAINED
    assert capture.targets[0].target_entity == victim.entity_id
    assert capture.hit_accumulator_ms == 850
    assert (
        source.runtime_provenance.field_evidence["capture_runtime"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )


def test_rich_runtime_projects_threshold_relocation_without_future_position(
    catalog,
) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    rich["objects"][6]["thresholdRelocationRuntime"] = {  # type: ignore[index]
        "schema": "native-threshold-relocation-runtime.v1",
        "actionDataGlobalId": 1_333_491_178,
        "phase": "relocating",
        "stage": 4,
        "relocationIndex": 1,
        "hideDurationMs": 1_000,
        "remainingMs": 100,
        "burrowed": True,
        "thresholdsPercent": [66, 33],
        "status": "authoritative",
    }

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog)
    )

    entity = next(item for item in observations[0].entities if item.owner == 0)
    relocation = entity.threshold_relocation_runtime
    assert relocation is not None
    assert relocation.phase == ThresholdRelocationPhase.RELOCATING
    assert relocation.stage == 4
    assert relocation.relocation_index == 1
    assert relocation.threshold_count == 2
    assert relocation.thresholds_percent == (66, 33)
    assert relocation.remaining_ms == 100
    assert relocation.burrowed is True
    assert (
        entity.runtime_provenance.field_evidence["threshold_relocation_runtime"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )
    assert (
        relocation.provenance.field_evidence["thresholds_percent"]
        == SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
    )
    assert (
        relocation.provenance.field_evidence["relocation_index"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )
    assert relocation.provenance.source_fields["relocation_index"] == (
        "objects[].thresholdRelocationRuntime.relocationIndex",
    )
    # The native future destination is intentionally absent from the FAIR contract.
    assert "destination" not in relocation.to_dict()


def test_effect_semantics_require_joint_buff_name_and_global_id(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    rich["objects"][6]["activeEffects"] = [  # type: ignore[index]
        {
            # 9000001 is Freeze, so it must not classify the name Rage.
            "buffGlobalId": 9_000_001,
            "name": "Rage",
            "remainingMs": 500,
            "sourceEntityKey": None,
            "sourceEntityValidated": True,
        }
    ]

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog)
    )

    effect = next(item for item in observations[0].entities if item.owner == 0).effect_states[0]
    assert effect.kind == EffectKind.UNKNOWN
    assert effect.attributes["classification"] == "native_buff_name_global_id_mismatch"
    assert effect.provenance.field_evidence["kind"] == SemanticEvidenceLevel.UNKNOWN


def test_hashed_high_bit_buff_uses_joint_uint32_identity(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    rich["objects"][6]["activeEffects"] = [  # type: ignore[index]
        {
            "buffGlobalId": 2_428_333_743,
            "name": "BabyDragon_EV1_wind_buff_negative",
            "remainingMs": 500,
            "sourceEntityKey": None,
            "sourceEntityValidated": True,
        }
    ]
    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog)
    )

    effect = next(item for item in observations[0].entities if item.owner == 0).effect_states[0]
    assert effect.kind == EffectKind.SLOW
    assert effect.attributes["classification"] == "static_catalog_resolved"
    assert effect.attributes["native_buff_global_id"] == 2_428_333_743


def test_projectile_is_typed_without_synthesizing_impact(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    rich["objects"][6]["projectile"] = {  # type: ignore[index]
        "projectileDataGlobalId": 10_000_123,
        "sourceEntityKey": [0, -2, 5_000_006],
        "sourceEntityValidated": True,
        "targetEntityKey": [1, -2, 5_000_004],
        "targetEntityValidated": True,
        "homingTargetEntityKey": [1, -2, 5_000_004],
        "homingTargetEntityValidated": True,
        "destinationX": 14_500,
        "destinationY": 25_500,
        "terminal": False,
        "nativePhase": "in_flight",
        "dragStage": "outbound",
    }
    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog)
    )
    projectile_entity = next(
        item for item in observations[0].entities if item.owner == 0
    )
    assert projectile_entity.entity_kind == "projectile"
    assert projectile_entity.projectile_state is not None
    assert projectile_entity.projectile_state.phase == ProjectilePhase.IN_FLIGHT
    assert projectile_entity.projectile_state.target_entity == 4
    assert projectile_entity.projectile_state.homing is True
    assert (
        projectile_entity.projectile_state.drag_stage
        == ProjectileDragStage.OUTBOUND
    )
    assert projectile_entity.projectile_state.impact_tick is None

    terminal_rich = deepcopy(rich)
    terminal_rich["objects"][6]["projectile"].update(  # type: ignore[index]
        {
            "terminal": True,
            "nativePhase": "terminal_or_finished_processing",
        }
    )
    terminal_observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, terminal_rich), catalog)
    )
    terminal = next(
        item for item in terminal_observations[0].entities if item.owner == 0
    ).projectile_state
    assert terminal is not None
    assert terminal.phase == ProjectilePhase.UNKNOWN
    assert terminal.impact_tick is None
    assert terminal.attributes["terminal_reason"] == "unknown"


def test_projectile_velocity_uses_consecutive_native_position_ticks(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    before_rich["objects"][6]["projectile"] = {  # type: ignore[index]
        "projectileDataGlobalId": 10_000_123,
        "sourceEntityKey": [0, -2, 5_000_006],
        "sourceEntityValidated": True,
        "targetEntityKey": [1, -2, 5_000_004],
        "targetEntityValidated": True,
        "homingTargetEntityKey": None,
        "homingTargetEntityValidated": True,
        "destinationX": 14_500,
        "destinationY": 25_500,
        "terminal": False,
        "nativePhase": "in_flight",
        "dragStage": None,
    }
    after = deepcopy(before)
    after["tick"] = int(before["tick"]) + 2
    after["objects"][6]["x"] = 8_100  # type: ignore[index]
    after["objects"][6]["y"] = 11_960  # type: ignore[index]
    after_rich = _rich(after)
    after_rich["objects"][6]["projectile"] = deepcopy(  # type: ignore[index]
        before_rich["objects"][6]["projectile"]  # type: ignore[index]
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    entity = next(item for item in observations[0].entities if item.owner == 0)
    assert entity.velocity == (50.0, -20.0)
    assert entity.projectile_state is not None
    assert entity.projectile_state.velocity == entity.velocity
    assert (
        entity.projectile_state.provenance.field_evidence["velocity"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )
    assert entity.projectile_state.provenance.source_fields["velocity"] == (
        "ObservationV1.objects[].x",
        "ObservationV1.objects[].y",
        "ObservationV1.tick",
    )


def test_player_runtime_is_private_and_kind5_owner_root_is_filtered(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _add_player_runtime(ordinary, rich)
    env = _environment(_RichNativeStub(ordinary, rich), catalog)

    observations, _ = _reset(
        env,
        match_config=_match_from_raw(ordinary),
    )
    actor_zero = observations[0]
    actor_one = observations[1]
    assert len(actor_zero.entities) == 2
    assert actor_zero.metadata["entity_count_before_limit"] == 2
    assert all(item.internal.get("object_index") != 500 for item in actor_zero.entities)

    private_player = next(item for item in actor_zero.players if item.owner == 0)
    ability = private_player.ability_runtime_states[0]
    assert ability.ability_id == "ArcherQueenRapid"
    assert ability.phase == AbilityPhase.READY
    assert ability.charges == 1
    assert (
        ability.provenance.field_evidence["charges"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )
    assert ability.attributes["remaining_charges_raw"] == 1
    assert ability.source_entity == next(
        item.entity_id
        for item in actor_zero.entities
        if item.card_id == 26_000_072
    )

    evolution = private_player.evolution_runtime_states[0]
    assert evolution.card_id == 26_000_001
    assert evolution.deck_slot == 1
    assert evolution.phase == EvolutionPhase.CYCLING
    assert evolution.cycle_remaining == 1
    assert evolution.active is None
    assert evolution.current_form_id is None
    assert evolution.provenance.field_evidence["active"] == SemanticEvidenceLevel.UNKNOWN
    assert all(item.evolution_state is None for item in actor_zero.entities)

    hidden_opponent_private = next(
        item for item in actor_one.players if item.owner == 0
    )
    assert hidden_opponent_private.ability_runtime_states == ()
    assert hidden_opponent_private.evolution_runtime_states == ()



def test_limited_availability_projects_as_queueable_ready_phase(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _add_player_runtime(ordinary, rich)
    runtime = rich["players"][0]["abilityRuntime"][0]  # type: ignore[index]
    runtime.update(
        {
            "buttonState": 4,
            "buttonStateLabel": "LimitedAvailability",
            "available": True,
        }
    )

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog),
        match_config=_match_from_raw(ordinary),
    )
    private_player = next(
        item for item in observations[0].players if item.owner == 0
    )
    ability = private_player.ability_runtime_states[0]

    assert ability.phase == AbilityPhase.READY
    assert ability.available is True


@pytest.mark.parametrize(
    ("base_card_id", "form_card_id", "ability_id", "configured_cooldown_ms"),
    (
        (26_000_000, 203_000_000, "Knight_hero_Ability", 0),
        (26_000_039, 203_000_039, "MegaMinion_Teleport_Ability", 1_000),
    ),
)
def test_hero_form_ability_joins_base_deck_card_and_live_source(
    catalog,
    base_card_id: int,
    form_card_id: int,
    ability_id: str,
    configured_cooldown_ms: int,
) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _add_player_runtime(ordinary, rich)
    ordinary["players"][0]["deck"][0]["cardId"] = base_card_id  # type: ignore[index]
    hero_cost = int(build_card_catalog().by_id[base_card_id].elixir_cost or 0)
    ordinary["players"][0]["hand"][0].update(  # type: ignore[index]
        {
            "cardId": base_card_id,
            "commandCardId": base_card_id,
            "deckSlot": 0,
            "cost": hero_cost,
            "cardParameter": (hero_cost << 28) | (1 << 22),
        }
    )
    rich["players"][0]["evolutionRuntime"][0]["cardId"] = base_card_id  # type: ignore[index]
    rich["players"][0]["evolutionRuntime"][0]["baseSpellGlobalId"] = base_card_id  # type: ignore[index]
    ordinary["objects"][6]["cardId"] = form_card_id  # type: ignore[index]
    rich["objects"][6]["cardId"] = form_card_id  # type: ignore[index]
    runtime = rich["players"][0]["abilityRuntime"][0]  # type: ignore[index]
    runtime.update(
        {
            "actionDataGlobalId": 88_000_101,
            "actionDataName": ability_id,
            "selectedCharacterDataGlobalId": form_card_id,
            "configuredCooldownMs": configured_cooldown_ms,
        }
    )

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog),
        match_config=_match_from_raw(
            ordinary,
            deck0_form_availability=(2, 0, 0, 0, 0, 0, 0, 0),
        ),
    )
    private_player = next(
        item for item in observations[0].players if item.owner == 0
    )
    ability = private_player.ability_runtime_states[0]
    assert ability.ability_id == ability_id
    assert ability.phase == AbilityPhase.READY
    assert ability.source_entity == next(
        item.entity_id
        for item in observations[0].entities
        if item.card_id == form_card_id
    )


def test_explicit_hero_ability_runtime_requires_owner_slot_bit_two(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _add_player_runtime(ordinary, rich)
    ordinary["players"][0]["deck"][0]["cardId"] = 26_000_000  # type: ignore[index]
    knight_cost = int(build_card_catalog().by_id[26_000_000].elixir_cost or 0)
    ordinary["players"][0]["hand"][0].update(  # type: ignore[index]
        {
            "cardId": 26_000_000,
            "commandCardId": 26_000_000,
            "deckSlot": 0,
            "cost": knight_cost,
            "cardParameter": (knight_cost << 28) | (1 << 22),
        }
    )
    rich["players"][0]["evolutionRuntime"][0]["cardId"] = 26_000_000  # type: ignore[index]
    rich["players"][0]["evolutionRuntime"][0]["baseSpellGlobalId"] = (  # type: ignore[index]
        26_000_000
    )
    ordinary["objects"][6]["cardId"] = 203_000_000  # type: ignore[index]
    rich["objects"][6]["cardId"] = 203_000_000  # type: ignore[index]
    rich["players"][0]["abilityRuntime"][0].update(  # type: ignore[index]
        {
            "actionDataGlobalId": 88_000_101,
            "actionDataName": "Knight_hero_Ability",
            "selectedCharacterDataGlobalId": 203_000_000,
            "configuredCooldownMs": 0,
        }
    )
    env = _environment(
        _AbilityTransitionNativeStub(
            ordinary,
            rich,
            deepcopy(ordinary),
            deepcopy(rich),
        ),
        catalog,
    )

    with pytest.raises(
        BattleEnvError,
        match="explicit-Hero ability is not enabled",
    ):
        _reset(env, match_config=_match_from_raw(ordinary))

    observation_only_native = _RichNativeStub(ordinary, rich)
    assert not callable(
        getattr(observation_only_native, "queue_ability_action_at", None)
    )
    observation_only_env = _environment(observation_only_native, catalog)
    with pytest.raises(
        BattleEnvError,
        match="explicit-Hero ability is not enabled",
    ):
        _reset(
            observation_only_env,
            match_config=_match_from_raw(ordinary),
        )


def _ability_transition_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    before = _ordinary()
    before_rich = _rich(before)
    _add_player_runtime(before, before_rich)
    after = deepcopy(before)
    after["tick"] = int(before["tick"]) + 1
    after["players"][0]["elixirRaw"] = 50_000  # type: ignore[index]
    after_rich = deepcopy(before_rich)
    after_rich["tick"] = after["tick"]
    after_rich["combatEvents"]["observationTick"] = after["tick"]  # type: ignore[index]
    runtime = after_rich["players"][0]["abilityRuntime"][0]  # type: ignore[index]
    runtime.update(
        {
            "buttonState": 10,
            "buttonStateLabel": "ChampionCasting",
            "available": False,
            "remainingChargesRaw": 0,
        }
    )
    return before, before_rich, after, after_rich


def test_battle_env_masks_and_queues_exact_next_tick_ability(catalog) -> None:
    before, before_rich, after, after_rich = _ability_transition_fixture()
    native = _AbilityTransitionNativeStub(
        before,
        before_rich,
        after,
        after_rich,
    )
    env = _environment(native, catalog)

    observations, _ = _reset(
        env,
        match_config=_match_from_raw(before),
    )
    assert env.match_config is not None
    assert env.match_config.deck0_form_availability == (0,) * 8
    source = observations[0].action_mask.ability_sources[0]
    assert observations[0].action_mask.kinds[
        ActionKind.ACTIVATE_ABILITY.value
    ] is True
    assert observations[1].action_mask.kinds[
        ActionKind.ACTIVATE_ABILITY.value
    ] is False

    action = ActionV1(
        owner=0,
        kind=ActionKind.ACTIVATE_ABILITY,
        source_entity=source,
        ability_id="ArcherQueenRapid",
        target_kind=TargetKind.NONE,
        next_decision_ticks=1,
    )
    next_observations, _rewards, _terminated, _truncated, _infos = env.step(
        {0: action, 1: ActionV1.wait(1)}
    )

    assert native.ability_actions == [(AbilityAction(0, -2, 5_000_006), 1)]
    assert next_observations[0].action_mask.kinds[
        ActionKind.ACTIVATE_ABILITY.value
    ] is False
    player = next(item for item in next_observations[0].players if item.owner == 0)
    assert player.ability_runtime_states[0].phase == AbilityPhase.CASTING
    assert any(
        event.event_type == "ability_state_change"
        for event in next_observations[0].events
    )
    assert any(
        event.event_type == "ability_command_queued"
        and event.entity_id == source
        for event in next_observations[0].events
    )
    assert all(
        event.event_type != "ability_command_queued"
        for event in next_observations[1].events
    )
    public_activations = [
        event
        for event in next_observations[1].events
        if event.event_type == "ability_activation" and event.owner == 0
    ]
    assert len(public_activations) == 1
    assert public_activations[0].entity_id == source
    assert public_activations[0].data["ability_id"] == "ArcherQueenRapid"
    assert public_activations[0].data["source"] == (
        "visible_runtime_transition"
    )
    assert "action_id" not in public_activations[0].data
    assert "remaining_cooldown_ms" not in public_activations[0].data
    assert all(
        not (
            event.event_type == "action_executed"
            and event.data.get("action_id") == action.action_id
        )
        for event in next_observations[1].events
    )
    executed = [
        event
        for event in next_observations[0].events
        if event.event_type == "action_executed"
        and event.data.get("action_id") == action.action_id
    ]
    assert len(executed) == 1
    assert executed[0].data["kind"] == ActionKind.ACTIVATE_ABILITY.value
    assert executed[0].data["fair_ability_activation_exact"] is True
    assert sum(
        event.event_type == "runtime_ability_activation"
        and event.data.get("action_id") == action.action_id
        for event in next_observations[0].events
    ) == 1
    assert env.awaiting_ability_action_ids == ()


def test_battle_env_waits_for_delayed_ability_apply_edge(catalog) -> None:
    before, before_rich, _after, _after_rich = _ability_transition_fixture()
    waiting = deepcopy(before)
    waiting["tick"] = int(before["tick"]) + 1
    waiting_rich = deepcopy(before_rich)
    waiting_rich["tick"] = waiting["tick"]
    waiting_rich["combatEvents"]["observationTick"] = waiting["tick"]  # type: ignore[index]
    native = _AbilityTransitionNativeStub(
        before,
        before_rich,
        waiting,
        waiting_rich,
    )
    env = _environment(native, catalog)
    observations, _ = _reset(
        env,
        match_config=_match_from_raw(before),
    )
    action = ActionV1(
        owner=0,
        kind=ActionKind.ACTIVATE_ABILITY,
        source_entity=observations[0].action_mask.ability_sources[0],
        ability_id="ArcherQueenRapid",
        target_kind=TargetKind.NONE,
        next_decision_ticks=1,
    )

    waiting_observations, _rewards, _terminated, _truncated, infos = env.step(
        {0: action, 1: ActionV1.wait(1, ticks=1)}
    )

    assert env.awaiting_ability_action_ids == (action.action_id,)
    assert infos[0]["pending_count"] == 1
    assert waiting_observations[0].pending_actions[0].status == (
        "awaiting_execution"
    )
    assert not any(
        event.event_type
        in {"action_executed", "action_rejected", "action_unattested"}
        and event.data.get("action_id") == action.action_id
        for event in waiting_observations[0].events
    )

    applied = deepcopy(waiting)
    applied["tick"] = int(before["tick"]) + 34
    applied_rich = deepcopy(waiting_rich)
    applied_rich["tick"] = applied["tick"]
    applied_rich["combatEvents"]["observationTick"] = applied["tick"]  # type: ignore[index]
    applied_runtime = applied_rich["players"][0]["abilityRuntime"][0]  # type: ignore[index]
    applied_runtime.update(
        {
            "buttonState": 6,
            "buttonStateLabel": "AllChargesConsumed",
            "remainingCooldownMs": 0,
            "remainingChargesRaw": 0,
            "available": False,
            # Entity churn is auxiliary; controller/action identity is stable.
            "championEntityKeys": [],
        }
    )
    native.after = applied
    native.after_rich = applied_rich
    applied_observations, _rewards, _terminated, _truncated, infos = env.step(
        {
            0: ActionV1.wait(0, ticks=33),
            1: ActionV1.wait(1, ticks=33),
        }
    )

    matching = [
        event
        for event in applied_observations[0].events
        if event.data.get("action_id") == action.action_id
    ]
    assert sum(event.event_type == "action_executed" for event in matching) == 1
    assert sum(
        event.event_type == "runtime_ability_activation" for event in matching
    ) == 1
    executed = next(
        event for event in matching if event.event_type == "action_executed"
    )
    assert executed.data["ability_execution_evidence"] == (
        "charge_consumed",
    )
    assert env.awaiting_ability_action_ids == ()
    assert infos[0]["pending_count"] == 0


def test_native_ability_source_disappearing_does_not_end_match(catalog, monkeypatch) -> None:
    before, before_rich, _after, _after_rich = _ability_transition_fixture()
    after = deepcopy(before)
    after["tick"] = int(before["tick"]) + 1
    after_rich = deepcopy(before_rich)
    after_rich["tick"] = after["tick"]
    after_rich["combatEvents"]["observationTick"] = after["tick"]  # type: ignore[index]
    native = _AbilityTransitionNativeStub(before, before_rich, after, after_rich)

    def reject_stale_source(_action: AbilityAction, *, execute_in_ticks: int = 1) -> None:
        raise RunnerError("could not resolve one ready native champion ability source")

    monkeypatch.setattr(native, "queue_ability_action_at", reject_stale_source)
    env = _environment(native, catalog)
    observations, _ = _reset(env, match_config=_match_from_raw(before))
    source = observations[0].action_mask.ability_sources[0]
    action = ActionV1(
        owner=0,
        kind=ActionKind.ACTIVATE_ABILITY,
        source_entity=source,
        ability_id="ArcherQueenRapid",
        target_kind=TargetKind.NONE,
        next_decision_ticks=1,
    )

    next_observations, _rewards, terms, truncs, _infos = env.step(
        {0: action, 1: ActionV1.wait(1, ticks=1)}
    )

    assert next_observations[0].tick == after["tick"]
    assert not terms["__all__"] and not truncs["__all__"]
    assert env.awaiting_ability_action_ids == ()
    assert any(
        event.event_type == "action_rejected"
        and event.entity_id == source
        and event.data.get("reason") == "native_ability_source_no_longer_ready"
        for event in next_observations[0].events
    )
    assert all(event.event_type != "ability_command_queued" for event in next_observations[0].events)


def test_battle_env_marks_ability_unattested_only_at_deadline(catalog) -> None:
    before, before_rich, _after, _after_rich = _ability_transition_fixture()
    waiting = deepcopy(before)
    waiting["tick"] = int(before["tick"]) + 1
    waiting_rich = deepcopy(before_rich)
    waiting_rich["tick"] = waiting["tick"]
    waiting_rich["combatEvents"]["observationTick"] = waiting["tick"]  # type: ignore[index]
    native = _AbilityTransitionNativeStub(
        before,
        before_rich,
        waiting,
        waiting_rich,
    )
    env = _environment(native, catalog)
    observations, _ = _reset(
        env,
        match_config=_match_from_raw(before),
    )
    action = ActionV1(
        owner=0,
        kind=ActionKind.ACTIVATE_ABILITY,
        source_entity=observations[0].action_mask.ability_sources[0],
        ability_id="ArcherQueenRapid",
        target_kind=TargetKind.NONE,
        next_decision_ticks=1,
    )
    interim, _rewards, _terminated, _truncated, _infos = env.step(
        {0: action, 1: ActionV1.wait(1, ticks=1)}
    )
    assert not any(
        event.event_type == "action_unattested"
        and event.data.get("action_id") == action.action_id
        for event in interim[0].events
    )

    deadline = env.next_ability_attestation_deadline_native_tick
    assert deadline is not None
    timed_out = deepcopy(waiting)
    timed_out["tick"] = deadline
    timed_out_rich = deepcopy(waiting_rich)
    timed_out_rich["tick"] = deadline
    timed_out_rich["combatEvents"]["observationTick"] = deadline  # type: ignore[index]
    native.after = timed_out
    native.after_rich = timed_out_rich
    delta = deadline - int(waiting["tick"])
    observations, _rewards, _terminated, _truncated, infos = env.step(
        {
            0: ActionV1.wait(0, ticks=delta),
            1: ActionV1.wait(1, ticks=delta),
        }
    )

    matching = [
        event
        for event in observations[0].events
        if event.data.get("action_id") == action.action_id
    ]
    assert sum(event.event_type == "action_unattested" for event in matching) == 1
    assert all(event.event_type != "action_rejected" for event in matching)
    assert all(event.event_type != "action_executed" for event in matching)
    assert env.awaiting_ability_action_ids == ()
    assert infos[0]["pending_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    (
        {
            "buttonState": 10,
            "buttonStateLabel": "ChampionCasting",
            "available": False,
            "remainingCooldownMs": 0,
            "remainingChargesRaw": 0,
        },
        {
            "buttonState": 0,
            "buttonStateLabel": "invalid/no match",
            "available": False,
        },
        {
            "buttonState": 6,
            "buttonStateLabel": "AllChargesConsumed",
            "available": False,
            "remainingCooldownMs": 0,
            "remainingChargesRaw": 0,
        },
    ),
)
def test_battle_env_masks_non_ready_native_ability(
    catalog,
    mutation: dict[str, object],
) -> None:
    before, before_rich, after, after_rich = _ability_transition_fixture()
    before_rich["players"][0]["abilityRuntime"][0].update(mutation)  # type: ignore[index]
    native = _AbilityTransitionNativeStub(
        before,
        before_rich,
        after,
        after_rich,
    )
    env = _environment(native, catalog)

    observations, _ = _reset(
        env,
        match_config=_match_from_raw(before),
    )

    assert observations[0].action_mask.ability_sources == ()
    assert observations[0].action_mask.kinds[
        ActionKind.ACTIVATE_ABILITY.value
    ] is False
    assert native.ability_actions == []


def test_battle_env_rejects_wrong_ability_id_and_queues_future_timing(catalog) -> None:
    before, before_rich, after, after_rich = _ability_transition_fixture()
    native = _AbilityTransitionNativeStub(
        before,
        before_rich,
        after,
        after_rich,
    )
    env = _environment(native, catalog)
    observations, _ = _reset(
        env,
        match_config=_match_from_raw(before),
    )
    source = observations[0].action_mask.ability_sources[0]

    with pytest.raises(BattleEnvError, match="not requested"):
        env.step(
            {
                0: ActionV1(
                    owner=0,
                    kind=ActionKind.ACTIVATE_ABILITY,
                    source_entity=source,
                    ability_id="wrong-ability",
                ),
                1: ActionV1.wait(1),
            }
        )
    assert native.ability_actions == []
    assert native.advanced is False

    future, _rewards, _terminated, _truncated, _infos = env.step(
        {
            0: ActionV1(
                owner=0,
                kind=ActionKind.ACTIVATE_ABILITY,
                source_entity=source,
                ability_id="ArcherQueenRapid",
                execute_offset_ticks=7,
            ),
            1: ActionV1.wait(1),
        }
    )
    assert native.ability_actions == [
        (AbilityAction(0, -2, 5_000_006), 7)
    ]
    assert native.advanced is True
    assert len(future[0].pending_actions) == 1
    assert future[0].pending_actions[0].expected_execution_tick == (
        int(before["tick"]) + 7
    )


def test_rich_transition_events_cover_shield_effect_target_stage_and_visibility(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    before_item = before_rich["objects"][6]  # type: ignore[index]
    before_item.update(
        {
            "shield": {"current": 100, "max": 100, "status": "authoritative"},
            "targetEntityValidated": True,
            "attackSequenceStage": 0,
        }
    )
    after = deepcopy(before)
    after["tick"] = int(before["tick"]) + 1
    after_rich = deepcopy(before_rich)
    after_rich["tick"] = after["tick"]
    after_rich["combatEvents"]["observationTick"] = after["tick"]  # type: ignore[index]
    after_item = after_rich["objects"][6]  # type: ignore[index]
    after_item.update(
        {
            "shield": {"current": 0, "max": 100, "status": "authoritative"},
            "targetEntityKey": [1, -2, 5_000_007],
            "targetEntityValidated": True,
            "attackSequenceStage": 1,
            "activeEffects": [
                {
                    "buffGlobalId": 9_000_000,
                    "name": "Rage",
                    "remainingMs": 1_000,
                    "sourceEntityKey": [0, -2, 5_000_006],
                    "sourceEntityValidated": True,
                }
            ],
            "invisibleCount": 1,
            "visibilityState": "invisible",
        }
    )
    native = _TransitionNativeStub(
        before,
        before_rich,
        after,
        after_rich,
    )
    env = _environment(native, catalog)
    observations, _ = _reset(env)
    initial_entity = next(
        item for item in observations[0].entities if item.card_id == 26_000_000
    )
    assert initial_entity.visibility_state is not None
    assert initial_entity.visibility_state.phase == VisibilityPhase.VISIBLE

    next_observations, _rewards, _terminated, _truncated, _infos = env.step(
        {0: ActionV1.wait(0), 1: ActionV1.wait(1)}
    )

    owner_zero_types = {event.event_type for event in next_observations[0].events}
    assert {
        "shield_damage",
        "shield_break",
        "effect_apply",
        "target_acquire",
        "attack_sequence_change",
        "visibility_change",
    }.issubset(owner_zero_types)
    hidden_entity = next(
        item for item in next_observations[0].entities if item.card_id == 26_000_000
    )
    assert hidden_entity.visibility_state is not None
    assert hidden_entity.visibility_state.phase == VisibilityPhase.HIDDEN
    opponent_view = next(
        item
        for item in next_observations[1].entities
        if item.card_id == 26_000_000
    )
    assert opponent_view.visibility_state is not None
    assert opponent_view.visibility_state.phase == VisibilityPhase.HIDDEN
    owner_one_types = {event.event_type for event in next_observations[1].events}
    assert "visibility_change" in owner_one_types
    assert "effect_apply" in owner_one_types

def test_unjoined_native_ability_does_not_invent_a_public_ability_id(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _add_player_runtime(ordinary, rich)
    rich["players"][0]["abilityRuntime"][0]["actionDataName"] = (  # type: ignore[index]
        "UnknownNativeAction"
    )

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog),
        match_config=_match_from_raw(ordinary),
    )
    private_player = next(
        item for item in observations[0].players if item.owner == 0
    )
    assert private_player.ability_runtime_states == ()
    assert (
        private_player.runtime_provenance.field_evidence[
            "ability_runtime_states"
        ]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )


def test_champion_absent_preserves_unique_ability_identity_without_source(
    catalog,
) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _add_player_runtime(ordinary, rich)
    runtime = rich["players"][0]["abilityRuntime"][0]  # type: ignore[index]
    runtime.update(
        {
            "buttonState": 1,
            "buttonStateLabel": "ChampionAbsent",
            "available": False,
            "championEntityKeys": [],
        }
    )

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog),
        match_config=_match_from_raw(ordinary),
    )
    private_player = next(
        item for item in observations[0].players if item.owner == 0
    )
    ability = private_player.ability_runtime_states[0]
    assert ability.ability_id == "ArcherQueenRapid"
    assert ability.phase == AbilityPhase.UNAVAILABLE
    assert ability.available is False
    assert ability.source_entity is None
    assert (
        ability.provenance.field_evidence["source_entity"]
        == SemanticEvidenceLevel.UNKNOWN
    )


@pytest.mark.parametrize("mismatch", ("tick", "generation", "slot", "identity"))
def test_rich_merge_fails_closed_on_state_or_identity_mismatch(
    catalog, mismatch: str
) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    if mismatch in {"tick", "generation"}:
        rich[mismatch] = int(rich[mismatch]) + 1
    elif mismatch == "slot":
        rich["objects"][6]["slot"] = 99  # type: ignore[index]
    else:
        rich["objects"][6]["entityKey"] = [0, 999, 0]  # type: ignore[index]
    env = _environment(_RichNativeStub(ordinary, rich), catalog)

    with pytest.raises(BattleEnvError, match="atomic telemetry merge failed"):
        _reset(env)


def test_native_render_cross_tick_capture_has_explicit_fail_closed_reason(
    catalog,
) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    rich["tick"] = int(rich["tick"]) + 1
    env = _environment(_RichNativeStub(ordinary, rich), catalog)

    with pytest.raises(BattleEnvError, match="atomic telemetry merge failed"):
        env.reset(
            match_config=MatchConfig(
                deck0=SEMANTIC_BASELINE_DECK,
                deck1=SEMANTIC_BASELINE_DECK,
            ),
            options={"render_mode": "native-render"},
        )


def test_battle_env_prefers_atomic_pair_and_adopts_its_ordinary_tick(catalog) -> None:
    stale = _ordinary()
    atomic = _ordinary()
    atomic["tick"] = 21
    rich = _rich(atomic)
    native = _AtomicNativeStub(stale, atomic, rich)
    env = _environment(native, catalog)

    _observations, info = env.reset(
        match_config=MatchConfig(
            deck0=SEMANTIC_BASELINE_DECK,
            deck1=SEMANTIC_BASELINE_DECK,
        ),
        options={"render_mode": "native-render"},
    )

    assert info["native_tick"] == 21
    assert env.raw_observation["tick"] == 21
    assert native.observe_atomic_calls == 1


def test_battle_env_rejects_pre_atomic_probe(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    native = _OldProbeNativeStub(ordinary, rich)

    with pytest.raises(BattleEnvError, match="atomic telemetry failed closed"):
        _reset(_environment(native, catalog))

    assert native.observe_rich_calls == 0


def test_battle_env_does_not_downgrade_a_real_atomic_failure(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    native = _FailedAtomicNativeStub(ordinary, rich)

    with pytest.raises(BattleEnvError, match="atomic telemetry failed closed"):
        _reset(_environment(native, catalog))

    assert native.observe_rich_calls == 0


def test_battle_env_step_refreshes_raw_and_rich_from_one_atomic_pair(catalog) -> None:
    before = _ordinary()
    before["tick"] = 0
    before["stateEpoch"] = 8
    before_rich = _rich(before)
    after = _ordinary()
    after_rich = _rich(after)
    native = _AtomicTransitionNativeStub(before, before_rich, after, after_rich)
    env = _environment(native, catalog)
    _reset(env)

    env.step({})

    assert env.raw_observation["tick"] == 20
    assert native.observe_atomic_calls == 2




def test_synchronized_native_render_advances_exactly_then_captures(catalog) -> None:
    before = _ordinary()
    before["tick"] = 0
    before["stateEpoch"] = 8
    before_rich = _rich(before)
    after = _ordinary()
    after["tick"] = 5
    after_rich = _rich(after)
    native = _SynchronizedAtomicTransitionNativeStub(
        before,
        before_rich,
        after,
        after_rich,
    )
    env = _environment(native, catalog)
    env.reset(
        match_config=MatchConfig(
            deck0=SEMANTIC_BASELINE_DECK,
            deck1=SEMANTIC_BASELINE_DECK,
        ),
        options={
            "render_mode": "native-render",
            "pause_native_render": True,
            "synchronize_native_render_steps": True,
            "native_render_speed": 4.0,
            "decision_ticks": 5,
            "event_driven_decisions": False,
        },
    )

    env.step({})

    assert native.pause_calls == 1
    assert native.speed_calls == [4.0]
    assert native.advance_calls == [5]
    assert native.observe_atomic_calls == 2
    assert env.raw_observation["tick"] == 5


def test_fair_view_keeps_invisible_entities_and_runtime_links_public(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    hidden_key = [1, -2, 5_000_007]
    rich["objects"][7].update(  # type: ignore[index,union-attr]
        {"invisibleCount": 1, "visibilityState": "invisible"}
    )
    rich["objects"][6].update(  # type: ignore[index,union-attr]
        {
            "targetEntityKey": hidden_key,
            "targetEntityValidated": True,
            "activeEffects": [
                {
                    "buffGlobalId": 9_000_000,
                    "name": "Rage",
                    "remainingMs": 300,
                    "sourceEntityKey": hidden_key,
                    "sourceEntityValidated": True,
                }
            ],
        }
    )
    native = _RichNativeStub(ordinary, rich)
    observations, _ = _reset(_environment(native, catalog))

    actor = observations[0]
    owner = observations[1]
    assert len(actor.entities) == 2
    assert actor.metadata["entity_count_before_limit"] == 2
    visible = next(item for item in actor.entities if item.owner == 0)
    assert visible.visible_target is not None
    assert visible.attack_state is not None
    assert visible.attack_state.target_entity is not None
    assert (
        visible.attack_state.provenance.field_evidence["target_entity"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )
    effect = visible.effect_states[0]
    assert effect.source_entity is not None
    assert effect.source_owner == 1
    assert effect.source_card_id == 26_000_002
    assert (
        effect.provenance.field_evidence["source_entity"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )

    hidden = next(item for item in owner.entities if item.owner == 1)
    assert hidden.visibility_state is not None
    assert hidden.visibility_state.phase == VisibilityPhase.HIDDEN
    assert owner.metadata["entity_count_before_limit"] == 2
    assert native.observe_rich_calls == 1


def test_invisible_spawn_is_public_to_both_players(
    catalog,
) -> None:
    before = _ordinary()
    before["tick"] = 0
    before["stateEpoch"] = 8
    before["objects"] = before["objects"][:-1]  # type: ignore[index]
    before["count"] = len(before["objects"])  # type: ignore[arg-type]
    before["returned"] = len(before["objects"])  # type: ignore[arg-type]
    before_rich = _rich(before)

    after = _ordinary()
    after_rich = _rich(after)
    after_rich["objects"][7].update(  # type: ignore[index,union-attr]
        {"invisibleCount": 1, "visibilityState": "invisible"}
    )
    native = _TransitionNativeStub(before, before_rich, after, after_rich)
    env = _environment(native, catalog)
    _reset(env)

    observations, _, _, _, infos = env.step({})

    actor = observations[0]
    owner = observations[1]
    assert any(event.event_type == "spawn" for event in actor.events)
    assert any(event.event_type == "spawn" for event in owner.events)
    opponent = next(player for player in actor.players if player.owner == 1)
    # A public spawned unit is board evidence, but it may have come from a
    # building or ability and therefore cannot reveal an opponent deck slot.
    assert opponent.revealed_cards == ()
    assert actor.opponent_belief.deck_probabilities == {}
    assert infos["__all__"]["event_wake"] is False
    assert native.observe_rich_calls == 2


def test_native_hero_form_entity_is_public_without_inventing_deck_reveal(
    catalog,
) -> None:
    before = _ordinary()
    before["tick"] = 0
    before["stateEpoch"] = 8
    before["objects"] = before["objects"][:-1]  # type: ignore[index]
    before["count"] = len(before["objects"])  # type: ignore[arg-type]
    before["returned"] = len(before["objects"])  # type: ignore[arg-type]
    before_rich = _rich(before)

    after = _ordinary()
    after["objects"][7]["cardId"] = 203_000_034  # type: ignore[index]
    after_rich = _rich(after)
    native = _TransitionNativeStub(before, before_rich, after, after_rich)
    env = _environment(native, catalog)
    match = MatchConfig(
        deck0=SEMANTIC_BASELINE_DECK,
        deck1=(26_000_034, *SEMANTIC_BASELINE_DECK[:7]),
        deck1_form_availability=(2, 0, 0, 0, 0, 0, 0, 0),
    )
    env.reset(match_config=match)

    observations, _, _, _, _ = env.step({})

    opponent = next(player for player in observations[0].players if player.owner == 1)
    assert opponent.revealed_cards == ()
    hero = next(entity for entity in observations[0].entities if entity.owner == 1)
    assert hero.card_id == 203_000_034




def test_live_reset_replays_opponent_card_play_into_public_tracker(
    catalog,
) -> None:
    ordinary = _ordinary()
    ordinary["tick"] = 150
    rich = _rich(ordinary)
    card_id = SEMANTIC_BASELINE_DECK[0]
    _set_combat_events(
        rich,
        epoch_first_sequence=40,
        events=[
            _card_play_combat_event(
                ordinary,
                sequence=40,
                tick=140,
                owner=1,
                card_id=card_id,
                deployment_sequence=9,
            )
        ],
    )
    native = _RichNativeStub(ordinary, rich)
    native.public_card_play_events_from_combat_ring = True
    env = _environment(native, catalog)

    observations, _ = _reset(env)

    for owner in (0, 1):
        plays = [
            event
            for event in observations[owner].events
            if event.event_type == "action_executed"
        ]
        assert len(plays) == 1
        assert plays[0].owner == 1
        assert plays[0].card_id == card_id
        assert plays[0].data["source"] == "native_card_play"
        assert "deck_slot" not in plays[0].data
        assert "card_parameter" not in plays[0].data
        assert "cost" not in plays[0].data

    costs = {
        item: float(env.card_specs[item].elixir_cost)
        for item in SEMANTIC_BASELINE_DECK
    }
    tracker = DeterministicPublicTracker(
        decks={0: SEMANTIC_BASELINE_DECK, 1: SEMANTIC_BASELINE_DECK},
        card_costs=costs,
        ability_cost_by_owner={0: 1.0, 1: 1.0},
        ability_cooldown_ms_by_owner={0: 1_000, 1: 1_000},
        evolution_cycle_required={},
    )
    tracker.reset(tick=130, initial_elixir={0: 6.0, 1: 6.0})
    tracker.update(observations[0])
    assert tracker.card_state(1, card_id).cycle_distance == 4
    assert tracker.elixir[1] < 4.0


def test_rich_binding_preserves_exact_mirror_root_and_effective_identity() -> None:
    ordinary = _ordinary()
    ordinary["tick"] = 150
    rich = _rich(ordinary)
    event = _card_play_combat_event(
        ordinary,
        sequence=40,
        tick=140,
        owner=1,
        card_id=SEMANTIC_BASELINE_DECK[0],
        deployment_sequence=9,
    )
    deployment = event["deploymentContext"]
    assert isinstance(deployment, dict)
    deployment.update(
        {
            "playedCardGlobalId": 28_000_006,
            "effectiveCardGlobalId": 26_000_014,
            "cardParameter": (5 << 28) | (3 << 22),
            "deckSlot": 2,
            "cost": 5,
            "formCode": 0,
            "formName": "BasicForm",
        }
    )
    _set_combat_events(
        rich,
        epoch_first_sequence=40,
        events=[event],
    )

    snapshot = bind_rich_telemetry(ordinary, rich)
    context = snapshot.combat_events.events[0].deployment_context

    assert context is not None
    assert context.played_card_global_id == 28_000_006
    assert context.effective_card_global_id == 26_000_014
    assert context.cost == 5


def test_live_public_mirror_hero_separates_native_and_policy_identity(
    catalog,
) -> None:
    ordinary = _ordinary()
    ordinary["tick"] = 150
    rich = _rich(ordinary)
    event = _card_play_combat_event(
        ordinary,
        sequence=40,
        tick=140,
        owner=1,
        card_id=SEMANTIC_BASELINE_DECK[0],
        deployment_sequence=9,
    )
    deployment = event["deploymentContext"]
    assert isinstance(deployment, dict)
    deployment.update(
        {
            "playedCardGlobalId": 28_000_006,
            "effectiveCardGlobalId": 203_000_034,
            "cardParameter": (6 << 28) | (3 << 22),
            "deckSlot": 2,
            "cost": 6,
            "formCode": 0,
            "formName": "BasicForm",
        }
    )
    _set_combat_events(
        rich,
        epoch_first_sequence=40,
        events=[event],
    )
    native = _RichNativeStub(ordinary, rich)
    native.public_card_play_events_from_combat_ring = True
    env = _environment(native, catalog)

    observations, _ = _reset(
        env,
        match_config=MatchConfig(
            deck0=SEMANTIC_BASELINE_DECK,
            deck1=(
                26_000_034,
                SEMANTIC_BASELINE_DECK[0],
                28_000_006,
                *SEMANTIC_BASELINE_DECK[1:6],
            ),
            deck1_form_availability=(2, 0, 0, 0, 0, 0, 0, 0),
        ),
    )

    executed = next(
        event
        for event in observations[0].events
        if event.event_type == "action_executed"
    )
    assert executed.card_id == 28_000_006
    assert executed.data["effective_card_id"] == 26_000_034
    assert executed.data["native_effective_card_id"] == 203_000_034
    assert executed.data["form_code"] == 2
    assert executed.data["native_form_code"] == 0

    invalid_native = _RichNativeStub(ordinary, rich)
    invalid_native.public_card_play_events_from_combat_ring = True
    invalid_env = _environment(invalid_native, catalog)
    with pytest.raises(BattleEnvError, match="episode deck"):
        _reset(
            invalid_env,
            match_config=MatchConfig(
                deck0=SEMANTIC_BASELINE_DECK,
                deck1=(26_000_034, *SEMANTIC_BASELINE_DECK[:7]),
                deck1_form_availability=(2, 0, 0, 0, 0, 0, 0, 0),
            ),
        )


def test_live_native_card_play_does_not_duplicate_local_execution(
    catalog,
) -> None:
    before = _ordinary()
    before["tick"] = 130
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=70, events=[])
    card_id = SEMANTIC_BASELINE_DECK[0]
    before["players"][0]["hand"][0].update(  # type: ignore[index]
        {
            "cardParameter": (3 << 28) | (1 << 22),
            "cost": 3,
        }
    )
    after = deepcopy(before)
    after["tick"] = 131
    replacement_card_id = SEMANTIC_BASELINE_DECK[4]
    after["players"][0]["hand"][0].update(  # type: ignore[index]
        {
            "cardId": replacement_card_id,
            "commandCardId": replacement_card_id,
            "deckSlot": 4,
            "cost": 4,
            "cardParameter": (4 << 28) | (5 << 22),
        }
    )
    after_rich = _rich(after)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=70,
        events=[
            _card_play_combat_event(
                after,
                sequence=70,
                tick=131,
                owner=0,
                card_id=card_id,
                deployment_sequence=12,
            )
        ],
    )
    native = _HandTransitionNativeStub(
        before,
        before_rich,
        after,
        after_rich,
    )
    native.public_card_play_events_from_combat_ring = True
    env = _environment(native, catalog)
    _reset(env)

    observations, _, _, _, _ = env.step(
        {
            0: ActionV1.play(
                owner=0,
                hand_slot=0,
                grid=(8, 10),
                action_id="local-play",
            ),
            1: ActionV1.wait(1, 1),
        }
    )

    executed = [
        event
        for event in observations[0].events
        if event.event_type == "action_executed"
    ]
    assert len(executed) == 1
    assert executed[0].data["action_id"] == "local-play"




def test_fair_entity_keeps_public_deployment_group_without_oracle_event(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=55, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=55,
        events=[_evolved_spawn_combat_event(after, sequence=55, tick=21)],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    for owner in (0, 1):
        assert not any(
            event.event_type == "spawn" for event in observations[owner].events
        )
        entity = next(
            item
            for item in observations[owner].entities
            if item.card_id == SEMANTIC_BASELINE_DECK[0]
        )
        assert entity.causal_group is not None
        assert entity.causal_group.kind == CausalGroupKind.DEPLOYMENT
        assert entity.causal_group.handle.endswith(":11")


def test_fair_multi_unit_deployment_shares_one_exact_deployment_root(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=80, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    goblins = [
        _causal_test_object(
            slot=8 + index,
            owner=0,
            card_id=26_000_002,
            x=7_000 + index * 500,
            y=11_000,
        )
        for index in range(4)
    ]
    after["objects"].extend(goblins)  # type: ignore[union-attr]
    after["count"] = after["returned"] = len(after["objects"])  # type: ignore[arg-type]
    after_rich = _rich(after)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=80,
        events=[
            _causal_spawn_combat_event(
                after,
                target_index=8 + index,
                sequence=80 + index,
                tick=21,
                deployment_card_id=26_000_002,
                deployment_sequence=41,
            )
            for index in range(4)
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    grouped = [
        entity
        for entity in observations[0].entities
        if entity.causal_group is not None
        and entity.causal_group.handle.endswith(":41")
    ]
    assert len(grouped) == 4
    assert {entity.causal_group.handle for entity in grouped} == {
        "deploy:3:9:41"
    }
    assert all(
        entity.causal_group.source_card_id == 26_000_002
        for entity in grouped
    )


def test_fair_runtime_children_share_parent_and_spawn_tick_root(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=90, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    additions = [
        _causal_test_object(
            slot=8 + index,
            owner=0,
            card_id=26_000_002,
            x=8_000 + index * 500,
            y=12_000,
        )
        for index in range(3)
    ]
    after["objects"].extend(additions)  # type: ignore[union-attr]
    after["count"] = after["returned"] = len(after["objects"])  # type: ignore[arg-type]
    after_rich = _rich(after)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=90,
        events=[
            _causal_spawn_combat_event(
                after,
                target_index=8,
                sequence=90,
                tick=21,
                deployment_card_id=26_000_002,
                deployment_sequence=42,
            ),
            _causal_spawn_combat_event(
                after,
                target_index=9,
                source_index=8,
                sequence=91,
                tick=21,
                cause_sequence=90,
            ),
            _causal_spawn_combat_event(
                after,
                target_index=10,
                source_index=8,
                sequence=92,
                tick=21,
                cause_sequence=90,
            ),
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    children = [
        entity
        for entity in observations[0].entities
        if entity.causal_group is not None
        and entity.causal_group.kind == CausalGroupKind.SPAWN_WAVE
    ]
    assert len(children) == 2
    assert len({entity.causal_group.handle for entity in children}) == 1
    assert len({entity.causal_group.parent_entity_id for entity in children}) == 1
    assert all(
        entity.causal_group.source_card_id == 26_000_002
        for entity in children
    )


def test_projectiles_without_an_exact_release_root_remain_singletons(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=100, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    projectiles = [
        _causal_test_object(
            slot=8 + index,
            owner=0,
            card_id=SEMANTIC_BASELINE_DECK[0],
            x=9_000 + index * 750,
            y=13_000 + index * 500,
        )
        for index in range(2)
    ]
    after["objects"].extend(projectiles)  # type: ignore[union-attr]
    after["count"] = after["returned"] = len(after["objects"])  # type: ignore[arg-type]
    after_rich = _rich(after)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=100,
        events=[
            _causal_spawn_combat_event(
                after,
                target_index=8 + index,
                source_index=6,
                sequence=100 + index,
                tick=21,
                projectile=True,
            )
            for index in range(2)
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    pellets = [
        entity
        for entity in observations[0].entities
        if entity.causal_group is not None
        and entity.causal_group.kind == CausalGroupKind.VOLLEY
    ]
    assert len(pellets) == 2
    assert len({entity.causal_group.handle for entity in pellets}) == 2
    assert all(
        entity.causal_group.source_card_id == SEMANTIC_BASELINE_DECK[0]
        for entity in pellets
    )


def test_projectile_deployments_use_exact_native_deployment_roots(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=120, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    source_cards = (28_000_000, 28_000_003)
    projectiles = [
        _causal_test_object(
            slot=8 + index,
            owner=0,
            card_id=source_card,
            x=9_000 + index * 750,
            y=13_000 + index * 500,
        )
        for index, source_card in enumerate(source_cards)
    ]
    after["objects"].extend(projectiles)  # type: ignore[union-attr]
    after["count"] = after["returned"] = len(after["objects"])  # type: ignore[arg-type]
    after_rich = _rich(after)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=120,
        events=[
            _causal_spawn_combat_event(
                after,
                target_index=8 + index,
                source_index=6,
                sequence=120 + index,
                tick=21,
                projectile=True,
                deployment_card_id=source_card,
                deployment_sequence=501 + index,
            )
            for index, source_card in enumerate(source_cards)
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    projectiles = [
        entity
        for entity in observations[0].entities
        if entity.causal_group is not None
        and entity.causal_group.kind == CausalGroupKind.VOLLEY
    ]
    assert len(projectiles) == 2
    assert {entity.causal_group.handle for entity in projectiles} == {
        "volley:3:9:deploy:501",
        "volley:3:9:deploy:502",
    }
    assert {
        entity.causal_group.source_card_id for entity in projectiles
    } == set(source_cards)


def test_projectile_deflect_rebinds_exact_current_owner_root(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=120, events=[])
    spawned = deepcopy(before)
    spawned["tick"] = 21
    source_card = 28_000_000
    projectiles = [
        _causal_test_object(
            slot=8 + index,
            owner=1,
            card_id=source_card,
            x=9_000 + index * 750,
            y=13_000 + index * 500,
        )
        for index in range(2)
    ]
    spawned["objects"].extend(projectiles)  # type: ignore[union-attr]
    spawned["count"] = spawned["returned"] = len(
        spawned["objects"]  # type: ignore[arg-type]
    )
    events = [
        _causal_spawn_combat_event(
            spawned,
            target_index=8 + index,
            source_index=7,
            sequence=120 + index,
            tick=21,
            projectile=True,
            deployment_card_id=source_card,
            deployment_sequence=501,
        )
        for index in range(2)
    ]
    spawned_rich = _rich(spawned)
    _set_combat_events(
        spawned_rich,
        epoch_first_sequence=120,
        events=deepcopy(events),
    )
    after = deepcopy(spawned)
    after["tick"] = 22

    # One member of the exact deployment volley is deflected.  The native
    # object identity survives while its current arena owner changes.
    reflected_object = after["objects"][8]  # type: ignore[index]
    reflected_object["owner"] = 0
    reflected = _combat_fact(reflected_object)
    reflected["objectKind"] = 4
    deflector = _combat_fact(after["objects"][6])  # type: ignore[index]
    shooter = _combat_fact(after["objects"][7])  # type: ignore[index]
    events.append(
        {
            "sequence": 122,
            "tick": 22,
            "generation": after["generation"],
            "stateEpoch": after["stateEpoch"],
            "kind": "projectile_deflect",
            "hookOffset": 0xF1629C,
            "callerOffset": None,
            "causeSequence": None,
            "pool": "none",
            "terminalReason": "none",
            "lethal": False,
            "deploymentContext": None,
            "target": shooter,
            "immediateSource": deflector,
            "source": deflector,
            "projectile": reflected,
            "related": deflector,
            "routeSourceBefore": shooter,
            "routeTargetBefore": deflector,
            "routeSourceAfter": deflector,
            "routeTargetAfter": shooter,
            "requestedAmount": None,
            "actualAmount": None,
            "preHp": None,
            "postHp": None,
            "preBuiltInShield": None,
            "postBuiltInShield": None,
            "preBuffShield": None,
            "postBuffShield": None,
            "destinationBefore": [9_000, 13_000],
            "destinationAfter": [9_000, 17_000],
        }
    )
    after_rich = _rich(after)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=120,
        events=events,
    )
    native = _TransitionNativeStub(
        before,
        before_rich,
        spawned,
        spawned_rich,
    )
    env = _environment(native, catalog)
    _reset(env)
    env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})
    native.after = after
    native.after_rich = after_rich
    native.advanced = False

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    groups = {
        group.key.handle: group
        for group in build_causal_groups(observations[0].entities)
        if group.key.kind == CausalGroupKind.VOLLEY
        and group.source_card_id == source_card
    }
    assert set(groups) == {
        "volley:3:9:deploy:501",
        "volley:3:9:deflect:122",
    }
    assert groups["volley:3:9:deploy:501"].owner == 1
    assert groups["volley:3:9:deflect:122"].owner == 0
    deflector_entity = next(
        entity
        for entity in observations[0].entities
        if entity.card_id == SEMANTIC_BASELINE_DECK[0]
        and entity.owner == 0
    )
    assert groups["volley:3:9:deflect:122"].parent_entity_id == (
        deflector_entity.entity_id
    )


def test_fair_area_spawn_uses_persistent_effect_root(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=110, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    area_card_id = SEMANTIC_BASELINE_DECK[6]
    after["objects"].append(  # type: ignore[union-attr]
        _causal_test_object(
            slot=8,
            owner=0,
            card_id=area_card_id,
            x=9_000,
            y=16_000,
        )
    )
    after["count"] = after["returned"] = len(after["objects"])  # type: ignore[arg-type]
    after_rich = _rich(after)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=110,
        events=[
            _causal_spawn_combat_event(
                after,
                target_index=8,
                sequence=110,
                tick=21,
                deployment_card_id=area_card_id,
                deployment_sequence=43,
                object_kind=3,
            )
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    area = next(
        entity
        for entity in observations[0].entities
        if entity.causal_group is not None
        and entity.causal_group.kind == CausalGroupKind.PERSISTENT_EFFECT
    )
    assert area.causal_group.handle == "effect:3:9:43"
    assert area.causal_group.source_card_id == area_card_id


def test_fair_area_children_inherit_one_persistent_effect_group(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=120, events=[])
    after = deepcopy(before)
    after["tick"] = 33
    area_card_id = SEMANTIC_BASELINE_DECK[6]
    area_index = len(after["objects"])  # type: ignore[arg-type]
    after["objects"].append(  # type: ignore[union-attr]
        _causal_test_object(
            slot=area_index,
            owner=0,
            card_id=area_card_id,
            x=9_000,
            y=16_000,
        )
    )
    child_start = len(after["objects"])  # type: ignore[arg-type]
    for index in range(12):
        after["objects"].append(  # type: ignore[union-attr]
            _causal_test_object(
                slot=child_start + index,
                owner=0,
                card_id=SEMANTIC_BASELINE_DECK[1],
                x=7_000 + index * 300,
                y=14_000 + index * 250,
            )
        )
    after["count"] = after["returned"] = len(after["objects"])  # type: ignore[arg-type]
    after_rich = _rich(after)
    events = [
        _causal_spawn_combat_event(
            after,
            target_index=area_index,
            sequence=120,
            tick=21,
            deployment_card_id=area_card_id,
            deployment_sequence=43,
            object_kind=3,
        )
    ]
    for index in range(12):
        event = _causal_spawn_combat_event(
            after,
            target_index=child_start + index,
            source_index=area_index,
            sequence=121 + index,
            tick=22 + index,
        )
        event["spawnProvenance"] = {
            "kind": "action_spawn_to_location",
            "hookOffset": 0xE7B864,
            "sourceDataGlobalId": 737_088_471,
        }
        events.append(event)
    _set_combat_events(
        after_rich,
        epoch_first_sequence=120,
        events=events,
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    grouped = [
        entity
        for entity in observations[0].entities
        if entity.causal_group is not None
        and entity.causal_group.kind == CausalGroupKind.PERSISTENT_EFFECT
        and entity.causal_group.handle == "effect:3:9:43"
    ]
    assert len(grouped) == 13
    assert {entity.causal_group.source_card_id for entity in grouped} == {
        area_card_id
    }
    root = next(entity for entity in grouped if entity.card_id == area_card_id)
    children = [entity for entity in grouped if entity.entity_id != root.entity_id]
    assert len(children) == 12
    assert {entity.causal_group.parent_entity_id for entity in children} == {
        root.entity_id
    }


def test_invisible_entity_damage_remains_public_to_both_players(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=70, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after["objects"][7]["hp"] = 650  # type: ignore[index]
    after_rich = _rich(after)
    after_rich["objects"][7].update(  # type: ignore[index,union-attr]
        {"invisibleCount": 1, "visibilityState": "invisible"}
    )
    _set_combat_events(
        after_rich,
        epoch_first_sequence=70,
        events=[
            _damage_combat_event(
                after,
                sequence=70,
                tick=21,
                hidden_target=True,
            )
        ],
    )
    native = _TransitionNativeStub(before, before_rich, after, after_rich)
    env = _environment(native, catalog)
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    for owner in (0, 1):
        damage = [
            item
            for item in observations[owner].events
            if item.event_type == "damage"
        ]
        assert len(damage) == 1
        combat = damage[0].combat
        assert combat is not None
        assert combat.kind == CombatEventKind.DAMAGE
        assert combat.source_entity is None
        assert combat.target_entity == damage[0].entity_id
        assert combat.amount == 50.0



def test_native_combat_cursor_fails_closed_before_oldest_retained_sequence(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_combat_events(before_rich, epoch_first_sequence=90, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_combat_events(after_rich, epoch_first_sequence=90, events=[])
    native = _TransitionNativeStub(before, before_rich, after, after_rich)
    env = _environment(native, catalog)
    _reset(env)
    env._combat_next_sequence = 89

    with pytest.raises(BattleEnvError, match="overwrote unseen"):
        env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})


def test_phase_runtime_projects_attack_movement_and_deployment(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_phase_runtime(
        before_rich,
        events=[],
        object_phase=_phase_object(tick=20),
    )

    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    after_rich["objects"][6].update(  # type: ignore[index,union-attr]
        {
            "targetEntityKey": [1, -2, 5_000_007],
            "targetEntityValidated": True,
        }
    )
    _set_phase_runtime(
        after_rich,
        events=[_phase_event(after, sequence=100, tick=21)],
        object_phase=_phase_object(
            tick=21,
            attack_timeline_ms=100,
            deploy_remaining_ms=500,
            deploy_previous_ms=545,
            attack_step=(50, 45),
            movement_step=(100, 91),
            deploy_step=(50, 45),
            effective_speed=91,
            movement_delta=10,
        ),
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    entity = next(
        item
        for item in observations[0].entities
        if item.card_id == SEMANTIC_BASELINE_DECK[0]
    )
    assert entity.attack_state is not None
    assert entity.attack_state.phase == AttackPhase.WINDUP
    assert entity.attack_state.phase_started_tick == 21
    assert entity.attack_state.target_entity is not None
    assert entity.movement_runtime is not None
    assert entity.movement_runtime["phase"] == "moving"
    assert entity.movement_runtime["effective_speed"] == 91
    assert entity.movement_runtime["effect_scaled_speed"] == 91
    assert entity.movement_runtime["classic_charge_phase"] == "accumulating"
    assert entity.movement_runtime["complete"] is True
    assert entity.deployment_runtime is not None
    assert entity.deployment_runtime["phase"] == "deploying"
    assert entity.deployment_runtime["observed_step_native_ms"] == 45
    assert entity.deployment_runtime["remaining_wall_ms"] == 600
    assert entity.deployment_runtime["complete"] is True
    assert (
        entity.runtime_provenance.field_evidence["movement_runtime"]
        == SemanticEvidenceLevel.NATIVE_DERIVED
    )
    persisted = entity.to_dict()
    assert persisted["movement_runtime"]["phase"] == "moving"
    assert persisted["deployment_runtime"]["phase"] == "deploying"


def test_phase_runtime_binding_can_skip_already_consumed_events() -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    _set_phase_runtime(
        rich,
        events=[
            _phase_event(ordinary, sequence=100, tick=20),
            _phase_event(
                ordinary,
                sequence=101,
                tick=20,
                kind="attack_release",
            ),
        ],
    )

    snapshot = bind_rich_telemetry(
        ordinary,
        rich,
        phase_event_floor=101,
    )

    assert snapshot.phase_runtime is not None
    assert [item.sequence for item in snapshot.phase_runtime.events] == [101]
    assert snapshot.phase_runtime.window.oldest_retained_sequence == 100


def test_rich_binding_can_skip_consumed_event_rings() -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    first = bind_rich_telemetry(ordinary, rich)

    snapshot = bind_rich_telemetry(
        ordinary,
        rich,
        combat_event_floor=first.combat_events.next_sequence,
        visibility_event_floor=(
            first.visibility_runtime.next_sequence
            if first.visibility_runtime is not None
            else None
        ),
        remaining_event_floor=(
            first.remaining_runtime.next_sequence
            if first.remaining_runtime is not None
            else None
        ),
    )

    assert snapshot.combat_events.events == ()
    if snapshot.visibility_runtime is not None:
        assert snapshot.visibility_runtime.events == ()
    if snapshot.remaining_runtime is not None:
        assert snapshot.remaining_runtime.events == ()


def test_phase_runtime_projects_attack_sequence_decay_state(catalog) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    rich["objects"][6]["attackSequenceStage"] = 2  # type: ignore[index]
    _set_phase_runtime(
        rich,
        events=[],
        object_phase=_phase_object(
            tick=20,
            attack_sequence_stage=2,
            attack_sequence_progress=12,
            attack_sequence_decay_remaining_ms=3_500,
        ),
    )

    observations, _ = _reset(
        _environment(_RichNativeStub(ordinary, rich), catalog)
    )

    entity = next(
        item
        for item in observations[0].entities
        if item.card_id == SEMANTIC_BASELINE_DECK[0]
    )
    assert entity.attack_state is not None
    assert entity.attack_state.sequence_index == 2
    assert entity.attack_state.sequence_progress == 12
    assert entity.attack_state.sequence_progress_limit == 50
    assert entity.attack_state.sequence_decay_remaining_ms == 3_500
    assert entity.attack_state.sequence_decay_duration_ms == 7_000


def test_phase_cursor_fails_only_when_unconsumed_sequence_is_overwritten(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_phase_runtime(
        before_rich,
        events=[],
        object_phase=_phase_object(tick=20),
    )
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_phase_runtime(
        after_rich,
        events=[],
        object_phase=_phase_object(tick=21),
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)
    env._phase_next_sequence = 99

    with pytest.raises(BattleEnvError, match="overwrote unseen"):
        env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})


def test_phase_rejected_capture_fails_closed_in_battle_env(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_phase_runtime(
        before_rich,
        events=[],
        object_phase=_phase_object(tick=20),
    )
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_phase_runtime(
        after_rich,
        events=[],
        object_phase=_phase_object(tick=21),
        rejected_count=1,
        complete=False,
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    with pytest.raises(BattleEnvError, match="rejected capture facts"):
        env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})


def test_visibility_runtime_emits_one_oracle_only_exact_transition(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_visibility_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    after_rich["objects"][6].update(  # type: ignore[index,union-attr]
        {"invisibleCount": 1, "visibilityState": "invisible"}
    )
    _set_visibility_runtime(
        after_rich,
        events=[
            _visibility_event(
                after,
                sequence=300,
                tick=21,
                kind="became_invisible",
            )
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    assert all(
        event.event_type != "visibility_change"
        for event in observations[0].events
    )
    exact = [
        event
        for event in env._events
        if event.event_type == "visibility_change"
    ]
    assert len(exact) == 1
    assert exact[0].data["native_event_id"] == "visibility:9:300"
    assert exact[0].data["oracle_only"] is True
    assert exact[0].data["from"] == "visible"
    assert exact[0].data["to"] == "hidden"


def test_visibility_runtime_preserves_same_interval_transient_edges(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_visibility_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_visibility_runtime(
        after_rich,
        events=[
            _visibility_event(
                after,
                sequence=300,
                tick=21,
                kind="became_invisible",
            ),
            _visibility_event(
                after,
                sequence=301,
                tick=21,
                kind="became_visible",
            ),
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})

    exact = [
        event
        for event in env._events
        if event.event_type == "visibility_change"
    ]
    assert [event.data["native_event_id"] for event in exact] == [
        "visibility:9:300",
        "visibility:9:301",
    ]
    assert [event.data["to"] for event in exact] == ["hidden", "visible"]


def test_visibility_runtime_cursor_is_fail_closed(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_visibility_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_visibility_runtime(after_rich, events=[])
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    assert env._visibility_event_identity == (3, 9)
    assert env._visibility_next_sequence == 300

    env._visibility_next_sequence = 299
    with pytest.raises(BattleEnvError, match="overwrote unseen"):
        env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})


def test_visibility_rejected_capture_fails_closed_in_battle_env(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_visibility_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_visibility_runtime(
        after_rich,
        events=[],
        rejected_count=1,
        complete=False,
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    with pytest.raises(BattleEnvError, match="rejected capture facts"):
        env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})


def test_remaining_runtime_accepts_only_exact_initial_tower_bootstrap(
    catalog,
) -> None:
    ordinary = _ordinary()
    rich = _rich(ordinary)
    towers = ordinary["objects"][:6]  # type: ignore[index]
    events: list[dict[str, object]] = []
    for index, tower in enumerate(towers):
        target = towers[index + 3] if index < 3 else towers[index - 3]
        entity_fact = _combat_fact(tower)
        target_fact = _combat_fact(target)
        entity_fact["objectKind"] = 5
        target_fact["objectKind"] = 5
        event = _remaining_resource_event(
            ordinary,
            sequence=200 + index,
            tick=0,
        )
        event.update(
            {
                "kind": "tower_aggro_acquire",
                "hookOffset": 0xF5C894,
                "entity": entity_fact,
                "source": _null_combat_fact(validated=False),
                "target": deepcopy(target_fact),
                "targetBefore": _null_combat_fact(),
                "targetAfter": target_fact,
                "resourcePreFixed": None,
                "resourcePostFixed": None,
                "resourceActualDeltaFixed": None,
                "amountArgument": None,
                "configuredAmountArgument": None,
                "resourceOwner": None,
                "resourceCause": "none",
            }
        )
        events.append(event)
    _set_remaining_runtime(
        rich,
        events=events,
        rejected_count=6,
        complete=False,
    )
    snapshot = bind_rich_telemetry(ordinary, rich)
    assert snapshot.remaining_runtime is not None
    assert BattleEnvV1._is_initial_tower_aggro_bootstrap_rejection(
        snapshot.remaining_runtime
    )

    # Cold headless creation followed by the one-tick renderer-alignment step
    # rebuilds the world and emits the same exact six transitions once more.
    repeated_events = [
        *events,
        *[
            {**deepcopy(event), "sequence": 206 + index}
            for index, event in enumerate(events)
        ],
    ]
    _set_remaining_runtime(
        rich,
        events=repeated_events,
        rejected_count=6,
        complete=False,
    )
    repeated = bind_rich_telemetry(ordinary, rich)
    assert repeated.remaining_runtime is not None
    assert BattleEnvV1._is_initial_tower_aggro_bootstrap_rejection(
        repeated.remaining_runtime
    )

    repeated_events[6]["targetAfter"] = deepcopy(events[1]["targetAfter"])
    malformed_repeat = bind_rich_telemetry(ordinary, rich)
    assert malformed_repeat.remaining_runtime is not None
    assert not BattleEnvV1._is_initial_tower_aggro_bootstrap_rejection(
        malformed_repeat.remaining_runtime
    )
    _set_remaining_runtime(
        rich,
        events=events,
        rejected_count=6,
        complete=False,
    )

    env = _environment(_RichNativeStub(ordinary, rich), catalog)
    env._allow_initial_remaining_runtime_rejections = True
    env._prime_remaining_event_cursor(snapshot)
    assert env._remaining_rejected_count_baseline == 6

    rich["remainingRuntime"]["rejectedCount"] = 7  # type: ignore[index]
    rejected = bind_rich_telemetry(ordinary, rich)
    assert rejected.remaining_runtime is not None
    assert not BattleEnvV1._is_initial_tower_aggro_bootstrap_rejection(
        rejected.remaining_runtime
    )


def test_remaining_runtime_cursor_emits_oracle_only_exact_event(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_remaining_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_remaining_runtime(
        after_rich,
        events=[
            _remaining_resource_event(
                after, sequence=200, tick=21
            )
        ],
    )
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )
    assert all(
        event.event_type != "runtime_resource_delta"
        for event in observations[0].events
    )
    event = next(
        item
        for item in env._events
        if item.event_type == "runtime_resource_delta"
    )
    assert event.data["resource_actual_delta_fixed"] == 10_000
    assert event.data["native_event_id"] == "remaining:9:200"
    assert event.data["oracle_only"] is True


def test_action_spawn_to_location_accepts_exact_character_source(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_remaining_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    source = _combat_fact(after["objects"][6])  # type: ignore[index]
    target = _combat_fact(after["objects"][7])  # type: ignore[index]
    source["objectKind"] = 5
    target["objectKind"] = 5
    event = _remaining_resource_event(after, sequence=200, tick=21)
    event.update(
        {
            "kind": "area_action_spawn",
            "hookOffset": 0xE7B864,
            "callerOffset": None,
            "entity": deepcopy(target),
            "source": source,
            "target": target,
            "objectKind": 5,
            "runtimeVtableOffset": 0x189C4E8,
            "dataAfterGlobalId": 34_000_032,
            "configuredDataGlobalId": 4_162_986_129,
            "resourcePreFixed": None,
            "resourcePostFixed": None,
            "resourceActualDeltaFixed": None,
            "amountArgument": None,
            "configuredAmountArgument": None,
            "resourceOwner": None,
            "resourceCause": "none",
            "committed": True,
        }
    )
    _set_remaining_runtime(after_rich, events=[event])
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)

    env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})

    oracle_event = next(
        item
        for item in env._events
        if item.event_type == "runtime_area_action_spawn"
    )
    assert oracle_event.data["runtime_vtable_offset"] == 0x189C4E8
    assert oracle_event.data["configured_data_global_id"] == 4_162_986_129


def test_remaining_runtime_cursor_rejects_unseen_rollover(catalog) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _set_remaining_runtime(before_rich, events=[])
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = _rich(after)
    _set_remaining_runtime(after_rich, events=[])
    env = _environment(
        _TransitionNativeStub(before, before_rich, after, after_rich),
        catalog,
    )
    _reset(env)
    env._remaining_next_sequence = 199

    with pytest.raises(BattleEnvError, match="overwrote unseen"):
        env.step({0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)})


def test_evolution_play_requires_executed_evo_receipt_and_ready_reset(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _add_player_runtime(before, before_rich)
    evolution_card_id = 26_000_001
    evolution_form_card_id = 26_100_001
    evolution_cost = int(
        catalog.by_id[evolution_card_id].elixir_cost or 0
    )
    before["players"][0]["hand"][1].update(  # type: ignore[index]
        {
            "cardId": evolution_card_id,
            "commandCardId": evolution_form_card_id,
            "deckSlot": 1,
            "cardParameter": (
                (evolution_cost << 28) | (2 << 22) | 1
            ),
            "cost": evolution_cost,
            "formCode": 1,
        }
    )
    before_evolution = before_rich["players"][0]["evolutionRuntime"][1]  # type: ignore[index]
    before_evolution.update(
        {
            "progress": 2,
            "cycleRemaining": 0,
            "ready": True,
        }
    )
    after = deepcopy(before)
    after["tick"] = 21
    replacement_card_id = SEMANTIC_BASELINE_DECK[4]
    replacement_cost = int(
        catalog.by_id[replacement_card_id].elixir_cost or 0
    )
    after["players"][0]["hand"][1].update(  # type: ignore[index]
        {
            "cardId": replacement_card_id,
            "commandCardId": replacement_card_id,
            "deckSlot": 4,
            "cardParameter": (
                (replacement_cost << 28) | (5 << 22)
            ),
            "cost": replacement_cost,
            "formCode": 0,
        }
    )
    after_rich = deepcopy(before_rich)
    after_rich["tick"] = 21
    after_rich["combatEvents"]["observationTick"] = 21  # type: ignore[index]
    after_evolution = after_rich["players"][0]["evolutionRuntime"][1]  # type: ignore[index]
    after_evolution.update(
        {
            "progress": 0,
            "cycleRemaining": 2,
            "ready": False,
        }
    )
    native = _HandTransitionNativeStub(
        before,
        before_rich,
        after,
        after_rich,
    )
    env = _environment(native, catalog)
    _reset(
        env,
        match_config=_match_from_raw(
            before,
            deck0_form_availability=(0, 1, 0, 0, 0, 0, 0, 0),
        ),
    )
    play = ActionV1.play(
        owner=0,
        hand_slot=1,
        grid=(8, 10),
        action_id="evo-play",
    )

    observations, _, _, _, _ = env.step(
        {0: play, 1: ActionV1.wait(1, 1)}
    )

    owner_events = observations[0].events
    queued = next(
        event
        for event in owner_events
        if event.event_type == "card_form_command_queued"
    )
    assert queued.data["form_code"] == 1
    assert queued.data["form_name"] == "EvoForm"
    played = next(
        event for event in owner_events if event.event_type == "evolution_played"
    )
    assert played.data["played_form"] == "EvoForm"
    assert played.data["deck_slot"] == 1
    assert all(
        event.event_type not in {
            "card_form_command_queued",
            "evolution_played",
        }
        for event in observations[1].events
    )


def test_evolution_ready_reset_without_evo_receipt_stays_unknown(
    catalog,
) -> None:
    before = _ordinary()
    before_rich = _rich(before)
    _add_player_runtime(before, before_rich)
    before_evolution = before_rich["players"][0]["evolutionRuntime"][1]  # type: ignore[index]
    before_evolution.update(
        {"progress": 2, "cycleRemaining": 0, "ready": True}
    )
    after = deepcopy(before)
    after["tick"] = 21
    after_rich = deepcopy(before_rich)
    after_rich["tick"] = 21
    after_rich["combatEvents"]["observationTick"] = 21  # type: ignore[index]
    after_evolution = after_rich["players"][0]["evolutionRuntime"][1]  # type: ignore[index]
    after_evolution.update(
        {"progress": 0, "cycleRemaining": 2, "ready": False}
    )
    native = _TransitionNativeStub(before, before_rich, after, after_rich)
    env = _environment(native, catalog)
    _reset(
        env,
        match_config=_match_from_raw(
            before,
            deck0_form_availability=(0, 1, 0, 0, 0, 0, 0, 0),
        ),
    )

    observations, _, _, _, _ = env.step(
        {0: ActionV1.wait(0, 1), 1: ActionV1.wait(1, 1)}
    )

    assert all(
        event.event_type != "evolution_played"
        for event in observations[0].events
    )
    progress = next(
        event
        for event in observations[0].events
        if event.event_type == "evolution_progress_change"
    )
    assert progress.data["played_form"] == "unknown"
