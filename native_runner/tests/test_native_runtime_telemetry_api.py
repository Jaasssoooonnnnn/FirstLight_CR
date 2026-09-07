from __future__ import annotations

from copy import deepcopy
import hashlib
import io
from typing import Any

import pytest

from native_runner import cr_native_env as native_env_module
from native_runner.cr_native_env import (
    NativeClashEnv,
    RunnerError,
)


def _record(status: str) -> dict[str, object]:
    return {
        "status": status,
        "source": "test-source",
        "validation": "test-validation",
        "confidence": "high" if status != "unavailable" else "none",
        "failClosed": True,
    }


def _empty_combat_events(
    *,
    generation: int = 1,
    state_epoch: int = 1,
    tick: int = 530,
) -> dict[str, object]:
    return {
        "ok": True,
        "schema": "native-combat-events.v1",
        "generation": generation,
        "stateEpoch": state_epoch,
        "observationTick": tick,
        "capacity": 1024,
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": _record("derived"),
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
    raw_object: dict[str, Any],
    *,
    native_object_id: int | None = None,
) -> dict[str, object]:
    effective_native_object_id = (
        int(raw_object["nativeObjectId"])
        if native_object_id is None
        else native_object_id
    )
    return {
        "validated": True,
        "present": True,
        "nativeObjectId": effective_native_object_id,
        "entityKey": raw_object["entityKey"],
        "owner": raw_object["owner"],
        "objectIndex": raw_object["objectIndex"],
        "secondaryIndex": raw_object["secondaryIndex"],
        "cardId": raw_object["cardId"],
        "objectKind": 1,
        "position": [raw_object["x"], raw_object["y"]],
        "visibilityValidated": True,
        "invisibleCount": raw_object["invisibleCount"],
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


def _add_damage_combat_event(response: dict[str, Any]) -> None:
    target = _combat_fact(response["objects"][0])
    absent = _null_combat_fact()
    event = {
        "sequence": 1,
        "tick": response["tick"],
        "generation": response["generation"],
        "stateEpoch": response["stateEpoch"],
        "kind": "damage",
        "hookOffset": 0xF642C4,
        "callerOffset": None,
        "causeSequence": None,
        "pool": "hitpoints",
        "terminalReason": "none",
        "lethal": False,
        "deploymentContext": None,
        "target": target,
        "immediateSource": target,
        "source": target,
        "projectile": absent,
        "related": absent,
        "routeSourceBefore": absent,
        "routeTargetBefore": absent,
        "routeSourceAfter": absent,
        "routeTargetAfter": absent,
        "requestedAmount": 100,
        "actualAmount": 100,
        "preHp": 3_000,
        "postHp": 2_900,
        "preBuiltInShield": 150,
        "postBuiltInShield": 150,
        "preBuffShield": None,
        "postBuffShield": None,
        "destinationBefore": None,
        "destinationAfter": None,
    }
    response["combatEvents"] = {
        **_empty_combat_events(
            generation=response["generation"],
            state_epoch=response["stateEpoch"],
            tick=response["tick"],
        ),
        "nextSequence": 2,
        "events": [event],
    }


def _add_generic_action_heal_combat_event(response: dict[str, Any]) -> None:
    _add_damage_combat_event(response)
    event = response["combatEvents"]["events"][0]
    event.update(
        {
            "kind": "heal",
            "hookOffset": 0xF64D14,
            "callerOffset": 0xF12D54,
            "requestedAmount": 218,
            "actualAmount": 218,
            "preHp": 255,
            "postHp": 473,
            "preBuiltInShield": 0,
            "postBuiltInShield": 0,
        }
    )


def _add_evolved_spawn_combat_event(response: dict[str, Any]) -> None:
    target = _combat_fact(response["objects"][0])
    absent = _null_combat_fact()
    deck_slot = 2
    cost = 4
    card_parameter = (cost << 28) | ((deck_slot + 1) << 22) | 1
    event = {
        "sequence": 1,
        "tick": response["tick"],
        "generation": response["generation"],
        "stateEpoch": response["stateEpoch"],
        "kind": "spawn",
        "hookOffset": 0xF25B8C,
        "callerOffset": None,
        "causeSequence": None,
        "pool": "none",
        "terminalReason": "none",
        "lethal": False,
        "deploymentContext": {
            "deploymentSequence": 7,
            "owner": target["owner"],
            "playedCardGlobalId": 26_000_001,
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
    response["combatEvents"] = {
        **_empty_combat_events(
            generation=response["generation"],
            state_epoch=response["stateEpoch"],
            tick=response["tick"],
        ),
        "nextSequence": 2,
        "events": [event],
    }


def _add_card_play_combat_event(response: dict[str, Any]) -> None:
    _add_evolved_spawn_combat_event(response)
    event = response["combatEvents"]["events"][0]
    absent = _null_combat_fact()
    deployment = event["deploymentContext"]
    deployment.update(
        {
            "effectiveCardGlobalId": deployment["playedCardGlobalId"],
            "cardParameter": (
                int(deployment["cost"]) << 28
                | (int(deployment["deckSlot"]) + 1) << 22
            ),
            "formCode": 0,
            "formName": "BasicForm",
        }
    )
    event.update(
        {
            "kind": "card_play",
            "hookOffset": 0xF38A68,
            "target": absent,
            "immediateSource": absent,
            "source": absent,
        }
    )


def _rich_response() -> dict[str, Any]:
    authoritative = {
        "entityCore",
        "targetCoordinates",
        "targetEntity",
        "hitpoints",
        "componentInventory",
        "shield",
        "attackSequenceStage",
        "buffs",
        "invisibility",
    }
    capability_names = authoritative | {
        "shield",
        "buffs",
        "debuffs",
        "slow",
        "rage",
        "stun",
        "freeze",
        "attackPhase",
        "deployPhase",
        "chargeStage",
        "sourceEntity",
        "projectile",
        "impact",
        "visibility",
        "invisibility",
        "abilityRuntime",
        "evolutionRuntime",
    }
    offsets = {
        0: (25_828_816, 16_107_640),
        1: (25_829_176, 16_159_644),
        2: (25_829_056, 16_134_016),
        3: (25_828_696, 16_095_916),
    }
    components = [
        {
            "slot": slot,
            "present": True,
            "ownerBacklinkValid": True,
            "vtableLibgOffset": vtable,
            "typeGetterLibgOffset": getter,
            "engineType": slot,
            "typeValid": True,
            "slotMatchesType": True,
        }
        for slot, (vtable, getter) in offsets.items()
    ]
    derived = {"projectile", "abilityRuntime", "evolutionRuntime"}
    return {
        "ok": True,
        "schema": "native-rich-telemetry.v3",
        "statusEnum": [
            "authoritative",
            "derived",
            "pending",
            "unavailable",
        ],
        "generation": 1,
        "stateEpoch": 1,
        "tick": 530,
        "provenance": {
            "nativeObjectId": _record("authoritative"),
            "dataGlobalId": _record("authoritative"),
            "entityKey": _record("derived"),
            "targetEntityKey": _record("authoritative"),
            "targetEntityValidated": _record("derived"),
            "shield.current": _record("authoritative"),
            "shield.max": _record("authoritative"),
            "attackSequenceStage": _record("authoritative"),
            "activeEffects": _record("authoritative"),
            "activeEffects.buffGlobalId": _record("authoritative"),
            "activeEffects.name": _record("authoritative"),
            "activeEffects.remainingMs": _record("authoritative"),
            "activeEffects.sourceEntityKey": _record("authoritative"),
            "activeEffects.sourceEntityValidated": _record("derived"),
            "invisibleCount": _record("authoritative"),
            "visibilityState": _record("derived"),
            "components.slots.vtableLibgOffset": _record("authoritative"),
            "components.slots.typeGetterLibgOffset": _record("authoritative"),
            "components.slots.engineType": _record("authoritative"),
            "projectile": _record("derived"),
            "projectile.projectileDataGlobalId": _record("derived"),
            "projectile.sourceEntityKey": _record("derived"),
            "projectile.targetEntityKey": _record("derived"),
            "projectile.homingTargetEntityKey": _record("derived"),
            "projectile.destination": _record("derived"),
            "projectile.terminal": _record("derived"),
            "projectile.nativePhase": _record("derived"),
            "projectile.dragStage": _record("authoritative"),
            "entityResourceRuntime": _record("authoritative"),
            "periodicAttackModifierRuntime": _record("authoritative"),
            "captureRuntime": _record("authoritative"),
            "thresholdRelocationRuntime": _record("authoritative"),
            "players.ownerRoot": _record("derived"),
            "players.abilityRuntime": _record("derived"),
            "players.abilityRuntime.buttonState": _record("derived"),
            "players.abilityRuntime.cooldown": _record("derived"),
            "players.abilityRuntime.charges": _record("derived"),
            "players.evolutionRuntime": _record("derived"),
            "players.evolutionRuntime.cycle": _record("derived"),
            "players.evolutionRuntime.playedForm": _record("unavailable"),
        },
        "capabilities": {
            name: _record(
                "authoritative"
                if name in authoritative
                else ("derived" if name in derived else "unavailable")
            )
            for name in capability_names
        },
        "count": 1,
        "objects": [
            {
                "slot": 0,
                "nativeObjectId": 500,
                "entityKey": [0, -2, 500],
                "owner": 0,
                "cardId": 26_000_003,
                "dataGlobalId": 34_000_003,
                "x": 9_000,
                "y": 12_000,
                "targetX": 14_500,
                "targetY": 25_500,
                "targetEntityKey": [0, -2, 500],
                "targetEntityValidated": True,
                "objectIndex": 14,
                "secondaryIndex": 67,
                "hp": 3_000,
                "maxHp": 3_000,
                "shield": {
                    "current": 150,
                    "max": 150,
                    "status": "authoritative",
                },
                "attackSequenceStage": 2,
                "activeEffects": [
                    {
                        "buffGlobalId": 9_000_003,
                        "name": "IceWizardSlowDown",
                        "remainingMs": 2_500,
                        "sourceEntityKey": [0, -2, 500],
                        "sourceEntityValidated": True,
                    }
                ],
                "invisibleCount": 0,
                "visibilityState": "visible",
                "projectile": None,
                "entityResourceRuntime": None,
                "periodicAttackModifierRuntime": None,
                "captureRuntime": None,
                "thresholdRelocationRuntime": None,
                "components": {
                    "valid": True,
                    "count": 4,
                    "capacity": 4,
                    "slots": components,
                },
            }
        ],
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
        "combatEvents": _empty_combat_events(),
        "returned": 1,
        "truncated": False,
    }


def _add_phase_runtime(response: dict[str, Any]) -> None:
    tick = int(response["tick"])
    for name in (
        "slow",
        "rage",
        "stun",
        "freeze",
        "attackPhase",
        "deployPhase",
        "chargeStage",
    ):
        response["capabilities"][name] = _record("derived")
    response["objects"][0]["phaseRuntime"] = {
        "schema": "native-phase-object.v1",
        "attackValidated": True,
        "movementValidated": True,
        "buffsValidated": True,
        "attackSequenceStage": 2,
        "attackTimelineMs": 4_100,
        "loadRemainingMs": 0,
        "deployRemainingMs": 0,
        "deployPreviousMs": 0,
        "configuredDeployTimeMs": 1_000,
        "hitSpeedMs": 400,
        "attackDashTimeMs": 0,
        "baseMovementSpeed": 60,
        "chargeSpeedMultiplier": 200,
        "classicChargeProgress": 10_000,
        "speedPositivePercent": 130,
        "speedNegativeMagnitude": 30,
        "hitSpeedPositivePercent": 130,
        "hitSpeedNegativeMagnitude": 30,
        "attackStepInput": 50,
        "attackStepOutput": 45,
        "attackStepTick": tick,
        "movementStepInput": 60,
        "movementStepOutput": 54,
        "movementStepTick": tick,
        "deployStepInput": None,
        "deployStepOutput": None,
        "deployStepTick": -1,
        "effectiveMovementSpeed": 108,
        "effectiveMovementSpeedTick": tick,
        "movementDelta": 20,
        "movementDeltaTick": tick,
    }
    response["phaseRuntime"] = {
        "ok": True,
        "schema": "native-phase-runtime.v1",
        "generation": response["generation"],
        "stateEpoch": response["stateEpoch"],
        "observationTick": response["tick"],
        "capacity": 4_096,
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": _record("derived"),
        "epochFirstSequence": 1,
        "oldestRetainedSequence": 1,
        "nextSequence": 1,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [],
    }


def _add_special_movement_runtime(response: dict[str, Any]) -> None:
    source_before = _combat_fact(response["objects"][0])
    source_after = deepcopy(source_before)
    source_after["position"] = [10_000, 13_000]
    target = deepcopy(source_before)
    target.update(
        {
            "nativeObjectId": 600,
            "entityKey": [1, -2, 600],
            "owner": 1,
            "objectIndex": 15,
            "secondaryIndex": 0,
            "cardId": 26_000_003,
            "position": [14_500, 25_500],
        }
    )
    response["specialMovementRuntime"] = {
        "ok": True,
        "schema": "native-special-movement-runtime.v1",
        "generation": response["generation"],
        "stateEpoch": response["stateEpoch"],
        "observationTick": response["tick"],
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
        "epochFirstSequence": 1,
        "oldestRetainedSequence": 1,
        "nextSequence": 2,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            {
                "sequence": 1,
                "tick": response["tick"],
                "kind": "ordinary_dash_execute",
                "mode": "dash",
                "stage": "execute",
                "hookOffset": 0xF609DC,
                "callerOffset": 0xF61BF0,
                "sourceBefore": source_before,
                "sourceAfter": source_after,
                "target": target,
                "requestedX": 14_100,
                "requestedY": 25_100,
                "targetRadius": 500,
                "operationFlag": 1,
                "cooldownBeforeMs": 1_000,
                "cooldownAfterMs": 1_000,
                "positionChanged": True,
                "completeContext": True,
            }
        ],
    }


def _add_action_movement_runtime(response: dict[str, Any]) -> None:
    response["objects"][0]["cardId"] = 26_000_074
    source = _combat_fact(response["objects"][0])
    target = deepcopy(source)
    target.update(
        {
            "nativeObjectId": 601,
            "entityKey": [1, -2, 601],
            "owner": 1,
            "objectIndex": 16,
            "secondaryIndex": 0,
            "cardId": 26_000_003,
            "position": [14_500, 25_500],
        }
    )
    response["actionMovementRuntime"] = {
        "ok": True,
        "schema": "native-action-movement-runtime.v1",
        "generation": response["generation"],
        "stateEpoch": response["stateEpoch"],
        "observationTick": response["tick"],
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
        "epochFirstSequence": 10,
        "oldestRetainedSequence": 10,
        "nextSequence": 11,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            {
                "sequence": 10,
                "tick": response["tick"],
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


def _add_character_state_runtime(response: dict[str, Any]) -> None:
    response["objects"][0]["cardId"] = 26_000_046
    source = _combat_fact(response["objects"][0])
    source["objectKind"] = 5
    response["characterStateRuntime"] = {
        "ok": True,
        "schema": "native-character-state-runtime.v1",
        "generation": response["generation"],
        "stateEpoch": response["stateEpoch"],
        "observationTick": response["tick"],
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
        "epochFirstSequence": 20,
        "oldestRetainedSequence": 20,
        "nextSequence": 21,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            {
                "sequence": 20,
                "tick": response["tick"],
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


def _add_visibility_runtime(
    response: dict[str, Any],
    *,
    kind: str = "became_invisible",
) -> None:
    became_invisible = kind == "became_invisible"
    before, after = ((0, 1) if became_invisible else (1, 0))
    response["objects"][0]["invisibleCount"] = after
    response["objects"][0]["visibilityState"] = (
        "invisible" if after else "visible"
    )
    subject = _combat_fact(response["objects"][0])
    response["visibilityRuntime"] = {
        "ok": True,
        "schema": "native-visibility-runtime.v1",
        "generation": response["generation"],
        "stateEpoch": response["stateEpoch"],
        "observationTick": response["tick"],
        "capacity": 4_096,
        "hookSetAttested": True,
        "transitionHookSetInstalled": True,
        "contextualGateHookSetInstalled": False,
        "capability": _record("derived"),
        "contextualGateCapability": _record("unavailable"),
        "ownerRelativeVisibility": {
            name: {
                "status": "unavailable",
                "reason": "no-exact-owner-conditioned-native-producer",
            }
            for name in ("publicByOwner", "targetableByOwner")
        },
        "epochFirstSequence": 1,
        "oldestRetainedSequence": 1,
        "nextSequence": 2,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            {
                "sequence": 1,
                "tick": response["tick"],
                "kind": kind,
                "hookOffset": 0xF5A2D4 if became_invisible else 0xF5A578,
                "callerOffset": 0xF59EAC if became_invisible else 0xF59308,
                "subject": subject,
                "buffGlobalId": 9_000_001,
                "invisibleCountBefore": before,
                "invisibleCountAfter": after,
                "scope": "native_invisibility_phase",
                "completeContext": True,
            }
        ],
    }


def _add_remaining_runtime(response: dict[str, Any]) -> None:
    source = _combat_fact(response["objects"][0])
    absent = _null_combat_fact()
    event = {
        "sequence": 1,
        "tick": response["tick"],
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
    response["remainingRuntime"] = {
        "ok": True,
        "schema": "native-remaining-runtime.v1",
        "generation": response["generation"],
        "stateEpoch": response["stateEpoch"],
        "observationTick": response["tick"],
        "capacity": 4_096,
        "hookSetAttested": True,
        "hookSetInstalled": True,
        "capability": _record("derived"),
        "ownerRelativeVisibility": {
            name: {
                "status": "unavailable",
                "reason": "no-exact-owner-conditioned-native-producer",
            }
            for name in ("publicByOwner", "targetableByOwner")
        },
        "epochFirstSequence": 1,
        "oldestRetainedSequence": 1,
        "nextSequence": 2,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [event],
    }


def _atomic_response() -> dict[str, Any]:
    rich = _rich_response()
    ordinary = {
        "ok": True,
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "tick": rich["tick"],
        "count": 1,
        "objects": [{"slot": 0}],
        "returned": 1,
        "truncated": False,
    }
    return {
        "ok": True,
        "schema": "native-observation-capture.v1",
        "atomic": True,
        "maxObjects": 256,
        "generation": rich["generation"],
        "stateEpoch": rich["stateEpoch"],
        "tick": rich["tick"],
        "ordinary": ordinary,
        "rich": rich,
    }


def _runtime_v2_response() -> dict[str, Any]:
    response = _rich_response()
    entity_key = [0, -2, 500]
    response["objects"][0]["projectile"] = {
        "projectileDataGlobalId": 10_000_123,
        "sourceEntityKey": entity_key,
        "sourceEntityValidated": True,
        "targetEntityKey": None,
        "targetEntityValidated": True,
        "homingTargetEntityKey": None,
        "homingTargetEntityValidated": True,
        "destinationX": 14_500,
        "destinationY": 25_500,
        "terminal": False,
        "nativePhase": "in_flight",
        "dragStage": None,
    }
    response["players"][0] = {
        "owner": 0,
        "ownerRootValidated": True,
        "ownerEntityKey": entity_key,
        "abilityRuntime": [
            {
                "controllerSlot": 1,
                "actionDataGlobalId": 88_000_001,
                "actionDataName": "ArcherQueenRapid",
                "selectedCharacterDataGlobalId": 26_000_072,
                "remainingCooldownMs": 0,
                "configuredCooldownMs": 17_000,
                "remainingChargesRaw": -1,
                "maxCharges": 0,
                "buttonState": 2,
                "buttonStateLabel": "Ready",
                "available": True,
                "championEntityKeys": [entity_key],
            }
        ],
        "evolutionRuntime": [
            {
                "deckSlot": 0,
                "cardId": 26_000_003,
                "baseSpellGlobalId": 26_000_003,
                "evolvable": True,
                "evolutionFormGlobalId": 26_100_003,
                "progress": 1,
                "cycleRequired": 2,
                "cycleRemaining": 1,
                "ready": False,
            }
        ],
    }
    return response


def _inspection_response(payload: bytes = bytes(range(16))) -> dict[str, Any]:
    return {
        "ok": True,
        "schema": "native-component-inspection.v1",
        "diagnosticOnly": True,
        "semanticCapability": False,
        "generation": 1,
        "stateEpoch": 1,
        "tick": 530,
        "objectSlot": 6,
        "nativeObjectId": 500,
        "entityKey": [0, -2, 500],
        "owner": 0,
        "cardId": 26_000_003,
        "objectIndex": 14,
        "secondaryIndex": 67,
        "componentSlot": 0,
        "vtableLibgOffset": 25_828_816,
        "typeGetterLibgOffset": 16_107_640,
        "engineType": 0,
        "typeValid": True,
        "slotMatchesType": True,
        "ownerBacklinkValid": True,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hex": payload.hex(),
    }


def _component3_entry_inspection_response(
    payload: bytes = bytes(range(0x70)),
) -> dict[str, Any]:
    return {
        "ok": True,
        "schema": "native-component3-entry-inspection.v1",
        "diagnosticOnly": True,
        "semanticCapability": False,
        "layoutSemantics": "unclassified",
        "generation": 1,
        "stateEpoch": 1,
        "tick": 530,
        "objectSlot": 6,
        "nativeObjectId": 500,
        "entityKey": [0, -2, 500],
        "owner": 0,
        "cardId": 26_000_003,
        "objectIndex": 14,
        "secondaryIndex": 67,
        "componentSlot": 3,
        "engineType": 3,
        "ownerBacklinkValid": True,
        "vtableLibgOffset": 25_828_696,
        "typeGetterLibgOffset": 16_095_916,
        "entryIndex": 0,
        "entrySize": 0x70,
        "vectorCount": 1,
        "vectorCapacity": 5,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hex": payload.hex(),
    }


class StubNativeClashEnv(NativeClashEnv):
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.responses = responses
        self.requests: list[str] = []

    def _request(self, command: str) -> dict[str, Any]:
        self.requests.append(command)
        return deepcopy(self.responses[command])


class _OversizedResponseConnection:
    def __init__(self) -> None:
        self.remaining = native_env_module.MAX_RUNNER_RESPONSE_BYTES + 1

    def __enter__(self) -> "_OversizedResponseConnection":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def sendall(self, _payload: bytes) -> None:
        return None

    def recv(self, maximum: int) -> bytes:
        if self.remaining == 0:
            return b""
        size = min(maximum, self.remaining)
        self.remaining -= size
        return b"x" * size


class _FixedResponseConnection:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.sent = b""

    def __enter__(self) -> "_FixedResponseConnection":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def sendall(self, payload: bytes) -> None:
        self.sent += payload

    def recv(self, _maximum: int) -> bytes:
        payload, self.payload = self.payload, b""
        return payload


class _PersistentResponseConnection:
    def __init__(self, payload: bytes) -> None:
        self.reader = io.BytesIO(payload)
        self.sent = b""
        self.closed = False
        self.socket_options: list[tuple[int, int, int]] = []

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.socket_options.append((level, option, value))

    def makefile(self, _mode: str) -> io.BytesIO:
        return self.reader

    def sendall(self, payload: bytes) -> None:
        self.sent += payload

    def close(self) -> None:
        self.closed = True


def test_observe_rich_validates_authoritative_component_and_target_schema() -> None:
    env = StubNativeClashEnv({"observe-rich": _rich_response()})

    result = env.observe_rich()

    assert result["capabilities"]["componentInventory"]["status"] == "authoritative"
    assert result["capabilities"]["shield"]["status"] == "authoritative"
    assert result["capabilities"]["buffs"]["status"] == "authoritative"
    assert result["capabilities"]["invisibility"]["status"] == "authoritative"
    assert result["objects"][0]["shield"] == {
        "current": 150,
        "max": 150,
        "status": "authoritative",
    }
    assert result["objects"][0]["targetEntityKey"] == [0, -2, 500]
    assert result["objects"][0]["attackSequenceStage"] == 2
    assert result["objects"][0]["activeEffects"][0]["name"] == "IceWizardSlowDown"


def test_observe_rich_validates_entity_owned_extra_spawn_resource() -> None:
    response = _rich_response()
    response["objects"][0]["entityResourceRuntime"] = {
        "kind": "extra_spawn_accumulator",
        "currentRaw": 3,
        "capacityRaw": 10,
        "baseRaw": 6,
        "limitRaw": 16,
        "normalized": 0.3,
        "status": "authoritative",
    }

    env = StubNativeClashEnv({"observe-rich": response})
    result = env.observe_rich()

    assert result["objects"][0]["entityResourceRuntime"]["currentRaw"] == 3
    assert env.requests == ["observe-rich"]


def test_observe_rich_validates_periodic_attack_modifier_linger() -> None:
    response = _rich_response()
    response["objects"][0]["periodicAttackModifierRuntime"] = {
        "schema": "native-periodic-attack-modifier-runtime.v1",
        "actionDataGlobalId": 3_398_518_713,
        "phase": "source_death_linger",
        "periodAttacks": 3,
        "completedAttacks": 2,
        "addedDamageRaw": 86,
        "lingerDurationMs": 5_000,
        "lingerRemainingMs": 5_000,
        "sourceNativeObjectId": 5_999_999,
        "sourceEntityKey": None,
        "sourceResolved": False,
        "status": "authoritative",
    }

    env = StubNativeClashEnv({"observe-rich": response})
    result = env.observe_rich()

    modifier = result["objects"][0]["periodicAttackModifierRuntime"]
    assert modifier["completedAttacks"] == 2
    assert modifier["lingerRemainingMs"] == 5_000
    assert env.requests == ["observe-rich"]


def test_observe_rich_validates_exact_capture_runtime() -> None:
    response = _rich_response()
    key = response["objects"][0]["entityKey"]
    response["objects"][0]["captureRuntime"] = {
        "schema": "native-capture-runtime.v1",
        "actionDataGlobalId": 1_941_152_550,
        "phase": "active",
        "dragDelayMs": 100,
        "grabPauseMs": 500,
        "captureDragTimeMs": 300,
        "configuredCooldownMs": 300,
        "cooldownRemainingMs": 0,
        "hitFrequencyMs": 1_000,
        # The native update can add one speed-scaled step after its >= check.
        "hitAccumulatorMs": 1_050,
        "completionResultCurrentUpdate": True,
        "firstCaptureHandled": True,
        "targets": [
            {
                "targetNativeObjectId": response["objects"][0]["nativeObjectId"],
                "targetEntityKey": key,
                "targetResolved": True,
                "phase": "contained",
                "elapsedMs": 950,
                "phaseBudgetRemainingMs": None,
            }
        ],
        "status": "authoritative",
    }

    env = StubNativeClashEnv({"observe-rich": response})
    result = env.observe_rich()

    capture = result["objects"][0]["captureRuntime"]
    assert capture["targets"][0]["phase"] == "contained"
    assert capture["hitAccumulatorMs"] == 1_050
    assert env.requests == ["observe-rich"]


def test_observe_rich_validates_threshold_relocation_runtime() -> None:
    response = _rich_response()
    response["objects"][0]["thresholdRelocationRuntime"] = {
        "schema": "native-threshold-relocation-runtime.v1",
        "actionDataGlobalId": 1_333_491_178,
        "phase": "relocating",
        "stage": 2,
        "relocationIndex": 0,
        "hideDurationMs": 1_000,
        "remainingMs": 1_000,
        "burrowed": True,
        "thresholdsPercent": [66, 33],
        "status": "authoritative",
    }

    env = StubNativeClashEnv({"observe-rich": response})
    result = env.observe_rich()

    relocation = result["objects"][0]["thresholdRelocationRuntime"]
    assert relocation["stage"] == 2
    assert relocation["thresholdsPercent"] == [66, 33]
    assert env.requests == ["observe-rich"]


def test_observe_rich_accepts_tagged_native_id_for_negative_indices_and_refs() -> None:
    response = _rich_response()
    item = response["objects"][0]
    item["nativeObjectId"] = 5_000_005
    item["objectIndex"] = -1
    item["secondaryIndex"] = -1
    item["entityKey"] = [0, -2, 5_000_005]
    item["targetEntityKey"] = [0, -2, 5_000_005]
    item["activeEffects"][0]["sourceEntityKey"] = [0, -2, 5_000_005]
    _add_damage_combat_event(response)

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    assert result["objects"][0]["entityKey"] == [0, -2, 5_000_005]
    assert result["combatEvents"]["events"][0]["target"]["objectIndex"] == -1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("entityKey", [0, -1, -1], "does not match source fields"),
        ("nativeObjectId", 0, r"invalid objects\[0\]\.nativeObjectId"),
    ),
)
def test_observe_rich_rejects_invalid_negative_index_identity(
    field: str,
    value: object,
    match: str,
) -> None:
    response = _rich_response()
    item = response["objects"][0]
    item["objectIndex"] = -1
    item["secondaryIndex"] = -1
    item["entityKey"] = [0, -2, 500]
    item[field] = value

    with pytest.raises(RunnerError, match=match):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_rejects_duplicate_native_object_id() -> None:
    response = _rich_response()
    second = deepcopy(response["objects"][0])
    second["slot"] = 1
    second["objectIndex"] = 15
    second["entityKey"] = [0, -2, 500]
    response["objects"].append(second)
    response["count"] = 2
    response["returned"] = 2

    with pytest.raises(RunnerError, match="duplicates a native object identity"):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_accepts_strict_native_phase_runtime() -> None:
    response = _rich_response()
    _add_phase_runtime(response)

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    assert result["phaseRuntime"]["capacity"] == 4_096
    assert result["phaseRuntime"]["complete"] is True
    assert result["objects"][0]["phaseRuntime"]["classicChargeProgress"] == 10_000
    assert result["capabilities"]["attackPhase"]["status"] == "derived"


def test_observe_rich_accepts_strict_special_movement_runtime_ring() -> None:
    response = _rich_response()
    _add_special_movement_runtime(response)

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    event = result["specialMovementRuntime"]["events"][0]
    assert event["kind"] == "ordinary_dash_execute"
    assert event["sourceBefore"]["entityKey"] == event["sourceAfter"]["entityKey"]
    assert event["positionChanged"] is True


def test_observe_rich_rejects_incoherent_special_movement_runtime() -> None:
    response = _rich_response()
    _add_special_movement_runtime(response)
    response["specialMovementRuntime"]["events"][0]["requestedY"] = "invalid"

    with pytest.raises(RunnerError, match="requestedY must be an integer"):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_accepts_strict_action_movement_runtime_ring() -> None:
    response = _rich_response()
    _add_action_movement_runtime(response)

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    event = result["actionMovementRuntime"]["events"][0]
    assert event["kind"] == "golden_knight_chain_hop_launch"
    assert event["stage"] == "hop_launch"
    assert event["chainIndex"] == 0


def test_observe_rich_rejects_incoherent_action_movement_runtime() -> None:
    response = _rich_response()
    _add_action_movement_runtime(response)
    response["actionMovementRuntime"]["events"][0][
        "actionClassVtableOffset"
    ] = 0x1895120

    with pytest.raises(
        RunnerError, match="actionClassVtableOffset is invalid"
    ):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_accepts_strict_character_state_runtime_ring() -> None:
    response = _rich_response()
    _add_character_state_runtime(response)

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    event = result["characterStateRuntime"]["events"][0]
    assert event["kind"] == "native_character_state_transition"
    assert event["previousState"] == 1
    assert event["committedState"] == 3


def test_observe_rich_rejects_incoherent_character_state_runtime() -> None:
    response = _rich_response()
    _add_character_state_runtime(response)
    response["characterStateRuntime"]["events"][0]["committedState"] = 2

    with pytest.raises(RunnerError, match="requested state was not committed"):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_accepts_strict_visibility_runtime_ring() -> None:
    response = _rich_response()
    _add_visibility_runtime(response)

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    envelope = result["visibilityRuntime"]
    assert envelope["complete"] is True
    assert envelope["events"][0]["kind"] == "became_invisible"
    assert envelope["contextualGateHookSetInstalled"] is False
    assert (
        envelope["ownerRelativeVisibility"]["targetableByOwner"]["status"]
        == "unavailable"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda envelope: envelope["events"][0].update(
            {"callerOffset": 0xF59EB0}
        ),
        lambda envelope: envelope["events"][0].update(
            {"invisibleCountAfter": 2}
        ),
        lambda envelope: envelope.update(
            {"contextualGateHookSetInstalled": True}
        ),
        lambda envelope: envelope["ownerRelativeVisibility"][
            "publicByOwner"
        ].update({"status": "derived"}),
        lambda envelope: envelope.update(
            {"rejectedCount": 1, "complete": True}
        ),
    ),
)
def test_observe_rich_rejects_incoherent_visibility_runtime(
    mutation,
) -> None:
    response = _rich_response()
    _add_visibility_runtime(response)
    mutation(response["visibilityRuntime"])

    with pytest.raises(RunnerError):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_accepts_strict_remaining_runtime_ring() -> None:
    response = _rich_response()
    _add_remaining_runtime(response)
    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()
    event = result["remainingRuntime"]["events"][0]
    assert event["resourceCause"] == "periodic"
    assert event["resourceActualDeltaFixed"] == 10_000
    assert (
        result["remainingRuntime"]["ownerRelativeVisibility"][
            "publicByOwner"
        ]["status"]
        == "unavailable"
    )








@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("events", 0, "callerOffset"), 0xF1ACD8),
        (("events", 0, "resourceActualDeltaFixed"), 9_999),
        (("ownerRelativeVisibility", "publicByOwner", "status"), "derived"),
        (("events", 0, "completeContext"), False),
    ],
)
def test_observe_rich_rejects_incoherent_remaining_runtime(
    path: tuple[object, ...],
    value: object,
) -> None:
    response = _rich_response()
    _add_remaining_runtime(response)
    target: Any = response["remainingRuntime"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(RunnerError):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda response: response["phaseRuntime"].update({"overflowCount": 1}),
        lambda response: response["phaseRuntime"].update(
            {"rejectedCount": 1, "complete": True}
        ),
        lambda response: response["objects"][0]["phaseRuntime"].update(
            {"movementDeltaTick": int(response["tick"]) + 1}
        ),
    ),
)
def test_observe_rich_rejects_incoherent_native_phase_runtime(
    mutation,
) -> None:
    response = _rich_response()
    _add_phase_runtime(response)
    mutation(response)

    with pytest.raises(RunnerError):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_accepts_exact_attack_sequence_decay_runtime() -> None:
    response = _rich_response()
    _add_phase_runtime(response)
    response["objects"][0]["phaseRuntime"].update(
        {
            "attackSequenceProgressRaw": 12,
            "attackSequenceProgressLimit": 50,
            "attackSequenceDecayRemainingMs": 3_500,
            "attackSequenceDecayDurationMs": 7_000,
        }
    )

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    phase = result["objects"][0]["phaseRuntime"]
    assert phase["attackSequenceProgressRaw"] == 12
    assert phase["attackSequenceDecayRemainingMs"] == 3_500


def test_observe_rich_accepts_validated_null_target_negative_control() -> None:
    response = _rich_response()
    response["tick"] = 131
    response["combatEvents"]["observationTick"] = 131
    response["objects"][0]["targetEntityKey"] = None
    response["objects"][0]["targetEntityValidated"] = True
    env = StubNativeClashEnv({"observe-rich": response})

    result = env.observe_rich()

    assert result["objects"][0]["targetEntityKey"] is None
    assert result["objects"][0]["targetEntityValidated"] is True


def test_observe_rich_accepts_fail_closed_null_runtime_state() -> None:
    response = _rich_response()
    response["objects"][0]["attackSequenceStage"] = None
    response["objects"][0]["activeEffects"] = None
    response["objects"][0]["invisibleCount"] = None
    response["objects"][0]["visibilityState"] = None
    env = StubNativeClashEnv({"observe-rich": response})

    result = env.observe_rich()

    assert result["objects"][0]["activeEffects"] is None
    assert result["objects"][0]["visibilityState"] is None


def test_observe_rich_accepts_card_specific_stage_and_nonexpiring_effect() -> None:
    response = _rich_response()
    response["objects"][0]["attackSequenceStage"] = 3
    response["objects"][0]["activeEffects"][0]["remainingMs"] = -1
    env = StubNativeClashEnv({"observe-rich": response})

    result = env.observe_rich()

    assert result["objects"][0]["attackSequenceStage"] == 3
    assert result["objects"][0]["activeEffects"][0]["remainingMs"] == -1


def test_observe_rich_validates_projectile_ability_and_evolution_v2() -> None:
    response = _runtime_v2_response()
    env = StubNativeClashEnv({"observe-rich": response})

    result = env.observe_rich()

    assert result["objects"][0]["projectile"]["nativePhase"] == "in_flight"
    ability = result["players"][0]["abilityRuntime"][0]
    assert ability["buttonStateLabel"] == "Ready"
    assert ability["remainingChargesRaw"] == -1
    evolution = result["players"][0]["evolutionRuntime"][0]
    assert evolution["cycleRemaining"] == 1
    assert evolution["ready"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("objects", 0, "projectile", "sourceEntityValidated"), False),
        (("objects", 0, "projectile", "nativePhase"), "impact"),
        (("players", 0, "ownerEntityKey"), [1, -2, 500]),
        (("players", 0, "abilityRuntime", 0, "buttonStateLabel"), "Casting"),
        (("players", 0, "abilityRuntime", 0, "available"), False),
        (("players", 0, "abilityRuntime", 0, "remainingChargesRaw"), 0),
        (("players", 0, "evolutionRuntime", 0, "cycleRemaining"), 0),
        (("players", 0, "evolutionRuntime", 0, "ready"), True),
        (("players", 0, "evolutionRuntime", 0, "progress"), 3),
    ],
)
def test_observe_rich_rejects_invalid_v2_runtime_semantics(
    path: tuple[object, ...], value: object
) -> None:
    response = _runtime_v2_response()
    cursor: Any = response
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(RunnerError):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


@pytest.mark.parametrize(
    ("button_state", "label"),
    list(enumerate(native_env_module.ABILITY_BUTTON_STATE_LABELS)),
)
def test_observe_rich_accepts_only_the_exact_ability_enum_labels(
    button_state: int, label: str
) -> None:
    response = _runtime_v2_response()
    ability = response["players"][0]["abilityRuntime"][0]
    ability["buttonState"] = button_state
    ability["buttonStateLabel"] = label
    ability["available"] = button_state in {2, 4}

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    assert result["players"][0]["abilityRuntime"][0]["buttonStateLabel"] == label


def test_observe_rich_rejects_unknown_v2_fields() -> None:
    response = _runtime_v2_response()
    response["objects"][0]["unexpected"] = True
    with pytest.raises(RunnerError, match="fields do not match"):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()

    response = _runtime_v2_response()
    response["objects"][0]["activeEffects"][0]["unexpected"] = True
    with pytest.raises(RunnerError, match="incomplete"):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_atomic_validates_complete_same_identity_envelopes() -> None:
    response = _atomic_response()
    env = StubNativeClashEnv({"observe-atomic": response})

    result = env.observe_atomic()

    assert result["ordinary"]["tick"] == 530
    assert result["rich"]["objects"][0]["attackSequenceStage"] == 2
    assert env.requests == ["observe-atomic"]


def test_observe_atomic_accepts_full_256_object_bound() -> None:
    response = _atomic_response()
    rich_object = response["rich"]["objects"][0]
    rich_objects = []
    for slot in range(256):
        item = deepcopy(rich_object)
        item["slot"] = slot
        item["nativeObjectId"] = 500 + slot
        item["objectIndex"] = slot
        item["secondaryIndex"] = 0
        item["entityKey"] = [0, -2, 500 + slot]
        item["targetEntityKey"] = [0, -2, 500]
        item["activeEffects"][0]["sourceEntityKey"] = [0, -2, 500]
        rich_objects.append(item)
    response["ordinary"]["objects"] = [{"slot": slot} for slot in range(256)]
    response["ordinary"]["count"] = 256
    response["ordinary"]["returned"] = 256
    response["rich"]["objects"] = rich_objects
    response["rich"]["count"] = 256
    response["rich"]["returned"] = 256
    env = StubNativeClashEnv({"observe-atomic": response})

    result = env.observe_atomic()

    assert result["ordinary"]["returned"] == 256
    assert result["rich"]["returned"] == 256


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("maxObjects",), 64, "object bound"),
        (("rich", "tick"), 531, "identity"),
        (("ordinary", "truncated"), True, "ordinary observation is incomplete"),
        (("rich", "truncated"), True, "rich observation is incomplete"),
        (("rich", "count"), 2, "non-truncated"),
    ],
)
def test_observe_atomic_rejects_mismatch_or_truncation(
    path: tuple[str, ...], value: object, message: str
) -> None:
    response = _atomic_response()
    cursor: dict[str, Any] = response
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    env = StubNativeClashEnv({"observe-atomic": response})

    with pytest.raises(RunnerError, match=message):
        env.observe_atomic()


def test_transport_rejects_response_above_16_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_env_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: _OversizedResponseConnection(),
    )

    with pytest.raises(RunnerError, match="exceeds the 16 MiB host bound"):
        NativeClashEnv().status()






def test_stop_resident_mode_restores_singleton_control_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FixedResponseConnection(
        b'{"ok":true,"mode":"legacy-headless"}\n'
    )
    monkeypatch.setattr(
        native_env_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: connection,
    )

    result = NativeClashEnv().stop_resident_mode()

    assert result["mode"] == "legacy-headless"
    assert connection.sent == b"multi-stop\n"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "native-rich-telemetry.v0"),
        (("statusEnum",), ["authoritative"]),
        (("stateEpoch",), True),
        (("objects", 0, "targetEntityValidated"), False),
        (("objects", 0, "components", "slots", 0, "engineType"), 32),
        (("objects", 0, "shield", "current"), 151),
        (("objects", 0, "shield", "status"), "derived"),
        (("objects", 0, "attackSequenceStage"), 256),
        (("objects", 0, "activeEffects", 0, "buffGlobalId"), 0),
        (
            ("objects", 0, "activeEffects", 0, "buffGlobalId"),
            0x1_0000_0000,
        ),
        (("objects", 0, "activeEffects", 0, "name"), "茅" * 65),
        (("objects", 0, "activeEffects", 0, "remainingMs"), -2),
        (("objects", 0, "activeEffects", 0, "sourceEntityKey"), [1, 2]),
        (("objects", 0, "invisibleCount"), 1),
        (("objects", 0, "visibilityState"), "hidden"),
    ],
)
def test_observe_rich_rejects_malformed_or_unvalidated_fields(
    path: tuple[object, ...], value: object
) -> None:
    response = _rich_response()
    cursor: Any = response
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    env = StubNativeClashEnv({"observe-rich": response})

    with pytest.raises(RunnerError):
        env.observe_rich()


def test_observe_rich_accepts_high_bit_uint32_buff_global_id() -> None:
    response = _rich_response()
    response["objects"][0]["activeEffects"][0][
        "buffGlobalId"
    ] = 2_428_333_743
    response["objects"][0]["activeEffects"][0][
        "name"
    ] = "BabyDragon_EV1_wind_buff_negative"
    env = StubNativeClashEnv({"observe-rich": response})

    result = env.observe_rich()

    assert (
        result["objects"][0]["activeEffects"][0]["buffGlobalId"]
        == 2_428_333_743
    )


def test_observe_rich_accepts_strict_native_combat_event_ring() -> None:
    response = _rich_response()
    _add_damage_combat_event(response)
    env = StubNativeClashEnv({"observe-rich": response})

    result = env.observe_rich()

    event = result["combatEvents"]["events"][0]
    assert event["kind"] == "damage"
    assert event["actualAmount"] == 100
    assert event["target"]["nativeObjectId"] == 500


def test_observe_rich_accepts_only_exact_generic_action_heal_caller() -> None:
    response = _rich_response()
    _add_generic_action_heal_combat_event(response)

    event = StubNativeClashEnv({"observe-rich": response}).observe_rich()[
        "combatEvents"
    ]["events"][0]

    assert event["callerOffset"] == 0xF12D54
    assert event["postHp"] - event["preHp"] == event["actualAmount"]

    response["combatEvents"]["events"][0]["callerOffset"] = 0xF12D50
    with pytest.raises(RunnerError, match="callerOffset is invalid for heal"):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_accepts_capture_time_owner_change_for_same_native_id() -> None:
    response = _rich_response()
    _add_damage_combat_event(response)
    first = response["combatEvents"]["events"][0]
    second = deepcopy(first)
    second["sequence"] = 2
    for fact_name in ("target", "immediateSource", "source"):
        second[fact_name]["owner"] = 1
        second[fact_name]["entityKey"] = [1, -2, 500]
    response["combatEvents"]["events"].append(second)
    response["combatEvents"]["nextSequence"] = 3

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    first_target, second_target = (
        event["target"] for event in result["combatEvents"]["events"]
    )
    assert first_target["nativeObjectId"] == second_target["nativeObjectId"] == 500
    assert first_target["entityKey"] == [0, -2, 500]
    assert second_target["entityKey"] == [1, -2, 500]


def test_observe_rich_accepts_exact_evolved_spawn_deployment_context() -> None:
    response = _rich_response()
    _add_evolved_spawn_combat_event(response)
    env = StubNativeClashEnv({"observe-rich": response})

    event = env.observe_rich()["combatEvents"]["events"][0]

    deployment = event["deploymentContext"]
    assert deployment["deploymentSequence"] == 7
    assert deployment["formName"] == "EvoForm"
    assert deployment["deckSlot"] == 2


def test_observe_rich_accepts_unique_card_play_deployment_context() -> None:
    response = _rich_response()
    _add_card_play_combat_event(response)

    event = StubNativeClashEnv({"observe-rich": response}).observe_rich()[
        "combatEvents"
    ]["events"][0]

    assert event["kind"] == "card_play"
    assert event["hookOffset"] == 0xF38A68
    assert event["deploymentContext"]["playedCardGlobalId"] == 26_000_001


def test_observe_rich_rejects_card_play_without_unique_deployment() -> None:
    missing = _rich_response()
    _add_card_play_combat_event(missing)
    missing["combatEvents"]["events"][0]["deploymentContext"] = None
    with pytest.raises(RunnerError, match="required deploymentContext"):
        StubNativeClashEnv({"observe-rich": missing}).observe_rich()

    duplicate = _rich_response()
    _add_card_play_combat_event(duplicate)
    second = deepcopy(duplicate["combatEvents"]["events"][0])
    second["sequence"] = 2
    duplicate["combatEvents"]["events"].append(second)
    duplicate["combatEvents"]["nextSequence"] = 3
    with pytest.raises(RunnerError, match="repeats a card_play"):
        StubNativeClashEnv({"observe-rich": duplicate}).observe_rich()


def test_observe_rich_accepts_exact_area_tick_spawn_provenance() -> None:
    response = _rich_response()
    _add_evolved_spawn_combat_event(response)
    combat_event = response["combatEvents"]["events"][0]
    child = combat_event["target"]
    area = deepcopy(child)
    area.update(
        {
            "nativeObjectId": 777,
            "entityKey": [0, -2, 777],
            "objectIndex": 77,
            "objectKind": 3,
            "cardId": 737_088_471,
        }
    )
    combat_event.update(
        {
            "deploymentContext": None,
            "immediateSource": area,
            "source": deepcopy(area),
            "spawnProvenance": {
                "kind": "area_tick_nested_spawn",
                "hookOffset": 0xF143A8,
                "sourceDataGlobalId": 737_088_471,
            },
        }
    )

    _add_remaining_runtime(response)
    runtime_event = response["remainingRuntime"]["events"][0]
    runtime_event.update(
        {
            "kind": "area_action_spawn",
            "hookOffset": 0xF143A8,
            "callerOffset": None,
            "entity": child,
            "source": area,
            "target": child,
            "objectKind": 3,
            "runtimeVtableOffset": 0x189C2E8,
            "dataAfterGlobalId": 119_617_828,
            "configuredDataGlobalId": 737_088_471,
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

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    assert result["combatEvents"]["events"][0]["causeSequence"] is None
    assert result["combatEvents"]["events"][0]["spawnProvenance"][
        "kind"
    ] == "area_tick_nested_spawn"
    assert result["remainingRuntime"]["events"][0]["kind"] == (
        "area_action_spawn"
    )


def test_observe_rich_accepts_exact_action_spawn_to_location_provenance() -> None:
    response = _rich_response()
    _add_evolved_spawn_combat_event(response)
    combat_event = response["combatEvents"]["events"][0]
    child = combat_event["target"]
    producer = deepcopy(child)
    producer.update(
        {
            "nativeObjectId": 777,
            "entityKey": [0, -2, 777],
            "objectIndex": 77,
            "objectKind": 5,
            "cardId": 27_000_010,
        }
    )
    combat_event.update(
        {
            "deploymentContext": None,
            "immediateSource": producer,
            "source": deepcopy(producer),
            "spawnProvenance": {
                "kind": "action_spawn_to_location",
                "hookOffset": 0xE7B864,
                "sourceDataGlobalId": 4_162_986_129,
            },
        }
    )

    _add_remaining_runtime(response)
    runtime_event = response["remainingRuntime"]["events"][0]
    runtime_event.update(
        {
            "kind": "area_action_spawn",
            "hookOffset": 0xE7B864,
            "callerOffset": None,
            "entity": child,
            "source": producer,
            "target": child,
            "objectKind": 5,
            "runtimeVtableOffset": 0x189C4E8,
            "dataAfterGlobalId": 119_617_828,
            "configuredDataGlobalId": 3_193_030_688,
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

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    assert result["combatEvents"]["events"][0]["spawnProvenance"] == {
        "kind": "action_spawn_to_location",
        "hookOffset": 0xE7B864,
        "sourceDataGlobalId": 4_162_986_129,
    }
    assert result["remainingRuntime"]["events"][0]["hookOffset"] == 0xE7B864


def test_observe_rich_accepts_exact_projectile_spawn_provenance() -> None:
    response = _rich_response()
    _add_evolved_spawn_combat_event(response)
    event = response["combatEvents"]["events"][0]
    child = event["target"]
    producer = deepcopy(child)
    producer.update(
        {
            "nativeObjectId": 777,
            "entityKey": [0, -2, 777],
            "objectIndex": 77,
            "objectKind": 5,
            "cardId": 27_000_010,
        }
    )
    child["objectKind"] = 4
    event.update(
        {
            "kind": "projectile_spawn",
            "deploymentContext": None,
            "immediateSource": producer,
            "source": deepcopy(producer),
            "projectile": deepcopy(child),
            "spawnProvenance": {
                "kind": "action_spawn_to_location",
                "hookOffset": 0xE7B864,
                "sourceDataGlobalId": 4_162_986_129,
            },
        }
    )

    result = StubNativeClashEnv({"observe-rich": response}).observe_rich()

    parsed = result["combatEvents"]["events"][0]
    assert parsed["kind"] == "projectile_spawn"
    assert parsed["spawnProvenance"]["kind"] == "action_spawn_to_location"


def test_observe_rich_rejects_area_tick_spawn_without_matching_source() -> None:
    response = _rich_response()
    _add_evolved_spawn_combat_event(response)
    event = response["combatEvents"]["events"][0]
    event.update(
        {
            "deploymentContext": None,
            "immediateSource": _null_combat_fact(),
            "spawnProvenance": {
                "kind": "area_tick_nested_spawn",
                "hookOffset": 0xF143A8,
                "sourceDataGlobalId": 737_088_471,
            },
        }
    )

    with pytest.raises(RunnerError, match="spawnProvenance"):
        StubNativeClashEnv({"observe-rich": response}).observe_rich()


def test_observe_rich_accepts_exact_mirror_spawn_deployment_context() -> None:
    response = _rich_response()
    _add_evolved_spawn_combat_event(response)
    deployment = response["combatEvents"]["events"][0]["deploymentContext"]
    deployment.update(
        {
            "playedCardGlobalId": 28_000_006,
            "effectiveCardGlobalId": 26_000_014,
            "cardParameter": (5 << 28) | ((4 + 1) << 22),
            "deckSlot": 4,
            "cost": 5,
            "formCode": 0,
            "formName": "BasicForm",
        }
    )
    env = StubNativeClashEnv({"observe-rich": response})

    accepted = env.observe_rich()["combatEvents"]["events"][0][
        "deploymentContext"
    ]

    assert accepted["playedCardGlobalId"] == 28_000_006
    assert accepted["effectiveCardGlobalId"] == 26_000_014


def test_observe_rich_rejects_nonmirror_basic_effective_identity_change() -> None:
    response = _rich_response()
    _add_evolved_spawn_combat_event(response)
    deployment = response["combatEvents"]["events"][0]["deploymentContext"]
    deployment.update(
        {
            "playedCardGlobalId": 28_000_005,
            "effectiveCardGlobalId": 26_000_014,
            "cardParameter": (5 << 28) | ((4 + 1) << 22),
            "deckSlot": 4,
            "cost": 5,
            "formCode": 0,
            "formName": "BasicForm",
        }
    )
    env = StubNativeClashEnv({"observe-rich": response})

    with pytest.raises(RunnerError, match="effective form identity"):
        env.observe_rich()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda deployment: deployment.update({"formName": "BasicForm"}),
        lambda deployment: deployment.update({"deckSlot": 3}),
        lambda deployment: deployment.update({"effectiveCardGlobalId": None}),
        lambda deployment: deployment.update({"consumeHookOffset": 0xF38A6C}),
    ],
)
def test_observe_rich_rejects_incoherent_spawn_deployment_context(
    mutation: Any,
) -> None:
    response = _rich_response()
    _add_evolved_spawn_combat_event(response)
    mutation(response["combatEvents"]["events"][0]["deploymentContext"])
    env = StubNativeClashEnv({"observe-rich": response})

    with pytest.raises(RunnerError, match="deploymentContext"):
        env.observe_rich()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda response: response["combatEvents"].update(
                {"rejectedCaptureCount": 1}
            ),
            "incomplete",
        ),
        (
            lambda response: response["combatEvents"].update(
                {"overflowCount": 1}
            ),
            "overflow",
        ),
        (
            lambda response: response["combatEvents"]["capability"].update(
                {"status": "pending"}
            ),
            "capability status",
        ),
        (
            lambda response: response["combatEvents"]["events"][0].update(
                {"actualAmount": 99}
            ),
            "delta",
        ),
        (
            lambda response: response["combatEvents"]["events"][0]["target"].update(
                {"validated": False}
            ),
            "unvalidated",
        ),
    ],
)
def test_observe_rich_rejects_incomplete_or_incoherent_combat_ring(
    mutation: Any,
    message: str,
) -> None:
    response = _rich_response()
    _add_damage_combat_event(response)
    mutation(response)
    env = StubNativeClashEnv({"observe-rich": response})

    with pytest.raises(RunnerError, match=message):
        env.observe_rich()
