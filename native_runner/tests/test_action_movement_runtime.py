from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pytest

from native_runner.action_movement_runtime import (
    ActionMovementRuntimeError,
    parse_action_movement_runtime,
)


def _fact(
    *,
    native_object_id: int,
    owner: int,
    object_index: int,
    card_id: int,
    position: tuple[int, int],
) -> dict[str, object]:
    return {
        "validated": True,
        "present": True,
        "nativeObjectId": native_object_id,
        "entityKey": [owner, -2, native_object_id],
        "owner": owner,
        "objectIndex": object_index,
        "secondaryIndex": 0,
        "cardId": card_id,
        "objectKind": 1,
        "position": list(position),
        "visibilityValidated": True,
        "invisibleCount": 0,
    }


def _golden_event() -> dict[str, Any]:
    source = _fact(
        native_object_id=101,
        owner=0,
        object_index=11,
        card_id=26_000_074,
        position=(5_000, 9_000),
    )
    target = _fact(
        native_object_id=202,
        owner=1,
        object_index=22,
        card_id=26_000_003,
        position=(8_000, 13_000),
    )
    return {
        "sequence": 7,
        "tick": 190,
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


def _boss_event() -> dict[str, Any]:
    before = _fact(
        native_object_id=303,
        owner=0,
        object_index=33,
        card_id=26_000_103,
        position=(7_000, 11_000),
    )
    after = deepcopy(before)
    after["position"] = [7_000, 5_000]
    return {
        "sequence": 8,
        "tick": 191,
        "kind": "boss_bandit_warp_position_commit",
        "mode": "warp",
        "stage": "position_commit",
        "hookOffset": 0xE82A34,
        "callerOffset": 0xF132A0,
        "actionDataGlobalId": 807_203_904,
        "actionName": "BossBandit_ability_warp",
        "actionClassVtableOffset": 0x1895120,
        "runtimeClassVtableOffset": None,
        "sourceBefore": before,
        "sourceAfter": after,
        "target": None,
        "chainIndex": None,
        "positionChanged": True,
        "completeContext": True,
    }


def _elite_event() -> dict[str, Any]:
    before = _fact(
        native_object_id=404,
        owner=1,
        object_index=44,
        card_id=203_000_062,
        position=(12_000, 20_000),
    )
    after = deepcopy(before)
    after["position"] = [12_000, 16_500]
    return {
        "sequence": 9,
        "tick": 192,
        "kind": "elite_archer_warp_position_commit",
        "mode": "warp",
        "stage": "position_commit",
        "hookOffset": 0xE82A34,
        "callerOffset": 0xF132A0,
        "actionDataGlobalId": 2_158_032_559,
        "actionName": "EliteArcherHero_Ability_Warp",
        "actionClassVtableOffset": 0x1895120,
        "runtimeClassVtableOffset": None,
        "sourceBefore": before,
        "sourceAfter": after,
        "target": None,
        "chainIndex": None,
        "positionChanged": True,
        "completeContext": True,
    }


def _envelope() -> dict[str, Any]:
    return {
        "ok": True,
        "schema": "native-action-movement-runtime.v1",
        "generation": 3,
        "stateEpoch": 9,
        "observationTick": 200,
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
        "epochFirstSequence": 7,
        "oldestRetainedSequence": 7,
        "nextSequence": 10,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [_golden_event(), _boss_event(), _elite_event()],
    }


def _parse(value: object):
    return parse_action_movement_runtime(
        value,
        generation=3,
        state_epoch=9,
        observation_tick=200,
    )


def test_parse_action_movement_runtime_preserves_exact_stage_boundaries() -> None:
    parsed = _parse(_envelope())

    assert parsed.complete is True
    assert parsed.capability_status == "derived"
    assert [event.kind for event in parsed.events] == [
        "golden_knight_chain_hop_launch",
        "boss_bandit_warp_position_commit",
        "elite_archer_warp_position_commit",
    ]
    golden, boss, elite = parsed.events
    assert golden.stage == "hop_launch"
    assert golden.target is not None
    assert golden.chain_index == 0
    assert golden.position_changed is False
    assert boss.stage == "position_commit"
    assert boss.source_before.position == (7_000, 11_000)
    assert boss.source_after.position == (7_000, 5_000)
    assert elite.source_before.card_id == 203_000_062


def test_parse_action_movement_runtime_accepts_fail_closed_unavailable_ring() -> None:
    envelope = _envelope()
    envelope.update(
        {
            "hookSetAttested": False,
            "hookSetInstalled": False,
            "capability": {
                **envelope["capability"],
                "status": "unavailable",
            },
            "nextSequence": 7,
            "complete": False,
            "events": [],
        }
    )

    parsed = _parse(envelope)

    assert parsed.capability_status == "unavailable"
    assert parsed.complete is False
    assert parsed.events == ()


Mutation = Callable[[dict[str, Any]], None]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda envelope: envelope.update({"capacity": 4_096}),
        lambda envelope: envelope["capability"].update(
            {"failClosed": False}
        ),
        lambda envelope: envelope.update({"hookSetAttested": False}),
        lambda envelope: envelope.update({"rejectedCount": 1}),
        lambda envelope: envelope["events"][0].update({"sequence": 8}),
        lambda envelope: envelope["events"][0].update({"tick": 201}),
        lambda envelope: envelope["events"][0].update(
            {"actionName": "GoldenKnight_Charge_Target"}
        ),
        lambda envelope: envelope["events"][0].update(
            {"actionDataGlobalId": 0}
        ),
        lambda envelope: envelope["events"][0].update(
            {"actionClassVtableOffset": 0x1895120}
        ),
        lambda envelope: envelope["events"][0].update(
            {"runtimeClassVtableOffset": None}
        ),
        lambda envelope: envelope["events"][0].update({"target": None}),
        lambda envelope: envelope["events"][0].update({"chainIndex": None}),
        lambda envelope: envelope["events"][0]["sourceBefore"].update(
            {"cardId": 26_000_072}
        ),
        lambda envelope: envelope["events"][1].update(
            {"stage": "execute"}
        ),
        lambda envelope: envelope["events"][1].update(
            {"positionChanged": False}
        ),
        lambda envelope: envelope["events"][1].update(
            {"runtimeClassVtableOffset": 0x189ECA0}
        ),
        lambda envelope: envelope["events"][1].update(
            {"target": deepcopy(envelope["events"][0]["target"])}
        ),
        lambda envelope: envelope["events"][2]["sourceAfter"].update(
            {"nativeObjectId": 405, "entityKey": [1, -2, 405]}
        ),
    ),
)
def test_parse_action_movement_runtime_rejects_incoherent_contract(
    mutation: Mutation,
) -> None:
    envelope = _envelope()
    mutation(envelope)

    with pytest.raises(ActionMovementRuntimeError):
        _parse(envelope)


def test_parse_action_movement_runtime_rejects_decreasing_ticks() -> None:
    envelope = _envelope()
    envelope["events"][1]["tick"] = 189

    with pytest.raises(
        ActionMovementRuntimeError, match="ticks are not monotonic"
    ):
        _parse(envelope)
