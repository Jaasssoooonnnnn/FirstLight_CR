from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pytest

from native_runner.special_movement_runtime import (
    SpecialMovementRuntimeError,
    parse_special_movement_runtime,
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


def _envelope() -> dict[str, Any]:
    source_before = _fact(
        native_object_id=101,
        owner=0,
        object_index=11,
        card_id=26_000_046,
        position=(5_000, 9_000),
    )
    source_after = deepcopy(source_before)
    source_after["position"] = [8_000, 12_000]
    target = _fact(
        native_object_id=202,
        owner=1,
        object_index=22,
        card_id=26_000_003,
        position=(9_000, 13_000),
    )
    return {
        "ok": True,
        "schema": "native-special-movement-runtime.v1",
        "generation": 3,
        "stateEpoch": 9,
        "observationTick": 200,
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
        "epochFirstSequence": 7,
        "oldestRetainedSequence": 7,
        "nextSequence": 8,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            {
                "sequence": 7,
                "tick": 190,
                "kind": "ordinary_dash_execute",
                "mode": "dash",
                "stage": "execute",
                "hookOffset": 0xF609DC,
                "callerOffset": 0xF61BF0,
                "sourceBefore": source_before,
                "sourceAfter": source_after,
                "target": target,
                "requestedX": 8_600,
                "requestedY": 12_600,
                "targetRadius": 500,
                "operationFlag": 1,
                "cooldownBeforeMs": 1_000,
                "cooldownAfterMs": 1_000,
                "positionChanged": True,
                "completeContext": True,
            }
        ],
    }


def _parse(value: object):
    return parse_special_movement_runtime(
        value,
        generation=3,
        state_epoch=9,
        observation_tick=200,
    )


def test_parse_special_movement_runtime_preserves_exact_execute_edge() -> None:
    parsed = _parse(_envelope())

    assert parsed.complete is True
    assert parsed.capability_status == "derived"
    assert len(parsed.events) == 1
    event = parsed.events[0]
    assert event.kind == "ordinary_dash_execute"
    assert event.source_before.entity_key == event.source_after.entity_key
    assert event.source_before.position == (5_000, 9_000)
    assert event.source_after.position == (8_000, 12_000)
    # The caller adjusts the target center by both radii before calling the
    # executor, so the requested destination is intentionally distinct.
    assert (event.requested_x, event.requested_y) != event.target.position
    assert event.position_changed is True


def test_parse_special_movement_runtime_accepts_fail_closed_unavailable_ring() -> None:
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
        lambda envelope: envelope["capability"].update({"failClosed": False}),
        lambda envelope: envelope.update({"hookSetAttested": False}),
        lambda envelope: envelope.update({"rejectedCount": 1}),
        lambda envelope: envelope["events"][0].update({"sequence": 8}),
        lambda envelope: envelope["events"][0].update({"tick": 201}),
        lambda envelope: envelope["events"][0].update(
            {"hookOffset": 0xF609E0}
        ),
        lambda envelope: envelope["events"][0].update(
            {"callerOffset": 0xF61BEC}
        ),
        lambda envelope: envelope["events"][0].update({"operationFlag": 0}),
        lambda envelope: envelope["events"][0].update({"completeContext": False}),
        lambda envelope: envelope["events"][0].update({"positionChanged": False}),
        lambda envelope: envelope["events"][0].update(
            {"requestedX": 1 << 31}
        ),
        lambda envelope: envelope["events"][0]["sourceAfter"].update(
            {"nativeObjectId": 102, "entityKey": [0, -2, 102]}
        ),
        lambda envelope: envelope["events"][0]["target"].update(
            {"present": False}
        ),
    ),
)
def test_parse_special_movement_runtime_rejects_incoherent_contract(
    mutation: Mutation,
) -> None:
    envelope = _envelope()
    mutation(envelope)

    with pytest.raises(SpecialMovementRuntimeError):
        _parse(envelope)


def test_parse_special_movement_runtime_rejects_decreasing_event_ticks() -> None:
    envelope = _envelope()
    second = deepcopy(envelope["events"][0])
    second.update({"sequence": 8, "tick": 189})
    envelope["events"].append(second)
    envelope["nextSequence"] = 9

    with pytest.raises(SpecialMovementRuntimeError, match="ticks are not monotonic"):
        _parse(envelope)
