from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pytest

from native_runner.character_state_runtime import (
    CharacterStateRuntimeError,
    parse_character_state_runtime,
)


def _fact(
    *,
    native_object_id: int = 101,
    card_id: int = 26_000_046,
    position: tuple[int, int] = (5_000, 9_000),
) -> dict[str, object]:
    return {
        "validated": True,
        "present": True,
        "nativeObjectId": native_object_id,
        "entityKey": [0, -2, native_object_id],
        "owner": 0,
        "objectIndex": 11,
        "secondaryIndex": 0,
        "cardId": card_id,
        "objectKind": 5,
        "position": list(position),
        "visibilityValidated": True,
        "invisibleCount": 0,
    }


def _event(
    *,
    sequence: int,
    tick: int,
    previous_state: int,
    committed_state: int,
    card_id: int = 26_000_046,
    caller_offset: int = 0xF60CC4,
) -> dict[str, Any]:
    before = _fact(
        native_object_id=100 + sequence,
        card_id=card_id,
    )
    return {
        "sequence": sequence,
        "tick": tick,
        "kind": "native_character_state_transition",
        "hookOffset": 0xF18654,
        "callerOffset": caller_offset,
        "previousState": previous_state,
        "requestedState": committed_state,
        "committedState": committed_state,
        "sourceBefore": before,
        "sourceAfter": deepcopy(before),
        "positionChanged": False,
        "completeContext": True,
    }


def _envelope() -> dict[str, Any]:
    return {
        "ok": True,
        "schema": "native-character-state-runtime.v1",
        "generation": 3,
        "stateEpoch": 9,
        "observationTick": 200,
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
        "epochFirstSequence": 7,
        "oldestRetainedSequence": 7,
        "nextSequence": 9,
        "overflowCount": 0,
        "rejectedCount": 0,
        "sequenceGapBeforeOldest": False,
        "complete": True,
        "events": [
            _event(
                sequence=7,
                tick=190,
                previous_state=1,
                committed_state=3,
            ),
            _event(
                sequence=8,
                tick=191,
                previous_state=3,
                committed_state=1,
                card_id=26_000_115,
                caller_offset=0xF18D24,
            ),
        ],
    }


def _parse(value: object):
    return parse_character_state_runtime(
        value,
        generation=3,
        state_epoch=9,
        observation_tick=200,
    )


def test_parse_character_state_runtime_preserves_raw_transitions() -> None:
    parsed = _parse(_envelope())

    assert parsed.complete is True
    assert parsed.capability_status == "derived"
    first, guard = parsed.events
    assert (first.previous_state, first.committed_state) == (1, 3)
    assert first.caller_offset == 0xF60CC4
    assert guard.source_before.card_id == 26_000_115
    assert guard.position_changed is False


def test_parse_character_state_runtime_accepts_unavailable_empty_ring() -> None:
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

    assert parsed.complete is False
    assert parsed.events == ()


Mutation = Callable[[dict[str, Any]], None]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda envelope: envelope.update({"capacity": 1_024}),
        lambda envelope: envelope["capability"].update(
            {"failClosed": False}
        ),
        lambda envelope: envelope.update({"hookSetAttested": False}),
        lambda envelope: envelope.update({"rejectedCount": 1}),
        lambda envelope: envelope["events"][0].update({"sequence": 8}),
        lambda envelope: envelope["events"][0].update({"tick": 201}),
        lambda envelope: envelope["events"][0].update({"hookOffset": 1}),
        lambda envelope: envelope["events"][0].update({"callerOffset": 0}),
        lambda envelope: envelope["events"][0].update(
            {"requestedState": 2}
        ),
        lambda envelope: envelope["events"][0].update(
            {"previousState": 3}
        ),
        lambda envelope: envelope["events"][0]["sourceBefore"].update(
            {"cardId": 26_000_003}
        ),
        lambda envelope: envelope["events"][0]["sourceBefore"].update(
            {"objectKind": 1}
        ),
        lambda envelope: envelope["events"][0]["sourceAfter"].update(
            {"nativeObjectId": 999, "entityKey": [0, -2, 999]}
        ),
        lambda envelope: envelope["events"][0].update(
            {"positionChanged": True}
        ),
    ),
)
def test_parse_character_state_runtime_rejects_incoherent_contract(
    mutation: Mutation,
) -> None:
    envelope = _envelope()
    mutation(envelope)

    with pytest.raises(CharacterStateRuntimeError):
        _parse(envelope)


def test_parse_character_state_runtime_rejects_decreasing_ticks() -> None:
    envelope = _envelope()
    envelope["events"][1]["tick"] = 189

    with pytest.raises(
        CharacterStateRuntimeError, match="ticks are not monotonic"
    ):
        _parse(envelope)
