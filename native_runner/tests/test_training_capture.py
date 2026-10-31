from __future__ import annotations

import json
import struct
from dataclasses import replace

import pytest

from native_runner.training_capture import (
    TrainingCaptureError,
    decode_training_capture,
)


_NONE_I32 = -(2**31)
_EMPTY_FACT = struct.pack("<II8i", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
_EMPTY_DEPLOYMENT = struct.pack(
    "<B3xQiIIIiii",
    0,
    0,
    -1,
    0,
    0,
    0,
    -1,
    -1,
    -1,
)


# Empty-window envelopes of the seven JSON runtime families appended by CRTF v5, taken
# from a real engine capture (identity fields are filled in per capture).
_V5_FAMILY_TEMPLATES = {'phaseRuntime': {'ok': True,
                  'schema': 'native-phase-runtime.v1',
                  'capacity': 4096,
                  'hookSetAttested': True,
                  'hookSetInstalled': True,
                  'capability': {'status': 'derived',
                                 'source': 'exact-build-native-phase-hook-ring',
                                 'validation': 'sha-build-id-prologue-abi-bounded-identity',
                                 'confidence': 'high',
                                 'failClosed': True},
                  'epochFirstSequence': 1,
                  'oldestRetainedSequence': 1,
                  'transmitFromSequence': 1,
                  'nextSequence': 1,
                  'overflowCount': 0,
                  'rejectedCount': 0,
                  'sequenceGapBeforeOldest': False,
                  'complete': True,
                  'events': []},
 'specialMovementRuntime': {'ok': True,
                            'schema': 'native-special-movement-runtime.v1',
                            'capacity': 1024,
                            'hookSetAttested': True,
                            'hookSetInstalled': True,
                            'capability': {'status': 'derived',
                                           'source': 'caller-filtered-native-special-movement-hook-ring',
                                           'validation': 'sha-build-id-prologue-abi-type0-owner-target-identity',
                                           'confidence': 'high',
                                           'failClosed': True},
                            'epochFirstSequence': 1,
                            'oldestRetainedSequence': 1,
                            'nextSequence': 1,
                            'overflowCount': 0,
                            'rejectedCount': 0,
                            'sequenceGapBeforeOldest': False,
                            'complete': True,
                            'events': []},
 'actionMovementRuntime': {'ok': True,
                           'schema': 'native-action-movement-runtime.v1',
                           'capacity': 1024,
                           'hookSetAttested': True,
                           'hookSetInstalled': True,
                           'capability': {'status': 'derived',
                                          'source': 'allowlisted-native-action-movement-hook-ring',
                                          'validation': 'sha-build-id-prologue-abi-action-name-global-id-class-vtable-source-identity',
                                          'confidence': 'high',
                                          'failClosed': True},
                           'epochFirstSequence': 1,
                           'oldestRetainedSequence': 1,
                           'nextSequence': 1,
                           'overflowCount': 0,
                           'rejectedCount': 0,
                           'sequenceGapBeforeOldest': False,
                           'complete': True,
                           'events': []},
 'characterStateRuntime': {'ok': True,
                           'schema': 'native-character-state-runtime.v1',
                           'capacity': 2048,
                           'hookSetAttested': True,
                           'hookSetInstalled': True,
                           'capability': {'status': 'derived',
                                          'source': 'allowlisted-native-character-state-hook-ring',
                                          'validation': 'sha-build-id-prologue-abi-kind5-card-identity-committed-state',
                                          'confidence': 'high',
                                          'failClosed': True},
                           'epochFirstSequence': 1,
                           'oldestRetainedSequence': 1,
                           'nextSequence': 1,
                           'overflowCount': 0,
                           'rejectedCount': 0,
                           'sequenceGapBeforeOldest': False,
                           'complete': True,
                           'events': []},
 'visibilityRuntime': {'ok': True,
                       'schema': 'native-visibility-runtime.v1',
                       'capacity': 4096,
                       'hookSetAttested': True,
                       'transitionHookSetInstalled': True,
                       'contextualGateHookSetInstalled': False,
                       'capability': {'status': 'derived',
                                      'source': 'exact-build-native-invisibility-transition-ring',
                                      'validation': 'sha-build-id-call-and-prologue-fingerprint-abi-count-edge',
                                      'confidence': 'high',
                                      'failClosed': True},
                       'contextualGateCapability': {'status': 'unavailable',
                                                    'source': 'no-safe-pc-relative-gate-bridge',
                                                    'validation': 'fail-closed-no-generic-trampoline',
                                                    'confidence': 'none',
                                                    'failClosed': True},
                       'ownerRelativeVisibility': {'publicByOwner': {'status': 'unavailable',
                                                                     'reason': 'no-exact-owner-conditioned-native-producer'},
                                                   'targetableByOwner': {'status': 'unavailable',
                                                                         'reason': 'no-exact-owner-conditioned-native-producer'}},
                       'epochFirstSequence': 1,
                       'oldestRetainedSequence': 1,
                       'transmitFromSequence': 1,
                       'nextSequence': 1,
                       'overflowCount': 0,
                       'rejectedCount': 0,
                       'sequenceGapBeforeOldest': False,
                       'complete': True,
                       'events': []},
 'remainingRuntime': {'ok': True,
                      'schema': 'native-remaining-runtime.v1',
                      'capacity': 4096,
                      'hookSetAttested': True,
                      'hookSetInstalled': True,
                      'capability': {'status': 'derived',
                                     'source': 'exact-build-native-remaining-hook-ring',
                                     'validation': 'sha-build-id-16-byte-fingerprint-abi-identity',
                                     'confidence': 'high',
                                     'failClosed': True},
                      'ownerRelativeVisibility': {'publicByOwner': {'status': 'unavailable',
                                                                    'reason': 'no-exact-owner-conditioned-native-producer'},
                                                  'targetableByOwner': {'status': 'unavailable',
                                                                        'reason': 'no-exact-owner-conditioned-native-producer'}},
                      'epochFirstSequence': 1,
                      'oldestRetainedSequence': 1,
                      'transmitFromSequence': 1,
                      'nextSequence': 1,
                      'overflowCount': 0,
                      'rejectedCount': 0,
                      'sequenceGapBeforeOldest': False,
                      'complete': True,
                      'events': []},
 'towerTroopRuntime': {'ok': True,
                       'schema': 'native-tower-troop-runtime.v1',
                       'hookSetAttested': True,
                       'hookSetInstalled': True,
                       'rejectedCount': 0,
                       'complete': True}}


def _json_families(*, generation: int = 7, state_epoch: int = 11, tick: int = 42, overrides=None) -> bytes:
    families = json.loads(json.dumps(_V5_FAMILY_TEMPLATES))
    for name, envelope in families.items():
        envelope.update({"generation": generation, "stateEpoch": state_epoch, "observationTick": tick})
        envelope.update((overrides or {}).get(name, {}))
    body = bytearray(struct.pack("<H", len(families)))
    for envelope in families.values():
        encoded = json.dumps(envelope, separators=(",", ":")).encode()
        body.extend(struct.pack("<I", len(encoded)))
        body.extend(encoded)
    return bytes(body)


def _fact(*, native_id: int, owner: int, card_id: int) -> bytes:
    return struct.pack(
        "<II8i",
        0b111,
        native_id,
        owner,
        native_id,
        -1,
        card_id,
        4,
        9000,
        16000,
        0,
    )


def _event(
    *,
    sequence: int,
    kind: int,
    hook_offset: int,
    deployment: bytes = _EMPTY_DEPLOYMENT,
    target: bytes = _EMPTY_FACT,
    immediate_source: bytes = _EMPTY_FACT,
    source: bytes = _EMPTY_FACT,
    projectile: bytes = _EMPTY_FACT,
    route_source_after: bytes = _EMPTY_FACT,
    destination_after: tuple[int, int] = (_NONE_I32, _NONE_I32),
    requested_amount: int = _NONE_I32,
    actual_amount: int = _NONE_I32,
    pre_hp: int = _NONE_I32,
    post_hp: int = _NONE_I32,
) -> bytes:
    return b"".join(
        (
            struct.pack(
                "<QiBBBBHQQQ8i",
                sequence,
                42,
                kind,
                1 if kind == 1 else 0,
                0,
                0,
                0,
                hook_offset,
                0,
                0,
                requested_amount,
                actual_amount,
                pre_hp,
                post_hp,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
            ),
            deployment,
            target,
            immediate_source,
            source,
            projectile,
            _EMPTY_FACT,  # related
            _EMPTY_FACT,  # route source before
            _EMPTY_FACT,  # route target before
            route_source_after,
            _EMPTY_FACT,  # route target after
            struct.pack("<4i", _NONE_I32, _NONE_I32, *destination_after),
        )
    )


def _tower_object(
    *,
    slot: int,
    native_id: int,
    owner: int,
    kind: int,
    runtime_values: tuple[int, ...],
    target_native_id: int = 0,
) -> bytes:
    if len(runtime_values) != 9:
        raise ValueError("tower runtime fixture requires nine state values")
    return b"".join(
        (
            struct.pack(
                "<IiI15i",
                (1 << 9) | (1 << 10),
                slot,
                native_id,
                owner,
                15_900_000 + slot,
                9000,
                16000,
                9000,
                16000,
                native_id,
                -1,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                -1,
            ),
            struct.pack("<I", 15_900_000 + slot),
            struct.pack(
                "<B3x10iI",
                kind,
                41,
                *runtime_values,
                target_native_id,
            ),
            struct.pack("<H", 0),
        )
    )


def _capture(
    combat_events: tuple[bytes, ...] = (),
    *,
    version: int = 6,
    objects: tuple[bytes, ...] = (),
    families: bytes | None = None,
) -> bytes:
    payload = bytearray(
        struct.pack(
            "<4sHHQQQiiiiiHH",
            b"CRTF",
            version,
            0,
            7,
            11,
            3,
            42,
            -1,
            0,
            0,
            0,
            2,
            len(objects),
        )
    )
    for owner in range(2):
        payload.extend(
            struct.pack(
                "<iiiIIH6Bi",
                owner,
                50_000,
                0,
                0,
                owner + 1,
                0b1101,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
            )
        )
    payload.extend(b"".join(objects))
    if version in (3, 4, 5, 6):
        payload.extend(
            struct.pack(
                "<IQQQQQH",
                0b111,
                1,
                1,
                1 + len(combat_events),
                0,
                0,
                len(combat_events),
            )
        )
        for event in combat_events:
            payload.extend(event)
    elif version != 1 or combat_events:
        raise ValueError("legacy capture fixture cannot contain combat events")
    payload.extend(
        struct.pack(
            "<IQQQQQH",
            0b111,
            1,
            1,
            1,
            0,
            0,
            0,
        )
    )
    if version >= 4:
        payload.extend(struct.pack("<IQ", 0b111, 0))
    if version >= 5:
        payload.extend(_json_families() if families is None else families)
    return bytes(payload)


def test_decode_training_capture_builds_bound_snapshot() -> None:
    capture = decode_training_capture(_capture())

    assert capture.ordinary["tick"] == 42
    assert capture.ordinary["generation"] == 7
    assert capture.ordinary["stateEpoch"] == 11
    assert capture.ordinary["players"][1]["elixir"] == 5.0
    assert capture.snapshot.tick == 42
    assert capture.snapshot.objects == ()
    assert capture.snapshot.players[0].ability_runtime == ()
    assert capture.snapshot.players[1].evolution_runtime == ()
    assert capture.snapshot.combat_events.complete
    assert capture.snapshot.combat_events.events == ()
    assert capture.snapshot.tower_troop_runtime is not None
    assert capture.snapshot.tower_troop_runtime.complete




def test_decode_training_capture_preserves_both_tower_troop_runtimes() -> None:
    dagger = _tower_object(
        slot=0,
        native_id=7001,
        owner=0,
        kind=1,
        runtime_values=(3, 8, 450, 900, 0, 0, 0, 0, -1),
    )
    chef = _tower_object(
        slot=1,
        native_id=7002,
        owner=1,
        kind=2,
        runtime_values=(0, 0, 0, 0, 2000, 7000, 230_000, 460_000, 350),
        target_native_id=7001,
    )

    capture = decode_training_capture(_capture(objects=(dagger, chef)))

    dagger_runtime = capture.snapshot.objects[0].tower_troop_runtime  # type: ignore[union-attr]
    chef_runtime = capture.snapshot.objects[1].tower_troop_runtime  # type: ignore[union-attr]
    assert dagger_runtime is not None
    assert dagger_runtime.charge_count == 3
    assert chef_runtime is not None
    assert chef_runtime.cooking_contribution == 230_000
    assert chef_runtime.target_native_object_id == 7001


def test_decode_training_capture_preserves_royal_chef_wasted_throw() -> None:
    chef = _tower_object(
        slot=0,
        native_id=7001,
        owner=0,
        kind=2,
        runtime_values=(0, 0, 0, 0, 0, 7000, 460_460, 460_000, -50),
    )

    capture = decode_training_capture(_capture(objects=(chef,)))

    runtime = capture.snapshot.objects[0].tower_troop_runtime  # type: ignore[union-attr]
    assert runtime is not None
    assert runtime.throw_delay_remaining_ms == -50
    assert runtime.target_native_object_id is None


def test_decode_training_capture_keeps_legacy_v1_fail_closed_by_default() -> None:
    with pytest.raises(
        TrainingCaptureError,
        match=r"unsupported training capture magic/version: b'CRTF'/1",
    ):
        decode_training_capture(_capture(version=1))




def test_decode_training_capture_preserves_reward_combat_facts() -> None:
    spell = _fact(native_id=17, owner=0, card_id=26000024)
    target = _fact(native_id=23, owner=1, card_id=26000042)
    card_parameter = (4 << 28) | (7 << 22)
    deployment = struct.pack(
        "<B3xQiIIIiii",
        1,
        9,
        0,
        26000024,
        26000024,
        card_parameter,
        6,
        4,
        0,
    )
    capture = decode_training_capture(
        _capture(
            (
                _event(
                    sequence=1,
                    kind=5,
                    hook_offset=0xF8C50C,
                    deployment=deployment,
                    target=spell,
                    projectile=spell,
                ),
                _event(
                    sequence=2,
                    kind=1,
                    hook_offset=0xFCB29C,
                    target=target,
                    immediate_source=spell,
                    source=spell,
                    projectile=spell,
                    requested_amount=100,
                    actual_amount=100,
                    pre_hp=1000,
                    post_hp=900,
                ),
            )
        )
    )

    events = capture.snapshot.combat_events.events
    assert [event.kind for event in events] == [
        "projectile_spawn",
        "damage",
    ]
    assert events[0].deployment_context is not None
    assert events[0].deployment_context.played_card_global_id == 26000024
    assert events[1].source.native_object_id == 17
    assert events[1].target.card_id == 26000042
    assert events[1].actual_amount == 100


def test_decode_training_capture_accepts_exact_mirror_deployment() -> None:
    deployment = struct.pack(
        "<B3xQiIIIiii",
        1,
        9,
        0,
        28_000_006,
        26_000_014,
        (5 << 28) | (3 << 22),
        2,
        5,
        0,
    )

    capture = decode_training_capture(
        _capture(
            (
                _event(
                    sequence=1,
                    kind=13,
                    hook_offset=0xF9EA68,
                    deployment=deployment,
                ),
            )
        )
    )

    context = capture.snapshot.combat_events.events[0].deployment_context
    assert context is not None
    assert context.played_card_global_id == 28_000_006
    assert context.effective_card_global_id == 26_000_014
    assert context.cost == 5


def test_decode_training_capture_rejects_nonmirror_basic_identity_change() -> None:
    deployment = struct.pack(
        "<B3xQiIIIiii",
        1,
        9,
        0,
        28_000_005,
        26_000_014,
        (5 << 28) | (3 << 22),
        2,
        5,
        0,
    )

    with pytest.raises(TrainingCaptureError, match="descriptor is invalid"):
        decode_training_capture(
            _capture(
                (
                    _event(
                        sequence=1,
                        kind=13,
                        hook_offset=0xF9EA68,
                        deployment=deployment,
                    ),
                )
            )
        )


def test_decode_training_capture_rejects_combat_sequence_gap() -> None:
    with pytest.raises(
        TrainingCaptureError,
        match=r"combat event\[1\] is invalid",
    ):
        decode_training_capture(
            _capture(
                (
                    _event(
                        sequence=1,
                        kind=13,
                        hook_offset=0xF9EA68,
                        deployment=struct.pack(
                            "<B3xQiIIIiii",
                            1,
                            1,
                            0,
                            26000004,
                            26000004,
                            (7 << 22),
                            6,
                            0,
                            0,
                        ),
                    ),
                    _event(
                        sequence=3,
                        kind=12,
                        hook_offset=0xF8EED8,
                        projectile=_fact(
                            native_id=17,
                            owner=0,
                            card_id=26000004,
                        ),
                    ),
                )
            )
        )


def test_decode_training_capture_rejects_trailing_bytes() -> None:
    with pytest.raises(
        TrainingCaptureError,
        match="trailing bytes",
    ):
        decode_training_capture(_capture() + b"\0")


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_decode_training_capture_rejects_obsolete_protocol(version):
    payload = bytearray(_capture())
    struct.pack_into("<H", payload, 4, version)
    with pytest.raises(TrainingCaptureError, match="unsupported training capture"):
        decode_training_capture(bytes(payload))


def test_decode_training_capture_v5_carries_every_json_runtime_family() -> None:
    snapshot = decode_training_capture(_capture()).snapshot

    for name in (
        "phase_runtime",
        "visibility_runtime",
        "remaining_runtime",
        "tower_troop_runtime",
        "special_movement_runtime",
        "action_movement_runtime",
        "character_state_runtime",
    ):
        assert getattr(snapshot, name) is not None, name
    assert snapshot.remaining_runtime.events == ()
    assert snapshot.phase_runtime.events == ()


def test_decode_training_capture_rejects_family_identity_mismatch() -> None:
    families = _json_families(overrides={"remainingRuntime": {"generation": 8}})
    with pytest.raises(TrainingCaptureError, match="runtime family contradicts"):
        decode_training_capture(_capture(families=families))


def test_decode_training_capture_rejects_missing_family() -> None:
    families = bytearray(_json_families())
    struct.pack_into("<H", families, 0, 6)
    with pytest.raises(TrainingCaptureError, match="JSON family count"):
        decode_training_capture(_capture(families=bytes(families)))


def test_decode_training_capture_rejects_malformed_family_json() -> None:
    families = bytearray(_json_families())
    families[6] = ord("!")  # first byte of the phaseRuntime JSON object
    with pytest.raises(TrainingCaptureError, match="not valid JSON"):
        decode_training_capture(_capture(families=bytes(families)))


def test_decode_training_capture_v6_preserves_projectile_routes_and_destinations() -> None:
    spell = _fact(native_id=17, owner=0, card_id=26000024)
    tower = _fact(native_id=5, owner=0, card_id=-1)
    capture = decode_training_capture(
        _capture(
            (
                _event(
                    sequence=1,
                    kind=5,
                    hook_offset=0xF8C50C,
                    target=spell,
                    projectile=spell,
                    route_source_after=tower,
                    destination_after=(3265, 18315),
                ),
            )
        )
    )

    event = capture.snapshot.combat_events.events[0]
    assert event.destination_after == (3265, 18315)
    assert event.destination_before is None
    assert event.route_source_after.present
    assert event.route_source_after.native_object_id == 5
    assert not event.route_target_after.present


def test_decode_training_capture_reuses_identical_combat_records_only() -> None:
    spell = _fact(native_id=17, owner=0, card_id=26000024)
    first = decode_training_capture(
        _capture((_event(sequence=1, kind=5, hook_offset=0xF8C50C, target=spell, projectile=spell),))
    )
    again = decode_training_capture(
        _capture((_event(sequence=1, kind=5, hook_offset=0xF8C50C, target=spell, projectile=spell),))
    )
    changed = decode_training_capture(
        _capture(
            (
                _event(
                    sequence=1,
                    kind=5,
                    hook_offset=0xF8C50C,
                    target=spell,
                    projectile=spell,
                    destination_after=(1, 2),
                ),
            )
        )
    )

    assert again.snapshot.combat_events.events[0] is first.snapshot.combat_events.events[0]
    assert changed.snapshot.combat_events.events[0] == replace(
        first.snapshot.combat_events.events[0], destination_after=(1, 2)
    )


def _mechanic_object(*, slot: int, native_id: int, runtimes: dict) -> bytes:
    encoded = json.dumps(runtimes).encode()
    return b"".join(
        (
            struct.pack(
                "<IiI15i",
                (1 << 9) | (1 << 11),
                slot,
                native_id,
                0,
                26_000_000 + slot,
                9000,
                16000,
                9000,
                16000,
                native_id,
                -1,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                _NONE_I32,
                -1,
            ),
            struct.pack("<I", 26_000_000 + slot),
            struct.pack("<I", len(encoded)),
            encoded,
            struct.pack("<H", 0),
        )
    )


def test_decode_training_capture_v6_parses_mechanic_runtimes_with_json_contract() -> None:
    resource = {
        "kind": "extra_spawn_accumulator",
        "currentRaw": 30,
        "capacityRaw": 100,
        "baseRaw": 100,
        "limitRaw": 200,
        "normalized": 0.3,
        "status": "authoritative",
    }
    runtimes = {
        "entityResourceRuntime": resource,
        "periodicAttackModifierRuntime": None,
        "captureRuntime": None,
        "thresholdRelocationRuntime": None,
    }

    capture = decode_training_capture(
        _capture(objects=(_mechanic_object(slot=0, native_id=4001, runtimes=runtimes),))
    )

    item = capture.snapshot.objects[0]
    assert item is not None
    assert item.entity_resource_runtime is not None
    assert item.entity_resource_runtime.current_raw == 30
    assert item.entity_resource_runtime.normalized == 0.3
    assert item.capture_runtime is None
    assert item.active_effects is None

    with pytest.raises(TrainingCaptureError, match="mechanic runtimes contradict"):
        decode_training_capture(
            _capture(objects=(_mechanic_object(slot=0, native_id=4001, runtimes={"captureRuntime": None}),))
        )
