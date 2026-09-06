from __future__ import annotations

import struct

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
    version: int = 4,
    objects: tuple[bytes, ...] = (),
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
    if version in (3, 4):
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
    if version == 4:
        payload.extend(struct.pack("<IQ", 0b111, 0))
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
                    hook_offset=0xF25B8C,
                    deployment=deployment,
                    target=spell,
                    projectile=spell,
                ),
                _event(
                    sequence=2,
                    kind=1,
                    hook_offset=0xF642C4,
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
                    hook_offset=0xF38A68,
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
                        hook_offset=0xF38A68,
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
                        hook_offset=0xF38A68,
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
                        hook_offset=0xF28470,
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


@pytest.mark.parametrize("version", [2, 3])
def test_decode_training_capture_rejects_obsolete_protocol(version):
    payload = bytearray(_capture())
    struct.pack_into("<H", payload, 4, version)
    with pytest.raises(TrainingCaptureError, match="unsupported training capture"):
        decode_training_capture(bytes(payload))
