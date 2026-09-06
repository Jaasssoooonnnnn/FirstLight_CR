from __future__ import annotations

from native_runner.paths import PACKAGE_ROOT
from native_runner.tests.asset_helpers import user_apk_bytes

import hashlib
import pytest


PROBE = PACKAGE_ROOT / "probe"
LAYOUT = PROBE / "character_state_runtime_layout.h"
TELEMETRY = PROBE / "character_state_runtime_telemetry.inc"
SOURCE = PROBE / "cr_replay_probe.cpp"


def test_character_state_exact_build_fingerprint() -> None:
    data = user_apk_bytes("lib/arm64-v8a/libg.so")
    assert hashlib.sha256(data).hexdigest() == (
        "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783"
    )
    assert data[0xF18654 : 0xF18654 + 16].hex() == (
        "ffc303d1fd7b09a9fc6f0aa9fa670ba9"
    )

def test_character_state_source_wiring() -> None:
    layout = LAYOUT.read_text(encoding="utf-8")
    for token in (
        'kSchema = "native-character-state-runtime.v1"',
        "kStateSetterOffset = 0x00f18654",
        "kCharacterStateOffset = 0x11c",
        "kMinimumState = 0",
        "kMaximumState = 16",
        "kChampionGuardCardId = 26000115",
    ):
        assert token in layout

    telemetry = TELEMETRY.read_text(encoding="utf-8")
    for token in (
        "character_state_setter_hook",
        "source_before.object_kind == 5",
        "committed_state != requested_state",
        "same_character_state_entity",
        "install_character_state_runtime_hook",
        "native_character_state_transition",
    ):
        assert token in telemetry
    assert "does not name them as dash/jump/warp/effect stages" in telemetry

    source = SOURCE.read_text(encoding="utf-8")
    assert '#include "character_state_runtime_telemetry.inc"' in source
    assert (
        "begin_character_state_runtime_epoch(generation, state_epoch)"
        in source
    )
    assert '\\"characterStateRuntime\\":' in source
    assert "install_character_state_runtime_hook(libg_base)" in source
