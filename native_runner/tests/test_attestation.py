from __future__ import annotations

import unittest

from native_runner.attestation import (
    RunnerAttestationError,
    attestation_from_response,
    verify_runner_attestation,
)
from native_runner.contracts import RunnerAttestationV1, content_hash
from native_runner.ruleset import RulesetManifestV1


def _manifest() -> RulesetManifestV1:
    hashes = [character * 64 for character in "abcdef0123456"]
    return RulesetManifestV1(
        ruleset_version="15.535.84",
        engine_build_hash=hashes[0],
        simulator_build_hash=hashes[1],
        card_schema_hash=hashes[2],
        observation_contract_hash=hashes[3],
        action_contract_hash=hashes[4],
        tick_rng_latency_rules_hash=hashes[5],
        card_specs_hash=hashes[6],
        ability_specs_hash=hashes[7],
        slot_rules_hash=hashes[8],
        legal_action_schema_hash=hashes[9],
        elixir_rules_hash=hashes[10],
        mode_map_hash=hashes[11],
        visual_asset_pack_hash=hashes[12],
        card_spec_count=10,
        contract_versions={"runner_attestation": "runner-attestation.v1"},
        files={
            "engine": {"release/lib/arm64-v8a/libg.so": "1" * 64},
            "simulator": {"native_runner/probe/out/libcrprobe.so": "2" * 64},
            "schema": {},
            "data": {"runtime-update/fingerprint.json": "3" * 64},
            "visual": {
                "runtime-update/fingerprint.json": "3" * 64,
                "runtime-update/assets.scdb": "4" * 64,
            },
        },
        config={
            "runtime_content_version": "15.535.84",
            "runtime_content_manifest_sha1": "5" * 40,
            "runtime_libg_build_id": "6" * 40,
        },
    )


def _attestation(**overrides: object) -> RunnerAttestationV1:
    identity = {
        "abi": "arm64-v8a",
        "capabilities": {
            "headless": True,
            "native_render": True,
            "snapshot_restore": True,
            "structured_observation": True,
            "two_sided_actions": True,
        },
        "content_assets_sha256": "4" * 64,
        "content_fingerprint_sha256": "3" * 64,
        "content_manifest_sha1": "5" * 40,
        "content_version": "15.535.84",
        "hook_install_ok": True,
        "hooks": {
            "game_state_load": True,
            "game_state_step": True,
            "native_replay_runner": True,
            "render_gate": True,
            "replay_json_setter": True,
        },
        "libg_build_id": "6" * 40,
        "libg_sha256": "1" * 64,
        "package_name": "nullsroyale.rel.free",
        "probe_sha256": "2" * 64,
        "protocol_version": "cr-native-control.v1",
        "version": "runner-attestation.v1",
    }
    identity.update(overrides)
    payload = {
        **identity,
        "content_ready": True,
        "content_assets_mapped": True,
        "production_ready": True,
        "attestation_digest": content_hash(identity),
    }
    return RunnerAttestationV1.from_mapping(payload)


class RunnerAttestationTests(unittest.TestCase):
    def test_live_identity_binds_every_manifested_native_file_and_build_id(self) -> None:
        actual = verify_runner_attestation(_attestation(), _manifest())
        self.assertTrue(actual.production_ready)
        self.assertEqual(len(actual.attestation_digest), 64)

    def test_probe_cannot_substitute_a_local_label_for_wrong_process(self) -> None:
        wrong = _attestation(probe_sha256="7" * 64)
        with self.assertRaisesRegex(RunnerAttestationError, "probe_sha256"):
            verify_runner_attestation(wrong, _manifest())

    def test_content_manifest_and_elf_build_id_are_hard_gates(self) -> None:
        for field, value in (
            ("content_manifest_sha1", "7" * 40),
            ("libg_build_id", "8" * 40),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(RunnerAttestationError, field):
                    verify_runner_attestation(
                        _attestation(**{field: value}), _manifest()
                    )

    def test_response_digest_tampering_is_rejected_by_wire_contract(self) -> None:
        payload = _attestation().to_dict()
        payload["attestation_digest"] = "0" * 64
        with self.assertRaisesRegex(RunnerAttestationError, "digest mismatch"):
            attestation_from_response({"ok": True, "attestation": payload})

    def test_nonproduction_attestation_cannot_bind(self) -> None:
        actual = _attestation()
        identity = actual.identity_dict()
        identity["hook_install_ok"] = False
        identity["hooks"] = dict(identity["hooks"])
        disabled = RunnerAttestationV1.from_mapping(
            {
                **identity,
                "content_ready": True,
                "content_assets_mapped": True,
                "production_ready": False,
                "attestation_digest": content_hash(identity),
            }
        )
        with self.assertRaisesRegex(RunnerAttestationError, "not production-ready"):
            verify_runner_attestation(disabled, _manifest())


if __name__ == "__main__":
    unittest.main()
