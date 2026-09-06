"""Fail-closed binding between a live runner process and a RulesetManifest."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import ContractError, RunnerAttestationV1
from .ruleset import RulesetManifestV1


class RunnerAttestationError(RuntimeError):
    """Raised when a live native process does not match its declared ruleset."""


def _unique_hash(files: Mapping[str, str], basename: str, *, group: str) -> str:
    matches = [str(digest) for name, digest in files.items() if Path(str(name)).name == basename]
    if len(matches) != 1:
        raise RunnerAttestationError(f"ruleset {group!r} must bind exactly one {basename}; found {len(matches)}")
    return matches[0]


def attestation_from_response(value: Mapping[str, Any]) -> RunnerAttestationV1:
    raw = value.get("attestation", value)
    if not isinstance(raw, Mapping):
        raise RunnerAttestationError("native runner response has no attestation object")
    try:
        return RunnerAttestationV1.from_mapping(raw)
    except (ContractError, TypeError, ValueError) as error:
        raise RunnerAttestationError(f"native runner returned an invalid attestation: {error}") from error


def verify_runner_attestation(
    attestation: RunnerAttestationV1 | Mapping[str, Any], manifest: RulesetManifestV1
) -> RunnerAttestationV1:
    """Require exact live hashes from the process represented by ``manifest``."""

    actual = attestation if isinstance(attestation, RunnerAttestationV1) else attestation_from_response(attestation)
    if not actual.production_ready:
        raise RunnerAttestationError("native runner is not production-ready: content, hook, or hash gate failed")
    if actual.package_name != "nullsroyale.rel.free" or actual.abi != "arm64-v8a":
        raise RunnerAttestationError(f"native runner package/ABI mismatch: {actual.package_name!r}/{actual.abi!r}")

    expected = {
        "libg_sha256": _unique_hash(manifest.files.get("engine", {}), "libg.so", group="engine"),
        "probe_sha256": _unique_hash(manifest.files.get("simulator", {}), "libcrprobe.so", group="simulator"),
        "content_fingerprint_sha256": _unique_hash(manifest.files.get("data", {}), "fingerprint.json", group="data"),
        "content_assets_sha256": _unique_hash(manifest.files.get("visual", {}), "assets.scdb", group="visual"),
        "content_version": str(manifest.config.get("runtime_content_version", "")),
        "content_manifest_sha1": str(manifest.config.get("runtime_content_manifest_sha1", "")),
        "libg_build_id": str(manifest.config.get("runtime_libg_build_id", "")),
    }
    mismatches = [
        f"{name}: expected {expected_value}, got {getattr(actual, name)}"
        for name, expected_value in expected.items()
        if getattr(actual, name) != expected_value
    ]
    visual_fingerprint = [
        str(digest)
        for name, digest in manifest.files.get("visual", {}).items()
        if Path(str(name)).name == "fingerprint.json"
    ]
    if visual_fingerprint and visual_fingerprint != [expected["content_fingerprint_sha256"]]:
        mismatches.append("ruleset visual/data fingerprint identities are missing or inconsistent")
    if mismatches:
        raise RunnerAttestationError("native runner does not match RulesetManifest: " + "; ".join(mismatches))
    return actual


def evidence_compatible_runner_attestations(left: RunnerAttestationV1, right: RunnerAttestationV1) -> bool:
    """Return whether two attestations may share incremental test evidence.

    Evidence compatibility is deliberately narrower than runtime identity:
    only the probe binary and its derived attestation digest may differ. The
    engine, content, package, ABI, protocol, capabilities, hook state, and
    production-readiness fields must remain byte-for-byte identical. Each
    evidence source still retains its own complete attestation.
    """

    if not left.production_ready or not right.production_ready:
        return False
    ignored = {"attestation_digest", "probe_sha256"}
    left_fields = {key: value for key, value in left.to_dict().items() if key not in ignored}
    right_fields = {key: value for key, value in right.to_dict().items() if key not in ignored}
    return left_fields == right_fields
