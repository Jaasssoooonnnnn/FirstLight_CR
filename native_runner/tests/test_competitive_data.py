"""Pinned-source identity and compiled fact integrity at the production boundary."""
from dataclasses import replace
import hashlib
import json
import os

import pytest

from native_runner import competitive_data
from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.card_specs import build_card_catalog
from native_runner.contracts import ContractError
from native_runner.effect_catalog import build_effect_catalog
from native_runner.projectile_catalog import build_projectile_catalog


def test_compiled_catalog_identities_match_frozen_release():
    cards = build_card_catalog()
    logic = build_static_card_logic_catalog(card_catalog=cards)
    effects = build_effect_catalog(logic)
    projectiles = build_projectile_catalog(logic)
    assert cards.content_hash() == "20b91834ef38ccc1b1664a5a1eb98b595ba3ead69a6bc4e44b0a46326af73a1e"
    assert logic.catalog_id == "c0280c054ee344063cd6d06024895b293a121cf28b078f073df78683937866bd"
    assert effects.catalog_id == "591aeead7043be97129607dcc77596fdf94cc7dd61b678e73c12a4cee8ee313e"
    assert projectiles.catalog_id == "0e4ede430b007eeb8b7c92d02bd41959965d39f76165c51e36d2470823f5f7f0"


def test_compiled_facts_work_without_original_assets(tmp_path):
    assert build_card_catalog(tmp_path).content_hash() == build_card_catalog().content_hash()
    assert not list(tmp_path.iterdir())
    with pytest.raises(ContractError, match="only the pinned competitive"):
        build_card_catalog(source_roots=(tmp_path,))


def test_changed_catalog_cannot_reuse_compiled_graph():
    cards = build_card_catalog()
    with pytest.raises(ContractError, match="card catalog differs"):
        build_static_card_logic_catalog(card_catalog=replace(cards, specs=cards.specs[:-1]))
    logic = build_static_card_logic_catalog(card_catalog=cards)
    changed = replace(logic, warnings=("changed",))
    for builder in (build_effect_catalog, build_projectile_catalog):
        with pytest.raises(ContractError, match="static logic differs"):
            builder(changed)


def test_source_verification_rejects_missing_and_changed_bytes(tmp_path, monkeypatch):
    resource = tmp_path / "resource.csv"
    original = b"competitive facts"
    resource.write_bytes(original)
    monkeypatch.setattr(competitive_data, "_manifest", lambda: {
        "source_files": {"resource.csv": hashlib.sha256(original).hexdigest()},
    })
    competitive_data.validate_competitive_sources(tmp_path)
    original_stat = resource.stat()
    resource.write_bytes(b"x" * len(original))
    os.utime(resource, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        competitive_data.validate_competitive_sources(tmp_path)
    resource.unlink()
    with pytest.raises(ContractError, match="unavailable"):
        competitive_data.validate_competitive_sources(tmp_path)


def test_cached_data_still_checks_file_integrity(tmp_path, monkeypatch):
    raw = json.dumps({"fact": 42}).encode()
    path = tmp_path / "example.json"
    path.write_bytes(raw)
    monkeypatch.setattr(competitive_data, "GENERATED_DATA_ROOT", tmp_path)
    monkeypatch.setattr(competitive_data, "_manifest", lambda: {
        "files": {"example": {"sha256": hashlib.sha256(raw).hexdigest()}},
    })
    assert competitive_data.load_competitive_data("example")["fact"] == 42
    path.write_bytes(b'{"fact": 43, "changed": true}')
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        competitive_data.load_competitive_data("example")


@pytest.mark.parametrize("abort", [False, True])
def test_initialization_reuses_verified_files_then_rechecks_changes(tmp_path, monkeypatch, abort):
    resource = tmp_path / "resource.csv"
    resource.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    calls = []
    file_digest = hashlib.file_digest

    def measured(stream, algorithm):
        calls.append(stream.name)
        return file_digest(stream, algorithm)

    monkeypatch.setattr(hashlib, "file_digest", measured)
    try:
        with competitive_data.competitive_data_initialization():
            competitive_data._verify_file(resource, digest)
            competitive_data._verify_file(resource, digest)
            if abort:
                raise RuntimeError("initialization failed")
    except RuntimeError:
        assert abort
    assert len(calls) == 1

    original_stat = resource.stat()
    resource.write_bytes(b"modified")
    os.utime(resource, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        competitive_data._verify_file(resource, digest)
    with competitive_data.competitive_data_initialization():
        with pytest.raises(ContractError, match="SHA-256 mismatch"):
            competitive_data._verify_file(resource, digest)
    assert len(calls) == 3


def test_initialization_does_not_reuse_failed_or_different_file_checks(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"original")
    second.write_bytes(b"modified")
    digest = hashlib.sha256(b"original").hexdigest()
    with competitive_data.competitive_data_initialization():
        competitive_data._verify_file(first, digest)
        with pytest.raises(ContractError, match="SHA-256 mismatch"):
            competitive_data._verify_file(second, digest)
        second.write_bytes(b"original")
        competitive_data._verify_file(second, digest)
        with pytest.raises(ContractError, match="SHA-256 mismatch"):
            competitive_data._verify_file(first, hashlib.sha256(b"modified").hexdigest())
