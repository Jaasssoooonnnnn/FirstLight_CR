from __future__ import annotations

from dataclasses import replace

import pytest

from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.contracts import ContractError
from native_runner.projectile_catalog import (
    NativeProjectileCatalogV1,
    build_projectile_catalog,
)


def _catalog() -> NativeProjectileCatalogV1:
    return build_projectile_catalog(build_static_card_logic_catalog())


def test_all_static_projectiles_have_exact_uint32_global_ids() -> None:
    catalog = _catalog()
    assert len(catalog.projectiles) == 140
    assert len(catalog.by_name) == 140
    assert len(catalog.by_global_id) == 140
    assert catalog.summary["explicit_oldformat_global_id_count"] == 96
    assert catalog.summary["fnv1a_global_id_count"] == 44
    assert catalog.summary["configured_homing_count"] == 5
    assert catalog.summary["runtime_active_homing_available"] is False
    assert catalog.summary["projectiles_with_unresolved_semantics"] == 0

    registry = catalog.global_id_registry
    assert registry["explicit_projectile_count"] == 96
    assert registry["fallback_algorithm"] == "fnv1a32_utf8_type_plus_name"
    assert registry["fallback_prefix"] == "Projectile"
    assert registry["fallback_offset_basis"] == 0x811C9DC5
    assert registry["fallback_prime"] == 0x01000193
    for definition in catalog.projectiles:
        assert definition.node_id == f"PROJECTILE.{definition.projectile_name}"
        assert definition.global_id_key == f"Projectile{definition.projectile_name}"
        assert definition.raw_record
        assert definition.resolved_record
        assert definition.source_records
        assert 1 <= definition.projectile_global_id <= 0xFFFFFFFF
        if definition.projectile_name in registry["projectile_entries"]:
            assert definition.global_id_strategy == "explicit_oldformat_registry"
            assert (
                definition.projectile_global_id
                == registry["projectile_entries"][definition.projectile_name]
            )
        else:
            assert definition.global_id_strategy == "fnv1a_type_name"


def test_configured_homing_is_data_driven_and_not_live_active_state() -> None:
    catalog = _catalog()
    expected = {
        "EliteArcherArrow": 10_000_043,
        "SuperEliteArcherArrow": 10_000_055,
        "FishermanProjectile": 10_000_064,
        "SuperArcherChargeArrow": 10_000_075,
        "EliteArcherArrow_Chess": 10_000_087,
    }
    configured = {
        item.projectile_name: item.projectile_global_id
        for item in catalog.projectiles
        if item.configured_homing
    }
    assert configured == expected
    for name in expected:
        definition = catalog.by_name[name]
        assert definition.homing_time_ms == 100
        assert definition.homing_min_distance == 5000

    runtime = catalog.resolve_runtime_projectile(
        {"projectileDataGlobalId": expected["FishermanProjectile"]}
    )
    assert runtime["definitionStatus"] == "resolved"
    assert runtime["projectileName"] == "FishermanProjectile"
    assert runtime["configuredHoming"] is True
    assert runtime["activeHoming"] is None
    assert (
        runtime["provenance"]["activeHoming"]
        == "unavailable_secondary_homing_target_not_exposed"
    )


def test_inherited_projectile_semantics_are_resolved_without_flattening_raw() -> None:
    catalog = _catalog()
    evolved = catalog.by_name["Archer_EV1_Arrow"]
    assert evolved.raw_record == {"Base": "ArcherArrow"}
    assert evolved.resolved_record["Name"] == "ArcherArrow"
    assert evolved.resolved_record["Base"] == "ArcherArrow"
    assert evolved.resolved_record["Speed"] == 600
    assert evolved.inheritance_chain == ("ArcherArrow", "Archer_EV1_Arrow")


def test_runtime_projectile_unknown_identity_fails_closed() -> None:
    runtime = _catalog().resolve_runtime_projectile(
        {"projectileDataGlobalId": 4_000_000_000}
    )
    assert runtime["definitionStatus"] == "unresolved"
    assert runtime["activeHoming"] is None
    assert "configuredHoming" not in runtime
    with pytest.raises(ContractError, match="global ID"):
        _catalog().resolve_runtime_projectile({"projectileDataGlobalId": True})


def test_projectile_catalog_round_trip_and_tamper_rejection(tmp_path) -> None:
    catalog = _catalog()
    path = catalog.save_content_addressed(tmp_path)
    loaded = NativeProjectileCatalogV1.load(path)
    assert loaded == catalog
    assert loaded.catalog_id == catalog.catalog_id
    assert catalog.save_content_addressed(tmp_path) == path

    definition = catalog.by_name["Archer_EV1_Arrow"]
    with pytest.raises(ContractError, match="FNV-1a"):
        replace(definition, projectile_global_id=definition.projectile_global_id + 1)

    value = catalog.to_dict()
    value["global_id_registry"]["fallback_prefix"] = "Buff"
    with pytest.raises(ContractError, match="registry identity"):
        NativeProjectileCatalogV1.from_mapping(value)
