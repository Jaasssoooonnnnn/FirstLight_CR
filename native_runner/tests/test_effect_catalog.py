from __future__ import annotations

from dataclasses import replace

import pytest

from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.contracts import ContractError
from native_runner.effect_catalog import (
    NativeEffectCatalogV1,
    NativeEffectDefinitionV1,
    build_effect_catalog,
)


def _catalog():
    return build_effect_catalog(build_static_card_logic_catalog())


def _application(effect, source_node_id: str):
    return next(
        item
        for item in effect.application_sources
        if item["source_node_id"] == source_node_id
    )


def test_all_static_buffs_have_exact_uint32_global_ids() -> None:
    catalog = _catalog()
    assert len(catalog.effects) == 160
    assert len(catalog.by_name) == 160
    assert len(catalog.by_global_id) == 160
    assert catalog.summary["effects_with_buff_global_id"] == 160
    assert catalog.summary["effects_without_buff_global_id"] == 0
    assert catalog.summary["explicit_oldformat_global_id_count"] == 70
    assert catalog.summary["fnv1a_global_id_count"] == 90
    assert catalog.summary["global_ids_with_high_bit_set"] == 49
    assert len(set(catalog.by_global_id)) == 160
    assert all(1 <= global_id <= 0xFFFFFFFF for global_id in catalog.by_global_id)

    registry = catalog.global_id_registry
    assert registry["explicit_buff_count"] == 70
    assert registry["fallback_algorithm"] == "fnv1a32_utf8_type_plus_name"
    assert registry["fallback_prefix"] == "Buff"
    assert registry["fallback_offset_basis"] == 0x811C9DC5
    assert registry["fallback_prime"] == 0x01000193
    for effect in catalog.effects:
        assert effect.node_id == f"BUFF.{effect.effect_name}"
        assert effect.global_id_key == f"Buff{effect.effect_name}"
        assert effect.raw_record
        assert effect.source_records
        assert effect.mechanic_tags
        assert set(effect.mechanic_evidence) == set(effect.mechanic_tags)
        assert all(effect.mechanic_evidence.values())
        if effect.effect_name in registry["buff_entries"]:
            assert effect.global_id_strategy == "explicit_oldformat_registry"
            assert (
                effect.buff_global_id
                == registry["buff_entries"][effect.effect_name]
            )
        else:
            assert effect.global_id_strategy == "fnv1a_type_name"


def test_inline_void_buffs_have_exact_runtime_identity_and_provenance() -> None:
    effects = _catalog().by_name
    expected_ids = {
        "DarkMagicAOE_Damage_lv3": 2_452_779_686,
        "DarkMagicAOE_Damage_lv2": 2_469_557_305,
        "DarkMagicAOE_Damage_lv1": 2_419_224_448,
    }
    for name, global_id in expected_ids.items():
        effect = effects[name]
        assert effect.buff_global_id == global_id
        assert effect.global_id_strategy == "fnv1a_type_name"
        assert "periodic_damage" in effect.mechanic_tags
        assert effect.modifiers["DamagePerSecond"] > 0
        application = _application(effect, "AEO.DarkMagicAOE")
        assert application["field_path"].startswith(
            "OnStartingAction.SubActions[1].OnDetectedUnitActionList["
        )
        assert application["field_path"].endswith("].SpawnData")
        assert application["application_kind"] == "action_spawn_buff"
        assert application["parameters"]["SpawnType"] == "BuffType"
        assert application["parameters"]["SpawnTime"] == 100

    # These current normal-mode effects are constructed by native mechanics
    # without a string applicator edge, but still have exact static Buff IDs.
    assert effects[
        "Hunter_EV1_bear_trap_snare_no_effect"
    ].buff_global_id == 2_180_748_724
    assert effects["RageDummyBuff"].buff_global_id == 2_778_964_406
    assert effects[
        "Vines_Trap_Snare_No_Effect"
    ].buff_global_id == 2_616_709_981


def test_representative_effect_semantics_and_applicators_are_data_driven() -> None:
    effects = _catalog().by_name
    assert effects["Rage"].buff_global_id == 9_000_000
    assert effects["Freeze"].buff_global_id == 9_000_001
    assert effects["IceWizardSlowDown"].buff_global_id == 9_000_003
    assert effects["ZapFreeze"].buff_global_id == 9_000_004
    assert (
        effects["BabyDragon_EV1_wind_buff_negative"].buff_global_id
        == 2_428_333_743
    )
    assert (
        effects["GoblinDemolisher_ResetTargetBuff"].buff_global_id
        == 3_263_358_259
    )

    rage = effects["Rage"]
    assert {"haste", "rage"}.issubset(rage.mechanic_tags)
    assert rage.modifiers["HitSpeedMultiplier"] == 130
    rage_aeo = _application(rage, "AEO.Rage")
    assert rage_aeo["parameters"]["BuffTime"] == 1000
    assert rage_aeo["parameters"]["LifeDuration"] == 4500

    freeze = effects["Freeze"]
    assert {"incapacitate", "freeze"}.issubset(freeze.mechanic_tags)
    assert freeze.modifiers["SpeedMultiplier"] == -100
    assert _application(freeze, "AEO.Freeze")["parameters"]["BuffTime"] == 4000

    slow = effects["IceWizardSlowDown"]
    assert "slow" in slow.mechanic_tags
    assert slow.modifiers["SpeedMultiplier"] == -30
    assert (
        _application(slow, "PROJECTILE.ice_wizardProjectile")["parameters"][
            "BuffTime"
        ]
        == 2500
    )

    stun = effects["ZapFreeze"]
    assert {"incapacitate", "stun", "attack_interrupt"}.issubset(
        stun.mechanic_tags
    )
    electro_wizard_spawn = _application(stun, "AEO.ElectroWizardZap")
    assert electro_wizard_spawn["parameters"]["BuffTime"] == 500
    assert (
        electro_wizard_spawn["parameters"]["StatsTags"]["BuffTime"]
        == "stun_duration"
    )
    electro_dragon = _application(stun, "PROJECTILE.ElectroDragonProjectile")
    assert electro_dragon["parameters"]["BuffTime"] == 500
    assert electro_dragon["parameters"]["AllowResetTarget"] is True
    assert "boolean_default=1" in " ".join(
        electro_dragon["parameter_provenance"]["AllowResetTarget"]
    )

    reset_target = effects["GoblinDemolisher_ResetTargetBuff"]
    assert "target_lock_control" in reset_target.mechanic_tags
    reset_action = _application(reset_target, "ACTION.ResetTauntEffect")
    assert reset_action["parameters"]["ClassType"] == "ActionTaunt"
    assert reset_action["parameters"]["ValidDuration"] == 50

    # Ronin's updated hit-speed multiplier is a partial slowdown.  The other
    # two -100 fields still prove movement/spawn locks, but no longer prove
    # full incapacitation, and the record name alone cannot prove stun.
    ronin = effects["ronin_reflect_stun_buff"]
    assert {"movement_lock", "slow", "spawn_lock"}.issubset(
        ronin.mechanic_tags
    )
    assert {"attack_lock", "incapacitate", "stun"}.isdisjoint(
        ronin.mechanic_tags
    )
    assert ronin.modifiers["HitSpeedMultiplier"] == -95
    assert ronin.modifiers["SpeedMultiplier"] == -100
    assert ronin.modifiers["SpawnSpeedMultiplier"] == -100
    assert (
        _application(ronin, "ACTION.ronin_reflect_stun")["parameters"][
            "SpawnTime"
        ]
        == 500
    )

    assert "invisibility" in effects["Invisibility"].mechanic_tags
    assert "periodic_damage" in effects["Poison"].mechanic_tags
    assert "death_spawn" in effects["VoodooCurse"].mechanic_tags
    assert "curse" in effects["VoodooCurse"].mechanic_tags


def test_hero_ice_golem_final_blast_is_classified_from_action_graph() -> None:
    effects = _catalog().by_name
    slow = effects["IceGolemiteHero_Slow_Buff_Base"]
    assert "slow" in slow.mechanic_tags
    assert {"freeze", "incapacitate"}.isdisjoint(slow.mechanic_tags)
    assert slow.modifiers["SpeedMultiplier"] == -30

    applications = tuple(
        item
        for item in slow.application_sources
        if item["source_node_id"]
        == "ACTION.IceGolemiteHero_Select_Slow_Buff"
    )
    assert {item["field_path"] for item in applications} == {
        "SubActions[0].SpawnData",
        "SubActions[1].SpawnData",
        "SubActions[2].SpawnData",
        "SubActions[3].SpawnData",
    }
    assert all(
        str(item["target_node_id"]).startswith(
            "EXT.IceGolemiteHero_Slow_Buff_"
        )
        for item in applications
    )
    triggers = {
        (trigger["source_node_id"], trigger["field_path"])
        for item in applications
        for trigger in item["trigger_sources"]
    }
    assert (
        "AEO.IceGolemiteHero_Freeze_AEO",
        "OnHitAction",
    ) in triggers

    # Freeze-named assets remain in the source pack, but the final AEO invokes
    # the slow selector.  The orphaned freeze selector has no final-AEO trigger.
    freeze_selector_applications = tuple(
        item
        for item in effects["Freeze"].application_sources
        if item["source_node_id"]
        == "ACTION.IceGolemiteHero_Select_Freeze_Buff"
    )
    assert freeze_selector_applications
    assert all(
        not item["trigger_sources"] for item in freeze_selector_applications
    )


def test_buff_base_inheritance_is_explicit_and_lossless() -> None:
    effect = _catalog().by_name["Zap_EV1_WithDamage"]
    assert effect.buff_global_id == 3_872_934_510
    assert effect.global_id_strategy == "fnv1a_type_name"
    assert effect.inheritance_chain == ("ZapFreeze", "Zap_EV1_WithDamage")
    assert effect.modifiers["SpeedMultiplier"] == -100
    assert effect.modifiers["DamagePerSecond"] == 75
    assert effect.raw_record["Base"] == "ZapFreeze"
    assert "SpeedMultiplier" not in effect.raw_record


def test_runtime_join_preserves_authoritative_identity_and_fails_closed() -> None:
    catalog = _catalog()
    resolved = catalog.resolve_runtime_effect(
        {
            "name": "Rage",
            "buffGlobalId": 9_000_000,
            "remainingMs": 1450,
            "sourceEntityKey": [0, 12, 0],
        }
    )
    assert resolved["definitionStatus"] == "resolved"
    assert resolved["effectNodeId"] == "BUFF.Rage"
    assert resolved["runtimeIdentity"]["remainingMs"] == 1450
    assert resolved["joinKey"] == "BUFF.buff_global_id+name"
    assert (
        resolved["provenance"]["staticIdentity"]
        == "explicit_oldformat_registry"
    )
    assert resolved["provenance"]["runtimeIdentity"].startswith("authoritative")

    mismatch = catalog.resolve_runtime_effect(
        {
            "name": "Rage",
            "buffGlobalId": 9_000_001,
            "remainingMs": 50,
            "sourceEntityKey": None,
        }
    )
    assert mismatch["definitionStatus"] == "identity_mismatch"
    assert "effectNodeId" not in mismatch
    assert mismatch["provenance"]["staticSemantics"].startswith("unavailable")

    unknown = catalog.resolve_runtime_effect(
        {
            "name": "FuturePatchBuff",
            "buffGlobalId": 99_999_999,
            "remainingMs": 50,
            "sourceEntityKey": None,
        }
    )
    assert unknown["definitionStatus"] == "unresolved"
    assert "mechanicTags" not in unknown

    high_bit = catalog.resolve_runtime_effect(
        {
            "name": "BabyDragon_EV1_wind_buff_negative",
            "buffGlobalId": 2_428_333_743,
            "remainingMs": 100,
            "sourceEntityKey": None,
        }
    )
    assert high_bit["definitionStatus"] == "resolved"
    assert high_bit["provenance"]["staticIdentity"] == "fnv1a_type_name"

    non_expiring = catalog.resolve_runtime_effect(
        {
            "name": "Invisibility",
            "buffGlobalId": 9_000_009,
            "remainingMs": -1,
            "sourceEntityKey": None,
        }
    )
    assert non_expiring["runtimeIdentity"]["remainingMs"] == -1

    with pytest.raises(ContractError, match="global ID is invalid"):
        catalog.resolve_runtime_effect(
            {
                "name": "FuturePatchBuff",
                "buffGlobalId": 0x1_0000_0000,
                "remainingMs": 50,
                "sourceEntityKey": None,
            }
        )


def test_effect_catalog_round_trip_is_canonical(tmp_path) -> None:
    catalog = _catalog()
    path = catalog.save_content_addressed(tmp_path)
    restored = NativeEffectCatalogV1.load(path)
    assert restored.to_json() == catalog.to_json()
    assert restored.catalog_id == catalog.catalog_id
    assert restored.by_global_id[9_000_004].effect_name == "ZapFreeze"
    assert (
        restored.by_global_id[2_428_333_743].effect_name
        == "BabyDragon_EV1_wind_buff_negative"
    )


def test_buff_global_id_integrity_fails_closed() -> None:
    catalog = _catalog()
    rage = catalog.by_name["Rage"]
    hashed = catalog.by_name["BabyDragon_EV1_wind_buff_negative"]

    with pytest.raises(ContractError, match="does not match exact FNV-1a"):
        replace(hashed, buff_global_id=hashed.buff_global_id + 1)

    with pytest.raises(ContractError, match="invalid ID hash key"):
        replace(hashed, global_id_key="BuffWrongName")

    bad_explicit = replace(rage, buff_global_id=9_999_999)
    with pytest.raises(ContractError, match="explicit old-format Buff registry"):
        NativeEffectCatalogV1(
            card_logic_catalog_id=catalog.card_logic_catalog_id,
            criteria_version=catalog.criteria_version,
            global_id_registry=catalog.global_id_registry,
            effects=tuple(
                bad_explicit if item.effect_name == "Rage" else item
                for item in catalog.effects
            ),
        )

    bad_mapping = hashed.to_dict()
    bad_mapping["mechanic_evidence"] = {}
    with pytest.raises(ContractError, match="require non-empty exact evidence"):
        NativeEffectDefinitionV1.from_mapping(bad_mapping)
