from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from native_runner.card_specs import (
    COMPOUND_DECLARED,
    FIELD_APPLICABILITY_VERSION,
    RAW_STATIC_ONLY,
    CardSpecCatalog,
    build_card_catalog,
)
from native_runner.contracts import (
    CardKind,
    ContractError,
    SemanticEvidenceLevel,
    TargetKind,
)
from native_runner.match_factory import MatchConfig
from native_runner.normal_form_evidence import (
    NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID,
    NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID,
)
from native_runner.ruleset import RulesetManifestV1, build_ruleset_manifest


from native_runner.paths import WORKSPACE_ROOT as WORKSPACE, PACKAGE_ROOT


def test_match_config_rejects_duplicate_cards() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        MatchConfig(deck0=(26000000,) * 8)


def test_ruleset_binds_the_configured_local_build_probe(tmp_path, monkeypatch):
    import hashlib

    probe = tmp_path / "libcrprobe.so"
    probe.write_bytes(b"configured local probe")
    monkeypatch.setenv("CR_PROBE", str(probe))
    manifest = build_ruleset_manifest(WORKSPACE)
    assert manifest.files["simulator"][probe.resolve().as_posix()] == hashlib.sha256(probe.read_bytes()).hexdigest()


def test_disabled_character_cards_are_statically_hidden_placeholders() -> None:
    catalog = build_card_catalog(WORKSPACE)
    for card_id, name in (
        (26_000_076, "CHAR_DISABLED_1"),
        (26_000_088, "CHAR_DISABLED_3"),
        (26_000_089, "CHAR_DISABLED_4"),
        (26_000_090, "CHAR_DISABLED_5"),
        (26_000_091, "CHAR_DISABLED_6"),
        (26_000_092, "CHAR_DISABLED_7"),
    ):
        spec = catalog.by_id[card_id]
        assert spec.name == name
        assert spec.attributes["NotInUse"] is True
        assert spec.attributes["NotVisible"] is True
        # The fallback Knight summon is evidence of an alias, not playable
        # card mechanics; runtime goldens retain this distinction.
        assert spec.attributes["SummonCharacter"] == "Knight"


def test_workspace_catalog_uses_engine_card_id_rows_and_explicit_unknowns(tmp_path: Path) -> None:
    catalog = build_card_catalog(WORKSPACE)
    assert len(catalog.specs) >= 151
    expected = {
        26000000: ("Knight", 3.0),
        26000018: ("MiniPekka", 4.0),
        26000024: ("RoyalGiant", 6.0),
        26000044: ("Hunter", 4.0),
        26000061: ("Fisherman", 3.0),
        26000084: ("ElectroSpirit", 1.0),
        28000000: ("Fireball", 4.0),
        28000011: ("Log", 2.0),
    }
    for card_id, (name, cost) in expected.items():
        spec = catalog.by_id[card_id]
        assert (spec.name, spec.elixir_cost) == (name, cost)
        assert spec.residual_key == f"card:{card_id}"
        assert spec.residual_init == 0.0
    knight = catalog.by_id[26000000]
    assert knight.kind == CardKind.TROOP
    assert (knight.hitpoints, knight.damage, knight.hit_speed_ms) == (690.0, 79.0, 1200)
    assert (knight.range_tiles, knight.move_speed, knight.deploy_time_ms) == (1.2, 60.0, 1000)
    assert knight.shield_hitpoints is None
    assert knight.ability_ids == ("Knight_hero_Ability",)
    knight_ability = next(
        ability
        for ability in catalog.abilities
        if ability.ability_id == "Knight_hero_Ability"
    )
    assert knight_ability.source_card_id == 26000000
    assert knight_ability.elixir_cost == 2.0
    assert knight_ability.cooldown_ms is None
    assert knight_ability.charges == 1
    assert "Cooldown" not in knight_ability.attributes["definition"]
    assert knight_ability.cast_time_ms == 1_200
    assert any(
        effect.custom_op == "Knight_hero_OnAbilityActivationGroup"
        for effect in knight_ability.effect_graph
    )
    mini_pekka = catalog.by_id[26000018]
    assert (mini_pekka.hitpoints, mini_pekka.damage, mini_pekka.hit_speed_ms) == (543.0, 295.0, 1600)
    fireball = catalog.by_id[28000000]
    assert (fireball.damage, fireball.radius_tiles, fireball.projectile_speed) == (269.0, 2.5, 600)
    heal_spirit = catalog.by_id[28_000_016]
    assert heal_spirit.attributes["SpellAsDeploy"] is True
    assert heal_spirit.attributes["SummonCharacter"] == "HealSpirit"
    assert heal_spirit.target_schema.placement_mask_key == "own_deployment_zone"
    assert catalog.by_id[28_000_011].target_schema.placement_mask_key == "full_arena"
    evolutions = [spec.evolution for spec in catalog.specs if spec.evolution is not None]
    assert len(evolutions) == 42
    assert all(item.cycle_required in (1, 2) for item in evolutions)
    assert all(item.effects for item in evolutions)
    assert catalog.by_id[26000000].evolution.evolution_form_id == "Knight_EV1"
    assert catalog.by_id[26000001].evolution.cycle_required == 2  # Archers
    assert catalog.by_id[26000004].evolution.cycle_required == 1  # P.E.K.K.A
    assert any(name.endswith("spells_evolved.csv") for name in catalog.source_files)
    archer_queen = catalog.by_id[26000072]
    assert archer_queen.kind == CardKind.HERO
    assert archer_queen.ability_ids == ("ArcherQueenRapid",)
    magic_archer = catalog.by_id[26000062]
    assert magic_archer.ability_ids == ("EliteArcherHero_Ability",)
    magic_archer_ability = next(
        ability
        for ability in catalog.abilities
        if ability.ability_id == "EliteArcherHero_Ability"
    )
    assert magic_archer_ability.source_card_id == 26000062
    assert (
        magic_archer_ability.elixir_cost,
        magic_archer_ability.cooldown_ms,
        magic_archer_ability.cast_time_ms,
    ) == (2.0, None, 950)
    abilities_by_id = {ability.ability_id: ability for ability in catalog.abilities}
    for card_id, binding in NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID.items():
        spec = catalog.by_id[card_id]
        ability = abilities_by_id[binding.ability_id]
        resolved_form = spec.attributes["resolved_hero_form"]

        assert spec.ability_ids == (binding.ability_id,)
        assert resolved_form["form_id"] == binding.hero_form_id
        assert resolved_form["ability_id"] == binding.ability_id
        assert ability.source_card_id == card_id
        assert ability.elixir_cost is not None
        assert ability.charges == 1
        if ability.ability_id == "MegaMinion_Teleport_Ability":
            # This value remains explicit in the authoritative runtime file;
            # MaxCharges, not omission, makes the ability single-use.
            assert ability.cooldown_ms == 1_000
        else:
            assert ability.cooldown_ms is None
        assert ability.cast_time_ms is not None
        assert ability.target_schema.allowed == (TargetKind.NONE,)
        assert ability.effect_graph
        assert ability.unknown_fields == ()
    for card_id, binding in NORMAL_MODE_DIRECT_HERO_BINDINGS_BY_CARD_ID.items():
        spec = catalog.by_id[card_id]
        ability = abilities_by_id[binding.ability_id]
        resolved = spec.attributes["resolved_direct_hero"]

        assert spec.kind is CardKind.HERO
        assert spec.ability_ids == (binding.ability_id,)
        assert resolved["ability_id"] == binding.ability_id
        assert resolved["ability_carrier"] == binding.carrier_name
        assert ability.source_card_id == card_id
        assert ability.elixir_cost is not None and ability.elixir_cost > 0
        if ability.ability_id == "BossBandit_ability":
            assert (ability.charges, ability.cooldown_ms) == (2, 3_000)
        else:
            assert (ability.charges, ability.cooldown_ms) == (1, None)
        assert ability.cast_time_ms is not None
        assert ability.target_schema.allowed == (TargetKind.NONE,)
        assert ability.effect_graph
        assert ability.unknown_fields == ()
    assert catalog.by_id[26_000_099].kind is CardKind.HERO
    assert catalog.by_id[26_000_099].ability_ids == ("goblinstein_ability",)
    assert catalog.by_id[26_000_101].kind is CardKind.TROOP
    assert catalog.by_id[26_000_101].ability_ids == ()
    assert len(catalog.abilities) == 24
    zap = catalog.by_id[28000008]
    assert (zap.damage, zap.radius_tiles) == (75.0, 2.5)
    assert any(
        mechanic.effect == "Damage"
        and mechanic.parameters["source"] == "resolved_area_effect"
        for mechanic in zap.mechanics
    )
    assert any(
        mechanic.custom_op == "ApplyBuff"
        and mechanic.parameters["buff"] == "ZapFreeze"
        and mechanic.parameters["duration_raw"] == 500
        for mechanic in zap.mechanics
    )
    assert any(name.endswith("characters/knight.toml") for name in catalog.source_files)
    assert any(name.endswith("characters/minipekka.toml") for name in catalog.source_files)
    assert any(name.endswith("characters/fireballspell.toml") for name in catalog.source_files)

    saved = catalog.save(tmp_path / "cards.json")
    assert catalog.save(saved) == saved
    restored = CardSpecCatalog.load(saved)
    assert restored.specs_hash == catalog.specs_hash
    assert restored.to_json() == catalog.to_json()
    missing_catalog_version = catalog.to_dict()
    missing_catalog_version.pop("version")
    with pytest.raises(ContractError, match="unsupported card catalog version"):
        CardSpecCatalog.from_mapping(missing_catalog_version)
    with pytest.raises(FileExistsError, match="immutable card catalog"):
        replace(catalog, source_release="different").save(saved)


def test_nullable_card_fields_distinguish_unknown_from_not_applicable() -> None:
    catalog = build_card_catalog(WORKSPACE)

    for spec in catalog.specs:
        categorical = spec.categorical_features
        applicability = categorical["field_applicability"]
        assert categorical["field_applicability_version"] == (
            FIELD_APPLICABILITY_VERSION
        )
        assert set(applicability) == set(spec.numeric_features)
        assert all(item["evidence"] for item in applicability.values())
        normalized_unknowns = {
            field_name
            for field_name, item in applicability.items()
            if item["status"] == SemanticEvidenceLevel.UNKNOWN.value
        }
        assert normalized_unknowns == (
            set(spec.unknown_fields) & set(spec.numeric_features)
        )
        assert set(categorical["not_applicable"]) == {
            field_name
            for field_name, item in applicability.items()
            if item["status"] == SemanticEvidenceLevel.NOT_APPLICABLE.value
        }
        assert set(categorical["compound_declared"]) == {
            field_name
            for field_name, item in applicability.items()
            if item["status"] == COMPOUND_DECLARED
        }
        assert set(categorical["raw_static_only"]) == {
            field_name
            for field_name, item in applicability.items()
            if item["status"] == RAW_STATIC_ONLY
        }

    knight = next(item for item in catalog.specs if item.name == "Knight")
    for field_name in (
        "duration_ms", "knockback", "projectile_speed", "radius_tiles",
        "shield_hitpoints",
    ):
        assert field_name not in knight.unknown_fields
        assert knight.categorical_features["field_applicability"][field_name][
            "status"
        ] == SemanticEvidenceLevel.NOT_APPLICABLE.value

    cannon = next(item for item in catalog.specs if item.name == "Cannon")
    assert cannon.duration_ms == 30_000
    assert cannon.categorical_features["field_applicability"]["duration_ms"][
        "status"
    ] == SemanticEvidenceLevel.STATIC_DECLARED.value
    assert cannon.categorical_features["field_applicability"]["move_speed"][
        "status"
    ] == SemanticEvidenceLevel.NOT_APPLICABLE.value

    fireball = next(item for item in catalog.specs if item.name == "Fireball")
    for field_name in ("projectile_speed", "radius_tiles", "knockback"):
        assert fireball.categorical_features["field_applicability"][field_name][
            "status"
        ] == SemanticEvidenceLevel.STATIC_DECLARED.value

    assert {spec.name for spec in catalog.specs if spec.unknown_fields} == {
        "IceWizard", "ElectroWizard"
    }

    three_musketeers = next(
        item for item in catalog.specs if item.name == "ThreeMusketeers"
    )
    assert three_musketeers.count == 3
    assert three_musketeers.summoned_forms == (
        "ThreeMusketeer_Rework_Character_1",
        "ThreeMusketeer_Rework_Character_2",
        "ThreeMusketeer_Rework_Character_3",
    )
    assert len(three_musketeers.attributes["resolved_summoned_forms"]) == 3

    for card_name in ("IceWizard", "ElectroWizard"):
        spec = next(item for item in catalog.specs if item.name == card_name)
        assert spec.summoned_forms == (card_name,)
        assert spec.count == 1
        assert spec.hitpoints is None
        assert "hitpoints" in spec.unknown_fields
        assert spec.attributes["deploy_form_resolution"]["method"] == (
            "typed_deploy_aeo_action_chain"
        )

    tri_wizards = next(
        item for item in catalog.specs if item.name == "TriWizards"
    )
    assert tri_wizards.summoned_forms == (
        "TriWizard", "ElectroWizard", "IceWizard"
    )
    assert tri_wizards.count is None
    assert tri_wizards.categorical_features["field_applicability"]["count"][
        "status"
    ] == RAW_STATIC_ONLY
    assert "SpawnInterval=300:periodic" in " ".join(
        tri_wizards.attributes["deploy_form_resolution"]["evidence"]
    )

    expected_compound_counts = {
        "GoblinGang": 6,
        "Rascals": 3,
        "Goblinstein": 2,
    }
    for card_name, expected_count in expected_compound_counts.items():
        spec = next(item for item in catalog.specs if item.name == card_name)
        assert spec.count == expected_count
        assert spec.categorical_features["field_applicability"]["hitpoints"][
            "status"
        ] == COMPOUND_DECLARED
        assert "hitpoints" not in spec.unknown_fields

    recruits_chess = next(
        item for item in catalog.specs if item.name == "RoyalRecruits_Chess"
    )
    recruits_speed = recruits_chess.categorical_features[
        "field_applicability"
    ]["move_speed"]
    assert recruits_chess.move_speed == 0.0
    assert recruits_speed["status"] == SemanticEvidenceLevel.STATIC_DECLARED.value
    assert "unit.Speed" not in recruits_speed["evidence"]
    assert any(
        item.startswith("native_loader_default:LogicCharacterData.Speed=0")
        for item in recruits_speed["evidence"]
    )
    assert any(
        item.startswith("libg.arm64.sha256:110aa2b5cac391c498645e072b0d8872")
        for item in recruits_speed["evidence"]
    )
    assert "Speed" not in recruits_chess.attributes["resolved_summoned_form"]

    dark_elixir = next(
        item for item in catalog.specs if item.name == "DarkElixir_Bottle"
    )
    dark_hitpoints = dark_elixir.categorical_features[
        "field_applicability"
    ]["hitpoints"]
    assert dark_elixir.hitpoints == 0.0
    assert dark_hitpoints["status"] == SemanticEvidenceLevel.STATIC_DECLARED.value
    assert "unit.Hitpoints" not in dark_hitpoints["evidence"]
    assert any(
        item.startswith("native_loader_default:LogicCharacterData.Hitpoints=0")
        for item in dark_hitpoints["evidence"]
    )
    assert any(
        item == "source_release:nr_15.535.13_release_ff1c6c29"
        for item in dark_hitpoints["evidence"]
    )
    assert "Hitpoints" not in dark_elixir.attributes["resolved_summoned_form"]


def test_ability_max_charges_is_projected_without_card_name_inference() -> None:
    catalog = build_card_catalog(WORKSPACE)
    ability = next(
        item for item in catalog.abilities
        if item.ability_id == "BossBandit_ability"
    )

    assert ability.charges == 2
    assert ability.attributes["definition"]["MaxCharges"] == 2


def test_runtime_file_replacement_drops_removed_august_fields() -> None:
    catalog = build_card_catalog(WORKSPACE)

    barbarians = next(item for item in catalog.specs if item.name == "Barbarians")
    base_unit = barbarians.attributes["resolved_summoned_form"]
    evolved_unit = barbarians.attributes["resolved_evolution_unit"]
    transform = barbarians.evolution.effects[0]
    assert evolved_unit["Hitpoints"] == base_unit["Hitpoints"] == 280
    assert "Hitpoints" not in transform.parameters["explicit_unit_overrides"]

    abilities = {item.ability_id: item for item in catalog.abilities}
    boss = abilities.pop("BossBandit_ability")
    assert (boss.charges, boss.cooldown_ms) == (2, 3_000)
    assert boss.attributes["definition"]["Cooldown"] == 3_000

    assert len(abilities) == 23
    assert all(item.charges == 1 for item in abilities.values())
    mega_minion = abilities.pop("MegaMinion_Teleport_Ability")
    assert mega_minion.cooldown_ms == 1_000
    assert mega_minion.attributes["definition"]["Cooldown"] == 1_000
    assert all(item.cooldown_ms is None for item in abilities.values())
    assert all(
        "Cooldown" not in item.attributes["definition"]
        for item in abilities.values()
    )


def test_ruleset_is_content_addressed_persistent_and_file_verifiable(tmp_path: Path) -> None:
    catalog = build_card_catalog(WORKSPACE)
    extra = tmp_path / "extra-rule.txt"
    extra.write_text("rule-v1", encoding="utf-8")
    manifest = build_ruleset_manifest(
        WORKSPACE,
        catalog=catalog,
        config={"test_marker": 1},
        data_paths=(extra,),
    )
    assert len(manifest.ruleset_id) == 64
    assert manifest.card_specs_hash == catalog.specs_hash
    assert manifest.card_spec_count == len(catalog.specs)
    assert manifest.verify_files(WORKSPACE) == ()
    assert str(WORKSPACE).replace("\\", "/") not in manifest.to_json()
    simulator_files = manifest.files["simulator"]
    package_relative = PACKAGE_ROOT.relative_to(WORKSPACE).as_posix()
    for required in (
        "native_runner/probe/build_probe.ps1",
        "native_runner/probe/combat_event_telemetry.inc",
        "native_runner/probe/phase_runtime_telemetry.inc",
        "native_runner/probe/visibility_runtime_telemetry.inc",
        "native_runner/probe/remaining_runtime_telemetry.inc",
        "native_runner/probe/native_call_arm64.S",
        "native_runner/phase_runtime.py",
        "native_runner/visibility_runtime.py",
        "native_runner/runtime_scope.py",
    ):
        assert required.replace("native_runner/", package_relative + "/", 1) in simulator_files
    ability_path = manifest.config["legal_actions"][
        "ability_activation_native_path"
    ]
    assert ability_path["available"] is True
    assert ability_path["target_arguments"] is False
    assert ability_path["fail_closed"] is True
    assert manifest.config["runtime_match_level_policy"] == {
        "version": "normalized-level-11.v1",
        "level_cap": 11,
        "minimum_card_level": 11,
        "king_tower_level": 11,
    }
    runtime_scope = manifest.config["runtime_evidence_scope"]
    assert runtime_scope["version"] == "normal-mode-runtime-policy-scope.v2"
    assert runtime_scope["publication_scope"] == "normal-mode-only"
    assert runtime_scope["source_card_count"] == 152
    assert runtime_scope["eligible_card_count"] == 122
    assert runtime_scope["inventory_role"] == "topology-capture-only"
    from native_runner.runtime_scope import NORMAL_MODE_CASE_SCOPE_SHA256
    assert runtime_scope["case_scope_id"] == NORMAL_MODE_CASE_SCOPE_SHA256
    assert "normal_mode_case_scope" not in runtime_scope

    path = manifest.save(tmp_path / "manifests")
    restored = RulesetManifestV1.load(path)
    assert restored.ruleset_id == manifest.ruleset_id
    assert restored.to_json() == manifest.to_json()
    assert manifest.save(path) == path

    value = json.loads(path.read_text(encoding="utf-8"))
    value["card_spec_count"] += 1
    with pytest.raises(ContractError, match="ruleset_id integrity"):
        RulesetManifestV1.from_mapping(value)

    missing_manifest_version = manifest.to_dict()
    missing_manifest_version.pop("version")
    with pytest.raises(ContractError, match="unsupported ruleset manifest"):
        RulesetManifestV1.from_mapping(missing_manifest_version)

    missing_manifest_id = manifest.to_dict()
    missing_manifest_id.pop("ruleset_id")
    with pytest.raises(ContractError, match="requires ruleset_id"):
        RulesetManifestV1.from_mapping(missing_manifest_id)

    extra.write_text("rule-v2", encoding="utf-8")
    errors = manifest.verify_files(WORKSPACE, raise_on_error=False)
    assert any("hash mismatch" in error and "extra-rule.txt" in error for error in errors)
