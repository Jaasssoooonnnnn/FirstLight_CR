from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT

from pathlib import Path

import pytest

from native_runner.card_logic import (
    StaticCardLogicCatalogV1,
    build_static_card_logic_catalog,
)


WORKSPACE = WORKSPACE_ROOT


@pytest.fixture(scope="module")
def logic_catalog() -> StaticCardLogicCatalogV1:
    return build_static_card_logic_catalog(WORKSPACE)


def test_static_logic_catalog_roots_every_card_and_keeps_full_namespaces(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    summary = logic_catalog.summary

    assert summary["card_count"] == 152
    assert summary["rooted_card_count"] == 152
    assert summary["node_count"] >= 2_700
    counts = summary["node_counts_by_namespace"]
    assert counts["ACTION"] >= 700
    assert counts["BUFF"] >= 85
    assert counts["AEO"] >= 74
    assert counts["PROJECTILE"] >= 83
    assert counts["CHARACTER"] >= 129
    assert counts["BUILDING"] >= 46
    assert counts["ABILITY"] >= 23
    assert summary["source_failure_count"] == 0
    assert summary["source_placeholder_count"] == 5
    assert {
        item["raw_sha256"] for item in logic_catalog.source_placeholders
    } == {
        "0573b69cc2c4df3d5e5a3f29e7d2929ae2655bfe1e03e2edba4a020c5ba56da8"
    }


def test_known_status_and_mechanic_chains_are_reachable(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    cards = logic_catalog.cards_by_name
    nodes = logic_catalog.nodes

    assert {
        "AEO.IceWizardCold",
        "ACTION.IceWizardCold_spawn",
        "CHARACTER.IceWizard",
        "PROJECTILE.ice_wizardProjectile",
        "BUFF.IceWizardSlowDown",
        "BUFF.IceWizardCold",
    }.issubset(cards["IceWizard"]["closure_node_ids"])
    assert {
        "AEO.ElectroWizardZap",
        "ACTION.ElectroWizardZap_spawn",
        "CHARACTER.ElectroWizard",
        "BUFF.ZapFreeze",
    }.issubset(cards["ElectroWizard"]["closure_node_ids"])
    assert {
        "CHARACTER.Ghost",
        "BUFF.Invisibility",
        "SPELL_EVOLVED.Ghost_EV1",
        "EXT.Ghost_EV1",
        "BUFF.Ghost_EV1_Invisibility",
    }.issubset(cards["Ghost"]["closure_node_ids"])
    assert {
        "CHARACTER.InfernoDragon",
        "EXT.InfernoDragon_EV1",
        "VARIABLE.InfernoDragon_EV1_AttackCount",
        "VARIABLE.InfernoDragon_EV1_AttackDecayCounter",
        "ACTION.InfernoDragon_EV1_UpdateAttackSequence",
    }.issubset(cards["InfernoDragon"]["closure_node_ids"])
    assert {
        "AEO.Rage",
        "BUFF.Rage",
        "BUILDING.RageBottle",
    }.issubset(cards["Rage"]["closure_node_ids"])
    assert {
        "AEO.Vines_AeO",
        "ACTION.Vines_Start_Action_Group",
        "ACTION.Vines_Target_Selector",
        "ACTION.Vines_Action_Group",
        "ACTION.Vines_Air_To_Ground",
        "ACTION.Vines_Select_Buff_Size",
    }.issubset(cards["Vines"]["closure_node_ids"])
    assert {
        "SPELL_CHARACTER.MergeMaiden_Mounted",
        "CHARACTER.MergeMaiden_Mounted",
        "PROJECTILE.MergeMaidenProjectile_Mounted",
        "SPELL_CHARACTER.MergeMaiden_Normal",
        "CHARACTER.MergeMaiden_Normal",
    }.issubset(cards["MergeMaiden"]["closure_node_ids"])

    ice_slow = nodes["BUFF.IceWizardSlowDown"]["merged_record"]
    ice_spawn = nodes["AEO.IceWizardCold"]["merged_record"]
    assert ice_slow["HitSpeedMultiplier"] == -30
    assert ice_slow["SpawnSpeedMultiplier"] == -30
    assert ice_slow["SpeedMultiplier"] == -30
    assert ice_spawn["BuffTime"] == 2_500
    assert ice_spawn["Damage"] == 33

    electro = nodes["CHARACTER.ElectroWizard"]["merged_record"]
    zap_freeze = nodes["BUFF.ZapFreeze"]["merged_record"]
    assert electro["BuffOnDamage"] == "ZapFreeze"
    assert electro["BuffOnDamageTime"] == 500
    assert zap_freeze["HitSpeedMultiplier"] == -100
    assert zap_freeze["SpawnSpeedMultiplier"] == -100
    assert zap_freeze["SpeedMultiplier"] == -100

    rage = nodes["BUFF.Rage"]["merged_record"]
    rage_area = nodes["AEO.Rage"]["merged_record"]
    assert rage["HitSpeedMultiplier"] == 130
    assert rage["SpawnSpeedMultiplier"] == 130
    assert rage["SpeedMultiplier"] == 130
    assert rage_area["LifeDuration"] == 4_500
    assert rage_area["BuffTime"] == 1_000

    ghost = nodes["CHARACTER.Ghost"]["merged_record"]
    invisibility = nodes["BUFF.Invisibility"]["merged_record"]
    assert ghost["BuffWhenNotAttacking"] == "Invisibility"
    assert ghost["BuffWhenNotAttackingTime"] == 2_000
    assert ghost["HideTimeMs"] == 400
    assert invisibility["Invisible"] is True

    inferno_dragon = nodes["CHARACTER.InfernoDragon"]["merged_record"]
    assert inferno_dragon["Damage"] == 14
    assert inferno_dragon["VariableDamage2"] == 47
    assert inferno_dragon["VariableDamage3"] == 165
    assert inferno_dragon["VariableDamageTime1"] == 2_000
    assert inferno_dragon["VariableDamageTime2"] == 2_000


def test_august_runtime_replacements_and_new_action_edges_are_closed(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    cards = logic_catalog.cards_by_name
    nodes = logic_catalog.nodes

    barbarian_evo = nodes["CHARACTER.Barbarian_EV1"]["merged_record"]
    assert "Hitpoints" not in barbarian_evo
    assert {
        fragment["source_record"]
        for fragment in nodes["CHARACTER.Barbarian_EV1"]["fragments"]
    } == {
        "runtime-update/csv_logic/characters_evo.toml#CHARACTER.Barbarian_EV1",
    }

    drill = cards["GoblinDrill"]["closure_node_ids"]
    assert {
        "BUILDING.GoblinDrill",
        "AEO.GoblinDrillDamage",
        "CHARACTER.GoblinDrillDig",
        "EXT.GoblinDrill_EV1_Dig",
        "EXT.GoblinDrill_EV1",
    }.issubset(drill)
    drill_damage = nodes["AEO.GoblinDrillDamage"]["merged_record"]
    assert (drill_damage["Damage"], drill_damage["CrownTowerDamagePercent"]) == (
        33,
        -100,
    )

    witch = cards["Witch"]["closure_node_ids"]
    assert {
        "ACTION.Witch_EV1_On_Skeleton_Destroyed",
        "ACTION.Witch_Soul_Drain",
        "ACTION.Witch_EV1_Heal_Action_Group",
        "ACTION.Witch_EV1_Apply_Heal_Buff_Action",
        "BUFF.Witch_EV1_Heal_Buff",
        "FILTER.friendly_skeletons_can_be_dead",
    }.issubset(witch)
    assert nodes["EXT.Witch_EV1"]["merged_record"]["SpawnPauseTime"] == 300_000
    witch_heal = nodes["BUFF.Witch_EV1_Heal_Buff"]["merged_record"]
    assert (witch_heal["HealPerSecond"], witch_heal["HitFrequency"]) == (
        1_200,
        50,
    )

    tombstone = cards["Tombstone"]["closure_node_ids"]
    assert {
        "ACTION.TombstoneHero_visual_dummy_break_tomb_group",
        "EXT.TombstoneHero_broken_visual_dummy",
        "ACTION.Tombstone_hero_Monster_TrackHealthValue_Action",
        "CHARACTER.TombstoneHero_Monster_Active",
    }.issubset(tombstone)
    tombstone_active = nodes[
        "CHARACTER.TombstoneHero_Monster_Active"
    ]["merged_record"]
    assert (
        tombstone_active["Damage"],
        tombstone_active["Hitpoints"],
        tombstone_active["SightRange"],
    ) == (165, 1_650, 7_000)

    goblins = cards["Goblins"]["closure_node_ids"]
    assert {
        "CHARACTER.GoblinHero_Flag_Building",
        "ACTION.GoblinHero_Ability_Activated_Group",
        "ACTION.GoblinHero_Spawn_Second_Wave_0",
        "ACTION.GoblinHero_Spawn_Second_Wave_1",
    }.issubset(goblins)
    goblin_wave = nodes[
        "ACTION.GoblinHero_Ability_Activated_Group"
    ]["merged_record"]
    assert goblin_wave["SubActions"][:3] == (
        "GoblinHero_Set_Custom_Tag",
        "GoblinHero_Spawn_Second_Wave_0",
        "GoblinHero_Spawn_Second_Wave_1",
    )

    archer = cards["Archer"]["closure_node_ids"]
    assert {
        "CHARACTER.Archer_EV1",
        "ACTION.Archer_EV1_AttackSelect",
        "PROJECTILE.Archer_EV1_Arrow",
        "PROJECTILE.Archer_EV1_ArrowDoubleDamage",
    }.issubset(archer)
    assert nodes["CHARACTER.Archer_EV1"]["merged_record"]["Projectile2"] == (
        "Archer_EV1_ArrowDoubleDamage"
    )
    assert nodes["PROJECTILE.Archer_EV1_ArrowDoubleDamage"]["merged_record"][
        "Damage"
    ] == ("%", 125)

    healer = cards["BattleHealer"]["closure_node_ids"]
    assert {
        "BUFF.BattleHealerAll",
        "BUFF.BattleHealerSpawnBuff",
    }.issubset(healer)
    assert nodes["CHARACTER.BattleHealer"]["merged_record"]["IgnoreBuff"] == (
        "BattleHealerAll",
        "BattleHealerSpawnBuff",
    )
    ignore_buff_references = [
        reference
        for reference in logic_catalog.references
        if reference["source_node_id"] == "CHARACTER.BattleHealer"
        and str(reference["field_path"]).startswith("IgnoreBuff[")
    ]
    assert {
        tuple(reference["target_node_ids"])
        for reference in ignore_buff_references
    } == {
        ("BUFF.BattleHealerAll",),
        ("BUFF.BattleHealerSpawnBuff",),
    }

    newly_typed_fields = {
        "ActionOnTargetReached",
        "ActionToRun",
        "ChampionCharacterData",
        "LinkedChampionCharacter",
        "NewCharacterData",
        "Projectile2",
        "Projectile3",
        "SpawnPathfindMorph",
    }
    references = [
        reference
        for reference in logic_catalog.references
        if str(reference["field_path"]).rsplit(".", 1)[-1].split("[", 1)[0]
        in newly_typed_fields
    ]
    assert {reference["field_path"] for reference in references}.issuperset(
        newly_typed_fields
    )
    assert all(reference["status"] == "resolved" for reference in references)


def test_level_profiles_cover_every_card_without_guessing_rounding(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    summary = logic_catalog.summary
    common = logic_catalog.rarity_curves["Common"]
    legendary = logic_catalog.rarity_curves["Legendary"]
    champion = logic_catalog.rarity_curves["Champion"]

    assert summary["cards_with_level_projection"] == 152
    assert tuple(common["card_levels"]) == tuple(range(1, 17))
    assert tuple(common["active_power_level_multipliers_percent"]) == (
        100,
        110,
        121,
        133,
        146,
        160,
        176,
        193,
        212,
        233,
        256,
        281,
        309,
        339,
        372,
        409,
    )
    assert tuple(legendary["card_levels"]) == tuple(range(9, 17))
    assert tuple(champion["card_levels"]) == tuple(range(11, 17))
    assert common["tournament_card_level"] == 11
    assert legendary["tournament_card_level"] == 11
    assert champion["tournament_card_level"] == 11
    assert common["rounding_rule"] is None
    assert "engine_rounding_rule" in common["unknown_fields"]

    representative_hitpoints = (
        ("Heal", "Rare", 84, 215),
        ("Giant", "Rare", 1_550, 3_968),
        ("Pekka", "Epic", 1_469, 3_760),
        ("Princess", "Legendary", 102, 261),
        ("ArcherQueen", "Champion", 391, 1_000),
    )
    for name, card_rarity, base_value, level_11_value in (
        representative_hitpoints
    ):
        card = logic_catalog.cards_by_name[name]
        card_projection = card["level_projection"]
        projection = card["field_level_projections"]["hitpoints"]

        assert card_projection["scope"] == "card_progression_only"
        assert card_projection["stat_value_projection_allowed"] is False
        assert card_projection["rarity_curve"] == card_rarity
        assert projection["status"] == "resolved"
        assert projection["base_value"] == base_value
        assert projection["carrier_rarity"] == "Common"
        assert projection["carrier_curve"]["node_id"] == "RARITY.Common"
        assert projection["level_11_multiplier_percent"] == 256
        assert projection["level_11_value"] == level_11_value
        assert projection["level_11_rounding_rule"] == (
            "integer_truncation_toward_zero"
        )

    heal = logic_catalog.cards_by_name["Heal"]
    heal_candidates = {
        (candidate["node_id"], candidate["field_path"]): candidate
        for candidate in heal["level_scaling_candidates"]
    }
    heal_hitpoints = heal_candidates[("CHARACTER.HealSpirit", "Hitpoints")]
    heal_damage = heal_candidates[
        ("PROJECTILE.HealSpiritProjectile", "Damage")
    ]
    assert heal_hitpoints["base_value_kind"] == "exact_numeric"
    assert heal_hitpoints["carrier_rarity_resolution"] == "direct"
    assert heal_hitpoints["carrier_rarity_source_node_id"] == (
        "CHARACTER.HealSpirit"
    )
    assert heal_hitpoints["level_11_value"] == 215
    assert heal_damage["carrier_rarity"] == "Common"
    assert heal_damage["level_11_value"] == 110

    inherited_princess_damage = next(
        candidate
        for candidate in logic_catalog.cards_by_name["Princess"][
            "level_scaling_candidates"
        ]
        if candidate["node_id"] == "EXT.Princess_EV1_DeathFreeze_Damage"
        and candidate["field_path"] == "Damage"
    )
    assert inherited_princess_damage["carrier_rarity_resolution"] == (
        "base_inheritance"
    )
    assert inherited_princess_damage["carrier_rarity_source_node_id"] == (
        "AEO.Princess_EV1_DeathFreeze"
    )
    assert inherited_princess_damage["carrier_rarity"] == "Common"


def test_level_scaling_candidates_resolve_percentage_override_values(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    node_id = "PROJECTILE.Archer_EV1_ArrowDoubleDamage"
    assert logic_catalog.nodes[node_id]["merged_record"]["Damage"] == ("%", 125)
    assert logic_catalog.nodes[node_id]["effective_record"]["Damage"] == 55
    candidate = next(
        candidate
        for candidate in logic_catalog.cards_by_name["Archer"][
            "level_scaling_candidates"
        ]
        if candidate["node_id"] == node_id
    )
    assert candidate["carrier_rarity_resolution"] == "base_inheritance"
    assert candidate["level_11_value"] == 140


def test_all_static_references_resolve_with_proved_semantic_aliases(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    summary = logic_catalog.summary

    assert summary["cards_with_unresolved_gameplay_references"] == 0
    assert summary["cards_with_unresolved_non_gameplay_references"] == 0
    assert summary["cards_with_ambiguous_references"] == 0
    assert summary["unresolved_reference_counts_by_class"] == {}
    assert summary["reference_counts_by_status"] == {
        "resolved": summary["reference_count"] - 1,
        "resolved_supplemental_mode_scoped": 1,
    }
    assert summary["cards_with_mode_scoped_supplemental_references"] == 1
    supplemental = [
        reference
        for reference in logic_catalog.references
        if reference["status"] == "resolved_supplemental_mode_scoped"
    ]
    assert [
        (
            item["source_node_id"],
            item["field_path"],
            item["raw_value"],
        )
        for item in supplemental
    ] == [
        (
            "ACTION.giantbuffer_collect_friend_troops",
            "TargetFilter",
            "friendly_troops_for_rune_giant",
        )
    ]
    rune_reference = supplemental[0]
    assert rune_reference["target_node_ids"] == ()
    assert rune_reference["supplemental_target_ids"] == (
        "FILTER.friendly_troops_for_rune_giant",
    )
    assert rune_reference["resolution_rule"] == (
        "unique_scdb_json_mode_scoped"
    )
    assert rune_reference["normal_mode_runtime_available"] is None
    assert rune_reference["requires_native_runtime_evidence"] is True

    rune_filter = logic_catalog.supplemental_mode_scoped_nodes[
        "FILTER.friendly_troops_for_rune_giant"
    ]
    assert rune_filter["scope"] == "mode_scoped"
    assert rune_filter["asset_type"] == "LogicChaosDataAsset"
    assert rune_filter["asset_path"] == (
        "logic/chaos_arena/season_2/game/zap_chaos_1"
    )
    assert rune_filter["source"] == "runtime-update/assets.scdb"
    assert rune_filter["source_sha256"] == (
        "4b99bfd5d5fb52fe79bba74cd1990ce08fc4f46064d905c64cb5c281b6474e7b"
    )
    assert rune_filter["record"]["ExcludeCharactersWithData"] == (
        "Wallbreaker",
        "Wallbreaker_EV1",
        "Wallbreaker_mini",
        "FireSpirits",
        "IceSpirits",
        "IceSpirits_EV1",
        "ElectroSpirit",
        "HealSpirit",
        "SkeletonBalloon",
        "SkeletonBalloon_EV1",
        "SuspiciousBush",
        "BattleRam",
        "GoblinDemolisher_kamikaze_form",
    )

    building_aliases = [
        reference
        for reference in logic_catalog.references
        if reference["resolution_rule"]
        == "qualified_character_data_building_alias"
    ]
    assert len(building_aliases) == 9
    assert {
        (reference["raw_value"], reference["target_node_ids"][0])
        for reference in building_aliases
    } == {
        ("CHARACTER.Cannon", "BUILDING.Cannon"),
        ("CHARACTER.GoblinCage", "BUILDING.GoblinCage"),
        ("CHARACTER.KingTower", "BUILDING.KingTower"),
        ("CHARACTER.PrincessTower", "BUILDING.PrincessTower"),
        ("CHARACTER.RageBarbarianBottle", "BUILDING.RageBarbarianBottle"),
    }
    assert all(
        logic_catalog.nodes[reference["target_node_ids"][0]][
            "merged_record"
        ]["IsBuilding"]
        is True
        for reference in building_aliases
    )


def test_dash_filter_exports_remain_raw_visual_fields_not_logic_edges(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    nodes = logic_catalog.nodes
    expected = {
        "CHARACTER.Assassin": "filter_bandit_charge",
        "CHARACTER.BossBandit": "filter_bandit_charge",
        "CHARACTER.MegaKnight": "filter_mega_knight_jump",
        "CHARACTER.SuperHogRider_Terry": "filter_mega_knight_jump",
        "EXT.MegaKnight_EV1": "filter_mega_knight_jump_evolution",
    }

    assert {
        node_id: nodes[node_id]["merged_record"]["DashFilter"]
        for node_id in expected
    } == expected
    assert not any(
        reference["field_path"] == "DashFilter"
        for reference in logic_catalog.references
    )


def test_elite_archer_hero_partial_table_is_folded_into_derived_object(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    references = [
        reference
        for reference in logic_catalog.references
        if reference["source_node_id"] == "SPELL_HERO.EliteArcher_hero"
        and reference["field_path"] == "SummonCharacter"
    ]
    assert len(references) == 1
    reference = references[0]
    assert reference["status"] == "resolved"
    assert reference["target_node_ids"] == ("EXT.EliteArcherHero",)

    extension = logic_catalog.nodes["EXT.EliteArcherHero"]
    assert "CHARACTER.EliteArcherHero" not in logic_catalog.nodes
    assert extension["semantic_overlay_source_node_ids"] == (
        "CHARACTER.EliteArcherHero",
    )
    assert extension["merged_record"]["Base"] == "CHARACTER.EliteArcher"
    assert len(extension["merged_record"]["AttackSequenceList"]) == 2
    assert {
        str(fragment["source_record"]).split("#", 1)[1]
        for fragment in extension["fragments"]
    } == {
        "EXT.EliteArcherHero",
        "CHARACTER.EliteArcherHero",
    }


def test_visual_action_lists_are_not_promoted_to_gameplay_edges(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    assert not any(
        "VisualActions" in str(reference["field_path"])
        for reference in logic_catalog.references
    )
    visual = [
        reference
        for reference in logic_catalog.references
        if reference["reference_class"] == "visual_native_action"
    ]
    assert visual
    assert all(reference["gameplay_reference"] is False for reference in visual)
    ghost_families = logic_catalog.cards_by_name["Ghost"]["mechanic_families"]
    assert "ActionPlayEffect" in ghost_families["visual_native_actions"]
    assert "native_action:ActionPlayEffect" not in ghost_families["gameplay"]


def test_native_actions_are_signature_typed_and_list_deployments_are_closed(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    summary = logic_catalog.summary
    typed = {
        action["class_type"]
        for card in logic_catalog.cards
        for action in card["typed_native_actions"]
    }

    assert summary["typed_native_action_count"] == 31
    assert summary["cards_with_typed_native_actions"] == 22
    assert {
        "ActionBlowdartGoblinEvoController",
        "ActionBossBanditAbility",
        "ActionCaptureCharacter",
        "ActionGiantBufferBuff",
        "ActionGoblinsteinAbility",
        "ActionHide",
        "ActionMusketeerSnipe",
        "ActionSkeletonBarrelPopBalloon",
    }.issubset(typed)
    hide_actions = [
        action
        for card in logic_catalog.cards
        for action in card["typed_native_actions"]
        if action["class_type"] == "ActionHide"
    ]
    assert hide_actions
    assert all(action["signature_fields"] for action in hide_actions)

    three = logic_catalog.cards_by_name["ThreeMusketeers"]
    assert {
        "EXT.ThreeMusketeer_Rework_Character_1",
        "EXT.ThreeMusketeer_Rework_Character_2",
        "EXT.ThreeMusketeer_Rework_Character_3",
    }.issubset(three["closure_node_ids"])
    assert "deploy_form_reference_absent" not in three["unknown_fields"]


def test_default_serialized_fields_do_not_become_mechanic_tags(
    logic_catalog: StaticCardLogicCatalogV1,
) -> None:
    arrows = logic_catalog.cards_by_name["Arrows"]["mechanic_families"]["gameplay"]
    balloon = logic_catalog.cards_by_name["Balloon"]["mechanic_families"]["gameplay"]

    assert not any("homing" in tag or "deflect" in tag for tag in arrows)
    assert not any("clone" in tag for tag in balloon)


def test_content_addressed_round_trip_is_byte_stable(
    logic_catalog: StaticCardLogicCatalogV1,
    tmp_path: Path,
) -> None:
    first = logic_catalog.save_content_addressed(tmp_path)
    second = logic_catalog.save_content_addressed(tmp_path)
    restored = StaticCardLogicCatalogV1.load(first)

    assert first == second
    assert first.stem == f"card-logic-{logic_catalog.catalog_id}"
    assert restored.catalog_id == logic_catalog.catalog_id
    assert restored.to_json() == logic_catalog.to_json()
