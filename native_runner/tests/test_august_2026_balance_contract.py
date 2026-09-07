from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT

import math
from typing import Any, Mapping

import pytest

from native_runner.card_logic import (
    StaticCardLogicCatalogV1,
    build_static_card_logic_catalog,
)
from native_runner.card_specs import CardSpecCatalog, build_card_catalog
from native_runner.effect_catalog import NativeEffectCatalogV1, build_effect_catalog
from native_runner.projectile_catalog import (
    NativeProjectileCatalogV1,
    build_projectile_catalog,
)
from native_runner.training.card_features import (
    build_card_feature_rows,
    level_11_value,
)


WORKSPACE = WORKSPACE_ROOT


@pytest.fixture(scope="module")
def card_catalog() -> CardSpecCatalog:
    return build_card_catalog(WORKSPACE)


@pytest.fixture(scope="module")
def static_logic(card_catalog: CardSpecCatalog) -> StaticCardLogicCatalogV1:
    return build_static_card_logic_catalog(WORKSPACE, card_catalog=card_catalog)


@pytest.fixture(scope="module")
def effect_catalog(
    static_logic: StaticCardLogicCatalogV1,
) -> NativeEffectCatalogV1:
    return build_effect_catalog(static_logic, WORKSPACE)


@pytest.fixture(scope="module")
def projectile_catalog(
    static_logic: StaticCardLogicCatalogV1,
) -> NativeProjectileCatalogV1:
    return build_projectile_catalog(static_logic, WORKSPACE)


def _record(
    static_logic: StaticCardLogicCatalogV1,
    node_id: str,
) -> Mapping[str, Any]:
    return static_logic.nodes[node_id]["merged_record"]


def _runtime_record(
    static_logic: StaticCardLogicCatalogV1,
    node_id: str,
    **expected: Any,
) -> Mapping[str, Any]:
    """Return the one raw runtime fragment carrying all changed fields."""

    matches = tuple(
        fragment
        for fragment in static_logic.nodes[node_id]["fragments"]
        if fragment["layer"] == "runtime"
        and all(fragment["record"].get(key) == value for key, value in expected.items())
        and all(key in fragment["record"] for key in expected)
    )
    assert len(matches) == 1, (node_id, expected, matches)
    assert str(matches[0]["source_record"]).startswith("runtime-update/")
    return matches[0]["record"]


def _level_11(
    card_catalog: CardSpecCatalog,
    card_id: int,
    raw_value: int | float,
    *,
    rarity: str = "Common",
) -> float:
    value = level_11_value(
        card_catalog.by_id[card_id],
        raw_value,
        carrier_rarity=rarity,
    )
    assert value is not None
    return value


def test_01_spirit_hitpoints_use_the_character_carrier_curve(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    # The tower-connection outcome is a live interaction.  Its static input is
    # the common-rarity 84 HP carrier; this test deliberately claims no live
    # targeting or connection result.
    spirits = (
        (26_000_031, "CHARACTER.FireSpirits"),
        (26_000_030, "CHARACTER.IceSpirits"),
        (28_000_016, "CHARACTER.HealSpirit"),
        (26_000_084, "CHARACTER.ElectroSpirit"),
    )
    rows = build_card_feature_rows(
        {card_id: card_catalog.by_id[card_id] for card_id, _ in spirits},
        static_logic=static_logic,
    )

    for card_id, node_id in spirits:
        raw = _runtime_record(static_logic, node_id, Hitpoints=84)
        assert raw["Rarity"] == "Common"
        assert card_catalog.by_id[card_id].hitpoints == 84
        assert rows[card_id].level_hitpoints == 215

    # Heal Spirit's card root is Rare, while its numeric character carrier is
    # Common.  This guards against projecting 84 through the wrong curve.
    assert card_catalog.by_id[28_000_016].categorical_features["rarity"] == "Rare"


def test_02_bound_abilities_are_single_use_except_boss_bandit(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    abilities = {ability.ability_id: ability for ability in card_catalog.abilities}
    assert len(abilities) == 24

    boss = abilities.pop("BossBandit_ability")
    assert (boss.charges, boss.cooldown_ms) == (2, 3_000)
    assert _record(static_logic, "ABILITY.BossBandit_ability")["MaxCharges"] == 2
    assert _record(static_logic, "ABILITY.BossBandit_ability")["Cooldown"] == 3_000

    mega_minion = abilities.pop("MegaMinion_Teleport_Ability")
    assert (mega_minion.charges, mega_minion.cooldown_ms) == (1, 1_000)
    mega_raw = _runtime_record(
        static_logic,
        "ABILITY.MegaMinion_Teleport_Ability",
        MaxCharges=1,
        Cooldown=1_000,
    )
    assert mega_raw["MaxCharges"] == 1

    assert len(abilities) == 22
    for ability_id, ability in abilities.items():
        assert (ability.charges, ability.cooldown_ms) == (1, None)
        raw = _record(static_logic, f"ABILITY.{ability_id}")
        assert raw["MaxCharges"] == 1
        assert "Cooldown" not in raw

    # The runtime replacement of character_abilities.csv carries only the
    # identity column for this row; it must not resurrect the old cooldown.
    mighty_fragments = static_logic.nodes[
        "ABILITY.MightyMinerLaneSwitch"
    ]["fragments"]
    release_row = next(
        fragment["record"]
        for fragment in mighty_fragments
        if "character_abilities.csv" in fragment["source_record"]
    )
    assert dict(release_row) == {"Name": "MightyMinerLaneSwitch"}


def test_03_goblinstein_doctor_and_tether_damage(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    doctor_projectile = _runtime_record(
        static_logic,
        "PROJECTILE.prj_goblinstein_doctor",
        Damage=53,
    )
    assert doctor_projectile["Rarity"] == "Common"
    assert _level_11(card_catalog, 26_000_099, 53) == 135

    tether = _runtime_record(
        static_logic,
        "ACTION.goblinstein_ability_action",
        TetherDamage=37,
        TetherHitInterval=500,
    )
    scaled_damage_per_hit = _level_11(card_catalog, 26_000_099, 37)
    assert scaled_damage_per_hit == 94
    assert scaled_damage_per_hit * 1_000 // tether["TetherHitInterval"] == 188


def test_04_little_prince_movement_grace_and_guard_damage(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    assert _runtime_record(
        static_logic,
        "ACTION.LP_SetGrace",
        Value="300",
    )["Variable"] == "LP_GraceTimer"
    ticker = _runtime_record(
        static_logic,
        "ACTION.LP_ConstantTicker",
        Interval=50,
    )
    assert ticker["ActionToExecute"]["SubActions"] == (
        "LP_DecrGrace",
        "LP_CheckReset",
        "LP_CheckCombatDisabled",
    )
    decrement = _runtime_record(
        static_logic,
        "ACTION.LP_DecrGrace",
        Value="max(0, LP_GraceTimer - 50 * is_moving)",
    )
    assert decrement["Variable"] == "LP_GraceTimer"

    guard = _runtime_record(
        static_logic,
        "CHARACTER.ChampionGuard",
        Damage=91,
    )
    assert guard["Rarity"] == "Common"
    assert _level_11(card_catalog, 26_000_093, 91) == 232


def test_05_mighty_miner_base_damage_uses_common_carrier(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    raw = _runtime_record(
        static_logic,
        "CHARACTER.MightyMiner",
        Damage=17,
    )
    assert raw["Rarity"] == "Common"
    row = build_card_feature_rows(
        {26_000_065: card_catalog.by_id[26_000_065]},
        static_logic=static_logic,
    )[26_000_065]
    assert (card_catalog.by_id[26_000_065].damage, row.level_damage) == (17, 43)


def test_06_ronin_first_hit_derivation_and_non_stun_parry(
    static_logic: StaticCardLogicCatalogV1,
    effect_catalog: NativeEffectCatalogV1,
) -> None:
    ronin = _runtime_record(
        static_logic,
        "CHARACTER.Ronin",
        HitSpeed=1_400,
        LoadTime=1_000,
    )
    assert ronin["HitSpeed"] - ronin["LoadTime"] == 400

    parry = effect_catalog.by_name["ronin_reflect_stun_buff"]
    assert parry.modifiers["HitSpeedMultiplier"] == -95
    assert {"slow", "movement_lock", "spawn_lock"}.issubset(parry.mechanic_tags)
    assert {"stun", "freeze", "incapacitate"}.isdisjoint(parry.mechanic_tags)


def test_07_xbow_hit_speed_damage_and_projectile_speed(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
    projectile_catalog: NativeProjectileCatalogV1,
) -> None:
    _runtime_record(static_logic, "BUILDING.Xbow", HitSpeed=400)
    _runtime_record(
        static_logic,
        "PROJECTILE.xbow_projectile",
        Damage=23,
        Speed=1_600,
    )
    projectile = projectile_catalog.by_name["xbow_projectile"]
    assert (projectile.raw_record["Damage"], projectile.raw_record["Speed"]) == (
        23,
        1_600,
    )
    row = build_card_feature_rows(
        {27_000_008: card_catalog.by_id[27_000_008]},
        static_logic=static_logic,
    )[27_000_008]
    assert (card_catalog.by_id[27_000_008].hit_speed_ms, row.level_damage) == (
        400,
        58,
    )
    assert card_catalog.by_id[27_000_008].projectile_speed == 1_600


def test_08_hero_ice_golem_third_blast_is_damage_plus_slow(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
    effect_catalog: NativeEffectCatalogV1,
) -> None:
    damage = _runtime_record(
        static_logic,
        "AEO.IceGolemiteHero_Damage_AEO",
        Damage=27,
    )
    assert damage["Rarity"] == "Common"
    assert _level_11(card_catalog, 26_000_038, 27) == 69

    final_blast = _runtime_record(
        static_logic,
        "AEO.IceGolemiteHero_Freeze_AEO",
        OnHitAction="IceGolemiteHero_Select_Slow_Buff",
    )
    assert final_blast["OnHitAction"] == "IceGolemiteHero_Select_Slow_Buff"
    slow = effect_catalog.by_name["IceGolemiteHero_Slow_Buff_Base"]
    assert set(slow.mechanic_tags) == {"slow"}
    assert slow.modifiers["SpeedMultiplier"] == -30
    triggers = {
        (trigger["source_node_id"], trigger["field_path"])
        for application in slow.application_sources
        for trigger in application["trigger_sources"]
    }
    assert ("AEO.IceGolemiteHero_Freeze_AEO", "OnHitAction") in triggers


def test_09_witch_heal_overheal_and_first_wave_identity(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    heal = _runtime_record(
        static_logic,
        "BUFF.Witch_EV1_Heal_Buff",
        HealPerSecond=1_200,
        HitFrequency=50,
        AllowedOverHealPerc=173,
    )
    raw_heal_per_skeleton = heal["HealPerSecond"] * heal["HitFrequency"] // 1_000
    assert raw_heal_per_skeleton == 60
    assert _level_11(card_catalog, 26_000_007, raw_heal_per_skeleton) == 153

    witch = _record(static_logic, "CHARACTER.Witch")
    assert (witch["Hitpoints"], witch["Rarity"]) == (328, "Common")
    level_hitpoints = _level_11(card_catalog, 26_000_007, witch["Hitpoints"])
    assert level_hitpoints == 839
    assert math.floor(level_hitpoints * heal["AllowedOverHealPerc"] / 100) == 1_451

    evolved = _runtime_record(
        static_logic,
        "EXT.Witch_EV1",
        SpawnCharacter="Witch_EV1_Healing_Skeleton",
        SpawnPauseTime=300_000,
    )
    assert evolved["Base"] == "CHARACTER.Witch"
    periodic = _runtime_record(
        static_logic,
        "ACTION.Witch_EV1_Interval_Spawn",
        SpawnData="Witch_EV1_Skeleton_Interval",
        Count=4,
    )
    assert periodic["SpawnData"] != evolved["SpawnCharacter"]
    filter_record = _record(static_logic, "FILTER.friendly_skeletons_can_be_dead")
    assert filter_record["IncludeCharactersWithData"] == (
        "Witch_EV1_Healing_Skeleton",
    )


def test_10_battle_healer_stats_radius_and_self_exclusion(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    healer = _runtime_record(
        static_logic,
        "CHARACTER.BattleHealer",
        Damage=105,
        HitSpeed=2_000,
        Hitpoints=750,
        LoadTime=1_600,
        IgnoreBuff=("BattleHealerAll", "BattleHealerSpawnBuff"),
    )
    assert healer["Rarity"] == "Common"
    rows = build_card_feature_rows(
        {26_000_068: card_catalog.by_id[26_000_068]},
        static_logic=static_logic,
    )
    assert (rows[26_000_068].level_damage, rows[26_000_068].level_hitpoints) == (
        268,
        1_920,
    )
    assert _runtime_record(
        static_logic,
        "AEO.BattleHealerHeal",
        Radius=3_000,
    )["OnlyOwnTroops"] is True
    assert _runtime_record(
        static_logic,
        "AEO.BattleHealerSpawnHeal",
        Radius=3_000,
    )["OnlyOwnTroops"] is True
    assert "BattleHealerSelf" not in healer["IgnoreBuff"]
    assert not any(
        reference["source_node_id"] == "CHARACTER.BattleHealer"
        and "BattleHealerSelf" in reference["target_node_ids"]
        for reference in static_logic.references
    )


def test_11_goblin_curse_encodes_fifteen_percent_slow(
    static_logic: StaticCardLogicCatalogV1,
    effect_catalog: NativeEffectCatalogV1,
) -> None:
    _runtime_record(
        static_logic,
        "BUFF.GoblinCurseDamage",
        SpeedMultiplier=-15,
    )
    effect = effect_catalog.by_name["GoblinCurseDamage"]
    assert "slow" in effect.mechanic_tags
    assert effect.modifiers["SpeedMultiplier"] == -15


def test_12_goblin_machine_stats_and_five_second_rocket_cadence(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
    projectile_catalog: NativeProjectileCatalogV1,
) -> None:
    machine = _runtime_record(
        static_logic,
        "CHARACTER.GoblinMachine",
        Damage=91,
        Hitpoints=885,
    )
    assert machine["Rarity"] == "Common"
    rows = build_card_feature_rows(
        {26_000_096: card_catalog.by_id[26_000_096]},
        static_logic=static_logic,
    )
    assert (rows[26_000_096].level_damage, rows[26_000_096].level_hitpoints) == (
        232,
        2_265,
    )
    rocket = _runtime_record(
        static_logic,
        "PROJECTILE.GoblinMachineRocketProjectile",
        Speed=350,
    )
    assert projectile_catalog.by_name[
        "GoblinMachineRocketProjectile"
    ].raw_record["Speed"] == rocket["Speed"]
    attack = _runtime_record(
        static_logic,
        "ACTION.goblin_machine_rocket",
        AttackDelay=1_000,
        AttackCooldown=4_000,
    )
    assert attack["AttackDelay"] + attack["AttackCooldown"] == 5_000


def test_13_rune_giant_stats_cast_path_and_mode_scoped_filter(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    giant = _runtime_record(
        static_logic,
        "CHARACTER.GiantBuffer",
        Hitpoints=1_100,
    )
    assert giant["Rarity"] == "Common"
    assert _level_11(card_catalog, 26_000_101, 1_100) == 2_816

    collector = _runtime_record(
        static_logic,
        "ACTION.giantbuffer_collect_friend_troops",
        UseAbility=False,
        TargetFilter="friendly_troops_for_rune_giant",
    )
    assert collector["ActionWhenUnitBuffed"] == "giantbuffer_enchanting_buff"
    reference = next(
        item
        for item in static_logic.references
        if item["source_node_id"] == "ACTION.giantbuffer_collect_friend_troops"
        and item["field_path"] == "TargetFilter"
    )
    assert reference["status"] == "resolved_supplemental_mode_scoped"
    assert reference["normal_mode_runtime_available"] is None
    assert reference["requires_native_runtime_evidence"] is True

    supplemental = static_logic.supplemental_mode_scoped_nodes[
        "FILTER.friendly_troops_for_rune_giant"
    ]
    assert supplemental["asset_type"] == "LogicChaosDataAsset"
    assert supplemental["scope"] == "mode_scoped"
    filter_record = supplemental["record"]
    assert filter_record["FilterTags"] == "NO_GIANTBUFFER_CHEF_ENCHANTMENT"
    assert {
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
    } == set(filter_record["ExcludeCharactersWithData"])


def test_14_void_cost_cadence_and_three_damage_tiers(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    assert card_catalog.by_id[28_000_023].elixir_cost == 5
    area = _runtime_record(
        static_logic,
        "AEO.DarkMagicAOE",
        LifeDuration=4_000,
    )
    laser = area["OnStartingAction"]["SubActions"][1]
    assert laser["ClassType"] == "ActionLaserBall"
    assert laser["HitFrequency"] == 1_200
    assert laser["MaxUnitPerActionList"] == (1, 4)

    tier_records = tuple(
        action["SpawnData"] for action in laser["OnDetectedUnitActionList"]
    )
    assert tuple(record["Rarity"] for record in tier_records) == (
        "Common",
        "Common",
        "Common",
    )
    assert tuple(record["DamagePerSecond"] for record in tier_records) == (
        2_720,
        1_150,
        600,
    )
    assert tuple(record["CrownTowerDamagePerHit"] for record in tier_records) == (
        38,
        20,
        14,
    )
    assert tuple(record["HitFrequency"] for record in tier_records) == (
        100,
        100,
        100,
    )

    raw_hits = tuple(
        record["DamagePerSecond"] * record["HitFrequency"] // 1_000
        for record in tier_records
    )
    assert raw_hits == (272, 115, 60)
    assert tuple(
        _level_11(card_catalog, 28_000_023, damage) for damage in raw_hits
    ) == (696, 294, 153)
    assert tuple(
        _level_11(card_catalog, 28_000_023, record["CrownTowerDamagePerHit"])
        for record in tier_records
    ) == (97, 51, 35)
    assert card_catalog.by_id[28_000_023].categorical_features["rarity"] == "Epic"


def test_15_hero_mega_minion_warp_and_permanent_tower_override(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    projectile = _runtime_record(
        static_logic,
        "EXT.MegaMinionSpit_DoubleDamage",
        Damage=156,
        CrownTowerDamagePercent=-75,
    )
    assert projectile["Base"] == "PROJECTILE.MegaMinionSpit"
    base_projectile = _record(static_logic, "PROJECTILE.MegaMinionSpit")
    assert base_projectile["Rarity"] == "Common"
    assert _level_11(card_catalog, 26_000_039, projectile["Damage"]) == 399
    assert _runtime_record(
        static_logic,
        "ACTION.MegaMinion_hero_CrownTowerBuff_Spawn",
        SpawnTime=99_999,
    )["SpawnData"] == "MegaMinion_hero_CrownTower_Buff"


def test_16_mortar_and_evolved_mortar_share_4700ms_hit_speed(
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    _runtime_record(
        static_logic,
        "BUILDING.Mortar",
        HitSpeed=4_700,
        LoadTime=3_700,
    )
    evolved = _runtime_record(
        static_logic,
        "BUILDING.Mortar_EV1",
        HitSpeed=("=", 4_700),
        LoadTime=3_700,
    )
    assert evolved["Base"] == "Mortar"


def test_17_barbarian_hp_evolution_bonus_removal_and_rage_duration(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    barbarian = _runtime_record(
        static_logic,
        "CHARACTER.Barbarian",
        Hitpoints=280,
    )
    assert barbarian["Rarity"] == "Common"
    rows = build_card_feature_rows(
        {26_000_008: card_catalog.by_id[26_000_008]},
        static_logic=static_logic,
    )
    assert rows[26_000_008].level_hitpoints == 716

    evolved_node = static_logic.nodes["CHARACTER.Barbarian_EV1"]
    evolved = evolved_node["merged_record"]
    assert evolved["Base"] == "Barbarian"
    assert evolved["BuffAfterHitsTime"] == (5_000,)
    assert "Hitpoints" not in evolved
    assert len(evolved_node["fragments"]) == 1
    assert evolved_node["fragments"][0]["source_record"] == (
        "runtime-update/csv_logic/characters_evo.toml#CHARACTER.Barbarian_EV1"
    )
    assert "Hitpoints" not in evolved_node["fragments"][0]["record"]


def test_18_goblins_damage_and_two_unit_banner_brigade(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    goblin = _runtime_record(
        static_logic,
        "CHARACTER.Goblin",
        Damage=49,
    )
    assert goblin["Rarity"] == "Common"
    rows = build_card_feature_rows(
        {26_000_002: card_catalog.by_id[26_000_002]},
        static_logic=static_logic,
    )
    assert rows[26_000_002].level_damage == 125

    group = _runtime_record(
        static_logic,
        "ACTION.GoblinHero_Ability_Activated_Group",
        SubActions=(
            "GoblinHero_Set_Custom_Tag",
            "GoblinHero_Spawn_Second_Wave_0",
            "GoblinHero_Spawn_Second_Wave_1",
            "GoblinHero_Flag_Self_Destruct",
        ),
    )
    wave_actions = group["SubActions"][1:3]
    assert len(wave_actions) == 2
    assert all(
        _record(static_logic, f"ACTION.{action}")["SpawnData"] == "Goblin_dummy"
        for action in wave_actions
    )
    dummy = _runtime_record(
        static_logic,
        "EXT.Goblin_dummy",
        DeployTime=1_000,
    )
    assert dummy["Base"] == "CHARACTER.Goblin"


def test_19_archer_evolution_power_shot_is_125_percent(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
    projectile_catalog: NativeProjectileCatalogV1,
) -> None:
    archer = _runtime_record(
        static_logic,
        "CHARACTER.Archer_EV1",
        Projectile2="Archer_EV1_ArrowDoubleDamage",
    )
    assert archer["Base"] == "Archer"
    power = projectile_catalog.by_name["Archer_EV1_ArrowDoubleDamage"]
    assert power.raw_record["Base"] == "ArcherArrow"
    assert power.raw_record["Damage"] == ("%", 125)
    assert power.inheritance_chain == (
        "ArcherArrow",
        "Archer_EV1_ArrowDoubleDamage",
    )
    base_level_damage = _level_11(card_catalog, 26_000_001, 44)
    assert base_level_damage == 112
    assert math.floor(base_level_damage * 125 / 100) == 140


def test_20_giant_snowball_evolution_rolls_4000_units(
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    action = _runtime_record(
        static_logic,
        "ACTION.SnowballSpell_EV1_rolling_projectile",
        DistanceY=4_000,
    )
    assert action["DistanceX"] == 0
    assert _runtime_record(
        static_logic,
        "EXT.SnowballSpell_EV1",
        MinDistance=4_000,
    )["SpawnProjectile"] == "SnowballSpell_EV1_Rolling"


def test_21_goblin_barrel_decoy_damage_projects_to_66(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
    projectile_catalog: NativeProjectileCatalogV1,
) -> None:
    decoy = projectile_catalog.by_name["GoblinBarrelSpell_EV1_Decoy"]
    assert decoy.raw_record["Base"] == "GoblinBarrelSpell"
    assert decoy.raw_record["SpawnCharacter"] == "GoblinDummy"
    dummy = _runtime_record(
        static_logic,
        "CHARACTER.GoblinDummy",
        Damage=26,
    )
    assert dummy["Rarity"] == "Common"
    assert _level_11(card_catalog, 28_000_004, 26) == 66


def test_22_goblin_cage_cycles_damage_and_capture_cooldown(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    evolved_spell = _record(static_logic, "SPELL_EVOLVED.GoblinCage_EV1")
    assert evolved_spell["DarkElixirCost"] == 2
    capture = _runtime_record(
        static_logic,
        "ACTION.GoblinCage_EV1_CaptureUnit",
        DamagePerHit=143,
        CaptureCooldown=300,
    )
    brawler = _record(static_logic, "CHARACTER.GoblinBrawler")
    assert brawler["Rarity"] == "Common"
    assert _level_11(card_catalog, 27_000_012, capture["DamagePerHit"]) == 366


def test_23_mega_knight_uppercuts_every_other_attack_at_4000_strength(
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    counter = _runtime_record(
        static_logic,
        "ACTION.MegaKnight_EV1_Update_Do_Uppercut_Counter",
        Value="(MegaKnight_EV1_Do_Uppercut_Counter + 1) % 2",
    )
    assert counter["Variable"] == "MegaKnight_EV1_Do_Uppercut_Counter"
    uppercut = _runtime_record(
        static_logic,
        "ACTION.MegaKnight_EV1_uppercut",
        PushBackStrength=4_000,
    )
    assert uppercut["ExecuteIfTrue"] == (
        "MegaKnight_EV1_Do_Uppercut_Counter % 2 == 0"
    )


def test_24_valkyrie_tornado_damage_keeps_base_inheritance(
    effect_catalog: NativeEffectCatalogV1,
) -> None:
    tornado = effect_catalog.by_name["Valkyrie_MiniTornado_EV1"]
    assert tornado.raw_record["Base"] == "Tornado"
    assert tornado.raw_record["DamagePerSecond"] == 42
    assert tornado.inheritance_chain == ("Tornado", "Valkyrie_MiniTornado_EV1")
    assert "periodic_damage" in tornado.mechanic_tags


@pytest.mark.parametrize(
    ("node_id", "projectile_name"),
    (
        ("PROJECTILE.BowlerProjectile", "BowlerProjectile"),
        ("PROJECTILE.AxeManProjectile", "AxeManProjectile"),
    ),
)
def test_25_26_bowler_and_executioner_projectile_range(
    static_logic: StaticCardLogicCatalogV1,
    projectile_catalog: NativeProjectileCatalogV1,
    node_id: str,
    projectile_name: str,
) -> None:
    _runtime_record(static_logic, node_id, ProjectileRange=7_000)
    assert projectile_catalog.by_name[projectile_name].resolved_record[
        "ProjectileRange"
    ] == 7_000


def test_27_goblin_drill_has_no_crown_tower_spawn_damage(
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    damage = _runtime_record(
        static_logic,
        "AEO.GoblinDrillDamage",
        Damage=33,
        CrownTowerDamagePercent=-100,
    )
    assert damage["CrownTowerDamagePercent"] == -100


@pytest.mark.parametrize(
    ("card_id", "node_id", "projectile_name", "raw_damage", "level_damage"),
    (
        (
            26_000_058,
            "PROJECTILE.WallbreakerProjectile",
            "WallbreakerProjectile",
            110,
            281,
        ),
        (
            26_000_062,
            "PROJECTILE.EliteArcherArrow",
            "EliteArcherArrow",
            53,
            135,
        ),
        (
            28_000_018,
            "PROJECTILE.RoyalDeliveryProjectile",
            "RoyalDeliveryProjectile",
            150,
            384,
        ),
    ),
)
def test_28_30_projectile_damage_carriers(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
    projectile_catalog: NativeProjectileCatalogV1,
    card_id: int,
    node_id: str,
    projectile_name: str,
    raw_damage: int,
    level_damage: int,
) -> None:
    _runtime_record(static_logic, node_id, Damage=raw_damage)
    projectile = projectile_catalog.by_name[projectile_name]
    assert projectile.raw_record["Rarity"] == "Common"
    assert projectile.raw_record["Damage"] == raw_damage
    assert _level_11(card_catalog, card_id, raw_damage) == level_damage


def test_31_zappies_hit_and_load_times(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    zappie = _runtime_record(
        static_logic,
        "CHARACTER.MiniZapMachine",
        HitSpeed=2_200,
        LoadTime=1_400,
    )
    assert zappie["Rarity"] == "Common"
    spec = card_catalog.by_id[26_000_052]
    assert (spec.name, spec.hit_speed_ms) == ("MiniSparkys", 2_200)


def test_32_hero_tombstone_damage_hitpoints_and_sight_range(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    monster = _runtime_record(
        static_logic,
        "CHARACTER.TombstoneHero_Monster_Active",
        Damage=165,
        Hitpoints=1_650,
        SightRange=7_000,
    )
    assert monster["Rarity"] == "Common"
    assert _level_11(card_catalog, 27_000_009, monster["Damage"]) == 422
    assert _level_11(card_catalog, 27_000_009, monster["Hitpoints"]) == 4_224


def test_33_minion_horde_evolution_slowdown_is_33_percent(
    static_logic: StaticCardLogicCatalogV1,
    effect_catalog: NativeEffectCatalogV1,
) -> None:
    _runtime_record(
        static_logic,
        "BUFF.MinionHorde_EV1_GhostBuff",
        SpeedMultiplier=-33,
        HitSpeedMultiplier=-33,
    )
    ghost = effect_catalog.by_name["MinionHorde_EV1_GhostBuff"]
    assert ghost.modifiers["SpeedMultiplier"] == -33
    assert ghost.modifiers["HitSpeedMultiplier"] == -33


def test_34_princess_evolution_every_other_attack_and_5500ms_slow(
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    assert _runtime_record(
        static_logic,
        "VARIABLE.Princess_EV1_reload_frequency",
        DefaultValue=2,
    )["DefaultValue"] == 2
    area = _runtime_record(
        static_logic,
        "EXT.Princess_EV1_projectile_aeo",
        LifeDuration=5_500,
    )
    assert area["Base"] == "AEO.Princess_EV1_DeathFreeze"


def test_35_spear_goblins_hit_and_load_times(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    spear = _runtime_record(
        static_logic,
        "CHARACTER.SpearGoblin",
        HitSpeed=1_600,
        LoadTime=1_100,
    )
    assert spear["Rarity"] == "Common"
    assert card_catalog.by_id[26_000_019].hit_speed_ms == 1_600


def test_36_suspicious_bush_goblin_hp_projects_to_337(
    card_catalog: CardSpecCatalog,
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    goblin = _runtime_record(
        static_logic,
        "CHARACTER.BushGoblin",
        Hitpoints=132,
    )
    assert goblin["Rarity"] == "Common"
    assert _level_11(card_catalog, 26_000_097, goblin["Hitpoints"]) == 337


def test_37_furnace_spawn_interval_is_5000ms(
    static_logic: StaticCardLogicCatalogV1,
) -> None:
    furnace = _runtime_record(
        static_logic,
        "CHARACTER.Furnace_rework",
        OnStartingAction="Furnace_rework_continuous_spawn",
    )
    assert furnace["OnStartingAction"] == "Furnace_rework_continuous_spawn"
    interval = _runtime_record(
        static_logic,
        "ACTION.Furnace_rework_continuous_spawn",
        Interval=5_000,
    )
    assert interval["AffectedBySpawnSpeed"] is True
