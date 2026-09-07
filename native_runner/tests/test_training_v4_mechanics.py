from __future__ import annotations

from functools import lru_cache

import pytest
import torch

from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.card_specs import build_card_catalog
from native_runner.effect_catalog import build_effect_catalog
from native_runner.projectile_catalog import build_projectile_catalog
from native_runner.training.v4.tensors import ActiveEffectSetV4
from native_runner.training.v4.catalog import AbilityCatalogV1, CardCatalogV1, EntityArchetypeCatalogV1, normal_mode_card_specs
from native_runner.training.v4.mechanics import EffectSemanticCatalogV1, MechanicProfileCatalogV1
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.components import (
    ActiveEffectEncoder,
    EffectSemanticEncoder,
)
from native_runner.training.v4.mechanics import (
    EFFECT_NUMERIC_FEATURE_NAMES,
    EFFECT_TAG_NAMES,
)


@lru_cache(maxsize=1)
def _catalogs():
    native = build_card_catalog()
    logic = build_static_card_logic_catalog(card_catalog=native)
    native_effects = build_effect_catalog(logic)
    projectiles = build_projectile_catalog(logic)
    specs = normal_mode_card_specs(
        native.by_id,
        require_frozen_baseline=True,
    )
    cards = CardCatalogV1.from_card_spec_catalog(native)
    abilities = AbilityCatalogV1.from_card_spec_catalog(native)
    archetypes = EntityArchetypeCatalogV1.from_card_specs(
        specs,
        static_logic=logic,
        projectile_catalog=projectiles,
    )
    effects = EffectSemanticCatalogV1.from_native_catalog(
        specs,
        static_logic=logic,
        native_effect_catalog=native_effects,
    )
    mechanics = MechanicProfileCatalogV1.from_compiled()
    return specs, cards, abilities, archetypes, effects, mechanics


def _effect_row(catalog: EffectSemanticCatalogV1, name: str) -> tuple[float, ...]:
    return catalog.semantic_features[catalog.effect_names.index(name)]


def _card_operations(name: str, form: int = 0):
    specs, cards, _, _, _, mechanics = _catalogs()
    card_id = next(card_id for card_id, spec in specs.items() if spec.name == name)
    profile_id = mechanics.card_form_profile_ids[cards.vocab_id(card_id)][form]
    return mechanics.profile_operations[profile_id - 1]


def test_effect_catalog_keeps_exact_slow_freeze_rage_and_minimum_hp_semantics() -> None:
    _, _, _, _, effects, _ = _catalogs()
    tag = {name: index for index, name in enumerate(EFFECT_TAG_NAMES)}
    number = {
        name: len(EFFECT_TAG_NAMES) + index
        for index, name in enumerate(EFFECT_NUMERIC_FEATURE_NAMES)
    }

    slow = _effect_row(effects, "IceWizardSlowDown")
    assert slow[tag["slow"]] == 1.0
    assert slow[tag["freeze"]] == 0.0
    assert slow[number["move_speed_delta"]] == pytest.approx(-0.30)
    assert slow[number["hit_speed_delta"]] == pytest.approx(-0.30)

    freeze = _effect_row(effects, "Freeze")
    assert freeze[tag["freeze"]] == 1.0
    assert freeze[tag["movement_lock"]] == 1.0
    assert freeze[number["move_speed_delta"]] == pytest.approx(-1.0)

    rage = _effect_row(effects, "Rage")
    assert rage[tag["rage"]] == 1.0
    assert rage[number["move_speed_delta"]] == pytest.approx(0.30)

    berserker = _effect_row(effects, "BerserkerHero_buff")
    assert berserker[tag["minimum_hp"]] == 1.0
    assert berserker[number["damage_multiplier_delta"]] == pytest.approx(0.64)

    minion_ghost = _effect_row(effects, "MinionHorde_EV1_GhostBuff")
    assert minion_ghost[tag["damage_immunity"]] == 1.0
    assert minion_ghost[tag["invisibility"]] == 1.0

    valkyrie_no_push = _effect_row(effects, "Valkyrie_NotPushed_BUF")
    assert valkyrie_no_push[tag["pushback_immunity"]] == 1.0

    bowler_lock = _effect_row(effects, "BowlerHero_ability_buff")
    assert bowler_lock[tag["target_lock_control"]] == 1.0


def test_effect_catalog_covers_the_complete_source_release() -> None:
    _, _, _, _, effects, _ = _catalogs()

    assert len(effects.effect_names) == 170
    assert "ChefTower_increase_level_buff" in effects.effect_names
    assert "WizardHeroAbilityBuff" in effects.effect_names
    assert {
        "BabyDragon_EV1_wind_buff_negative",
        "BabyDragon_EV1_wind_buff_positive",
        "Cannon_EV1_barrage_damage_buff",
        "Ghost_EV1_Summon_Invisibility",
        "GoblinDemolisher_ResetTargetBuff",
        "DarkMagicAOE_Damage_lv3",
        "DarkMagicAOE_Damage_lv2",
        "DarkMagicAOE_Damage_lv1",
        "Hunter_EV1_bear_trap_snare_no_effect",
        "RageDummyBuff",
        "Vines_Trap_Snare_No_Effect",
    }.issubset(effects.effect_names)
    assert "Event_LoveBuff" in effects.effect_names


def test_mechanic_profiles_are_form_scoped_and_structured() -> None:
    _, _, abilities, _, _, mechanics = _catalogs()

    ice_wizard = _card_operations("IceWizard")
    assert any(
        operation.operation == "apply_effect"
        and operation.trigger == "projectile_target_buff"
        for operation in ice_wizard
    )
    assert any("grounding" in operation.mechanic_tags for operation in _card_operations("Vines"))
    hunter = _card_operations("Hunter")
    assert any("multi_projectile" in operation.mechanic_tags for operation in hunter)
    assert any(operation.numeric_features[0] == 1.0 for operation in hunter)
    assert any("recoil" in operation.mechanic_tags for operation in _card_operations("Firecracker"))

    goblin_gang = [
        operation
        for operation in _card_operations("GoblinGang")
        if operation.operation == "spawn_character"
        and operation.trigger == "on_deploy"
    ]
    assert len(goblin_gang) == 2
    assert {operation.numeric_features[0] for operation in goblin_gang} == {0.3}
    assert len({operation.produced_archetype_id for operation in goblin_gang}) == 2

    knight_normal = _card_operations("Knight", 0)
    knight_evolution = _card_operations("Knight", 1)
    assert knight_normal != knight_evolution
    assert not any(operation.operation == "apply_effect" for operation in knight_normal)
    assert any(operation.operation == "apply_effect" for operation in knight_evolution)

    ability_profile = mechanics.ability_profile_ids[
        abilities.vocab_id("Knight_hero_Ability")
    ]
    ability_operations = mechanics.profile_operations[ability_profile - 1]
    assert any("taunt" in operation.mechanic_tags for operation in ability_operations)


def test_mechanic_profiles_keep_distinct_control_and_state_machines() -> None:
    _, _, abilities, _, _, mechanics = _catalogs()

    inferno = _card_operations("InfernoTower")
    assert any("damage_ramp" in operation.mechanic_tags for operation in inferno)
    assert not any("attack_sequence" in operation.mechanic_tags for operation in inferno)

    monk = _card_operations("Monk")
    assert any("attack_sequence" in operation.mechanic_tags for operation in monk)
    assert not any("damage_ramp" in operation.mechanic_tags for operation in monk)

    assert any("parry" in operation.mechanic_tags for operation in _card_operations("Ronin"))
    assert not any(
        "reflect" in operation.mechanic_tags
        for operation in _card_operations("Fireball")
    )
    deflect_profile = mechanics.ability_profile_ids[abilities.vocab_id("Deflect")]
    assert any(
        "reflect" in operation.mechanic_tags
        for operation in mechanics.profile_operations[deflect_profile - 1]
    )

    goblin_giant_evolution = _card_operations("GoblinGiant", 1)
    assert any(
        operation.trigger == "native_action_spawn"
        and operation.operation == "spawn_character"
        and operation.produced_archetype_id > 1
        for operation in goblin_giant_evolution
    )
    goblin_drill_evolution = _card_operations("GoblinDrill", 1)
    assert any(
        operation.trigger == "state_transition"
        and operation.operation == "transform_character"
        for operation in goblin_drill_evolution
    )
    relocation_stages = [
        operation
        for operation in goblin_drill_evolution
        if operation.operation == "health_threshold_relocate_stage"
    ]
    assert [operation.numeric_features[10] for operation in relocation_stages] == [
        0.66,
        0.33,
    ]
    assert [operation.numeric_features[14] for operation in relocation_stages] == [
        0.0,
        0.1,
    ]
    assert not any(
        operation.operation == "health_threshold_relocate_stage"
        for operation in _card_operations("GoblinDrill", 0)
    )


def test_rune_giant_static_contract_is_capacity_period_and_linger() -> None:
    operations = _card_operations("GiantBuffer")

    assert any(
        operation.operation == "typed:nearest_ally_collect_and_buff"
        for operation in operations
    )
    assert any(
        operation.operation == "physics:MaxFriendlyTroops"
        and operation.numeric_features[0] == pytest.approx(0.2)
        for operation in operations
    )
    assert any(
        operation.operation == "typed:periodic_attack_damage_modifier"
        for operation in operations
    )
    assert any(
        operation.operation == "physics:AttackAmount"
        and operation.numeric_features[0] == pytest.approx(0.3)
        for operation in operations
    )
    assert any(
        operation.operation == "physics:FinishIfInstigatorDies"
        and operation.numeric_features[1] == pytest.approx(0.5)
        for operation in operations
    )
    assert any(
        operation.operation == "physics:InstigatorDepth"
        and operation.numeric_features[14] == pytest.approx(0.2)
        for operation in operations
    )
    assert not any(
        operation.operation == "typed:limited_attack_damage_modifier"
        for operation in operations
    )


def test_merge_maiden_keeps_two_exact_elixir_variants() -> None:
    operations = _card_operations("MergeMaiden")
    variants = [
        operation
        for operation in operations
        if operation.operation == "select_card_variant"
    ]
    spawns = [
        operation
        for operation in operations
        if operation.operation == "spawn_character"
        and operation.trigger == "on_deploy"
    ]

    assert [operation.numeric_features[0] for operation in variants] == [0.6, 0.3]
    assert [operation.numeric_features[1] for operation in variants] == [0.12, 0.12]
    assert len(spawns) == 2
    assert {operation.numeric_features[0] for operation in spawns} == {0.1}
    assert len({operation.produced_archetype_id for operation in spawns}) == 2


def test_capture_immunity_pull_and_false_spawn_regressions() -> None:
    goblin_cage = _card_operations("GoblinCage", form=1)
    death_spawns = [
        operation
        for operation in goblin_cage
        if operation.trigger == "on_death"
        and operation.operation == "spawn_character"
    ]
    assert len(death_spawns) == 1
    assert death_spawns[0].numeric_features[0] == pytest.approx(0.1)
    assert any("capture" in operation.mechanic_tags for operation in goblin_cage)
    assert any("pull" in operation.mechanic_tags for operation in goblin_cage)

    lava_hound = _card_operations("LavaHound")
    immunities = [
        operation for operation in lava_hound if operation.operation == "ignore_effect"
    ]
    assert len(immunities) == 2
    assert len({operation.effect_vocab_id for operation in immunities}) == 2
    assert all("effect_immunity" in operation.mechanic_tags for operation in immunities)

    fisherman = _card_operations("Fisherman")
    assert any("pull" in operation.mechanic_tags for operation in fisherman)
    exact = {operation.condition: operation.numeric_features for operation in fisherman}
    assert exact["field:SpecialMinRange"][4] == pytest.approx(0.35)
    assert exact["field:SpecialRange"][5] == pytest.approx(0.70)
    assert exact["field:DragBackSpeed"][7] == pytest.approx(0.85)
    assert exact["field:DragSelfSpeed"][7] == pytest.approx(0.45)

    mega_knight = _card_operations("MegaKnight")
    assert not any("periodic_spawn" in operation.mechanic_tags for operation in mega_knight)
    assert any(
        operation.condition == "field:DashDamage"
        and operation.numeric_features[6] == pytest.approx(0.537)
        for operation in mega_knight
    )


def test_ordered_spawn_and_delayed_area_schedules_are_not_deduplicated() -> None:
    graveyard = _card_operations("Graveyard")
    graveyard_spawns = [
        operation
        for operation in graveyard
        if operation.operation == "spawn_character"
        and operation.trigger == "native_action_spawn"
    ]
    graveyard_delays = sorted(
        operation.numeric_features[1]
        for operation in graveyard
        if operation.operation == "timed_control_edge"
        and operation.condition == "edge:SubActions"
    )
    assert len(graveyard_spawns) == 12
    assert {operation.numeric_features[0] for operation in graveyard_spawns} == {0.1}
    assert graveyard_delays == pytest.approx(
        [0.22, 0.27, 0.33, 0.38, 0.44, 0.49, 0.55, 0.60, 0.65, 0.71, 0.76, 0.82]
    )

    evolved_zap = _card_operations("Zap", form=1)
    zap_delays = {
        operation.numeric_features[1]
        for operation in evolved_zap
        if operation.operation == "timed_control_edge"
    }
    assert 0.145 in zap_delays


def test_elite_archer_parallel_projectiles_keep_exact_identity_and_count() -> None:
    _, _, _, archetypes, _, _ = _catalogs()
    node_ids = (
        "EXT.EliteArcherHero_Ability_Power_Shot_Projectile_Middle",
        "EXT.EliteArcherHero_Ability_Triple_Shot_Projectile",
        "EXT.EliteArcherHero_arrow_projectile",
    )
    expected = {
        node_id: archetypes.node_vocab_id(node_id)
        for node_id in node_ids
    }
    assert all(value > 1 for value in expected.values())
    assert len(set(expected.values())) == 3

    hero = _card_operations("EliteArcher", form=2)
    side_projectiles = [
        operation
        for operation in hero
        if operation.operation == "spawn_projectile"
        and operation.condition == "resource:ProjectileType"
    ]
    assert len(side_projectiles) == 1
    assert side_projectiles[0].produced_archetype_id == expected[
        "EXT.EliteArcherHero_Ability_Triple_Shot_Projectile"
    ]
    assert side_projectiles[0].numeric_features[0] == pytest.approx(0.2)


def test_dark_magic_keeps_three_exact_target_count_damage_bands() -> None:
    operations = _card_operations("DarkMagic")
    damage_rows = [
        operation
        for operation in operations
        if operation.operation == "amount:DamagePerSecond"
    ]
    crown_rows = [
        operation
        for operation in operations
        if operation.operation == "amount:CrownTowerDamagePerHit"
    ]

    assert sorted(operation.numeric_features[6] for operation in damage_rows) == pytest.approx(
        [1.536, 2.944, 6.963]
    )
    assert sorted(operation.numeric_features[6] for operation in crown_rows) == pytest.approx(
        [0.035, 0.051, 0.097]
    )
    assert len({operation.control_path for operation in damage_rows}) == 3
    assert any(
        operation.operation == "typed:target_count_damage_controller"
        for operation in operations
    )


def test_damage_ramps_keep_every_level_11_stage_and_transition_time() -> None:
    expected = {
        "InfernoTower": (0.043, 0.158, 0.847),
        "InfernoDragon": (0.035, 0.120, 0.422),
        "MightyMiner": (0.043, 0.204, 0.409),
    }
    for card_name, damage in expected.items():
        stages = sorted(
            (
                operation
                for operation in _card_operations(card_name)
                if operation.operation.startswith("damage_ramp_stage_")
            ),
            key=lambda operation: operation.operation,
        )
        assert [operation.trigger for operation in stages] == [
            "locked_target_elapsed"
        ] * 3
        assert [operation.numeric_features[1] for operation in stages] == [
            0.2,
            0.2,
            0.0,
        ]
        assert [operation.numeric_features[6] for operation in stages] == list(
            damage
        )

    monk = sorted(
        (
            operation
            for operation in _card_operations("Monk")
            if operation.operation.startswith("attack_sequence_step_")
        ),
        key=lambda operation: operation.operation,
    )
    assert [operation.numeric_features[6] for operation in monk] == [
        0.140,
        0.140,
        0.422,
    ]

    evolved_inferno_dragon = sorted(
        (
            operation
            for operation in _card_operations("InfernoDragon", form=1)
            if operation.operation.startswith("attack_sequence_step_")
        ),
        key=lambda operation: operation.operation,
    )
    assert [operation.trigger for operation in evolved_inferno_dragon] == [
        "attack_sequence"
    ] * 4
    assert [
        operation.numeric_features[6] for operation in evolved_inferno_dragon
    ] == [0.035, 0.120, 0.422, 0.844]
    assert not any(
        operation.operation.startswith("damage_ramp_stage_")
        for operation in _card_operations("InfernoDragon", form=1)
    )
    assert all(
        "attack_sequence" in operation.mechanic_tags
        for operation in evolved_inferno_dragon
    )
    assert not any(
        "damage_ramp" in operation.mechanic_tags
        for operation in _card_operations("InfernoDragon", form=1)
    )


def test_active_effect_encoder_scatter_is_strictly_local() -> None:
    _, _, _, _, effects, _ = _catalogs()
    config = ModelConfigV4()
    semantic_encoder = EffectSemanticEncoder(config, effects).eval()
    active_encoder = ActiveEffectEncoder(config).eval()
    slow_id = effects.vocab_id(
        effects.buff_global_ids[effects.effect_names.index("IceWizardSlowDown")]
    )
    rage_id = effects.vocab_id(
        effects.buff_global_ids[effects.effect_names.index("Rage")]
    )
    active = ActiveEffectSetV4(
        effect_vocab_id=torch.tensor([[slow_id, rage_id]]),
        parent_type=torch.tensor([[0, 1]]),
        parent_index=torch.tensor([[1, 0]]),
        source_owner_type=torch.tensor([[1, 0]]),
        runtime_features=torch.tensor(
            [
                [0.25, 1.0, 0.0, 0.1, 1.0, 0.3, 1.0],
                [0.40, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        mask=torch.tensor([[True, True]]),
    )

    with torch.no_grad():
        static = semantic_encoder.all_effects()
        child, tower = active_encoder(
            active,
            effect_memory=static,
            child_count=3,
            tower_count=2,
        )

    assert torch.count_nonzero(child[0, 1]) > 0
    assert torch.count_nonzero(child[0, 0]) == 0
    assert torch.count_nonzero(child[0, 2]) == 0
    assert torch.count_nonzero(tower[0, 0]) > 0
    assert torch.count_nonzero(tower[0, 1]) == 0
