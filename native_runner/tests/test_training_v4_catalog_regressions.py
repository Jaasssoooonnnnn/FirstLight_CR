from __future__ import annotations

from functools import lru_cache

from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.card_specs import CardSpecCatalog, build_card_catalog
from native_runner.effect_catalog import build_effect_catalog
from native_runner.projectile_catalog import build_projectile_catalog
from native_runner.training.card_features import (
    CARD_MECHANIC_FEATURE_NAMES,
    CARD_STATIC_FEATURE_NAMES,
)
from native_runner.training.v4.catalog import (
    AbilityCatalogV1,
    CardCatalogV1,
    EntityArchetypeCatalogV1,
    NORMAL_MODE_POLICY_CARD_COUNT,
    UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID,
    normal_mode_card_specs,
)
from native_runner.training.v4.config import ABILITY_FEATURE_NAMES
from native_runner.training.v4.mechanics import (
    EFFECT_SEMANTIC_FEATURE_NAMES,
    MECHANIC_NUMERIC_FEATURE_NAMES,
    MECHANIC_TAG_NAMES,
    UNKNOWN_EFFECT_VOCAB_ID,
    EffectSemanticCatalogV1,
    MechanicProfileCatalogV1,
)
from native_runner.training.v4.tensorizer import FORM_HERO, FORM_HERO_ACTIVE


@lru_cache(maxsize=1)
def _production_catalogs() -> tuple[
    CardSpecCatalog,
    CardCatalogV1,
    AbilityCatalogV1,
    EntityArchetypeCatalogV1,
    EffectSemanticCatalogV1,
    MechanicProfileCatalogV1,
]:
    cards = build_card_catalog()
    scoped = normal_mode_card_specs(
        cards.by_id,
        require_frozen_baseline=True,
    )
    static_logic = build_static_card_logic_catalog(card_catalog=cards)
    projectiles = build_projectile_catalog(static_logic)
    native_effects = build_effect_catalog(static_logic)
    card_features = CardCatalogV1.from_card_spec_catalog(
        cards,
        static_logic=static_logic,
    )
    abilities = AbilityCatalogV1.from_card_spec_catalog(cards)
    archetypes = EntityArchetypeCatalogV1.from_card_specs(
        scoped,
        static_logic=static_logic,
        projectile_catalog=projectiles,
    )
    effects = EffectSemanticCatalogV1.from_native_catalog(
        scoped,
        static_logic=static_logic,
        native_effect_catalog=native_effects,
    )
    return (
        cards,
        card_features,
        abilities,
        archetypes,
        effects,
        MechanicProfileCatalogV1.from_compiled(),
    )


def test_production_entity_archetypes_contain_only_runtime_identities() -> None:
    cards, _card_features, _abilities, archetypes, _effects, _mechanics = (
        _production_catalogs()
    )
    expected_scope = tuple(
        sorted(
            normal_mode_card_specs(
                cards.by_id,
                require_frozen_baseline=True,
            )
        )
    )

    assert len(archetypes.card_scope) == NORMAL_MODE_POLICY_CARD_COUNT
    assert archetypes.card_scope == expected_scope
    assert not any(key.startswith("card:") for key in archetypes.archetype_keys)
    assert sum(key.startswith("form:") for key in archetypes.archetype_keys) == 221
    assert sum(key.startswith("projectile:") for key in archetypes.archetype_keys) == 130
    assert sum(key.startswith("area:") for key in archetypes.archetype_keys) == 76
    assert len(archetypes.archetype_keys) == 427
    assert archetypes.vocab_size == 429
    assert all(
        archetypes.character_vocab_id(global_id) != UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
        for global_id, _form_id in archetypes.character_global_ids
    )
    runtime_nodes = (
        "AEO.BabyDragon_EV1_wind_aeo",
        "EXT.Cannon_EV1_barrage_projectile",
        "EXT.ChefTower_pancake_projectile",
        "EXT.ChefTower_spatula_projectile",
        "EXT.Ghost_EV1_Summon_Left",
        "EXT.Ghost_EV1_Summon_Right",
        "EXT.SpearGoblin_Dummy",
        "PROJECTILE.CannoneerProjectile",
        "PROJECTILE.KingProjectile",
        "PROJECTILE.TowerKnifeThrowerProjectile",
        "PROJECTILE.TowerPrincessProjectile",
    )
    assert all(
        archetypes.node_vocab_id(node_id)
        != UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
        for node_id in runtime_nodes
    )
    ghost_left = archetypes.node_vocab_id("EXT.Ghost_EV1_Summon_Left")
    ghost_right = archetypes.node_vocab_id("EXT.Ghost_EV1_Summon_Right")
    assert ghost_left == ghost_right == archetypes.form_vocab_id(
        "Ghost_EV1_Summon_Guardian"
    )
    assert archetypes.form_vocab_id("Ghost_EV1_Summon_Left") == ghost_left
    assert archetypes.form_vocab_id("Ghost_EV1_Summon_Right") == ghost_left

    # Similar damage does not prove identical projectile mechanics.  King and
    # Princess tower shots have different native speed/gravity records.
    assert archetypes.node_vocab_id("PROJECTILE.KingProjectile") != (
        archetypes.node_vocab_id("PROJECTILE.TowerPrincessProjectile")
    )

    # Card scope remains an independent compatibility contract; it no longer
    # forces one unreachable EntityArchetype row per playable card.
    empty = EntityArchetypeCatalogV1((1, 2), ())
    assert empty.card_scope == (1, 2)
    assert empty.vocab_size == 2


def test_production_archetypes_cover_exact_hero_evolution_and_ability_forms() -> None:
    cards, _card_features, abilities, archetypes, _effects, mechanics = (
        _production_catalogs()
    )
    scoped = normal_mode_card_specs(
        cards.by_id,
        require_frozen_baseline=True,
    )
    expected_forms: set[str] = set()
    for spec in scoped.values():
        for attribute_name in ("resolved_hero_form", "resolved_direct_hero"):
            resolved = spec.attributes.get(attribute_name)
            if resolved is not None:
                expected_forms.add(str(resolved["ability_carrier"]))
        if spec.evolution is None:
            continue
        for operation in spec.evolution.effects:
            summoned_form = operation.parameters.get("summoned_form")
            if isinstance(summoned_form, str):
                expected_forms.add(summoned_form)
            overrides = operation.parameters.get("explicit_unit_overrides", {})
            for field_name, override in overrides.items():
                if "Character" not in str(field_name):
                    continue
                evolved = override.get("evolved")
                if isinstance(evolved, str):
                    expected_forms.add(evolved)
    for ability in cards.abilities:
        if ability.source_card_id not in scoped:
            continue
        for operation in ability.effect_graph:
            form = operation.parameters.get("form")
            if operation.effect.lower() == "spawn" and isinstance(form, str):
                expected_forms.add(form)

    assert expected_forms
    assert all(
        archetypes.form_vocab_id(form) != UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
        for form in expected_forms
    )
    assert archetypes.form_vocab_id("MightyMinerBomb") != (
        UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
    )
    for non_entity_name in (
        "AngryBarbarian_EV1_RageAEO",
        "Furnace_EV1_Spawn_Spirit_Projectile",
        "Vines_Trap_Snare_Small",
    ):
        assert (
            archetypes.form_vocab_id(non_entity_name)
            == UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
        )
    assert not [
        operation
        for operations in mechanics.profile_operations
        for operation in operations
        if operation.operation in {"spawn_character", "transform_character"}
        and operation.produced_archetype_id == UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
    ]

    mighty_miner_profile = mechanics.ability_profile_ids[
        abilities.vocab_id("MightyMinerLaneSwitch")
    ]
    assert any(
        operation.operation == "spawn_character"
        and operation.produced_archetype_id
        == archetypes.form_vocab_id("MightyMinerBomb")
        for operation in mechanics.profile_operations[mighty_miner_profile - 1]
    )


def test_resolved_hero_forms_use_their_exact_static_profile() -> None:
    cards, card_features, _abilities, _archetypes, _effects, mechanics = (
        _production_catalogs()
    )
    for card_id in card_features.raw_card_ids:
        spec = cards.by_id[card_id]
        resolved = spec.attributes.get("resolved_hero_form")
        if resolved is None:
            continue
        profile_key = f"root:SPELL_HERO.{resolved['form_id']}:ability=0"
        expected_profile = 1 + mechanics.profile_keys.index(profile_key)
        mapping = mechanics.card_form_profile_ids[card_features.vocab_id(card_id)]
        assert mapping[FORM_HERO] == expected_profile
        assert mapping[FORM_HERO_ACTIVE] == expected_profile

    for card_id in (26_000_014, 26_000_017):
        mapping = mechanics.card_form_profile_ids[card_features.vocab_id(card_id)]
        assert mapping[FORM_HERO] != mapping[0]


def test_competitive_ability_effects_are_in_the_effect_catalog() -> None:
    cards, _card_features, abilities, _archetypes, effects, mechanics = (
        _production_catalogs()
    )
    static_logic = build_static_card_logic_catalog(card_catalog=cards)
    native_effects = build_effect_catalog(static_logic)
    wizard_buff = next(
        effect
        for effect in native_effects.effects
        if effect.effect_name == "WizardHeroAbilityBuff"
    )
    effect_vocab_id = effects.vocab_id(wizard_buff.buff_global_id)
    assert effect_vocab_id != UNKNOWN_EFFECT_VOCAB_ID

    ability_profile = mechanics.ability_profile_ids[
        abilities.vocab_id("WizardHeroAbility")
    ]
    assert any(
        operation.operation == "apply_effect"
        and operation.effect_vocab_id == effect_vocab_id
        for operation in mechanics.profile_operations[ability_profile - 1]
    )


def test_native_missing_ability_cooldown_is_known_zero_only() -> None:
    cards, _card_features, abilities, _archetypes, _effects, _mechanics = (
        _production_catalogs()
    )
    by_id = {spec.ability_id: spec for spec in cards.abilities}
    missing_cooldown = 0
    explicit_cooldown = 0

    assert len(abilities.ability_ids) == 24
    for ability_id, row in zip(
        abilities.ability_ids,
        abilities.static_features,
        strict=True,
    ):
        spec = by_id[ability_id]
        if spec.cooldown_ms is None:
            missing_cooldown += 1
            assert row[1] == 0.0
        else:
            explicit_cooldown += 1
            assert row[1] == float(spec.cooldown_ms) / 30_000.0

        assert row[0] == float(spec.elixir_cost) / 10.0
        assert row[2] == float(spec.cast_time_ms) / 5_000.0
        assert row[3] == float(spec.charges) / 4.0

    assert missing_cooldown == 22
    assert explicit_cooldown == 2


def test_production_catalog_contents_are_frozen() -> None:
    _cards, card_features, abilities, archetypes, effects, mechanics = (
        _production_catalogs()
    )

    assert card_features.catalog_id == (
        "3c90f5d943c0be7337c8f867f3e10ce9b37c0d0acd73ffb9ddc2119726b5a289"
    )
    assert abilities.catalog_id == (
        "9f8f1ff4bac217d0cfcd352ff05d888fbaca29849b4588797d52f0f556d1a9ed"
    )
    assert archetypes.catalog_id == (
        "8945cd3e8e35b758b673f114d7c55ab8b263b5a1b6b8bfaf27573362fea5d714"
    )
    assert effects.catalog_id == (
        "15ca0fba148b6f524401c033799b7d76a4e4643deb7fae71b5ee61e669c68bee"
    )
    assert mechanics.catalog_id == (
        "3c20b32a8a1b030837833e97200f48b91c7eb8c4da5501dc9791fb06a8dcf0b3"
    )
    assert len(effects.effect_names) == 170
    assert effects.vocab_size == 172
    assert mechanics.profile_count == 634
    assert sum(map(len, mechanics.profile_operations)) == 14_606


def test_production_numeric_catalogs_have_no_dead_or_duplicate_columns() -> None:
    _cards, card_features, abilities, _archetypes, effects, mechanics = (
        _production_catalogs()
    )

    mechanic_tables = mechanics.tensor_tables()

    for names, rows in (
        (CARD_STATIC_FEATURE_NAMES, card_features.static_features),
        (CARD_MECHANIC_FEATURE_NAMES, card_features.mechanic_features),
        (ABILITY_FEATURE_NAMES, abilities.static_features),
        (EFFECT_SEMANTIC_FEATURE_NAMES, effects.semantic_features),
        (MECHANIC_TAG_NAMES, mechanic_tables["tags"].tolist()),
        (MECHANIC_NUMERIC_FEATURE_NAMES, mechanic_tables["numeric"].tolist()),
    ):
        columns = tuple(zip(*rows, strict=True))
        assert len(columns) == len(names)
        assert all(len(set(column)) > 1 for column in columns)
        assert len(set(columns)) == len(columns)
