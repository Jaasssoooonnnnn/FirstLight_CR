from __future__ import annotations

from functools import lru_cache

from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.card_specs import build_card_catalog
from native_runner.projectile_catalog import build_projectile_catalog
from native_runner.training.v4.catalog import (
    ENTITY_ARCHETYPE_STATIC_DAMAGE_LEVEL,
    EntityArchetypeCatalogV1,
    normal_mode_card_specs,
)


@lru_cache(maxsize=1)
def _catalogs():
    cards = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=cards)
    projectiles = build_projectile_catalog(static_logic)
    archetypes = EntityArchetypeCatalogV1.from_card_specs(
        normal_mode_card_specs(cards.by_id, require_frozen_baseline=True),
        static_logic=static_logic,
        projectile_catalog=projectiles,
    )
    return static_logic, projectiles, archetypes


def _form_metadata(form_id: str):
    _static, _projectiles, archetypes = _catalogs()
    return archetypes.metadata_for_vocab_id(archetypes.form_vocab_id(form_id))


def test_exact_character_metadata_uses_level_11_carrier_stats() -> None:
    assert ENTITY_ARCHETYPE_STATIC_DAMAGE_LEVEL == 11

    bat = _form_metadata("Bat")
    assert (bat.child_kind, bat.child_kind_known) == ("character", True)
    assert (bat.is_airborne, bat.is_airborne_known) == (True, True)
    assert (bat.static_damage_basis, bat.static_damage_basis_known) == (81.0, True)

    dark_witch = _form_metadata("DarkWitch")
    assert (dark_witch.is_airborne, dark_witch.is_airborne_known) == (False, True)
    assert (
        dark_witch.static_damage_basis,
        dark_witch.static_damage_basis_known,
    ) == (314.0, True)

    # Ranged character damage comes from its one exact direct-attack
    # projectile edge, never from a parent playable-card fallback.
    archer = _form_metadata("Archer")
    assert (archer.static_damage_basis, archer.static_damage_basis_known) == (
        112.0,
        True,
    )

    rascal_girl = _form_metadata("RascalGirl")
    assert (
        rascal_girl.static_damage_basis,
        rascal_girl.static_damage_basis_known,
    ) == (125.0, True)


def test_exact_inheritance_and_building_kind_are_data_driven() -> None:
    evolved_pekka = _form_metadata("Pekka_EV1")
    assert (evolved_pekka.child_kind, evolved_pekka.child_kind_known) == (
        "character",
        True,
    )
    assert evolved_pekka.static_damage_basis == 842.0
    assert evolved_pekka.static_damage_basis_known is True

    cannon = _form_metadata("Cannon")
    assert (cannon.child_kind, cannon.child_kind_known) == ("building", True)
    assert (cannon.is_airborne, cannon.is_airborne_known) == (False, True)
    assert (cannon.static_damage_basis, cannon.static_damage_basis_known) == (
        202.0,
        True,
    )


def test_projectile_damage_requires_an_authoritative_numeric_field() -> None:
    _static, projectiles, archetypes = _catalogs()

    arrow_definition = projectiles.by_name["ArcherArrow"]
    arrow = archetypes.metadata_for_vocab_id(
        archetypes.projectile_vocab_id(arrow_definition.projectile_global_id)
    )
    assert (arrow.child_kind, arrow.child_kind_known) == ("projectile", True)
    assert (arrow.is_airborne, arrow.is_airborne_known) == (False, False)
    assert (arrow.static_damage_basis, arrow.static_damage_basis_known) == (
        112.0,
        True,
    )
    assert arrow.projectile_radius_known is False

    hunter_definition = projectiles.by_name["HunterProjectile"]
    hunter = archetypes.metadata_for_vocab_id(
        archetypes.projectile_vocab_id(hunter_definition.projectile_global_id)
    )
    assert (hunter.static_damage_basis, hunter.static_damage_basis_known) == (
        84.0,
        True,
    )
    assert (hunter.projectile_radius_tiles, hunter.projectile_radius_known) == (
        0.3,
        True,
    )

    firecracker_definition = projectiles.by_name["FirecrackerProjectile"]
    firecracker = archetypes.metadata_for_vocab_id(
        archetypes.projectile_vocab_id(
            firecracker_definition.projectile_global_id
        )
    )
    assert firecracker.child_kind == "projectile"
    assert firecracker.static_damage_basis == 0.0
    assert firecracker.static_damage_basis_known is False


def test_area_archetypes_keep_exact_geometry_and_lifetime() -> None:
    _static, _projectiles, archetypes = _catalogs()

    poison = archetypes.metadata_for_vocab_id(
        archetypes.node_vocab_id("AEO.Poison")
    )
    assert (poison.child_kind, poison.child_kind_known) == ("area", True)
    assert (poison.projectile_radius_tiles, poison.projectile_radius_known) == (
        3.5,
        True,
    )
    assert (poison.lifetime_ms, poison.lifetime_known) == (8_000.0, True)

    graveyard = archetypes.metadata_for_vocab_id(
        archetypes.node_vocab_id("AEO.Graveyard_rework")
    )
    assert (graveyard.projectile_radius_tiles, graveyard.lifetime_ms) == (
        4.0,
        9_000.0,
    )
