from __future__ import annotations

import pytest

from native_runner.contracts import (
    ABILITY_RUNTIME_STATE_FIELDS,
    ATTACK_STATE_FIELDS,
    COMBAT_EVENT_FIELDS,
    EFFECT_STATE_FIELDS,
    ENTITY_RUNTIME_SEMANTIC_FIELDS,
    EVOLUTION_RUNTIME_STATE_FIELDS,
    PROJECTILE_STATE_FIELDS,
    SHIELD_STATE_FIELDS,
    VISIBILITY_STATE_FIELDS,
    AbilityPhase,
    AbilityRuntimeStateV1,
    AttackPhase,
    AttackStateV1,
    CombatEventKind,
    CombatEventV1,
    ContractError,
    EffectKind,
    EffectStateV1,
    EntityStateV1,
    EvolutionPhase,
    EvolutionRuntimeStateV1,
    ProjectilePhase,
    ProjectileStateV1,
    SemanticEvidenceLevel,
    SemanticProvenanceV1,
    ShieldStateV1,
    VisibilityPhase,
    VisibilityStateV1,
)


def _native(fields: frozenset[str], tick: int = 10) -> SemanticProvenanceV1:
    return SemanticProvenanceV1(
        {
            name: SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
            for name in sorted(fields)
        },
        observed_tick=tick,
    )


def _entity_runtime_provenance() -> SemanticProvenanceV1:
    return SemanticProvenanceV1({
        "shield_state": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
        "effect_states": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
        "attack_state": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
        "projectile_state": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
        "visibility_state": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
        "ability_states": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
        "evolution_state": SemanticEvidenceLevel.NATIVE_AUTHORITATIVE,
    })


def test_typed_runtime_state_round_trips_without_losing_provenance() -> None:
    shield = ShieldStateV1(
        hitpoints=120,
        max_hitpoints=200,
        kind="spawn_shield",
        source_entity=7,
        broken=False,
        broken_tick=None,
        provenance=_native(SHIELD_STATE_FIELDS),
    )
    effect = EffectStateV1(
        effect_id="rage:7:11",
        kind=EffectKind.RAGE,
        source_entity=7,
        source_card_id=28_000_002,
        source_owner=0,
        started_tick=10,
        end_tick=110,
        remaining_ms=5_000,
        stacks=1,
        magnitude=1.35,
        stage="active",
        active=True,
        provenance=_native(EFFECT_STATE_FIELDS),
    )
    attack = AttackStateV1(
        phase=AttackPhase.CHARGING,
        target_entity=99,
        target_position=(8_500, 20_000),
        phase_started_tick=10,
        phase_remaining_ms=500,
        cooldown_remaining_ms=0,
        sequence_index=2,
        charge_stage=2,
        charge_elapsed_ms=2_250,
        damage_multiplier=4.0,
        locked=True,
        interrupted=False,
        provenance=_native(ATTACK_STATE_FIELDS),
    )
    projectile = ProjectileStateV1(
        projectile_id="p:17",
        phase=ProjectilePhase.IN_FLIGHT,
        source_entity=7,
        source_card_id=26_000_037,
        target_entity=99,
        target_position=(8_500, 20_000),
        spawn_tick=10,
        expected_impact_tick=14,
        impact_tick=None,
        velocity=(0, 1_000),
        damage=47,
        radius_tiles=0,
        homing=True,
        deflected=False,
        provenance=_native(PROJECTILE_STATE_FIELDS),
    )
    visibility = VisibilityStateV1(
        phase=VisibilityPhase.VISIBLE,
        public_by_owner={"0": True, "1": True},
        targetable_by_owner={"0": True, "1": True},
        area_damage_eligible=True,
        transition_remaining_ms=0,
        provenance=_native(VISIBILITY_STATE_FIELDS),
    )
    ability = AbilityRuntimeStateV1(
        ability_id="RapidShot",
        source_entity=7,
        phase=AbilityPhase.COOLDOWN,
        elixir_cost=1,
        cooldown_ms=15_000,
        remaining_cooldown_ms=5_000,
        charges=0,
        available=False,
        cast_started_tick=10,
        active_until_tick=30,
        target_entity=99,
        target_position=(8_500, 20_000),
        provenance=_native(ABILITY_RUNTIME_STATE_FIELDS),
    )
    evolution = EvolutionRuntimeStateV1(
        card_id=26_000_000,
        deck_slot=0,
        phase=EvolutionPhase.EVOLVED,
        base_form_id="Knight",
        current_form_id="KnightEvo",
        next_form_id="Knight",
        cycle_required=2,
        cycle_remaining=2,
        ready=False,
        active=True,
        deployments_in_cycle=0,
        provenance=_native(EVOLUTION_RUNTIME_STATE_FIELDS),
    )
    entity = EntityStateV1(
        entity_id=7,
        owner=0,
        card_id=26_000_037,
        entity_kind="troop",
        position=(8_500, 16_000),
        hitpoints=900,
        max_hitpoints=1_000,
        shield=120,
        shield_state=shield,
        effect_states=(effect,),
        attack_state=attack,
        projectile_state=projectile,
        visibility_state=visibility,
        ability_states=(ability,),
        evolution_state=evolution,
        runtime_provenance=_entity_runtime_provenance(),
    )

    restored = EntityStateV1.from_mapping(entity.to_dict())

    assert restored == entity
    assert restored.attack_state is not None
    assert restored.attack_state.charge_stage == 2
    assert restored.effect_states[0].remaining_ms == 5_000
    assert restored.visibility_state is not None
    restored.visibility_state.assert_public_for(1)
    declared_domains = _entity_runtime_provenance().field_evidence.keys()
    restored.assert_runtime_semantics(declared_domains, require_authoritative=True)
    unknown_domains = ENTITY_RUNTIME_SEMANTIC_FIELDS.difference(declared_domains)
    assert set(restored.runtime_provenance.unknown_fields) == unknown_domains
    with pytest.raises(ContractError, match="movement_runtime"):
        restored.assert_runtime_semantics(
            unknown_domains,
            require_authoritative=True,
        )


def test_runtime_defaults_are_explicitly_unknown_and_fail_closed() -> None:
    entity = EntityStateV1(
        entity_id=1,
        owner=0,
        card_id=26_000_000,
        entity_kind="troop",
        position=(1, 2),
    )

    assert set(entity.runtime_provenance.unknown_fields) == set(
        ENTITY_RUNTIME_SEMANTIC_FIELDS
    )
    assert entity.visible is None
    with pytest.raises(ContractError, match="fails closed"):
        entity.assert_runtime_semantics(("visibility_state",))


def test_entity_state_preserves_native_overheal_above_static_maximum() -> None:
    entity = EntityStateV1(
        entity_id=54,
        owner=1,
        card_id=13_000_007,
        entity_kind="troop",
        position=(14_098, 16_233),
        hitpoints=1_586,
        max_hitpoints=1_341,
    )

    assert entity.hitpoints == 1_586
    assert entity.max_hitpoints == 1_341
    assert entity.hitpoints / entity.max_hitpoints > 1.0


def test_runtime_nested_mappings_reject_unknown_fields() -> None:
    payload = AttackStateV1().to_dict()
    payload["invented_native_flag"] = True

    with pytest.raises(ContractError, match="unknown attack state fields"):
        AttackStateV1.from_mapping(payload)

    incomplete = SemanticProvenanceV1.unknown_all(("phase",))
    with pytest.raises(ContractError, match="cover every field exactly"):
        AttackStateV1(provenance=incomplete)


def test_typed_state_requires_positive_aggregate_provenance() -> None:
    effect = EffectStateV1(
        effect_id="slow:1",
        kind=EffectKind.SLOW,
        provenance=SemanticProvenanceV1.unknown_all(EFFECT_STATE_FIELDS),
    )
    with pytest.raises(ContractError, match="without positive provenance"):
        EntityStateV1(
            entity_id=1,
            owner=0,
            card_id=26_000_000,
            entity_kind="troop",
            position=(1, 2),
            effect_states=(effect,),
        )


def test_combat_event_taxonomy_carries_cause_target_and_visibility() -> None:
    event = CombatEventV1(
        kind=CombatEventKind.ATTACK_INTERRUPT,
        source_entity=20,
        source_card_id=26_000_042,
        target_entity=30,
        target_card_id=26_000_037,
        position=(8_500, 16_000),
        amount=0,
        effect_id="stun:20:30",
        projectile_id="zap:20:30",
        ability_id=None,
        evolution_form_id=None,
        visible_by_owner={"0": True, "1": True},
        provenance=_native(COMBAT_EVENT_FIELDS),
    )

    assert CombatEventV1.from_mapping(event.to_dict()) == event
