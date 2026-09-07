from __future__ import annotations

from dataclasses import fields, replace
from functools import lru_cache

import torch
import pytest

from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.card_specs import build_card_catalog
from native_runner.effect_catalog import build_effect_catalog
from native_runner.contracts import (
    ActionKind,
    ActionMaskV1,
    AbilityPhase,
    AbilityRuntimeStateV1,
    AttackPhase,
    AttackStateV1,
    CardSpecV1,
    CAPTURE_RUNTIME_STATE_FIELDS,
    CaptureActionPhase,
    CaptureRuntimeStateV1,
    CaptureTargetPhase,
    CaptureTargetStateV1,
    CausalGroupKind,
    CausalGroupRefV1,
    CombatEventKind,
    CombatEventV1,
    ContractError,
    DaggerDuchessRuntimeStateV1,
    ENTITY_RUNTIME_SEMANTIC_FIELDS,
    ENTITY_RESOURCE_STATE_FIELDS,
    EntityResourceStateV1,
    EntityStateV1,
    EffectKind,
    EffectStateV1,
    EventV1,
    EvolutionRuntimeStateV1,
    ObservationTier,
    ObservationV1,
    PLAYER_RUNTIME_SEMANTIC_FIELDS,
    PERIODIC_ATTACK_MODIFIER_STATE_FIELDS,
    PeriodicAttackModifierPhase,
    PeriodicAttackModifierStateV1,
    PlayerStateV1,
    PROJECTILE_STATE_FIELDS,
    ProjectileDragStage,
    ProjectilePhase,
    ProjectileStateV1,
    SemanticEvidenceLevel,
    SemanticProvenanceV1,
    RoyalChefRuntimeStateV1,
    TOWER_RUNTIME_SEMANTIC_FIELDS,
    THRESHOLD_RELOCATION_STATE_FIELDS,
    ThresholdRelocationPhase,
    ThresholdRelocationStateV1,
    TowerStateV1,
)
from native_runner.match_factory import STANDARD_DECK
from native_runner.projectile_catalog import build_projectile_catalog
from native_runner.training.tracking import DeterministicPublicTracker
from native_runner.training.v4.tensors import ActionSequenceV4, GATE_ACT
from native_runner.training.v4.catalog import AbilityCatalogV1, CardCatalogV1, EntityArchetypeCatalogV1, PAD_CARD_VOCAB_ID, UNKNOWN_CARD_VOCAB_ID, UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID, normal_mode_card_specs
from native_runner.training.v4.mechanics import EffectSemanticCatalogV1, MechanicProfileCatalogV1
from native_runner.training.v4.decoding import ShadowCandidateLegality, decode_action_sequence_v4
from native_runner.training.v4.model import UniversalCardPolicyV4
from native_runner.training.v4.tensorizer import UniversalObservationTensorizerV4
from native_runner.training.v4.config import MATCH_SCALAR_NAMES
from native_runner.training.v4.tensorizer import (
    EVENT_TYPE,
    FORM_EVOLUTION,
    FORM_HERO_ACTIVE,
    IGNORED_EVENT_TYPES,
)

STANDARD_CARD_COSTS = {
    26_000_000: 3.0,
    26_000_001: 3.0,
    26_000_005: 3.0,
    28_000_001: 3.0,
    28_000_000: 4.0,
    26_000_003: 5.0,
    26_000_014: 4.0,
    26_000_018: 4.0,
}


def _towers() -> tuple[TowerStateV1, ...]:
    return tuple(
        TowerStateV1(
            entity_id=entity_id,
            owner=owner,
            tower_kind=kind,
            position=(x, y),
            hitpoints=3000.0,
            max_hitpoints=3000.0,
            tower_troop_id=(None if kind == "king" else 159_000_000),
        )
        for entity_id, owner, kind, x, y in (
            (1, 0, "king", 9000, 3000),
            (2, 0, "princess_left", 3500, 6500),
            (3, 0, "princess_right", 14500, 6500),
            (4, 1, "king", 9000, 29000),
            (5, 1, "princess_left", 3500, 25500),
            (6, 1, "princess_right", 14500, 25500),
        )
    )


def test_event_catalog_classifies_every_exact_non_unknown_combat_kind() -> None:
    classified = set(EVENT_TYPE).union(IGNORED_EVENT_TYPES)

    assert {
        kind.value for kind in CombatEventKind if kind != CombatEventKind.UNKNOWN
    }.issubset(classified)
    assert sorted(EVENT_TYPE.values()) == list(range(1, 39))


@lru_cache(maxsize=1)
def _semantic_bundle() -> tuple[
    dict[int, CardSpecV1],
    CardCatalogV1,
    AbilityCatalogV1,
    EntityArchetypeCatalogV1,
    EffectSemanticCatalogV1,
    MechanicProfileCatalogV1,
]:
    native = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=native)
    projectile_catalog = build_projectile_catalog(static_logic)
    native_effect_catalog = build_effect_catalog(static_logic)
    specs = normal_mode_card_specs(native.by_id, require_frozen_baseline=True)
    cards = CardCatalogV1.from_card_spec_catalog(native)
    abilities = AbilityCatalogV1.from_card_spec_catalog(native)
    archetypes = EntityArchetypeCatalogV1.from_card_specs(
        specs,
        static_logic=static_logic,
        projectile_catalog=projectile_catalog,
    )
    effects = EffectSemanticCatalogV1.from_native_catalog(
        specs,
        static_logic=static_logic,
        native_effect_catalog=native_effect_catalog,
    )
    mechanics = MechanicProfileCatalogV1.from_compiled()
    return specs, cards, abilities, archetypes, effects, mechanics


def _semantic_catalog() -> tuple[
    dict[int, CardSpecV1],
    CardCatalogV1,
    AbilityCatalogV1,
    EntityArchetypeCatalogV1,
]:
    specs, cards, abilities, archetypes, _, _ = _semantic_bundle()
    return specs, cards, abilities, archetypes


def _deck_roles(
    overrides: dict[int, tuple[bool, bool]] | None = None,
) -> dict[int, tuple[bool, bool]]:
    result = {card_id: (False, False) for card_id in STANDARD_DECK}
    result.update(overrides or {})
    return result


def _tensorizer(
    specs: dict[int, CardSpecV1],
    catalog: CardCatalogV1,
    abilities: AbilityCatalogV1,
    archetypes: EntityArchetypeCatalogV1,
    *,
    deck_roles: dict[int, tuple[bool, bool]] | None = None,
    tracker: DeterministicPublicTracker | None = None,
    initial_elixir: dict[int, float] | None = None,
    start: bool = True,
    reject_unknown_public_semantics: bool = False,
) -> UniversalObservationTensorizerV4:
    effects = _semantic_bundle()[4]
    tensorizer = UniversalObservationTensorizerV4(
        actor_owner=0,
        card_catalog=catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        effect_catalog=effects,
        card_specs=specs,
        deck=STANDARD_DECK,
        tracker=tracker,
        deck_roles=_deck_roles(deck_roles),
        reject_unknown_public_semantics=reject_unknown_public_semantics,
    )
    if start:
        tensorizer.start_episode(
            _observation(skeleton_count=0),
            initial_elixir=(
                initial_elixir
                if tracker is not None
                else None
            ),
        )
    return tensorizer


def _tracker(
    specs: dict[int, CardSpecV1],
    *,
    evolution_card_id: int | None = None,
) -> DeterministicPublicTracker:
    tracker = DeterministicPublicTracker(
        decks={0: STANDARD_DECK, 1: STANDARD_DECK},
        card_costs={
            card_id: float(specs[card_id].elixir_cost)
            for card_id in STANDARD_DECK
        },
        ability_cost_by_owner={0: 1.0, 1: 1.0},
        ability_cooldown_ms_by_owner={0: 1_000, 1: 1_000},
        evolution_cycle_required=(
            {}
            if evolution_card_id is None
            else {evolution_card_id: 2}
        ),
    )
    return tracker


def _ability_entity() -> EntityStateV1:
    ability = AbilityRuntimeStateV1(
        ability_id="Knight_hero_Ability",
        source_entity=501,
        phase=AbilityPhase.READY,
        elixir_cost=2.0,
        available=True,
        attributes={
            "source_card_id": STANDARD_DECK[0],
            "controller_slot": 0,
        },
    )
    return EntityStateV1(
        entity_id=501,
        owner=0,
        card_id=STANDARD_DECK[0],
        entity_kind="hero",
        position=(9000, 12000),
        hitpoints=1000.0,
        max_hitpoints=1766.0,
        causal_group=CausalGroupRefV1(
            kind=CausalGroupKind.DEPLOYMENT,
            handle="deploy:1:501",
            source_card_id=STANDARD_DECK[0],
        ),
        ability_states=(ability,),
        runtime_provenance=SemanticProvenanceV1(
            field_evidence={
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "ability_states"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )


def _observation(*, skeleton_count: int = 15) -> ObservationV1:
    rows = [[True] * 18 for _ in range(32)]
    placements = {
        str(slot): {
            "card_id": card_id,
            "effective_cost": STANDARD_CARD_COSTS[card_id],
            "form_code": 0,
            "shape": [32, 18],
            "row_major": rows,
        }
        for slot, card_id in enumerate(STANDARD_DECK[:4])
    }
    skeletons = tuple(
        EntityStateV1(
            entity_id=1000 + index,
            owner=0,
            card_id=STANDARD_DECK[1],
            entity_kind="troop",
            position=(5000, 9000),
            hitpoints=100.0,
            max_hitpoints=100.0,
            age_ms=250,
            source_entity=999,
            causal_group=CausalGroupRefV1(
                kind=CausalGroupKind.DEPLOYMENT,
                handle="deploy:1:999",
                source_card_id=STANDARD_DECK[1],
            ),
        )
        for index in range(skeleton_count)
    )
    return ObservationV1(
        tier=ObservationTier.FAIR,
        owner=0,
        tick=20,
        episode_id="v4-test-episode",
        phase="regulation",
        players=(
            PlayerStateV1(
                owner=0,
                elixir_exact=10.0,
                elixir_visible=10.0,
                hand=STANDARD_DECK[:4],
                deck=STANDARD_DECK,
                cycle=STANDARD_DECK[4:],
                private_state_visible=True,
                ability_runtime_states=_ability_entity().ability_states,
                metadata={
                    "hand_slot_by_card": {
                        str(card_id): slot
                        for slot, card_id in enumerate(STANDARD_DECK[:4])
                    },
                    "hand_runtime_by_slot": {
                        str(slot): {"form_code": 0}
                        for slot in range(4)
                    },
                },
                runtime_provenance=SemanticProvenanceV1(
                    field_evidence={
                        field: (
                            SemanticEvidenceLevel.NATIVE_DERIVED
                            if field == "ability_runtime_states"
                            else SemanticEvidenceLevel.UNKNOWN
                        )
                        for field in PLAYER_RUNTIME_SEMANTIC_FIELDS
                    }
                ),
            ),
            PlayerStateV1(
                owner=1,
                revealed_cards=(STANDARD_DECK[1],),
            ),
        ),
        towers=_towers(),
        entities=(*skeletons, _ability_entity()),
        action_mask=ActionMaskV1(
            kinds={
                ActionKind.WAIT.value: True,
                ActionKind.PLAY_CARD.value: True,
                ActionKind.ACTIVATE_ABILITY.value: True,
            },
            hand_slots=(True, True, True, True),
            placement_masks=placements,
            ability_sources=(501,),
            reasons={"effective_elixir": 10.0, "reserved_elixir": 0.0},
        ),
    )


def test_live_tensorizer_builds_sorted_cards_grouped_children_and_candidates() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    tensorizer = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        deck_roles={
            STANDARD_DECK[0]: (True, False),
            STANDARD_DECK[1]: (False, True),
        },
    )

    batch = tensorizer.tensorize(_observation())

    expected = sorted(STANDARD_DECK)
    assert batch.own_cards.card_vocab_id[0].tolist() == [
        catalog.vocab_id(card_id) for card_id in expected
    ]
    assert int(batch.groups.mask.sum()) == 2
    assert int(batch.groups.child_mask.sum()) == 16
    group_counts = batch.groups.features[0, batch.groups.mask[0], 0] * 16.0
    assert sorted(group_counts.tolist()) == [1.0, 15.0]
    assert batch.explicit_spatial_planes[0, 0].sum() == 16.0
    assert int(batch.candidates.mask.sum()) == 5
    assert (
        int((batch.candidates.variant == 1).logical_and(batch.candidates.mask).sum())
        == 1
    )
    ability_row = int((batch.candidates.variant[0] == 1).nonzero(as_tuple=False)[0])
    assert int(batch.candidates.native_source_entity[0, ability_row]) == 501
    assert batch.opponent_cards.mask[0].sum() == 2
    assert int(batch.opponent_cards.card_vocab_id[0, 8]) == UNKNOWN_CARD_VOCAB_ID
    assert float(batch.opponent_cards.runtime_features[0, 8, 15]) == pytest.approx(
        7.0 / 8.0
    )
    elite_row = int(
        (batch.own_cards.card_vocab_id[0] == catalog.vocab_id(STANDARD_DECK[0]))
        .nonzero(as_tuple=False)[0]
    )
    assert bool(batch.own_cards.role_bits[0, elite_row, 0])
    assert int(batch.own_cards.runtime_form[0, elite_row]) == 0

    model = UniversalCardPolicyV4(
        catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        effect_catalog=_semantic_bundle()[4],
        mechanic_profile_catalog=_semantic_bundle()[5],
    ).eval()
    state = model.initial_state(1)
    with torch.no_grad():
        output = model.act(batch, state)
    assert output.value.shape == (1,)

    action = ActionSequenceV4(
        gate=torch.tensor([GATE_ACT]),
        micro_action_count=torch.tensor([1]),
        candidate_index=torch.tensor([[ability_row, -1]]),
        candidate_uid=torch.tensor([[int(batch.candidates.uid[0, ability_row]), -1]]),
        target_cell=torch.tensor([[-1, -1]]),
        delay_offset_bin=torch.tensor([[0, -1]]),
    )
    compound = decode_action_sequence_v4(
        action,
        batch.candidates,
        row=0,
        observation=_observation(),
        catalog=catalog,
        deck=STANDARD_DECK,
        card_costs=tensorizer.card_costs,
        ability_id_by_vocab_id=tensorizer.ability_id_by_vocab_id,
    )
    assert compound.actions[0].kind == ActionKind.ACTIVATE_ABILITY
    assert compound.actions[0].source_entity == 501

    tensorizer.record_action(action, batch)
    next_batch = tensorizer.tensorize(_observation())
    assert int(next_batch.previous_action.gate[0]) == GATE_ACT
    assert int(next_batch.previous_action.micro_action_count[0]) == 1
    assert int(next_batch.previous_action.ability_vocab_id[0, 0]) > 0
    assert int(next_batch.previous_action.effective_form[0, 0]) == int(
        batch.candidates.effective_form[0, ability_row]
    )
    assert bool(next_batch.previous_action.source_mask[0, 0])


def test_tensorizer_episode_lifecycle_binds_identity_order_and_tracker() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    tracker = _tracker(specs)
    tensorizer = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        tracker=tracker,
        start=False,
    )
    observation = _observation(skeleton_count=0)

    with pytest.raises(RuntimeError, match="start_episode"):
        tensorizer.tensorize(observation)

    tensorizer.start_episode(
        observation,
        initial_elixir={0: 7.0, 1: 6.0},
    )
    assert tracker.elixir_interval(0) == (7.0, 7.0)
    assert tracker.elixir_interval(1) == (6.0, 6.0)
    tensorizer.tensorize(observation)

    with pytest.raises(ValueError, match="episode_id"):
        tensorizer.tensorize(
            replace(observation, episode_id="different-episode")
        )
    with pytest.raises(ValueError, match="move backwards"):
        tensorizer.tensorize(replace(observation, tick=19))

    tensorizer.end_episode()
    with pytest.raises(RuntimeError, match="not initialized"):
        tracker.elixir_interval(1)
    with pytest.raises(RuntimeError, match="start_episode"):
        tensorizer.tensorize(observation)

    next_observation = replace(
        observation,
        episode_id="v4-test-next-episode",
        tick=40,
    )
    tensorizer.start_episode(
        next_observation,
        initial_elixir={0: 5.0, 1: 4.0},
    )
    assert tracker.elixir_interval(0) == (5.0, 5.0)
    assert tracker.elixir_interval(1) == (4.0, 4.0)


def test_enemy_elixir_missingness_uses_full_interval_not_exact_zero() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    observation = _observation(skeleton_count=0)

    unknown = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
    ).tensorize(observation)
    exact_zero = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        tracker=_tracker(specs),
        initial_elixir={0: 10.0, 1: 0.0},
    ).tensorize(observation)

    assert unknown.match_scalars[0, 15:17].tolist() == [1.0, 1.0]
    assert exact_zero.match_scalars[0, 15:17].tolist() == [0.0, 0.0]


def test_true_side_scalar_uses_absolute_owner_independent_of_mirror() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    side_index = MATCH_SCALAR_NAMES.index("actor_is_true_red")

    def side_value(actor_owner: int, horizontal_mirror: bool) -> float:
        tensorizer = UniversalObservationTensorizerV4(
            actor_owner=actor_owner,
            card_catalog=catalog,
            ability_catalog=abilities,
            entity_archetype_catalog=archetypes,
            effect_catalog=_semantic_bundle()[4],
            card_specs=specs,
            deck=STANDARD_DECK,
            tracker=None,
            deck_roles=_deck_roles(),
            horizontal_mirror=horizontal_mirror,
        )
        observation = _observation(skeleton_count=0)
        if actor_owner == 1:
            observation = replace(
                observation,
                owner=1,
                players=tuple(
                    replace(player, owner=1 - player.owner)
                    for player in observation.players
                ),
            )
        scalars = tensorizer._scalars(
            observation,
            group_overflow=0,
            child_overflow=0,
            event_count=0,
            event_overflow=0,
            candidate_count=0,
        )
        return float(scalars[0, side_index])

    assert MATCH_SCALAR_NAMES[-1] == "actor_is_true_red"
    assert side_value(0, False) == 1.0
    assert side_value(0, True) == 1.0
    assert side_value(1, False) == 0.0
    assert side_value(1, True) == 0.0


@pytest.mark.parametrize(
    ("actor_owner", "horizontal_mirror"),
    ((0, True), (1, False), (1, True)),
)
def test_inline_spatial_perspective_matches_full_contract_transform(
    actor_owner: int,
    horizontal_mirror: bool,
) -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    observation = _observation(skeleton_count=1)
    if actor_owner == 1:
        observation = replace(
            observation,
            owner=1,
            players=tuple(
                replace(player, owner=1 - player.owner)
                for player in observation.players
            ),
            towers=tuple(
                replace(tower, owner=1 - tower.owner)
                for tower in observation.towers
            ),
            entities=tuple(
                replace(
                    entity,
                    owner=(
                        None if entity.owner is None else 1 - entity.owner
                    ),
                )
                for entity in observation.entities
            ),
        )
    observation = replace(
        observation,
        entities=(
            replace(observation.entities[0], velocity=(1250.0, -750.0)),
            *observation.entities[1:],
        ),
    )

    def build() -> UniversalObservationTensorizerV4:
        result = UniversalObservationTensorizerV4(
            actor_owner=actor_owner,
            card_catalog=catalog,
            ability_catalog=abilities,
            entity_archetype_catalog=archetypes,
            effect_catalog=_semantic_bundle()[4],
            card_specs=specs,
            deck=STANDARD_DECK,
            tracker=None,
            deck_roles=_deck_roles(),
            horizontal_mirror=horizontal_mirror,
        )
        result.start_episode(observation)
        return result

    inline = build()
    reference = build()
    # Independent reference: rotate the native scene geometrically, then let
    # the same tensorizer consume it with identity spatial coordinates.
    flip_x = (actor_owner == 1) != horizontal_mirror
    flip_y = actor_owner == 1

    def point(value):
        if value is None:
            return None
        x, y = value
        return (18000 - x if flip_x else x, 32000 - y if flip_y else y)

    def vector(value):
        if value is None:
            return None
        x, y = value
        return (-x if flip_x else x, -y if flip_y else y)

    def lane(kind):
        if not flip_x:
            return kind
        return kind.replace("left", "SWAP").replace("right", "left").replace("SWAP", "right")

    transformed = replace(
        reference.perspective.observation_policy_metadata_to_model(observation),
        towers=tuple(replace(tower, position=point(tower.position),
                             tower_kind=lane(tower.tower_kind))
                     for tower in observation.towers),
        entities=tuple(replace(entity, position=point(entity.position),
                               movement_target=point(entity.movement_target),
                               velocity=vector(entity.velocity))
                       for entity in observation.entities),
        events=tuple(replace(event, position=point(event.position))
                     for event in observation.events),
    )

    class IdentitySpatialPerspective:
        flip_x = False
        flip_y = False

        @staticmethod
        def observation_policy_metadata_to_model(
            value: ObservationV1,
        ) -> ObservationV1:
            return value

    reference.perspective = IdentitySpatialPerspective()  # type: ignore[assignment]
    actual = inline.tensorize(observation)
    expected = reference.tensorize(transformed)
    for item in fields(actual):
        actual_value = getattr(actual, item.name)
        expected_value = getattr(expected, item.name)
        if hasattr(actual_value, "__dataclass_fields__"):
            for nested in fields(actual_value):
                torch.testing.assert_close(
                    getattr(actual_value, nested.name),
                    getattr(expected_value, nested.name),
                )
        else:
            torch.testing.assert_close(actual_value, expected_value)


def test_tower_troop_identity_activation_geometry_and_target_relation() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    towers = list(_towers())
    tower_runtime_provenance = SemanticProvenanceV1(
        field_evidence={
            name: (
                SemanticEvidenceLevel.NATIVE_DERIVED
                if name == "tower_troop_runtime"
                else SemanticEvidenceLevel.UNKNOWN
            )
            for name in TOWER_RUNTIME_SEMANTIC_FIELDS
        },
        observed_tick=20,
    )
    towers[0] = replace(towers[0], status=("activated",), visible_target=501)
    towers[1] = replace(
        towers[1],
        tower_troop_id=159_000_002,
        tower_troop_runtime=DaggerDuchessRuntimeStateV1(
            charge_count=2,
            max_charge_count=8,
            recharge_elapsed_ms=450,
            recharge_duration_ms=900,
        ),
        runtime_provenance=tower_runtime_provenance,
    )
    towers[3] = replace(
        towers[3],
        tower_troop_id=159_000_004,
        tower_troop_runtime=RoyalChefRuntimeStateV1(
            start_delay_remaining_ms=0,
            start_delay_duration_ms=7_000,
            cooking_contribution=345_000,
            contribution_needed=460_000,
            surviving_side_towers=2,
            throw_delay_remaining_ms=100,
            target_entity=501,
        ),
        runtime_provenance=tower_runtime_provenance,
    )
    observation = replace(_observation(skeleton_count=0), towers=tuple(towers))

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(observation)

    assert int(batch.towers.tower_troop_type[0, 1]) == 3
    assert int(batch.towers.tower_troop_type[0, 3]) == 4
    assert int(batch.towers.tower_troop_type[0, 4]) == 1
    assert float(batch.towers.features[0, 0, 4]) == 1.0
    assert float(batch.towers.features[0, 1, 15]) == pytest.approx(0.25)
    assert float(batch.towers.features[0, 1, 16]) == pytest.approx(0.5)
    assert float(batch.towers.features[0, 3, 17]) == 0.0
    assert float(batch.towers.features[0, 3, 18]) == pytest.approx(0.75)
    assert float(batch.towers.features[0, 3, 19]) == 1.0
    assert float(batch.towers.features[0, 3, 20]) == 1.0
    assert tuple(batch.towers.extent[0, 0].tolist()) == pytest.approx(
        (7.0, 1.0, 11.0, 5.0)
    )
    assert tuple(batch.towers.extent[0, 1].tolist()) == pytest.approx(
        (2.0, 5.0, 5.0, 8.0)
    )
    group_start = 6 + 8 + 9
    ability_group = int(
        (
            batch.groups.mask[0]
            & (batch.groups.features[0, :, 9] == 1.0)
        ).nonzero(as_tuple=False)[0]
    )
    edges = {
        (int(source), int(target), int(kind))
        for source, target, kind, active in zip(
            batch.relation_edges.source[0],
            batch.relation_edges.target[0],
            batch.relation_edges.relation_type[0],
            batch.relation_edges.mask[0],
            strict=True,
        )
        if bool(active)
    }
    assert (0, group_start + ability_group, 1) in edges
    assert (group_start + ability_group, 0, 2) in edges
    assert (3, group_start + ability_group, 10) in edges
    assert (group_start + ability_group, 3, 11) in edges


def test_unknown_opponent_summary_contains_only_unrevealed_count() -> None:
    class TrackerStub:
        @staticmethod
        def revealed_cards(owner: int) -> tuple[int, ...]:
            return ()

        @staticmethod
        def elixir_interval(owner: int) -> tuple[float, float]:
            return 4.0, 9.0

        @staticmethod
        def card_state(owner: int, card_id: int) -> object:
            raise KeyError(card_id)

    specs, catalog, abilities, archetypes = _semantic_catalog()
    tensorizer = UniversalObservationTensorizerV4(
        actor_owner=0,
        card_catalog=catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        effect_catalog=_semantic_bundle()[4],
        card_specs=specs,
        deck=STANDARD_DECK,
        tracker=TrackerStub(),  # type: ignore[arg-type]
        deck_roles=_deck_roles(),
    )

    _, opponent, _ = tensorizer._card_sets(_observation())

    assert int(opponent.card_vocab_id[0, 8]) == UNKNOWN_CARD_VOCAB_ID
    assert bool(opponent.mask[0, 8])
    assert float(opponent.runtime_features[0, 8, 15]) == pytest.approx(7.0 / 8.0)
    assert torch.count_nonzero(opponent.runtime_features[0, 8, :15]) == 0


def test_own_runtime_exposes_hand_and_next_card_not_hidden_cycle_order() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    first = _observation(skeleton_count=0)
    players = list(first.players)
    players[0] = replace(
        players[0],
        next_card=STANDARD_DECK[4],
        cycle=STANDARD_DECK[4:],
    )
    first = replace(first, players=tuple(players))
    reversed_players = list(first.players)
    reversed_players[0] = replace(
        reversed_players[0],
        cycle=tuple(reversed(STANDARD_DECK[4:])),
    )
    second = replace(first, players=tuple(reversed_players))

    def tensorize(observation: ObservationV1):
        tensorizer = UniversalObservationTensorizerV4(
            actor_owner=0,
            card_catalog=catalog,
            ability_catalog=abilities,
            entity_archetype_catalog=archetypes,
            effect_catalog=_semantic_bundle()[4],
            card_specs=specs,
            deck=STANDARD_DECK,
            tracker=_tracker(specs),
            deck_roles=_deck_roles(),
        )
        tensorizer.start_episode(
            observation,
            initial_elixir={0: 10.0, 1: 10.0},
        )
        return tensorizer.tensorize(observation)

    first_batch = tensorize(first)
    second_batch = tensorize(second)
    torch.testing.assert_close(
        first_batch.own_cards.runtime_features,
        second_batch.own_cards.runtime_features,
    )
    assert torch.count_nonzero(
        first_batch.own_cards.runtime_features[..., 6:9]
    ) == 0
    next_row = int(
        (
            first_batch.own_cards.card_vocab_id[0]
            == catalog.vocab_id(STANDARD_DECK[4])
        ).nonzero(as_tuple=False)[0]
    )
    assert float(first_batch.own_cards.runtime_features[0, next_row, 1]) == 1.0


def test_event_payload_form_precedes_current_source_entity_form() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    source = _ability_entity()
    observation = replace(
        _observation(skeleton_count=0),
        entities=(source,),
        events=(
            EventV1(
                tick=20,
                event_type="damage",
                owner=0,
                entity_id=source.entity_id,
                card_id=STANDARD_DECK[0],
                combat=CombatEventV1(
                    kind=CombatEventKind.DAMAGE,
                    source_entity=source.entity_id,
                    source_card_id=STANDARD_DECK[0],
                    evolution_form_id="EvoForm",
                ),
                runtime_provenance=SemanticProvenanceV1(
                    field_evidence={
                        "combat": SemanticEvidenceLevel.NATIVE_DERIVED
                    },
                    observed_tick=20,
                ),
            ),
            EventV1(
                tick=20,
                event_type="ability_state_change",
                owner=0,
                entity_id=source.entity_id,
                card_id=STANDARD_DECK[0],
                data={"form_code": FORM_HERO_ACTIVE},
            ),
        ),
    )

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(observation)
    forms_by_event_type = {
        int(event_type): int(source_form)
        for event_type, source_form, active in zip(
            batch.events.event_type[0],
            batch.events.source_form[0],
            batch.events.mask[0],
            strict=True,
        )
        if bool(active)
    }

    assert forms_by_event_type[EVENT_TYPE["damage"]] == FORM_EVOLUTION
    assert (
        forms_by_event_type[EVENT_TYPE["ability_state_change"]]
        == FORM_HERO_ACTIVE
    )


def test_own_evolution_event_does_not_reveal_same_opponent_card() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    evolution_card_id = STANDARD_DECK[1]
    observation = _observation(skeleton_count=0)
    own_evolution = EventV1(
        tick=20,
        event_type=CombatEventKind.EVOLUTION_DEPLOY.value,
        owner=0,
        entity_id=900,
        card_id=evolution_card_id,
        combat=CombatEventV1(
            kind=CombatEventKind.EVOLUTION_DEPLOY,
            source_entity=900,
            source_card_id=evolution_card_id,
            evolution_form_id="Evolution",
        ),
        runtime_provenance=SemanticProvenanceV1(
            {"combat": SemanticEvidenceLevel.NATIVE_DERIVED},
            observed_tick=20,
        ),
    )
    tensorizer = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        tracker=_tracker(specs, evolution_card_id=evolution_card_id),
        initial_elixir={0: 10.0, 1: 10.0},
    )

    tensorizer.tensorize(replace(observation, events=(own_evolution,)))
    next_batch = tensorizer.tensorize(replace(observation, tick=25, events=()))
    opponent_row = int(
        (
            next_batch.opponent_cards.card_vocab_id[0]
            == catalog.vocab_id(evolution_card_id)
        ).nonzero(as_tuple=False)[0]
    )

    assert float(
        next_batch.opponent_cards.runtime_features[0, opponent_row, 3]
    ) == 0.0

    opponent_episode = replace(
        observation,
        episode_id="v4-test-opponent-evolution",
    )
    tensorizer.start_episode(
        opponent_episode,
        initial_elixir={0: 10.0, 1: 10.0},
    )
    opponent_evolution = replace(own_evolution, owner=1)
    public_batch = tensorizer.tensorize(
        replace(opponent_episode, events=(opponent_evolution,))
    )
    opponent_row = int(
        (
            public_batch.opponent_cards.card_vocab_id[0]
            == catalog.vocab_id(evolution_card_id)
        ).nonzero(as_tuple=False)[0]
    )
    assert float(
        public_batch.opponent_cards.runtime_features[0, opponent_row, 3]
    ) == 1.0


def test_unknown_opponent_summary_is_padding_after_full_reveal() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    observation = _observation()
    players = list(observation.players)
    players[1] = replace(players[1], revealed_cards=STANDARD_DECK)

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, players=tuple(players))
    )

    assert int(batch.opponent_cards.card_vocab_id[0, 8]) == 0
    assert not bool(batch.opponent_cards.mask[0, 8])
    assert torch.count_nonzero(batch.opponent_cards.runtime_features[0, 8]) == 0


def test_full_catalogs_cover_competitive_abilities_forms_and_projectiles() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()

    assert len(specs) == len(catalog.raw_card_ids) == 122
    assert abilities.card_scope == archetypes.card_scope == catalog.raw_card_ids
    assert len(abilities.ability_ids) == 24
    assert any(any(value != 0.0 for value in row) for row in abilities.static_features)
    assert archetypes.vocab_size <= 512
    for form in (
        "Goblin_Stab",
        "SpearGoblin",
        "RascalBoy",
        "RascalGirl",
        "Golemite",
        "LavaPups",
        "ElixirGolem2",
        "ElixirGolem4",
    ):
        assert archetypes.form_vocab_id(form) != UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
    assert any(key.startswith("projectile:") for key in archetypes.archetype_keys)


def test_tensorizer_requires_complete_deck_roles() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()

    with pytest.raises(ValueError, match="classify every actor deck card"):
        UniversalObservationTensorizerV4(
            actor_owner=0,
            card_catalog=catalog,
            ability_catalog=abilities,
            entity_archetype_catalog=archetypes,
            effect_catalog=_semantic_bundle()[4],
            card_specs=specs,
            deck=STANDARD_DECK,
            deck_roles={STANDARD_DECK[0]: (False, False)},
        )


def test_own_card_runtime_form_comes_from_exact_current_hand_contract() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    observation = _observation()
    players = list(observation.players)
    metadata = dict(players[0].metadata)
    runtime_by_slot = {
        key: dict(value) for key, value in metadata["hand_runtime_by_slot"].items()
    }
    runtime_by_slot["0"]["form_code"] = 2
    metadata["hand_runtime_by_slot"] = runtime_by_slot
    players[0] = replace(players[0], metadata=metadata)
    placement = {
        key: dict(value)
        for key, value in observation.action_mask.placement_masks.items()
    }
    placement["0"]["form_code"] = 2
    observation = replace(
        observation,
        players=tuple(players),
        action_mask=replace(
            observation.action_mask,
            placement_masks=placement,
        ),
    )

    batch = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        deck_roles={STANDARD_DECK[0]: (True, False)},
    ).tensorize(observation)

    row = int(
        (batch.own_cards.card_vocab_id[0] == catalog.vocab_id(STANDARD_DECK[0]))
        .nonzero(as_tuple=False)[0]
    )
    assert int(batch.own_cards.runtime_form[0, row]) == 2


def test_deploy_candidate_cost_comes_from_exact_live_placement_contract() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    observation = _observation()
    placement = {
        key: dict(value)
        for key, value in observation.action_mask.placement_masks.items()
    }
    placement["0"]["effective_cost"] = 2.5
    observation = replace(
        observation,
        action_mask=replace(
            observation.action_mask,
            placement_masks=placement,
        ),
    )

    batch = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        deck_roles={STANDARD_DECK[0]: (True, False)},
    ).tensorize(observation)

    candidate = int(
        (
            batch.candidates.native_visible_card_id[0]
            == STANDARD_DECK[0]
        ).nonzero(as_tuple=False)[0]
    )
    assert float(batch.candidates.cost[0, candidate]) == pytest.approx(2.5)


def test_distinct_child_forms_use_independent_archetype_ids() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    gang_card_id = 26_000_041
    reference = CausalGroupRefV1(
        kind=CausalGroupKind.DEPLOYMENT,
        handle="deploy:goblin-gang:1",
        source_card_id=gang_card_id,
    )
    gang = (
        EntityStateV1(
            entity_id=610,
            owner=0,
            card_id=gang_card_id,
            entity_kind="troop",
            position=(8_000, 12_000),
            native_data_global_id=3_565_953_160,
            causal_group=reference,
        ),
        EntityStateV1(
            entity_id=611,
            owner=0,
            card_id=gang_card_id,
            entity_kind="troop",
            position=(10_000, 12_000),
            native_data_global_id=34_000_019,
            causal_group=reference,
        ),
    )
    observation = replace(
        _observation(skeleton_count=0),
        entities=(*gang, _ability_entity()),
    )

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        observation
    )
    group_row = int(
        (
            batch.groups.mask[0]
            & (
                batch.groups.source_card_vocab_id[0]
                == catalog.require_vocab_id(gang_card_id)
            )
        ).nonzero(as_tuple=False)[0]
    )
    child_rows = (
        batch.groups.child_mask[0]
        & (batch.groups.child_group_index[0] == group_row)
    ).nonzero(as_tuple=False).flatten()
    assert set(batch.groups.child_archetype_id[0, child_rows].tolist()) == {
        archetypes.form_vocab_id("Goblin_Stab"),
        archetypes.form_vocab_id("SpearGoblin"),
    }
    assert all(
        int(value) != UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
        for value in batch.groups.child_archetype_id[0, child_rows]
    )


def test_event_rows_group_one_cause_and_encoder_consumes_references() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    events = tuple(
        EventV1(
            tick=20,
            event_type="damage",
            owner=0,
            entity_id=501,
            card_id=STANDARD_DECK[0],
            position=(9_000 + index * 10, 12_000),
            combat=CombatEventV1(
                kind=CombatEventKind.DAMAGE,
                source_entity=501,
                source_card_id=STANDARD_DECK[0],
                target_entity=4,
                amount=100.0 + index,
                attributes={"cause_event_id": "attack:501:77"},
            ),
            runtime_provenance=SemanticProvenanceV1(
                {"combat": SemanticEvidenceLevel.NATIVE_DERIVED}
            ),
        )
        for index in range(10)
    )
    observation = replace(_observation(), events=events)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        observation
    )

    assert int(batch.events.mask.sum()) == 1
    event_row = int(batch.events.mask[0].nonzero(as_tuple=False)[0])
    assert float(batch.events.features[0, event_row, 4]) == pytest.approx(
        10.0 / 16.0
    )
    assert float(batch.events.features[0, event_row, 1]) == pytest.approx(
        sum(100.0 + index for index in range(10)) / 5_000.0
    )
    assert int(batch.events.source_group_index[0, event_row]) >= 0
    assert int(batch.events.target_token_index[0, event_row]) >= 0

    model = UniversalCardPolicyV4(
        catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        effect_catalog=_semantic_bundle()[4],
        mechanic_profile_catalog=_semantic_bundle()[5],
    ).eval()
    with torch.no_grad():
        original = model.encode_observation(batch).event_summary
        changed_events = replace(
            batch.events,
            source_group_index=torch.where(
                batch.events.mask,
                torch.zeros_like(batch.events.source_group_index),
                torch.full_like(batch.events.source_group_index, -1),
            ),
            target_token_index=torch.where(
                batch.events.mask,
                torch.zeros_like(batch.events.target_token_index),
                torch.full_like(batch.events.target_token_index, -1),
            ),
        )
        changed = model.encode_observation(
            replace(batch, events=changed_events)
        ).event_summary
    assert not torch.allclose(original, changed)


def test_projectile_impacts_share_one_event_row_via_exact_volley_group() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    source_card_id = STANDARD_DECK[1]
    volley = CausalGroupRefV1(
        kind=CausalGroupKind.VOLLEY,
        handle="volley:hunter:impact:77",
        source_card_id=source_card_id,
    )
    projectiles = tuple(
        EntityStateV1(
            entity_id=800 + index,
            owner=0,
            card_id=source_card_id,
            entity_kind="projectile",
            position=(8_900 + 100 * index, 13_000),
            causal_group=volley,
            projectile_state=ProjectileStateV1(
                projectile_id=f"pellet:{index}",
                phase=ProjectilePhase.IMPACTED,
                source_card_id=source_card_id,
                target_entity=1,
            ),
            runtime_provenance=SemanticProvenanceV1(
                {
                    field: (
                        SemanticEvidenceLevel.NATIVE_DERIVED
                        if field == "projectile_state"
                        else SemanticEvidenceLevel.UNKNOWN
                    )
                    for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
                }
            ),
        )
        for index in range(3)
    )
    events = tuple(
        EventV1(
            tick=20,
            event_type=CombatEventKind.PROJECTILE_IMPACT.value,
            owner=0,
            entity_id=projectile.entity_id,
            card_id=source_card_id,
            position=projectile.position,
            combat=CombatEventV1(
                kind=CombatEventKind.PROJECTILE_IMPACT,
                source_entity=projectile.entity_id,
                source_card_id=source_card_id,
                target_entity=1,
                projectile_id=projectile.projectile_state.projectile_id,
                amount=84.0,
            ),
            runtime_provenance=SemanticProvenanceV1(
                {"combat": SemanticEvidenceLevel.NATIVE_DERIVED}
            ),
        )
        for projectile in projectiles
        if projectile.projectile_state is not None
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(
            observation,
            entities=projectiles,
            events=events,
            action_mask=replace(observation.action_mask, ability_sources=()),
        )
    )

    assert int(batch.events.mask.sum()) == 1
    row = int(batch.events.mask[0].nonzero(as_tuple=False)[0])
    assert float(batch.events.features[0, row, 4]) == pytest.approx(3.0 / 16.0)
    assert float(batch.events.features[0, row, 1]) == pytest.approx(252.0 / 5_000.0)
    assert int(batch.events.source_group_index[0, row]) >= 0
    assert int(batch.events.target_token_index[0, row]) == 0


def test_rolling_event_window_contributes_only_to_its_first_policy_turn() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes)
    event = EventV1(
        tick=20,
        event_type="damage",
        owner=0,
        entity_id=501,
        card_id=STANDARD_DECK[0],
        combat=CombatEventV1(
            kind=CombatEventKind.DAMAGE,
            source_entity=501,
            source_card_id=STANDARD_DECK[0],
            target_entity=4,
            amount=100.0,
            attributes={"cause_event_id": "attack:501:rolling"},
        ),
        runtime_provenance=SemanticProvenanceV1(
            {"combat": SemanticEvidenceLevel.NATIVE_DERIVED}
        ),
    )

    first = tensorizer.tensorize(replace(_observation(), events=(event,)))
    second = tensorizer.tensorize(
        replace(_observation(), tick=25, events=(event,))
    )

    assert int(first.events.mask.sum()) == 1
    assert int(second.events.mask.sum()) == 0
    assert float(first.match_scalars[0, 8]) == pytest.approx(1.0 / 32.0)
    assert float(second.match_scalars[0, 8]) == 0.0


def test_fair_snapshot_damage_does_not_relabel_victim_as_source() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    event = EventV1(
        tick=20,
        event_type="damage",
        owner=0,
        entity_id=501,
        card_id=STANDARD_DECK[0],
        combat=CombatEventV1(
            kind=CombatEventKind.DAMAGE,
            target_entity=501,
            target_card_id=STANDARD_DECK[0],
            amount=100.0,
        ),
        runtime_provenance=SemanticProvenanceV1(
            {"combat": SemanticEvidenceLevel.NATIVE_DERIVED}
        ),
    )

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(_observation(), events=(event,))
    )
    row = int(batch.events.mask[0].nonzero(as_tuple=False)[0])
    assert int(batch.events.source_card_vocab_id[0, row]) == 0
    assert int(batch.events.source_group_index[0, row]) == -1
    assert float(batch.events.features[0, row, 2]) == 0.0
    assert float(batch.events.features[0, row, 3]) == 1.0


def test_public_effect_transition_uses_exact_source_and_affected_target() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    event = EventV1(
        tick=20,
        event_type="effect_apply",
        owner=1,
        entity_id=4,
        card_id=STANDARD_DECK[1],
        data={"source_entity": 501, "amount": 40.0},
    )
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(_observation(), events=(event,))
    )
    row = int(batch.events.mask[0].nonzero(as_tuple=False)[0])

    assert int(batch.events.source_group_index[0, row]) >= 0
    assert int(batch.events.target_token_index[0, row]) == 3
    assert int(batch.events.source_card_vocab_id[0, row]) == catalog.vocab_id(
        STANDARD_DECK[0]
    )
    assert float(batch.events.features[0, row, 1]) == pytest.approx(40.0 / 5_000.0)


def test_shared_native_ability_controller_masks_every_candidate_in_group() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    first = _ability_entity()
    second_state = replace(first.ability_states[0], source_entity=502)
    second = replace(
        first,
        entity_id=502,
        position=(10_000, 12_000),
        ability_states=(second_state,),
    )
    observation = _observation(skeleton_count=0)
    players = list(observation.players)
    players[0] = replace(
        players[0],
        ability_runtime_states=(first.ability_states[0], second_state),
    )
    observation = replace(
        observation,
        players=tuple(players),
        entities=(first, second),
        action_mask=replace(
            observation.action_mask,
            ability_sources=(501, 502),
        ),
    )
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes)
    batch = tensorizer.tensorize(observation)
    rows = (
        batch.candidates.mask[0]
        & (batch.candidates.variant[0] == 1)
    ).nonzero(as_tuple=False).flatten()
    assert rows.numel() == 2
    assert (
        batch.candidates.exclusion_group_id[0, rows[0]]
        == batch.candidates.exclusion_group_id[0, rows[1]]
    )
    shadow = ShadowCandidateLegality(batch.candidates, tensorizer.config)
    shadow.apply(
        torch.tensor([True]),
        torch.tensor([int(rows[0])]),
        torch.tensor([-1]),
        torch.tensor([0]),
        step=0,
    )
    assert not bool(shadow.candidate_mask()[0, rows[1]])


def test_generic_relation_builder_emits_every_available_v4_relation() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    parent = _ability_entity()
    spawn = EntityStateV1(
        entity_id=620,
        owner=0,
        card_id=STANDARD_DECK[1],
        entity_kind="troop",
        position=(8_000, 14_000),
        source_entity=parent.entity_id,
        causal_group=CausalGroupRefV1(
            kind=CausalGroupKind.SPAWN_WAVE,
            handle="spawn:parent:501:1",
            source_card_id=STANDARD_DECK[1],
            parent_entity_id=parent.entity_id,
        ),
    )
    projectile = EntityStateV1(
        entity_id=621,
        owner=0,
        card_id=STANDARD_DECK[1],
        entity_kind="projectile",
        position=(9_000, 15_000),
        source_entity=parent.entity_id,
        causal_group=CausalGroupRefV1(
            kind=CausalGroupKind.VOLLEY,
            handle="volley:parent:501:1",
            source_card_id=STANDARD_DECK[1],
            parent_entity_id=parent.entity_id,
        ),
        projectile_state=ProjectileStateV1(
            projectile_id="projectile:621",
            phase=ProjectilePhase.IN_FLIGHT,
            source_entity=parent.entity_id,
            source_card_id=STANDARD_DECK[1],
            target_entity=4,
        ),
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "projectile_state"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    taunted = EntityStateV1(
        entity_id=622,
        owner=1,
        card_id=STANDARD_DECK[2],
        entity_kind="troop",
        position=(10_000, 16_000),
        visible_target=parent.entity_id,
        effect_states=(
                EffectStateV1(
                    effect_id="public-taunt",
                    kind=EffectKind.TARGETING_MODIFIER,
                    source_entity=parent.entity_id,
                ),
        ),
        causal_group=CausalGroupRefV1(
            kind=CausalGroupKind.DEPLOYMENT,
            handle="deploy:enemy:622",
            source_card_id=STANDARD_DECK[2],
        ),
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "effect_states"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    observation = replace(
        _observation(skeleton_count=0),
        entities=(parent, spawn, projectile, taunted),
    )

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        observation
    )

    relation_types = set(
        batch.relation_edges.relation_type[
            batch.relation_edges.mask
        ].tolist()
    )
    assert set(range(1, 10)).issubset(relation_types)


def test_tensorizer_preserves_exact_active_effect_identity_and_runtime() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    effect = EffectStateV1(
        effect_id="buff:9000003",
        kind=EffectKind.SLOW,
        source_entity=999,
        source_owner=1,
        remaining_ms=2_500,
        stacks=1,
        magnitude=-0.30,
        active=True,
        attributes={"native_buff_global_id": 9_000_003},
    )
    entity = replace(
        _ability_entity(),
        effect_states=(effect,),
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field in {"ability_states", "effect_states"}
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(_observation(skeleton_count=0), entities=(entity,))
    )
    effects = _semantic_bundle()[4]

    assert int(batch.active_effects.mask.sum()) == 1
    assert int(batch.active_effects.effect_vocab_id[0, 0]) == effects.vocab_id(
        9_000_003
    )
    assert int(batch.active_effects.parent_type[0, 0]) == 0
    assert int(batch.active_effects.parent_index[0, 0]) == 0
    assert int(batch.active_effects.source_owner_type[0, 0]) == 1
    assert batch.active_effects.runtime_features[0, 0].tolist() == pytest.approx(
        [0.25, 1.0, 0.0, 0.1, 1.0, -0.30, 1.0]
    )


def test_active_effect_remaining_time_is_safe_for_float16_storage() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    effect = EffectStateV1(
        effect_id="buff:9000003",
        kind=EffectKind.SLOW,
        remaining_ms=999_999_999,
        active=True,
        attributes={"native_buff_global_id": 9_000_003},
    )
    entity = replace(
        _ability_entity(),
        effect_states=(effect,),
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field in {"ability_states", "effect_states"}
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(_observation(skeleton_count=0), entities=(entity,))
    )

    assert batch.active_effects.runtime_features[0, 0, 0].item() == 30.0
    stored = batch.to_storage("cpu", float_dtype=torch.float16)
    assert torch.isfinite(stored.active_effects.runtime_features).all()


def test_spatial_geometry_rasterizes_buildings_and_bounds_persistent_areas() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    projectile = EntityStateV1(
        entity_id=704,
        owner=0,
        card_id=STANDARD_DECK[1],
        entity_kind="projectile",
        position=(12_000, 14_000),
        projectile_state=ProjectileStateV1(
            projectile_id="projectile:704",
            phase=ProjectilePhase.IN_FLIGHT,
            source_card_id=STANDARD_DECK[1],
            damage=100.0,
        ),
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "projectile_state"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    entities = (
        _ability_entity(),
        EntityStateV1(
            entity_id=701,
            owner=0,
            card_id=STANDARD_DECK[1],
            entity_kind="troop",
            position=(6_000, 10_000),
            hitpoints=100.0,
            max_hitpoints=200.0,
        ),
            EntityStateV1(
                entity_id=702,
                owner=0,
                card_id=26_000_029,
                native_data_global_id=34_000_028,
                entity_kind="troop",
            position=(8_000, 12_000),
            hitpoints=500.0,
            max_hitpoints=1_000.0,
        ),
        EntityStateV1(
            entity_id=703,
            owner=0,
            card_id=27_000_000,
            entity_kind="building",
            position=(10_500, 12_500),
            hitpoints=400.0,
            max_hitpoints=800.0,
        ),
        projectile,
        EntityStateV1(
            entity_id=705,
            owner=1,
            card_id=STANDARD_DECK[5],
            entity_kind="troop",
            position=(14_000, 20_000),
            hitpoints=50.0,
            max_hitpoints=200.0,
        ),
        EntityStateV1(
            entity_id=706,
            owner=None,
            card_id=None,
            entity_kind="effect",
            position=(9_000, 16_000),
            hitpoints=9_999.0,
            max_hitpoints=9_999.0,
        ),
        EntityStateV1(
            entity_id=707,
            owner=1,
            card_id=28_000_009,
            entity_kind="spell",
            position=(9_000, 16_000),
            causal_group=CausalGroupRefV1(
                kind=CausalGroupKind.PERSISTENT_EFFECT,
                handle="effect:poison:707",
                source_card_id=28_000_009,
            ),
        ),
    )
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(_observation(skeleton_count=0), entities=entities)
    )
    planes = batch.explicit_spatial_planes[0]

    assert float(planes[0].sum()) == 2.0
    assert float(planes[1].sum()) == 1.0
    assert float(planes[2].sum()) == 9.0
    assert torch.all(planes[2, 11:14, 9:12] == 1.0)
    assert float(planes[3].sum()) == 1.0
    assert planes.shape[0] == 14
    assert float(planes[7].sum()) == 1.0
    assert float(planes.sum()) < 28.0

    building_child = int(
        (batch.groups.child_type[0] == 1).logical_and(batch.groups.child_mask[0])
        .nonzero(as_tuple=False)[0]
    )
    building_group = int(batch.groups.child_group_index[0, building_child])
    assert batch.groups.extent[0, building_group].tolist() == pytest.approx(
        [9.0, 11.0, 12.0, 14.0]
    )
    assert batch.groups.features[0, building_group, 7:9].tolist() == pytest.approx(
        [3.0, 3.0]
    )
    assert batch.groups.child_extent[0, building_child].tolist() == pytest.approx(
        [9.0, 11.0, 12.0, 14.0]
    )
    assert float(batch.groups.child_radius[0, building_child]) == 0.0

    area_group = int(
        (batch.groups.group_type[0] == 3)
        .logical_and(
            batch.groups.source_card_vocab_id[0]
            == catalog.require_vocab_id(28_000_009)
        )
        .logical_and(batch.groups.mask[0])
        .nonzero(as_tuple=False)[0]
    )
    assert batch.groups.extent[0, area_group].tolist() == pytest.approx(
        [5.5, 12.5, 12.5, 19.5]
    )
    assert batch.groups.features[0, area_group, 7:9].tolist() == pytest.approx(
        [7.0, 7.0]
    )
    area_child = int(
        (batch.groups.child_group_index[0] == area_group)
        .logical_and(batch.groups.child_mask[0])
        .nonzero(as_tuple=False)[0]
    )
    assert batch.groups.child_extent[0, area_child].tolist() == pytest.approx(
        [5.5, 12.5, 12.5, 19.5]
    )
    assert float(batch.groups.child_radius[0, area_child]) == pytest.approx(3.5)


def test_exact_character_archetype_overrides_parent_building_classification() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    bat_global_id = next(
        global_id
        for global_id, form_id in archetypes.character_global_ids
        if form_id == "Bat"
    )
    spawned_bat = EntityStateV1(
        entity_id=707,
        owner=0,
        card_id=27_000_000,
        native_data_global_id=bat_global_id,
        # The live adapter can expose the parent building's coarse kind.  The
        # exact Character identity is authoritative for child/spatial class.
        entity_kind="building",
        position=(9_000, 14_000),
        hitpoints=81.0,
        max_hitpoints=81.0,
    )
    observation = _observation(skeleton_count=0)
    action_mask = replace(observation.action_mask, ability_sources=())
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(
            observation,
            entities=(spawned_bat,),
            action_mask=action_mask,
        )
    )

    child_row = int(batch.groups.child_mask[0].nonzero(as_tuple=False)[0])
    assert int(batch.groups.child_type[0, child_row]) == 6
    assert float(batch.explicit_spatial_planes[0, 1].sum()) == 1.0
    assert float(batch.explicit_spatial_planes[0, 2].sum()) == 0.0


def test_exact_global_id_overrides_incorrect_live_entity_namespace() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    area_global_id = 22_000_005
    area_vocab_id = archetypes.area_vocab_id(area_global_id)
    assert area_vocab_id != UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
    area = EntityStateV1(
        entity_id=708,
        owner=0,
        card_id=26_000_035,
        native_data_global_id=area_global_id,
        # Runtime telemetry can label materialized AreaEffectData as a troop.
        # Its exact LogicData GlobalID is the authoritative namespace.
        entity_kind="troop",
        position=(9_000, 14_000),
        causal_group=CausalGroupRefV1(
            kind=CausalGroupKind.PERSISTENT_EFFECT,
            handle="effect:rage:708",
            source_card_id=26_000_035,
        ),
    )
    observation = _observation(skeleton_count=0)
    tensorizer = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        reject_unknown_public_semantics=True,
    )

    batch = tensorizer.tensorize(
        replace(
            observation,
            entities=(area,),
            action_mask=replace(observation.action_mask, ability_sources=()),
        )
    )

    child_row = int(batch.groups.child_mask[0].nonzero(as_tuple=False)[0])
    assert int(batch.groups.child_archetype_id[0, child_row]) == area_vocab_id
    assert int(batch.groups.child_type[0, child_row]) == 4


def test_production_strict_mode_rejects_unknown_public_entity() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    unknown_global_id = 4_294_000_001
    entity = EntityStateV1(
        entity_id=709,
        owner=0,
        card_id=STANDARD_DECK[1],
        native_data_global_id=unknown_global_id,
        entity_kind="troop",
        position=(9_000, 14_000),
    )
    observation = _observation(skeleton_count=0)
    tensorizer = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        reject_unknown_public_semantics=True,
    )

    with pytest.raises(
        ValueError,
        match=r"public entity resolved to UNKNOWN archetype:.*4294000001",
    ):
        tensorizer.tensorize(
            replace(
                observation,
                entities=(entity,),
                action_mask=replace(observation.action_mask, ability_sources=()),
            )
        )


def test_production_strict_mode_rejects_unknown_active_effect() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    bat_global_id = next(
        global_id
        for global_id, form_id in archetypes.character_global_ids
        if form_id == "Bat"
    )
    effect = EffectStateV1(
        effect_id="buff:4294000002",
        kind=EffectKind.SLOW,
        source_entity=710,
        source_owner=1,
        active=True,
        attributes={"native_buff_global_id": 4_294_000_002},
    )
    entity = EntityStateV1(
        entity_id=710,
        owner=0,
        card_id=STANDARD_DECK[1],
        native_data_global_id=bat_global_id,
        entity_kind="troop",
        position=(9_000, 14_000),
        effect_states=(effect,),
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "effect_states"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    observation = _observation(skeleton_count=0)
    tensorizer = _tensorizer(
        specs,
        catalog,
        abilities,
        archetypes,
        reject_unknown_public_semantics=True,
    )

    with pytest.raises(
        ValueError,
        match=r"active public effect resolved to UNKNOWN:.*4294000002",
    ):
        tensorizer.tensorize(
            replace(
                observation,
                entities=(entity,),
                action_mask=replace(observation.action_mask, ability_sources=()),
            )
        )


def test_persistent_effect_children_do_not_each_inherit_the_source_area() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    reference = CausalGroupRefV1(
        kind=CausalGroupKind.PERSISTENT_EFFECT,
        handle="effect:graveyard:spawn-wave",
        source_card_id=28_000_010,
    )
    skeletons = (
        EntityStateV1(
            entity_id=708,
            owner=0,
            card_id=26_000_010,
            entity_kind="troop",
            position=(8_000, 12_000),
            causal_group=reference,
        ),
        EntityStateV1(
            entity_id=709,
            owner=0,
            card_id=26_000_010,
            entity_kind="troop",
            position=(10_000, 13_000),
            causal_group=reference,
        ),
    )
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(
            _observation(skeleton_count=0),
            entities=(*skeletons, _ability_entity()),
        )
    )
    group_row = int(
        (
            batch.groups.source_card_vocab_id[0]
            == catalog.require_vocab_id(28_000_010)
        )
        .logical_and(batch.groups.mask[0])
        .nonzero(as_tuple=False)[0]
    )

    assert batch.groups.extent[0, group_row].tolist() == pytest.approx(
        [8.0, 12.0, 10.0, 13.0]
    )
    assert batch.groups.features[0, group_row, 7:9].tolist() == pytest.approx(
        [2.0, 1.0]
    )


def test_non_spell_effect_does_not_reuse_an_ambiguous_card_radius() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    goblins = 26_000_002
    assert specs[goblins].radius_tiles is None
    effect = EntityStateV1(
        entity_id=710,
        owner=0,
        card_id=goblins,
        entity_kind="effect",
        position=(9_000, 16_000),
        causal_group=CausalGroupRefV1(
            kind=CausalGroupKind.PERSISTENT_EFFECT,
            handle="effect:goblins:ambiguous-radius",
            source_card_id=goblins,
        ),
    )
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(
            _observation(skeleton_count=0),
            entities=(effect, _ability_entity()),
        )
    )
    group_row = int(
        (
            batch.groups.source_card_vocab_id[0]
            == catalog.require_vocab_id(goblins)
        )
        .logical_and(batch.groups.mask[0])
        .nonzero(as_tuple=False)[0]
    )
    child_row = int(
        (batch.groups.child_group_index[0] == group_row)
        .logical_and(batch.groups.child_mask[0])
        .nonzero(as_tuple=False)[0]
    )

    assert batch.groups.child_extent[0, child_row].tolist() == pytest.approx(
        [9.0, 16.0, 9.0, 16.0]
    )
    assert float(batch.groups.child_radius[0, child_row]) == 0.0


def test_child_overflow_keeps_ability_source_and_full_spatial_density() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes)

    batch = tensorizer.tensorize(_observation(skeleton_count=170))

    assert int(batch.groups.child_mask.sum()) == 160
    ability_row = int((batch.candidates.variant[0] == 1).nonzero(as_tuple=False)[0])
    assert int(batch.candidates.source_child_index[0, ability_row]) >= 0
    assert batch.explicit_spatial_planes[0, 0].sum() == 171.0
    assert batch.explicit_spatial_planes[0, 4].sum() == pytest.approx(
        (170.0 * 100.0 + 1000.0) / 10_000.0
    )
    assert batch.explicit_spatial_planes[0, 7:].sum() == 0.0
    assert batch.match_scalars[0, 10] == 11.0 / 160.0


def test_generic_volley_aggregates_direction_dispersion_and_expected_damage() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    source_card_id = STANDARD_DECK[1]
    reference = CausalGroupRefV1(
        kind=CausalGroupKind.VOLLEY,
        handle="volley:1:attack:77",
        source_card_id=source_card_id,
        parent_entity_id=900,
    )
    projectiles = tuple(
        EntityStateV1(
            entity_id=700 + index,
            owner=0,
            card_id=source_card_id,
            entity_kind="projectile",
            position=(8_000 + index * 2_000, 10_000 + index * 1_000),
            velocity=(1_000 + index * 500, 2_000 - index * 250),
            causal_group=reference,
            projectile_state=ProjectileStateV1(
                projectile_id=f"pellet:{index}",
                phase=ProjectilePhase.IN_FLIGHT,
                source_entity=900,
                source_card_id=source_card_id,
                spawn_tick=20,
                velocity=(1_000 + index * 500, 2_000 - index * 250),
                damage=100.0 + index * 50.0,
                provenance=SemanticProvenanceV1(
                    {
                        field: SemanticEvidenceLevel.NATIVE_DERIVED
                        for field in PROJECTILE_STATE_FIELDS
                    }
                ),
            ),
            runtime_provenance=SemanticProvenanceV1(
                {
                    field: (
                        SemanticEvidenceLevel.NATIVE_DERIVED
                        if field == "projectile_state"
                        else SemanticEvidenceLevel.UNKNOWN
                    )
                    for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
                }
            ),
        )
        for index in range(2)
    )
    observation = replace(
        _observation(skeleton_count=0),
        entities=(*projectiles, _ability_entity()),
    )
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes)

    batch = tensorizer.tensorize(observation)

    source_vocab_id = catalog.require_vocab_id(source_card_id)
    volley_row = int(
        (
            batch.groups.mask[0]
            & (batch.groups.source_card_vocab_id[0] == source_vocab_id)
            & (batch.groups.group_type[0] == 2)
        ).nonzero(as_tuple=False)[0]
    )
    features = batch.groups.features[0, volley_row]
    assert float(features[0]) == pytest.approx(2.0 / 16.0)
    assert features[11] > 0.0
    assert features[12] > 0.0
    assert features[14] > 0.0
    assert features[15] > 0.0
    assert features[16] > 0.0
    assert float(features[18]) == pytest.approx(2.0 / 16.0)
    assert float(features[19]) == pytest.approx((100.0 + 150.0) / 5_000.0)
    assert float(features[20]) == pytest.approx(1.0)


def test_live_projectile_archetype_supplies_known_damage_and_radius() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    source_card_id = 26_000_044  # Hunter
    reference = CausalGroupRefV1(
        kind=CausalGroupKind.VOLLEY,
        handle="volley:hunter:77",
        source_card_id=source_card_id,
    )
    projectiles = tuple(
        EntityStateV1(
            entity_id=780 + index,
            owner=0,
            card_id=source_card_id,
            entity_kind="projectile",
            position=(8_000 + index * 500, 10_000),
            causal_group=reference,
            projectile_state=ProjectileStateV1(
                projectile_id=f"hunter:{index}",
                phase=ProjectilePhase.IN_FLIGHT,
                source_card_id=source_card_id,
                attributes={"native_projectile_data_global_id": 10_000_044},
            ),
            runtime_provenance=SemanticProvenanceV1(
                {
                    field: (
                        SemanticEvidenceLevel.NATIVE_DERIVED
                        if field == "projectile_state"
                        else SemanticEvidenceLevel.UNKNOWN
                    )
                    for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
                }
            ),
        )
        for index in range(2)
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, entities=(*projectiles, _ability_entity()))
    )

    volley_row = int((batch.groups.group_type[0] == 2).nonzero(as_tuple=False)[0])
    group = batch.groups.features[0, volley_row]
    assert float(group[19]) == pytest.approx(168.0 / 5_000.0)
    assert float(group[20]) == pytest.approx(1.0)
    child_rows = batch.groups.child_mask[0] & (
        batch.groups.child_archetype_id[0]
        == archetypes.projectile_vocab_id(10_000_044)
    )
    child = batch.groups.child_features[0, child_rows]
    assert child[:, 13].tolist() == pytest.approx([84.0 / 5_000.0] * 2)
    assert child[:, 14].tolist() == pytest.approx([0.3 / 5.0] * 2)
    assert child[:, 33].tolist() == [1.0, 1.0]
    assert child[:, 34].tolist() == [1.0, 1.0]


def test_fisherman_projectile_drag_stage_reaches_child_memory() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    source_card_id = 26_000_061
    reference = CausalGroupRefV1(
        kind=CausalGroupKind.VOLLEY,
        handle="volley:fisherman:drag-stage",
        source_card_id=source_card_id,
    )
    projectiles = tuple(
        EntityStateV1(
            entity_id=790 + index,
            owner=0,
            card_id=source_card_id,
            entity_kind="projectile",
            position=(9_000, 14_000 + index * 100),
            causal_group=reference,
            projectile_state=ProjectileStateV1(
                projectile_id=f"fisherman:{index}",
                phase=ProjectilePhase.IN_FLIGHT,
                source_card_id=source_card_id,
                drag_stage=stage,
                attributes={"native_projectile_data_global_id": 10_000_064},
            ),
            runtime_provenance=SemanticProvenanceV1(
                {
                    field: (
                        SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                        if field == "projectile_state"
                        else SemanticEvidenceLevel.UNKNOWN
                    )
                    for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
                }
            ),
        )
        for index, stage in enumerate(
            (
                ProjectileDragStage.OUTBOUND,
                ProjectileDragStage.DRAG_BACK_ACTIVE,
            )
        )
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, entities=(*projectiles, _ability_entity()))
    )

    child_rows = batch.groups.child_mask[0] & (
        batch.groups.child_archetype_id[0]
        == archetypes.projectile_vocab_id(10_000_064)
    )
    child = batch.groups.child_features[0, child_rows]
    assert child[:, 43].tolist() == [0.0, 1.0]
    assert child[:, 44].tolist() == [1.0, 1.0]


def test_entity_extra_spawn_accumulator_reaches_child_memory() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    resource = EntityResourceStateV1(
        kind="extra_spawn_accumulator",
        current_raw=3,
        capacity_raw=10,
        normalized=0.3,
        provenance=SemanticProvenanceV1(
            {
                field: SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                for field in ENTITY_RESOURCE_STATE_FIELDS
            }
        ),
    )
    entity = EntityStateV1(
        entity_id=799,
        owner=0,
        card_id=26_000_069,
        entity_kind="hero",
        position=(9_000, 14_000),
        resource_states=(resource,),
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                    if field == "resource_states"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, entities=(entity, _ability_entity()))
    )

    child = batch.groups.child_features[0, batch.groups.child_mask[0]]
    resource_rows = child[:, 46] == 1.0
    assert int(resource_rows.sum()) == 1
    assert float(child[resource_rows, 45].item()) == pytest.approx(0.3)


def test_capture_runtime_reaches_child_memory_and_relation_edges() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    capture = CaptureRuntimeStateV1(
        phase=CaptureActionPhase.ACTIVE,
        configured_cooldown_ms=300,
        cooldown_remaining_ms=0,
        hit_frequency_ms=1_000,
        hit_accumulator_ms=850,
        targets=(
            CaptureTargetStateV1(
                phase=CaptureTargetPhase.CONTAINED,
                elapsed_ms=950,
                target_entity=800,
            ),
        ),
        provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "targets"
                    else SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                )
                for field in CAPTURE_RUNTIME_STATE_FIELDS
            }
        ),
    )
    source = EntityStateV1(
        entity_id=799,
        owner=0,
        card_id=27_000_012,
        entity_kind="building",
        position=(9_000, 14_000),
        capture_runtime=capture,
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "capture_runtime"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    target = EntityStateV1(
        entity_id=800,
        owner=1,
        card_id=26_000_003,
        entity_kind="troop",
        position=(9_000, 15_000),
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, entities=(source, target, _ability_entity()))
    )

    child = batch.groups.child_features[0, batch.groups.child_mask[0]]
    capture_rows = child[:, 47] == 1.0
    assert int(capture_rows.sum()) == 1
    capture_features = child[capture_rows][0]
    assert capture_features[51].item() == 1.0
    assert capture_features[54].item() == pytest.approx(0.85)
    relation_types = batch.relation_edges.relation_type[
        0, batch.relation_edges.mask[0]
    ]
    assert 12 in relation_types.tolist()
    assert 13 in relation_types.tolist()


def test_threshold_relocation_runtime_reaches_child_memory() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    relocation = ThresholdRelocationStateV1(
        phase=ThresholdRelocationPhase.RELOCATING,
        stage=4,
        relocation_index=1,
        thresholds_percent=(66, 33),
        hide_duration_ms=1_000,
        remaining_ms=100,
        burrowed=True,
        provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field in {"phase", "relocation_index", "burrowed"}
                    else SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                )
                for field in THRESHOLD_RELOCATION_STATE_FIELDS
            }
        ),
    )
    source = EntityStateV1(
        entity_id=801,
        owner=0,
        card_id=27_000_013,
        entity_kind="building",
        position=(9_000, 14_000),
        threshold_relocation_runtime=relocation,
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "threshold_relocation_runtime"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, entities=(source, _ability_entity()))
    )

    child = batch.groups.child_features[0, batch.groups.child_mask[0]]
    relocation_rows = child[:, 56] == 1.0
    assert int(relocation_rows.sum()) == 1
    features = child[relocation_rows][0]
    assert features[58].item() == 1.0
    assert features[60].item() == 1.0
    assert features[61].item() == pytest.approx(0.1)
    assert features[62].item() == pytest.approx(0.5)
    assert features[63].item() == pytest.approx(0.33)


def test_attack_sequence_decay_reaches_child_memory() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    source = EntityStateV1(
        entity_id=802,
        owner=0,
        card_id=26_000_037,
        entity_kind="troop",
        position=(9_000, 14_000),
        attack_state=AttackStateV1(
            phase=AttackPhase.CHANNEL,
            sequence_index=2,
            sequence_progress=12,
            sequence_progress_limit=50,
            sequence_decay_remaining_ms=3_500,
            sequence_decay_duration_ms=7_000,
        ),
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "attack_state"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, entities=(source, _ability_entity()))
    )

    child = batch.groups.child_features[0, batch.groups.child_mask[0]]
    runtime_rows = child[:, 65] == 1.0
    assert int(runtime_rows.sum()) == 1
    features = child[runtime_rows][0]
    assert features[64].item() == pytest.approx(12.0 / 50.0)
    assert features[66].item() == pytest.approx(0.5)
    assert features[67].item() == 1.0


def test_periodic_attack_modifier_linger_reaches_child_memory() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    modifier = PeriodicAttackModifierStateV1(
        phase=PeriodicAttackModifierPhase.SOURCE_DEATH_LINGER,
        period_attacks=3,
        completed_attacks=2,
        source_entity=None,
        linger_remaining_ms=4_950,
        linger_duration_ms=5_000,
        provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.UNKNOWN
                    if field == "source_entity"
                    else SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                )
                for field in PERIODIC_ATTACK_MODIFIER_STATE_FIELDS
            }
        ),
    )
    source = EntityStateV1(
        entity_id=803,
        owner=0,
        card_id=26_000_000,
        entity_kind="troop",
        position=(9_000, 14_000),
        periodic_attack_modifier=modifier,
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "periodic_attack_modifier"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, entities=(source, _ability_entity()))
    )

    child = batch.groups.child_features[0, batch.groups.child_mask[0]]
    modifier_rows = child[:, 68] == 1.0
    assert int(modifier_rows.sum()) == 1
    features = child[modifier_rows][0]
    assert features[69].item() == pytest.approx(2.0 / 3.0)
    assert features[70].item() == 1.0
    assert features[71].item() == pytest.approx(0.99)


def test_periodic_attack_modifier_source_alive_keeps_exact_source_relation() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    rune = EntityStateV1(
        entity_id=804,
        owner=0,
        card_id=STANDARD_DECK[0],
        entity_kind="troop",
        position=(9_000, 14_000),
    )
    modifier = PeriodicAttackModifierStateV1(
        phase=PeriodicAttackModifierPhase.SOURCE_ALIVE,
        period_attacks=3,
        completed_attacks=1,
        source_entity=rune.entity_id,
        linger_remaining_ms=0,
        linger_duration_ms=5_000,
        provenance=SemanticProvenanceV1(
            {
                field: SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
                for field in PERIODIC_ATTACK_MODIFIER_STATE_FIELDS
            }
        ),
    )
    carrier = EntityStateV1(
        entity_id=805,
        owner=0,
        card_id=STANDARD_DECK[1],
        entity_kind="troop",
        position=(9_500, 14_000),
        periodic_attack_modifier=modifier,
        runtime_provenance=SemanticProvenanceV1(
            {
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "periodic_attack_modifier"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            }
        ),
    )
    observation = _observation(skeleton_count=0)
    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        replace(observation, entities=(rune, carrier, _ability_entity()))
    )

    child = batch.groups.child_features[0, batch.groups.child_mask[0]]
    modifier_rows = child[:, 68] == 1.0
    assert int(modifier_rows.sum()) == 1
    features = child[modifier_rows][0]
    assert features[69].item() == pytest.approx(1.0 / 3.0)
    assert features[70].item() == 0.0
    assert features[71].item() == 0.0
    relation_types = batch.relation_edges.relation_type[
        0, batch.relation_edges.mask[0]
    ].tolist()
    assert 3 in relation_types
    assert 4 in relation_types


@pytest.mark.parametrize(
    "thresholds",
    ((66.9, 33.1), (True, False), ("66", "33")),
)
def test_threshold_relocation_rejects_coerced_thresholds(
    thresholds: tuple[object, object],
) -> None:
    with pytest.raises(ContractError, match="exact integers"):
        ThresholdRelocationStateV1(
            phase=ThresholdRelocationPhase.WAITING_THRESHOLD,
            stage=1,
            relocation_index=0,
            thresholds_percent=thresholds,  # type: ignore[arg-type]
            hide_duration_ms=1_000,
            remaining_ms=0,
            burrowed=False,
        )


def test_tensorizer_rejects_observed_card_outside_its_v4_scope() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    off_scope_card_id = 28_000_019  # GlobalClone event-mode card.
    observation = replace(
        _observation(skeleton_count=0),
        entities=(
            EntityStateV1(
                entity_id=800,
                owner=0,
                card_id=off_scope_card_id,
                entity_kind="effect",
                position=(9_000, 16_000),
                causal_group=CausalGroupRefV1(
                    kind=CausalGroupKind.PERSISTENT_EFFECT,
                    handle="effect:event-mode:1",
                    source_card_id=off_scope_card_id,
                ),
            ),
            _ability_entity(),
        ),
    )
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes)

    with pytest.raises(ValueError, match="outside this V4 catalog"):
        tensorizer.tensorize(observation)


@pytest.mark.parametrize("internal_form_id", (26_000_104, 26_000_105))
def test_spirit_empress_hidden_forms_use_visible_policy_card(
    internal_form_id: int,
) -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    visible_card_id = 28_000_025
    observation = replace(
        _observation(skeleton_count=0),
        entities=(
            EntityStateV1(
                entity_id=803,
                owner=0,
                card_id=internal_form_id,
                entity_kind="troop",
                position=(9_000, 16_000),
                causal_group=CausalGroupRefV1(
                    kind=CausalGroupKind.SPAWN_WAVE,
                    handle=f"spawn:spirit-empress:{internal_form_id}",
                    source_card_id=internal_form_id,
                ),
            ),
            _ability_entity(),
        ),
    )

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        observation
    )

    assert catalog.require_vocab_id(visible_card_id) in (
        batch.groups.source_card_vocab_id[0, batch.groups.mask[0]].tolist()
    )


def test_unidentified_internal_child_form_uses_archetype_unknown() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    source_card_id = STANDARD_DECK[1]
    internal_form_id = 203_000_034
    observation = replace(
        _observation(skeleton_count=0),
        entities=(
            EntityStateV1(
                entity_id=801,
                owner=0,
                card_id=internal_form_id,
                entity_kind="troop",
                position=(9_000, 16_000),
                causal_group=CausalGroupRefV1(
                    kind=CausalGroupKind.SPAWN_WAVE,
                    handle="spawn:internal-form:1",
                    source_card_id=source_card_id,
                    parent_entity_id=700,
                ),
            ),
            _ability_entity(),
        ),
        events=(
            EventV1(
                tick=20,
                event_type="spawn",
                owner=0,
                entity_id=801,
                card_id=internal_form_id,
            ),
        ),
    )
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes)

    batch = tensorizer.tensorize(observation)

    source_vocab_id = catalog.require_vocab_id(source_card_id)
    assert UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID in batch.groups.child_archetype_id[
        0,
        batch.groups.child_mask[0],
    ].tolist()
    assert (
        source_vocab_id
        in batch.events.source_card_vocab_id[
            0,
            batch.events.mask[0],
        ].tolist()
    )


def test_tower_despawn_internal_data_is_not_a_policy_card() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    observation = replace(
        _observation(skeleton_count=0),
        events=(
            EventV1(
                tick=20,
                event_type="death_or_despawn",
                owner=0,
                entity_id=2,
                card_id=13_000_001,
            ),
        ),
    )

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        observation
    )

    assert PAD_CARD_VOCAB_ID in batch.events.source_card_vocab_id[
        0,
        batch.events.mask[0],
    ].tolist()


def test_despawned_native_character_data_is_not_a_policy_card() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    observation = replace(
        _observation(skeleton_count=0),
        events=(
            EventV1(
                tick=20,
                event_type="death_or_despawn",
                owner=0,
                entity_id=9_999,
                card_id=13_000_001,
            ),
        ),
    )

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(
        observation
    )

    assert PAD_CARD_VOCAB_ID in batch.events.source_card_vocab_id[
        0,
        batch.events.mask[0],
    ].tolist()


def test_unrooted_internal_entity_data_is_not_an_off_scope_policy_card() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    internal_form_id = 203_999_999
    observation = replace(
        _observation(skeleton_count=0),
        entities=(
            EntityStateV1(
                entity_id=802,
                owner=0,
                card_id=internal_form_id,
                native_data_global_id=internal_form_id,
                entity_kind="troop",
                position=(9_000, 16_000),
            ),
            _ability_entity(),
        ),
        events=(
            EventV1(
                tick=20,
                event_type="spawn",
                owner=0,
                entity_id=802,
                card_id=internal_form_id,
            ),
        ),
    )
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes)

    batch = tensorizer.tensorize(observation)

    internal_rows = (
        (batch.groups.child_archetype_id[0] == UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)
        & batch.groups.child_mask[0]
    ).nonzero(as_tuple=False)
    assert len(internal_rows) >= 1
    internal_group_rows = batch.groups.child_group_index[
        0,
        internal_rows[:, 0],
    ]
    assert bool(
        (
            batch.groups.source_card_vocab_id[0, internal_group_rows]
            == PAD_CARD_VOCAB_ID
        ).any()
    )
    assert PAD_CARD_VOCAB_ID in batch.events.source_card_vocab_id[
        0,
        batch.events.mask[0],
    ].tolist()


def test_despawned_internal_form_event_recovers_base_policy_card() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    internal_form_id = 203_000_039
    base_card_id = 26_000_039
    observation = replace(
        _observation(skeleton_count=0),
        entities=(_ability_entity(),),
        events=(
            EventV1(
                tick=20,
                event_type="death_or_despawn",
                owner=0,
                entity_id=802,
                card_id=internal_form_id,
            ),
        ),
    )
    tensorizer = _tensorizer(specs, catalog, abilities, archetypes)

    batch = tensorizer.tensorize(observation)

    assert catalog.require_vocab_id(base_card_id) in batch.events.source_card_vocab_id[
        0,
        batch.events.mask[0],
    ].tolist()


def test_evolution_form_spawn_event_uses_base_card_vocabulary() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    source_card_id = STANDARD_DECK[1]
    evolution_form_id = 203_000_076
    entity = EntityStateV1(
        entity_id=802,
        owner=0,
        card_id=evolution_form_id,
        entity_kind="troop",
        position=(9_000, 16_000),
        evolution_state=EvolutionRuntimeStateV1(
            card_id=source_card_id,
            active=True,
        ),
        runtime_provenance=SemanticProvenanceV1(
            field_evidence={
                field: (
                    SemanticEvidenceLevel.NATIVE_DERIVED
                    if field == "evolution_state"
                    else SemanticEvidenceLevel.UNKNOWN
                )
                for field in ENTITY_RUNTIME_SEMANTIC_FIELDS
            },
            observed_tick=20,
        ),
    )
    observation = replace(
        _observation(skeleton_count=0),
        entities=(entity, _ability_entity()),
        events=(
            EventV1(
                tick=20,
                event_type="spawn",
                owner=0,
                entity_id=entity.entity_id,
                card_id=evolution_form_id,
            ),
        ),
    )

    batch = _tensorizer(specs, catalog, abilities, archetypes).tensorize(observation)

    assert catalog.require_vocab_id(source_card_id) in batch.events.source_card_vocab_id[
        0,
        batch.events.mask[0],
    ].tolist()


def test_opaque_causal_handle_never_changes_model_tensors() -> None:
    specs, catalog, abilities, archetypes = _semantic_catalog()
    first = _observation(skeleton_count=3)
    second = replace(
        first,
        entities=tuple(
            replace(
                entity,
                causal_group=replace(
                    entity.causal_group,
                    handle="deploy:opaque:changed",
                ),
            )
            if entity.entity_id >= 1_000 and entity.causal_group is not None
            else entity
            for entity in first.entities
        ),
    )

    first_batch = _tensorizer(
        specs, catalog, abilities, archetypes
    ).tensorize(first)
    second_batch = _tensorizer(
        specs, catalog, abilities, archetypes
    ).tensorize(second)

    for item in fields(first_batch.groups):
        assert torch.equal(
            getattr(first_batch.groups, item.name),
            getattr(second_batch.groups, item.name),
        )
