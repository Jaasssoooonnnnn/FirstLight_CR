from __future__ import annotations

from native_runner.contracts import (
    CausalGroupKind,
    CausalGroupRefV1,
    EntityStateV1,
)
from native_runner.training.v4.grouping import build_causal_groups


def test_simple_public_action_handle_groups_one_deployment() -> None:
    causal_group = CausalGroupRefV1(
        kind=CausalGroupKind.DEPLOYMENT,
        handle="deploy:4:7",
        source_card_id=26000012,
    )
    skeletons = tuple(
        EntityStateV1(
            entity_id=100 + index,
            owner=0,
            card_id=26000012,
            entity_kind="troop",
            position=(4.0 + index * 0.1, 8.0),
            age_ms=250,
            causal_group=causal_group,
        )
        for index in range(15)
    )
    groups = build_causal_groups(skeletons)
    assert len(groups) == 1
    assert groups[0].child_entity_ids == tuple(range(100, 115))
    assert groups[0].key.handle == "deploy:4:7"
    assert groups[0].source_card_id == 26000012


def test_fallback_grouping_is_deterministic_without_an_action_handle() -> None:
    first = EntityStateV1(
        entity_id=9,
        owner=1,
        card_id=26000001,
        entity_kind="troop",
        position=(7.9, 20.0),
        age_ms=100,
    )
    second = EntityStateV1(
        entity_id=3,
        owner=1,
        card_id=26000001,
        entity_kind="troop",
        position=(8.1, 20.0),
        age_ms=100,
    )
    forward = build_causal_groups((first, second))
    reverse = build_causal_groups((second, first))
    assert forward == reverse
    assert len(forward) == 2
    assert tuple(group.child_entity_ids for group in forward) == ((3,), (9,))


def test_card_and_child_form_are_attributes_not_group_equality() -> None:
    reference = CausalGroupRefV1(
        kind=CausalGroupKind.SPAWN_WAVE,
        handle="spawn:8:44",
        source_card_id=26000009,
        parent_entity_id=70,
    )
    children = (
        EntityStateV1(
            entity_id=72,
            owner=0,
            card_id=26000009,
            entity_kind="troop",
            position=(8.0, 12.0),
            causal_group=reference,
        ),
        EntityStateV1(
            entity_id=71,
            owner=0,
            card_id=26000001,
            entity_kind="hero",
            position=(9.0, 12.0),
            causal_group=reference,
        ),
    )

    groups = build_causal_groups(children)

    assert len(groups) == 1
    assert groups[0].child_entity_ids == (71, 72)
    assert groups[0].parent_entity_id == 70


def test_volley_group_retains_multiple_exact_parent_entities() -> None:
    entities = tuple(
        EntityStateV1(
            entity_id=200 + index,
            owner=0,
            card_id=26000058,
            entity_kind="projectile",
            position=(8.0 + index, 12.0),
            causal_group=CausalGroupRefV1(
                kind=CausalGroupKind.VOLLEY,
                handle="volley:50:50:deploy:615",
                source_card_id=26000058,
                parent_entity_id=1005000094 + index,
            ),
        )
        for index in range(2)
    )

    groups = build_causal_groups(entities)

    assert len(groups) == 1
    assert groups[0].child_entity_ids == (200, 201)
    assert groups[0].parent_entity_ids == (1005000094, 1005000095)
    assert groups[0].parent_entity_id is None


def test_causal_group_reference_round_trips_with_entity_contract() -> None:
    entity = EntityStateV1(
        entity_id=44,
        owner=1,
        card_id=26000044,
        entity_kind="projectile",
        position=(9.0, 18.0),
        causal_group=CausalGroupRefV1(
            kind=CausalGroupKind.VOLLEY,
            handle="volley:3:9:cause:77",
            source_card_id=26000044,
            parent_entity_id=12,
        ),
    )

    restored = EntityStateV1.from_mapping(entity.to_dict())

    assert restored == entity
    assert restored.causal_group is not None
    assert restored.causal_group.handle == "volley:3:9:cause:77"
