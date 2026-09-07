"""Exact public causal grouping for V4 battle entities.

Only the opaque ``(kind, handle)`` address decides membership.  Card identity,
child archetype, position, and time are descriptive attributes and can never
merge unrelated entities.  An entity without an exact public reference becomes
its own conservative singleton group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ...contracts import CausalGroupKind, EntityStateV1


@dataclass(frozen=True, slots=True)
class CausalGroupKeyV4:
    kind: CausalGroupKind
    handle: str


@dataclass(frozen=True, slots=True)
class CausalGroupV4:
    key: CausalGroupKeyV4
    owner: int | None
    source_card_id: int | None
    parent_entity_ids: tuple[int, ...]
    child_entity_ids: tuple[int, ...]

    @property
    def parent_entity_id(self) -> int | None:
        """Return the parent only when the whole group has one unique parent."""

        return self.parent_entity_ids[0] if len(self.parent_entity_ids) == 1 else None


def _singleton_kind(entity: EntityStateV1) -> CausalGroupKind:
    if entity.projectile_state is not None or entity.entity_kind == "projectile":
        return CausalGroupKind.VOLLEY
    if entity.entity_kind in {"area", "effect"}:
        return CausalGroupKind.PERSISTENT_EFFECT
    return CausalGroupKind.DEPLOYMENT


def causal_group_key(entity: EntityStateV1) -> CausalGroupKeyV4:
    """Return the exact public address, or an entity-local singleton address."""

    reference = entity.causal_group
    if reference is None:
        return CausalGroupKeyV4(kind=_singleton_kind(entity), handle=f"singleton:{int(entity.entity_id)}")
    return CausalGroupKeyV4(kind=reference.kind, handle=reference.handle)


def _merge_optional(current: int | None, candidate: int | None, *, label: str, key: CausalGroupKeyV4) -> int | None:
    if current is None:
        return candidate
    if candidate is None or candidate == current:
        return current
    raise ValueError(f"causal group {key.kind.value}:{key.handle} has conflicting {label}: {current} != {candidate}")


def build_causal_groups(entities: Iterable[EntityStateV1]) -> tuple[CausalGroupV4, ...]:
    """Group entities solely by their public causal address."""

    grouped: dict[CausalGroupKeyV4, tuple[int | None, int | None, set[int], list[int]]] = {}
    seen_entity_ids: set[int] = set()
    for entity in entities:
        entity_id = int(entity.entity_id)
        if entity_id in seen_entity_ids:
            raise ValueError(f"duplicate entity ID {entity_id} in causal grouping")
        seen_entity_ids.add(entity_id)
        key = causal_group_key(entity)
        reference = entity.causal_group
        source_card_id = reference.source_card_id if reference is not None else entity.card_id
        parent_entity_id = reference.parent_entity_id if reference is not None else entity.source_entity
        current = grouped.get(key)
        if current is None:
            grouped[key] = (
                entity.owner,
                source_card_id,
                ({int(parent_entity_id)} if parent_entity_id is not None else set()),
                [entity_id],
            )
            continue
        owner, source_card, parent_entities, child_ids = current
        owner = _merge_optional(owner, entity.owner, label="owner", key=key)
        source_card = _merge_optional(
            source_card, source_card_id, label=f"source_card_id at entity {entity_id}", key=key
        )
        if parent_entity_id is not None:
            parent_entities.add(int(parent_entity_id))
        child_ids.append(entity_id)
        grouped[key] = (owner, source_card, parent_entities, child_ids)

    return tuple(
        CausalGroupV4(
            key=key,
            owner=owner,
            source_card_id=source_card_id,
            parent_entity_ids=tuple(sorted(parent_entity_ids)),
            child_entity_ids=tuple(sorted(child_ids)),
        )
        for key, (owner, source_card_id, parent_entity_ids, child_ids) in sorted(
            grouped.items(), key=lambda item: (item[0].kind.value, item[0].handle)
        )
    )
