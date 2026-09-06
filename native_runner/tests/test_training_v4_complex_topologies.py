from __future__ import annotations

import pytest

from native_runner.card_specs import build_card_catalog
from native_runner.contracts import (
    CausalGroupKind,
    CausalGroupRefV1,
    EntityStateV1,
)
from native_runner.training.v4.grouping import build_causal_groups
from native_runner.training.v4.catalog import normal_mode_card_specs


# This is a coverage ledger, not production card-specific logic.  Each row is
# one representative normal-mode root whose children may differ in form,
# archetype, position, or lifetime while sharing one exact public handle.
COMPLEX_TOPOLOGY_CASES = (
    ("SkeletonArmy", 26_000_012, CausalGroupKind.DEPLOYMENT, (26_000_010,) * 4),
    (
        "GoblinGang",
        26_000_041,
        CausalGroupKind.DEPLOYMENT,
        (26_000_002, 26_000_019) * 2,
    ),
    ("Rascals", 26_000_053, CausalGroupKind.DEPLOYMENT, (26_000_053,) * 3),
    (
        "GoblinGiant",
        26_000_060,
        CausalGroupKind.DEPLOYMENT,
        (26_000_060, 26_000_019, 26_000_019),
    ),
    ("Goblinstein", 26_000_099, CausalGroupKind.DEPLOYMENT, (26_000_099,) * 2),
    ("Golem", 26_000_009, CausalGroupKind.SPAWN_WAVE, (26_000_009,) * 2),
    ("LavaHound", 26_000_029, CausalGroupKind.SPAWN_WAVE, (26_000_029,) * 6),
    ("BattleRam", 26_000_036, CausalGroupKind.SPAWN_WAVE, (26_000_036,) * 2),
    ("SkeletonBalloon", 26_000_056, CausalGroupKind.SPAWN_WAVE, (26_000_010,) * 4),
    ("ElixirGolem", 26_000_067, CausalGroupKind.SPAWN_WAVE, (26_000_067,) * 2),
    ("Phoenix", 26_000_087, CausalGroupKind.SPAWN_WAVE, (26_000_087,)),
    ("SuspiciousBush", 26_000_097, CausalGroupKind.SPAWN_WAVE, (26_000_002,) * 2),
    ("Witch", 26_000_007, CausalGroupKind.SPAWN_WAVE, (26_000_010,) * 4),
    ("GoblinDrill", 27_000_013, CausalGroupKind.SPAWN_WAVE, (26_000_002,) * 2),
    ("Graveyard", 28_000_010, CausalGroupKind.PERSISTENT_EFFECT, (26_000_010,) * 3),
    ("GoblinBarrel", 28_000_004, CausalGroupKind.SPAWN_WAVE, (26_000_002,) * 3),
    ("Hunter", 26_000_044, CausalGroupKind.VOLLEY, (26_000_044,) * 10),
    ("Firecracker", 26_000_064, CausalGroupKind.VOLLEY, (26_000_064,) * 5),
    ("Clone", 28_000_013, CausalGroupKind.SPAWN_WAVE, (26_000_001, 26_000_002)),
    ("GoblinCurse", 28_000_024, CausalGroupKind.SPAWN_WAVE, (26_000_001, 26_000_002)),
)
NORMAL_MODE_SPECS = normal_mode_card_specs(build_card_catalog().by_id)


@pytest.mark.parametrize(
    ("name", "source_card_id", "kind", "child_card_ids"),
    COMPLEX_TOPOLOGY_CASES,
)
def test_representative_complex_card_uses_one_explicit_public_root(
    name: str,
    source_card_id: int,
    kind: CausalGroupKind,
    child_card_ids: tuple[int, ...],
) -> None:
    assert NORMAL_MODE_SPECS[source_card_id].name == name
    reference = CausalGroupRefV1(
        kind=kind,
        handle=f"golden:{name}:root",
        source_card_id=source_card_id,
        parent_entity_id=900 if kind != CausalGroupKind.DEPLOYMENT else None,
    )
    children = tuple(
        EntityStateV1(
            entity_id=1_000 + index,
            owner=0,
            card_id=card_id,
            entity_kind=("projectile" if kind == CausalGroupKind.VOLLEY else "troop"),
            position=(4.0 + index, 8.0 + (index % 2)),
            causal_group=reference,
        )
        for index, card_id in enumerate(child_card_ids)
    )

    groups = build_causal_groups(reversed(children))

    assert len(groups) == 1
    assert groups[0].key.kind == kind
    assert groups[0].source_card_id == source_card_id
    assert groups[0].child_entity_ids == tuple(
        range(1_000, 1_000 + len(child_card_ids))
    )
