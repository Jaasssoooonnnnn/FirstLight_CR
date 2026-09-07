from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from native_runner.contracts import (
    ActionKind,
    ActionMaskV1,
    AbilityPhase,
    AbilityRuntimeStateV1,
    ENTITY_RUNTIME_SEMANTIC_FIELDS,
    EntityStateV1,
    ObservationTier,
    ObservationV1,
    PLAYER_RUNTIME_SEMANTIC_FIELDS,
    PlayerStateV1,
    SemanticEvidenceLevel,
    SemanticProvenanceV1,
)
from native_runner.training.v4.tensors import ActionCandidatesV4, ActionSequenceV4, CANDIDATE_ABILITY, CANDIDATE_DEPLOY, GATE_ACT, GATE_WAIT, TARGET_GRID, TARGET_NONE, target_cell_from_xy
from native_runner.training.v4.catalog import CardCatalogV1
from native_runner.training.v4.native_actions import MIRROR_CARD_ID
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.decoding import ShadowCandidateLegality, candidate_uid_v4, decode_action_sequence_v4


DECK = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008)
CARD_COSTS = {card_id: float(index % 4 + 1) for index, card_id in enumerate(DECK)}
MERGE_MAIDEN_CARD_ID = 28_000_025


def _catalog(*raw_card_ids: int) -> CardCatalogV1:
    return CardCatalogV1.from_raw_ids(raw_card_ids or DECK)


def _ability_entity(
    *,
    owner: int,
    source_entity: int,
    card_id: int,
    ability_id: str,
    cost: float,
) -> EntityStateV1:
    return EntityStateV1(
        entity_id=source_entity,
        owner=owner,
        card_id=card_id,
        entity_kind="hero",
        position=(9000.0, 12000.0),
        ability_states=(
            AbilityRuntimeStateV1(
                ability_id=ability_id,
                source_entity=source_entity,
                phase=AbilityPhase.READY,
                elixir_cost=cost,
                available=True,
                attributes={
                    "source_card_id": card_id,
                    "controller_slot": source_entity,
                },
            ),
        ),
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


def _observation(
    *,
    owner: int = 0,
    deck: tuple[int, ...] = DECK,
    ability_entities: tuple[EntityStateV1, ...] = (),
    mirror_effective_card_id: int | None = None,
) -> ObservationV1:
    rows = [[True] * 18 for _ in range(32)]
    placement_masks: dict[str, dict[str, object]] = {}
    hand_runtime: dict[str, dict[str, int | float]] = {}
    for slot, visible_card_id in enumerate(deck[:4]):
        effective_card_id = visible_card_id
        effective_cost = CARD_COSTS.get(visible_card_id, 0.0)
        runtime: dict[str, int | float] = {
            "effective_cost": effective_cost,
            "form_code": 0,
        }
        if visible_card_id == MIRROR_CARD_ID:
            assert mirror_effective_card_id is not None
            effective_card_id = mirror_effective_card_id
            effective_cost = CARD_COSTS[effective_card_id] + 1.0
            runtime = {
                "visible_card_id": visible_card_id,
                "effective_card_id": effective_card_id,
                "effective_cost": effective_cost,
                "form_code": 0,
            }
            hand_runtime[str(slot)] = {
                "hand_slot": slot,
                **runtime,
            }
        placement_masks[str(slot)] = {
            "card_id": effective_card_id,
            "shape": [32, 18],
            "row_major": rows,
            "model_subcell_offset": [0.25, -0.25],
            **runtime,
        }
    return ObservationV1(
        tier=ObservationTier.FAIR,
        owner=owner,
        tick=20,
        players=(
            PlayerStateV1(
                owner=owner,
                elixir_exact=10.0,
                elixir_visible=10.0,
                hand=deck[:4],
                deck=deck,
                cycle=deck[4:],
                private_state_visible=True,
                ability_runtime_states=tuple(
                    state
                    for entity in ability_entities
                    for state in entity.ability_states
                ),
                metadata={
                    "hand_slot_by_card": {
                        str(card_id): slot
                        for slot, card_id in enumerate(deck[:4])
                    },
                    **(
                        {"hand_runtime_by_slot": hand_runtime}
                        if hand_runtime
                        else {}
                    ),
                },
                runtime_provenance=SemanticProvenanceV1(
                    field_evidence={
                        field: (
                            SemanticEvidenceLevel.NATIVE_DERIVED
                            if field == "ability_runtime_states"
                            and ability_entities
                            else SemanticEvidenceLevel.UNKNOWN
                        )
                        for field in PLAYER_RUNTIME_SEMANTIC_FIELDS
                    }
                ),
            ),
            PlayerStateV1(owner=1 - owner),
        ),
        entities=ability_entities,
        action_mask=ActionMaskV1(
            kinds={
                ActionKind.WAIT.value: True,
                ActionKind.PLAY_CARD.value: True,
                ActionKind.ACTIVATE_ABILITY.value: bool(ability_entities),
            },
            hand_slots=(True, True, True, True),
            placement_masks=placement_masks,
            ability_sources=tuple(
                int(entity.entity_id) for entity in ability_entities
            ),
            reasons={"effective_elixir": 10.0, "reserved_elixir": 0.0},
        ),
    )


def _candidates(
    catalog: CardCatalogV1,
    rows: tuple[dict[str, int | float], ...],
) -> ActionCandidatesV4:
    config = ModelConfigV4()
    count = len(rows)
    placement = torch.zeros(1, count, 32, 18, dtype=torch.bool)
    for index, row in enumerate(rows):
        if int(row["target_mode"]) == TARGET_GRID:
            placement[0, index] = True
    uids = [
        candidate_uid_v4(
            variant=int(row["variant"]),
            visible_card_vocab_id=int(row["visible_card_vocab_id"]),
            effective_card_vocab_id=int(row["effective_card_vocab_id"]),
            native_hand_slot=int(row.get("native_hand_slot", -1)),
            native_source_entity=int(row.get("native_source_entity", -1)),
            ability_vocab_id=int(row.get("ability_vocab_id", 0)),
        )
        for row in rows
    ]
    return ActionCandidatesV4(
        mask=torch.ones(1, count, dtype=torch.bool),
        elixir=torch.tensor([10.0]),
        uid=torch.tensor([uids], dtype=torch.long),
        variant=torch.tensor([[int(row["variant"]) for row in rows]]),
        visible_card_vocab_id=torch.tensor(
            [[int(row["visible_card_vocab_id"]) for row in rows]]
        ),
        effective_card_vocab_id=torch.tensor(
            [[int(row["effective_card_vocab_id"]) for row in rows]]
        ),
        effective_form=torch.tensor(
            [[int(row.get("effective_form", 0)) for row in rows]]
        ),
        own_card_row=torch.tensor(
            [[int(row.get("own_card_row", -1)) for row in rows]]
        ),
        source_group_index=torch.full((1, count), -1, dtype=torch.long),
        source_child_index=torch.full((1, count), -1, dtype=torch.long),
        ability_vocab_id=torch.tensor(
            [[int(row.get("ability_vocab_id", 0)) for row in rows]]
        ),
        cost=torch.tensor([[float(row["cost"]) for row in rows]]),
        runtime_features=torch.zeros(
            1,
            count,
            config.candidate_runtime_feature_dim,
        ),
        target_mode=torch.tensor(
            [[int(row["target_mode"]) for row in rows]]
        ),
        placement=placement,
        is_building=torch.zeros(1, count, dtype=torch.bool),
        building_half_width=torch.zeros(1, count),
        building_half_height=torch.zeros(1, count),
        building_offset_x=torch.zeros(1, count),
        building_offset_y=torch.zeros(1, count),
        exclusion_group_id=torch.tensor(
            [[int(row.get("exclusion_group_id", index)) for index, row in enumerate(rows)]]
        ),
        native_hand_slot=torch.tensor(
            [[int(row.get("native_hand_slot", -1)) for row in rows]]
        ),
        native_source_entity=torch.tensor(
            [[int(row.get("native_source_entity", -1)) for row in rows]]
        ),
        native_visible_card_id=torch.tensor(
            [[int(row.get("native_visible_card_id", -1)) for row in rows]]
        ),
    )


def _sequence(
    candidates: ActionCandidatesV4,
    *,
    selected_rows: tuple[int, ...],
    target_cells: tuple[int, ...],
    offset_bins: tuple[int, ...],
) -> ActionSequenceV4:
    count = len(selected_rows)
    indices = [-1, -1]
    uids = [-1, -1]
    targets = [-1, -1]
    delays = [-1, -1]
    for step, candidate_row in enumerate(selected_rows):
        indices[step] = candidate_row
        uids[step] = int(candidates.uid[0, candidate_row])
        targets[step] = target_cells[step]
        delays[step] = offset_bins[step]
    return ActionSequenceV4(
        gate=torch.tensor([GATE_ACT if count else GATE_WAIT]),
        micro_action_count=torch.tensor([count]),
        candidate_index=torch.tensor([indices]),
        candidate_uid=torch.tensor([uids]),
        target_cell=torch.tensor([targets]),
        delay_offset_bin=torch.tensor([delays]),
    )


def _deploy_row(
    catalog: CardCatalogV1,
    *,
    card_id: int,
    effective_card_id: int | None = None,
    hand_slot: int,
    cost: float,
    own_card_row: int,
) -> dict[str, int | float]:
    effective = card_id if effective_card_id is None else effective_card_id
    return {
        "variant": CANDIDATE_DEPLOY,
        "visible_card_vocab_id": catalog.vocab_id(card_id),
        "effective_card_vocab_id": catalog.vocab_id(effective),
        "own_card_row": own_card_row,
        "cost": cost,
        "target_mode": TARGET_GRID,
        "native_hand_slot": hand_slot,
        "native_visible_card_id": card_id,
    }


def test_fixed_wait_decodes_to_exactly_five_ticks() -> None:
    catalog = _catalog()
    candidates = _candidates(
        catalog,
        (
            _deploy_row(
                catalog,
                card_id=DECK[0],
                hand_slot=0,
                cost=CARD_COSTS[DECK[0]],
                own_card_row=0,
            ),
        ),
    )
    sequence = _sequence(
        candidates,
        selected_rows=(),
        target_cells=(),
        offset_bins=(),
    )

    compound = decode_action_sequence_v4(
        sequence,
        candidates,
        row=0,
        observation=_observation(),
        catalog=catalog,
        deck=DECK,
        card_costs=CARD_COSTS,
    )

    assert len(compound.actions) == 1
    assert compound.actions[0].kind == ActionKind.WAIT
    assert compound.actions[0].next_decision_ticks == 5
    assert "duration" not in " ".join(compound.actions[0].metadata).lower()


def test_non_mirror_deploy_uses_exact_live_variant_cost() -> None:
    deck = (MERGE_MAIDEN_CARD_ID, *DECK[:7])
    card_costs = {**CARD_COSTS, MERGE_MAIDEN_CARD_ID: 6.0}
    catalog = _catalog(*deck)
    candidates = _candidates(
        catalog,
        (
            _deploy_row(
                catalog,
                card_id=MERGE_MAIDEN_CARD_ID,
                hand_slot=0,
                cost=3.0,
                own_card_row=0,
            ),
        ),
    )
    observation = _observation(deck=deck)
    placement = {
        key: dict(value)
        for key, value in observation.action_mask.placement_masks.items()
    }
    placement["0"]["effective_cost"] = 3.0
    observation = replace(
        observation,
        action_mask=replace(
            observation.action_mask,
            placement_masks=placement,
        ),
    )
    target_cell = int(
        target_cell_from_xy(torch.tensor([9]), torch.tensor([10]))[0]
    )
    sequence = _sequence(
        candidates,
        selected_rows=(0,),
        target_cells=(target_cell,),
        offset_bins=(0,),
    )

    compound = decode_action_sequence_v4(
        sequence,
        candidates,
        row=0,
        observation=observation,
        catalog=catalog,
        deck=deck,
        card_costs=card_costs,
    )

    action = compound.actions[0]
    assert action.card_id == MERGE_MAIDEN_CARD_ID
    assert action.metadata["policy_effective_cost"] == pytest.approx(3.0)


def test_two_deploys_use_row_major_cells_and_absolute_offsets() -> None:
    catalog = _catalog()
    candidates = _candidates(
        catalog,
        (
            _deploy_row(
                catalog,
                card_id=DECK[0],
                hand_slot=0,
                cost=CARD_COSTS[DECK[0]],
                own_card_row=0,
            ),
            _deploy_row(
                catalog,
                card_id=DECK[1],
                hand_slot=1,
                cost=CARD_COSTS[DECK[1]],
                own_card_row=1,
            ),
        ),
    )
    cells = tuple(
        int(value)
        for value in target_cell_from_xy(
            torch.tensor([2, 4]),
            torch.tensor([3, 5]),
            width=18,
            height=32,
        )
    )
    sequence = _sequence(
        candidates,
        selected_rows=(0, 1),
        target_cells=cells,
        offset_bins=(1, 2),
    )

    compound = decode_action_sequence_v4(
        sequence,
        candidates,
        row=0,
        observation=_observation(owner=1),
        catalog=catalog,
        deck=DECK,
        card_costs=CARD_COSTS,
    )

    assert [action.target_grid for action in compound.actions] == [
        (15, 28),
        (13, 26),
    ]
    assert [action.execute_offset_ticks for action in compound.actions] == [2, 3]
    assert [action.next_decision_ticks for action in compound.actions] == [5, 5]
    assert compound.actions[0].subcell_offset == pytest.approx((-0.25, 0.25))
    assert compound.actions[1].metadata["policy_delay_offset_ms"] == 100
    assert "policy_cumulative_delay_ms" not in compound.actions[1].metadata


def test_candidate_uid_rebinds_reordered_ability_to_exact_source() -> None:
    catalog = _catalog()
    entities = (
        _ability_entity(
            owner=0,
            source_entity=501,
            card_id=DECK[0],
            ability_id="ability-alpha",
            cost=2.0,
        ),
        _ability_entity(
            owner=0,
            source_entity=777,
            card_id=DECK[1],
            ability_id="ability-beta",
            cost=3.0,
        ),
    )
    rows = tuple(
        {
            "variant": CANDIDATE_ABILITY,
            "visible_card_vocab_id": catalog.vocab_id(entity.card_id),
            "effective_card_vocab_id": catalog.vocab_id(entity.card_id),
            "own_card_row": index,
            "ability_vocab_id": index + 10,
            "cost": float(index + 2),
            "target_mode": TARGET_NONE,
            "native_source_entity": entity.entity_id,
        }
        for index, entity in enumerate(entities)
    )
    candidates = _candidates(catalog, rows)
    sequence = _sequence(
        candidates,
        selected_rows=(1,),
        target_cells=(-1,),
        offset_bins=(0,),
    )
    permutation = torch.tensor([1, 0])
    reordered_values = {}
    for item in fields(candidates):
        value = getattr(candidates, item.name)
        reordered_values[item.name] = (
            value.index_select(1, permutation)
            if value.ndim >= 2 and value.shape[1] == 2
            else value
        )
    reordered = ActionCandidatesV4(**reordered_values)

    compound = decode_action_sequence_v4(
        sequence,
        reordered,
        row=0,
        observation=_observation(ability_entities=entities),
        catalog=catalog,
        deck=DECK,
        card_costs=CARD_COSTS,
        ability_id_by_vocab_id={10: "ability-alpha", 11: "ability-beta"},
    )

    assert len(compound.actions) == 1
    assert compound.actions[0].kind == ActionKind.ACTIVATE_ABILITY
    assert compound.actions[0].source_entity == 777
    assert compound.actions[0].ability_id == "ability-beta"


def test_mirror_keeps_visible_native_slot_and_effective_semantics_separate() -> None:
    mirror_deck = (MIRROR_CARD_ID, *DECK[:7])
    catalog = _catalog(*mirror_deck)
    copied = DECK[0]
    mirror_cost = CARD_COSTS[copied] + 1.0
    candidates = _candidates(
        catalog,
        (
            _deploy_row(
                catalog,
                card_id=MIRROR_CARD_ID,
                effective_card_id=copied,
                hand_slot=0,
                cost=mirror_cost,
                own_card_row=0,
            ),
        ),
    )
    cell = int(
        target_cell_from_xy(
            torch.tensor([9]),
            torch.tensor([10]),
            width=18,
            height=32,
        )[0]
    )
    sequence = _sequence(
        candidates,
        selected_rows=(0,),
        target_cells=(cell,),
        offset_bins=(0,),
    )

    compound = decode_action_sequence_v4(
        sequence,
        candidates,
        row=0,
        observation=_observation(
            deck=mirror_deck,
            mirror_effective_card_id=copied,
        ),
        catalog=catalog,
        deck=mirror_deck,
        card_costs=CARD_COSTS,
    )

    action = compound.actions[0]
    assert action.hand_slot == 0
    assert action.card_id == MIRROR_CARD_ID
    assert action.metadata["policy_visible_card_id"] == MIRROR_CARD_ID
    assert action.metadata["policy_effective_card_id"] == copied
    assert action.metadata["policy_effective_cost"] == pytest.approx(mirror_cost)


def test_mirror_is_never_legal_as_the_second_micro_action() -> None:
    mirror_deck = (MIRROR_CARD_ID, *DECK[:7])
    catalog = _catalog(*mirror_deck)
    copied = DECK[0]
    candidates = _candidates(
        catalog,
        (
            _deploy_row(
                catalog,
                card_id=copied,
                hand_slot=1,
                cost=2.0,
                own_card_row=1,
            ),
            _deploy_row(
                catalog,
                card_id=MIRROR_CARD_ID,
                effective_card_id=copied,
                hand_slot=0,
                cost=3.0,
                own_card_row=0,
            ),
        ),
    )
    shadow = ShadowCandidateLegality(candidates, ModelConfigV4())
    assert shadow.candidate_mask()[0].tolist() == [True, True]

    shadow.apply(
        torch.tensor([True]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        step=0,
    )

    assert shadow.candidate_mask()[0].tolist() == [False, False]
