from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from native_runner.contracts import (
    ActionKind,
    ActionMaskV1,
    ActionV1,
    ObservationTier,
    ObservationV1,
    PlayerStateV1,
    TargetKind,
)
from native_runner.perspective import PerspectiveTransformV1
from native_runner.training.v4.catalog import CardCatalogV1
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.decoding import (
    ShadowCandidateLegality,
    decode_action_sequence_v4,
)
from native_runner.training.v4.expert import (
    ExpertActionAlignmentError,
    TimedExpertActionV4,
    build_expert_action_batch,
)
from native_runner.training.v4.tensors import (
    ActionCandidatesV4,
    ActionSequenceV4,
    CANDIDATE_ABILITY,
    CANDIDATE_DEPLOY,
    GATE_ACT,
    TARGET_GRID,
    TARGET_NONE,
    target_cell_from_xy,
)
from native_runner.tests.test_training_v4_model import _batch


DECK = tuple(26_000_000 + index for index in range(8))
CARD_COSTS = {card_id: 3.0 for card_id in DECK}


def _candidate_set(
    *,
    variants: tuple[int, ...],
    target_modes: tuple[int, ...],
    own_rows: tuple[int, ...],
    native_slots: tuple[int, ...],
    native_sources: tuple[int, ...],
    exclusions: tuple[int, ...],
    card_ids: tuple[int, ...],
    placements: torch.Tensor | None = None,
    building_offsets: tuple[tuple[float, float], ...] | None = None,
) -> ActionCandidatesV4:
    config = ModelConfigV4()
    count = len(variants)
    catalog = CardCatalogV1.from_raw_ids(DECK)
    placement = (
        placements.clone()
        if placements is not None
        else torch.zeros(1, count, 32, 18, dtype=torch.bool)
    )
    offsets = building_offsets or ((0.0, 0.0),) * count
    is_building = torch.tensor(
        [[bool(target_modes[index] == TARGET_GRID) for index in range(count)]],
        dtype=torch.bool,
    )
    return ActionCandidatesV4(
        mask=torch.ones(1, count, dtype=torch.bool),
        elixir=torch.tensor([10.0]),
        uid=torch.tensor([[100 + index for index in range(count)]]),
        variant=torch.tensor([variants]),
        visible_card_vocab_id=torch.tensor(
            [[catalog.vocab_id(card_id) for card_id in card_ids]]
        ),
        effective_card_vocab_id=torch.tensor(
            [[catalog.vocab_id(card_id) for card_id in card_ids]]
        ),
        effective_form=torch.zeros(1, count, dtype=torch.long),
        own_card_row=torch.tensor([own_rows]),
        source_group_index=torch.full((1, count), -1, dtype=torch.long),
        source_child_index=torch.full((1, count), -1, dtype=torch.long),
        ability_vocab_id=torch.tensor(
            [[1 if variant == CANDIDATE_ABILITY else 0 for variant in variants]]
        ),
        cost=torch.tensor(
            [[
                CARD_COSTS[card_id] if variant == CANDIDATE_DEPLOY else 1.0
                for card_id, variant in zip(card_ids, variants, strict=True)
            ]]
        ),
        runtime_features=torch.zeros(
            1,
            count,
            config.candidate_runtime_feature_dim,
        ),
        target_mode=torch.tensor([target_modes]),
        placement=placement,
        is_building=is_building,
        building_half_width=is_building.to(torch.float32),
        building_half_height=is_building.to(torch.float32),
        building_offset_x=torch.tensor([[item[0] for item in offsets]]),
        building_offset_y=torch.tensor([[item[1] for item in offsets]]),
        exclusion_group_id=torch.tensor([exclusions]),
        native_hand_slot=torch.tensor([native_slots]),
        native_source_entity=torch.tensor([native_sources]),
        native_visible_card_id=torch.tensor(
            [[
                card_id if variant == CANDIDATE_DEPLOY else -1
                for card_id, variant in zip(card_ids, variants, strict=True)
            ]]
        ),
    )


def test_shadow_allows_at_most_one_ability_per_compound_action() -> None:
    config = ModelConfigV4()
    placements = torch.zeros(1, 3, 32, 18, dtype=torch.bool)
    placements[0, 2] = True
    candidates = _candidate_set(
        variants=(CANDIDATE_ABILITY, CANDIDATE_ABILITY, CANDIDATE_DEPLOY),
        target_modes=(TARGET_NONE, TARGET_NONE, TARGET_GRID),
        own_rows=(0, 1, 2),
        native_slots=(-1, -1, 0),
        native_sources=(501, 502, -1),
        exclusions=(2001, 2002, 1000),
        card_ids=DECK[:3],
        placements=placements,
    )

    shadow = ShadowCandidateLegality(candidates, config)
    shadow.apply(
        torch.tensor([True]),
        torch.tensor([0]),
        torch.tensor([-1]),
        torch.tensor([0]),
        step=0,
    )
    legal = shadow.candidate_mask()[0]
    assert not bool(legal[1])
    assert bool(legal[2])

    deploy_first = ShadowCandidateLegality(candidates, config)
    deploy_first.apply(
        torch.tensor([True]),
        torch.tensor([2]),
        torch.tensor([0]),
        torch.tensor([0]),
        step=0,
    )
    assert bool(deploy_first.candidate_mask()[0, 0])
    assert bool(deploy_first.candidate_mask()[0, 1])


@pytest.mark.parametrize(
    ("mask", "expected_index"),
    (
        ((True, True), 1),
        ((True, False), 0),
    ),
)
def test_inferred_expert_ability_uses_newest_available_source(
    mask: tuple[bool, bool],
    expected_index: int,
) -> None:
    candidates = _candidate_set(
        variants=(CANDIDATE_ABILITY, CANDIDATE_ABILITY),
        target_modes=(TARGET_NONE, TARGET_NONE),
        own_rows=(0, 1),
        native_slots=(-1, -1),
        native_sources=(1_000_000_501, 1_000_000_777),
        exclusions=(2001, 2002),
        card_ids=DECK[:2],
    )
    candidates = replace(
        candidates,
        mask=torch.tensor([mask], dtype=torch.bool),
    )
    batch = replace(_batch(batch_size=1), candidates=candidates)
    action = ActionV1(
        owner=0,
        kind=ActionKind.ACTIVATE_ABILITY,
        source_entity=0,
        metadata={"source_entity_inferred_at_render": True},
    )

    sequence = build_expert_action_batch(
        batch,
        ((TimedExpertActionV4(source_tick=130, action=action),),),
        decision_ticks_by_row=(130,),
        perspectives=(PerspectiveTransformV1(actor_owner=0),),
    )

    assert int(sequence.candidate_index[0, 0]) == expected_index


def test_expert_grid_target_must_be_legal_under_shadow_placement() -> None:
    placements = torch.zeros(1, 1, 32, 18, dtype=torch.bool)
    placements[0, 0, 4, 3] = True
    candidates = _candidate_set(
        variants=(CANDIDATE_DEPLOY,),
        target_modes=(TARGET_GRID,),
        own_rows=(0,),
        native_slots=(0,),
        native_sources=(-1,),
        exclusions=(1000,),
        card_ids=DECK[:1],
        placements=placements,
    )
    batch = replace(_batch(batch_size=1), candidates=candidates)
    action = ActionV1(
        owner=0,
        kind=ActionKind.PLAY_CARD,
        hand_slot=0,
        card_id=DECK[0],
        target_kind=TargetKind.GRID,
        target_grid=(4, 4),
    )

    with pytest.raises(
        ExpertActionAlignmentError,
        match="expert GRID target is illegal",
    ):
        build_expert_action_batch(
            batch,
            ((TimedExpertActionV4(source_tick=130, action=action),),),
            decision_ticks_by_row=(130,),
            perspectives=(PerspectiveTransformV1(actor_owner=0),),
        )


def _native_observation(owner: int) -> ObservationV1:
    rows = tuple(tuple(True for _ in range(18)) for _ in range(32))
    placement = {
        str(slot): {
            "card_id": DECK[slot],
            "effective_cost": CARD_COSTS[DECK[slot]],
            "form_code": 0,
            "row_major": rows,
            "model_subcell_offset": (0.5, 0.5),
        }
        for slot in range(2)
    }
    return ObservationV1(
        tier=ObservationTier.FAIR,
        tick=10,
        owner=owner,
        players=(
            PlayerStateV1(
                owner=owner,
                elixir_exact=10.0,
                elixir_visible=10.0,
                private_state_visible=True,
            ),
            PlayerStateV1(owner=1 - owner),
        ),
        action_mask=ActionMaskV1(
            kinds={
                ActionKind.WAIT.value: True,
                ActionKind.PLAY_CARD.value: True,
            },
            hand_slots=(True, True, False, False),
            placement_masks=placement,
            reasons={"effective_elixir": 10.0},
        ),
    )


@pytest.mark.parametrize("owner", (0, 1))
@pytest.mark.parametrize("horizontal_mirror", (False, True))
def test_building_offset_matches_shadow_and_native_decode(
    owner: int,
    horizontal_mirror: bool,
) -> None:
    config = ModelConfigV4()
    native_observation = _native_observation(owner)
    perspective = PerspectiveTransformV1(
        actor_owner=owner,
        horizontal_mirror=horizontal_mirror,
    )
    model_observation = perspective.observation_policy_metadata_to_model(native_observation)
    entries = model_observation.action_mask.placement_masks
    first_entry = entries["0"]
    model_offset = tuple(float(value) for value in first_entry["model_subcell_offset"])
    assert model_offset == (
        -0.5 if horizontal_mirror else 0.5,
        0.5,
    )

    placement = torch.tensor(
        [
            [
                first_entry["row_major"],
                entries["1"]["row_major"],
            ]
        ],
        dtype=torch.bool,
    )
    candidates = _candidate_set(
        variants=(CANDIDATE_DEPLOY, CANDIDATE_DEPLOY),
        target_modes=(TARGET_GRID, TARGET_GRID),
        own_rows=(0, 1),
        native_slots=(0, 1),
        native_sources=(-1, -1),
        exclusions=(1000, 1001),
        card_ids=DECK[:2],
        placements=placement,
        building_offsets=(model_offset, model_offset),
    )
    target_x, target_y = 5, 8
    target_cell = int(
        target_cell_from_xy(
            torch.tensor([target_x]),
            torch.tensor([target_y]),
        )[0]
    )

    shadow = ShadowCandidateLegality(candidates, config)
    shadow.apply(
        torch.tensor([True]),
        torch.tensor([0]),
        torch.tensor([target_cell]),
        torch.tensor([0]),
        step=0,
    )
    assert not bool(shadow.placement()[0, 1, target_y, target_x])
    assert bool(shadow.placement()[0, 1, 0, 0])

    sequence = ActionSequenceV4(
        gate=torch.tensor([GATE_ACT]),
        micro_action_count=torch.tensor([1]),
        candidate_index=torch.tensor([[0, -1]]),
        candidate_uid=torch.tensor([[100, -1]]),
        target_cell=torch.tensor([[target_cell, -1]]),
        delay_offset_bin=torch.tensor([[0, -1]]),
    )
    decoded = decode_action_sequence_v4(
        sequence,
        candidates,
        row=0,
        observation=native_observation,
        catalog=CardCatalogV1.from_raw_ids(DECK),
        deck=DECK,
        card_costs=CARD_COSTS,
        horizontal_mirror=horizontal_mirror,
        config=config,
    )
    action = decoded.actions[0]
    expected_native_grid = (
        17 - target_x if perspective.flip_x else target_x,
        31 - target_y if perspective.flip_y else target_y,
    )
    assert action.target_grid == expected_native_grid
    assert action.subcell_offset == pytest.approx(
        (0.5 if owner == 0 else -0.5, 0.5 if owner == 0 else -0.5)
    )

    model_center = (
        target_x + 0.5 + model_offset[0],
        target_y + 0.5 + model_offset[1],
    )
    native_center = (
        action.target_grid[0] + 0.5 + action.subcell_offset[0],
        action.target_grid[1] + 0.5 + action.subcell_offset[1],
    )
    assert native_center == pytest.approx(
        (
            18.0 - model_center[0] if perspective.flip_x else model_center[0],
            32.0 - model_center[1] if perspective.flip_y else model_center[1],
        )
    )
