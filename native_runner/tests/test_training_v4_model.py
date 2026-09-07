from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from native_runner.training.v4.tensors import ActiveEffectSetV4, ActionCandidatesV4, ActionSequenceV4, BattleGroupSetV4, CardSetV4, EventSetV4, GATE_ACT, PreviousActionV4, RelationEdgesV4, TARGET_GRID, TARGET_NONE, TowerSetV4, UniversalSemanticBatchV4, target_cell_from_xy, target_xy_from_cell
from native_runner.training.v4.mechanics import ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES, EffectSemanticCatalogV1, MechanicProfileCatalogV1
from native_runner.training.v4.catalog import AbilityCatalogV1, CardCatalogV1, EntityArchetypeCatalogV1
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.model import UniversalCardPolicyV4
from native_runner.training.v4.decoding import delay_offset_ms, wait_ticks
from native_runner.training.v4.components import SpatialScatterEncoder
from native_runner.training.v4.decoding import ShadowCandidateLegality
from native_runner.training.v4.factory import build_production_model_v4


def _catalog() -> CardCatalogV1:
    return CardCatalogV1.from_raw_ids((9004, 1002, 7007, 3003, 5005, 8008, 6006, 2001))


def _ability_catalog(catalog: CardCatalogV1) -> AbilityCatalogV1:
    return AbilityCatalogV1(
        catalog.raw_card_ids,
        ("ability-a",),
        ((0.0,) * ModelConfigV4().ability_feature_dim,),
    )


def _model(catalog: CardCatalogV1 | None = None) -> UniversalCardPolicyV4:
    actual = catalog or _catalog()
    abilities = _ability_catalog(actual)
    archetypes = EntityArchetypeCatalogV1(
        actual.raw_card_ids,
        ("form:fixture_a", "form:fixture_b"),
    )
    effects = EffectSemanticCatalogV1.empty(actual.raw_card_ids)
    return UniversalCardPolicyV4(
        actual,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        effect_catalog=effects,
        mechanic_profile_catalog=MechanicProfileCatalogV1.empty(
            card_catalog=actual,
            ability_catalog=abilities,
            entity_archetype_catalog=archetypes,
        ),
    )


def _batch(batch_size: int = 2) -> UniversalSemanticBatchV4:
    config = ModelConfigV4()
    catalog = _catalog()
    own_ids = torch.tensor(
        [[catalog.vocab_id(card_id) for card_id in catalog.raw_card_ids]]
        * batch_size,
        dtype=torch.long,
    )
    own_cards = CardSetV4(
        card_vocab_id=own_ids,
        runtime_form=torch.zeros(batch_size, 8, dtype=torch.long),
        role_bits=torch.zeros(batch_size, 8, 2, dtype=torch.bool),
        runtime_features=torch.zeros(
            batch_size,
            8,
            config.card_runtime_feature_dim,
        ),
        mask=torch.ones(batch_size, 8, dtype=torch.bool),
    )
    opponent_mask = torch.zeros(batch_size, 9, dtype=torch.bool)
    opponent_mask[:, :4] = True
    opponent_ids = torch.zeros(batch_size, 9, dtype=torch.long)
    opponent_ids[:, :3] = own_ids[:, :3]
    opponent_ids[:, 3] = 1  # OpponentUnknownSummary uses UNKNOWN semantics.
    opponent_cards = CardSetV4(
        card_vocab_id=opponent_ids,
        runtime_form=torch.zeros(batch_size, 9, dtype=torch.long),
        role_bits=torch.zeros(batch_size, 9, 2, dtype=torch.bool),
        runtime_features=torch.zeros(
            batch_size,
            9,
            config.card_runtime_feature_dim,
        ),
        mask=opponent_mask,
    )
    tower_position = torch.tensor(
        [
            [2.5, 3.0],
            [9.0, 2.0],
            [15.5, 3.0],
            [2.5, 29.0],
            [9.0, 30.0],
            [15.5, 29.0],
        ]
    ).repeat(batch_size, 1, 1)
    towers = TowerSetV4(
        tower_type=torch.tensor([[0, 1, 1, 0, 1, 1]] * batch_size),
        tower_troop_type=torch.tensor(
            [[0, 1, 1, 0, 1, 1]] * batch_size
        ),
        owner_type=torch.tensor([[0, 0, 0, 1, 1, 1]] * batch_size),
        features=torch.zeros(batch_size, 6, config.tower_feature_dim),
        position=tower_position,
        extent=torch.cat((tower_position - 0.5, tower_position + 0.5), dim=-1),
        mask=torch.ones(batch_size, 6, dtype=torch.bool),
    )
    group_position = torch.tensor(
        [[5.0, 8.0], [12.0, 20.0], [8.5, 16.0]]
    ).repeat(batch_size, 1, 1)
    child_position = torch.tensor(
        [[4.8, 8.0], [5.2, 8.0], [12.0, 20.0], [8.0, 16.0], [9.0, 16.0]]
    ).repeat(batch_size, 1, 1)
    groups = BattleGroupSetV4(
        source_card_vocab_id=own_ids[:, :3].clone(),
        group_type=torch.tensor([[0, 1, 2]] * batch_size),
        owner_type=torch.tensor([[0, 1, 0]] * batch_size),
        runtime_form=torch.zeros(batch_size, 3, dtype=torch.long),
        features=torch.zeros(batch_size, 3, config.group_feature_dim),
        position=group_position,
        extent=torch.cat((group_position - 1.0, group_position + 1.0), dim=-1),
        mask=torch.ones(batch_size, 3, dtype=torch.bool),
        child_archetype_id=torch.tensor([[1, 1, 2, 3, 3]] * batch_size),
        child_group_index=torch.tensor([[0, 0, 1, 2, 2]] * batch_size),
        child_type=torch.tensor([[0, 0, 0, 1, 1]] * batch_size),
        child_features=torch.zeros(batch_size, 5, config.child_feature_dim),
        child_position=child_position,
        child_extent=torch.cat((child_position, child_position), dim=-1),
        child_radius=torch.zeros(batch_size, 5),
        child_mask=torch.ones(batch_size, 5, dtype=torch.bool),
    )
    events = EventSetV4(
        event_type=torch.tensor([[1, 2]] * batch_size),
        owner_type=torch.tensor([[0, 1]] * batch_size),
        source_card_vocab_id=own_ids[:, :2].clone(),
        source_form=torch.zeros(batch_size, 2, dtype=torch.long),
        source_group_index=torch.tensor([[0, 1]] * batch_size),
        target_token_index=torch.tensor([[23, 24]] * batch_size),
        features=torch.zeros(batch_size, 2, config.event_feature_dim),
        mask=torch.ones(batch_size, 2, dtype=torch.bool),
    )
    relation_edges = RelationEdgesV4(
        source=torch.tensor([[23, 24, 0]] * batch_size),
        target=torch.tensor([[24, 23, 23]] * batch_size),
        relation_type=torch.tensor([[1, 2, 3]] * batch_size),
        mask=torch.ones(batch_size, 3, dtype=torch.bool),
    )
    previous = PreviousActionV4(
        gate=torch.zeros(batch_size, dtype=torch.long),
        micro_action_count=torch.zeros(batch_size, dtype=torch.long),
        variant=torch.zeros(batch_size, 2, dtype=torch.long),
        visible_card_vocab_id=torch.zeros(batch_size, 2, dtype=torch.long),
        effective_card_vocab_id=torch.zeros(batch_size, 2, dtype=torch.long),
        effective_form=torch.zeros(batch_size, 2, dtype=torch.long),
        ability_vocab_id=torch.zeros(batch_size, 2, dtype=torch.long),
        target_cell=torch.full((batch_size, 2), -1, dtype=torch.long),
        delay_offset_bin=torch.full((batch_size, 2), -1, dtype=torch.long),
        source_position=torch.zeros(batch_size, 2, 2),
        source_mask=torch.zeros(batch_size, 2, dtype=torch.bool),
        action_mask=torch.zeros(batch_size, 2, dtype=torch.bool),
    )
    candidate_mask = torch.zeros(batch_size, 5, dtype=torch.bool)
    candidate_mask[0, :4] = True
    placement = torch.zeros(batch_size, 5, 32, 18, dtype=torch.bool)
    placement[0, 0] = True
    placement[0, 1] = True
    placement[0, 3] = True
    candidates = ActionCandidatesV4(
        mask=candidate_mask,
        elixir=torch.tensor([10.0] + [0.0] * (batch_size - 1)),
        uid=torch.tensor([[101, 102, 103, 104, 0]] * batch_size),
        variant=torch.tensor([[0, 0, 1, 0, 0]] * batch_size),
        visible_card_vocab_id=torch.cat(
            (own_ids[:, :4], torch.zeros(batch_size, 1, dtype=torch.long)),
            dim=1,
        ),
        effective_card_vocab_id=torch.cat(
            (own_ids[:, :4], torch.zeros(batch_size, 1, dtype=torch.long)),
            dim=1,
        ),
        effective_form=torch.zeros(batch_size, 5, dtype=torch.long),
        own_card_row=torch.tensor([[0, 1, 2, 3, -1]] * batch_size),
        source_group_index=torch.tensor([[-1, -1, 0, -1, -1]] * batch_size),
        source_child_index=torch.tensor([[-1, -1, 0, -1, -1]] * batch_size),
        ability_vocab_id=torch.tensor([[0, 0, 1, 0, 0]] * batch_size),
        cost=torch.tensor([[3.0, 4.0, 1.0, 9.0, 0.0]] * batch_size),
        runtime_features=torch.zeros(
            batch_size,
            5,
            config.candidate_runtime_feature_dim,
        ),
        target_mode=torch.tensor(
            [[TARGET_GRID, TARGET_GRID, TARGET_NONE, TARGET_GRID, TARGET_NONE]]
            * batch_size
        ),
        placement=placement,
        is_building=torch.tensor(
            [[False, True, False, False, False]] * batch_size
        ),
        building_half_width=torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0]] * batch_size
        ),
        building_half_height=torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0]] * batch_size
        ),
        building_offset_x=torch.zeros(batch_size, 5),
        building_offset_y=torch.zeros(batch_size, 5),
        exclusion_group_id=torch.tensor([[10, 11, 12, 13, -1]] * batch_size),
        native_hand_slot=torch.tensor([[0, 1, -1, 3, -1]] * batch_size),
        native_source_entity=torch.tensor([[-1, -1, 501, -1, -1]] * batch_size),
        native_visible_card_id=torch.tensor(
            [[catalog.raw_card_ids[0], catalog.raw_card_ids[1], catalog.raw_card_ids[2], catalog.raw_card_ids[3], -1]]
            * batch_size
        ),
    )
    padding = ~candidates.mask
    for name in (
        "uid",
        "variant",
        "visible_card_vocab_id",
        "effective_card_vocab_id",
        "effective_form",
        "ability_vocab_id",
        "cost",
        "runtime_features",
        "target_mode",
        "placement",
        "is_building",
        "building_half_width",
        "building_half_height",
        "building_offset_x",
        "building_offset_y",
    ):
        getattr(candidates, name)[padding] = 0
    for name in (
        "own_card_row",
        "source_group_index",
        "source_child_index",
        "exclusion_group_id",
        "native_hand_slot",
        "native_source_entity",
        "native_visible_card_id",
    ):
        getattr(candidates, name)[padding] = -1
    return UniversalSemanticBatchV4(
        match_scalars=torch.zeros(batch_size, config.generic_scalar_dim),
        own_cards=own_cards,
        opponent_cards=opponent_cards,
        towers=towers,
        groups=groups,
        active_effects=ActiveEffectSetV4(
            effect_vocab_id=torch.zeros(batch_size, 0, dtype=torch.long),
            parent_type=torch.zeros(batch_size, 0, dtype=torch.long),
            parent_index=torch.full((batch_size, 0), -1, dtype=torch.long),
            source_owner_type=torch.zeros(batch_size, 0, dtype=torch.long),
            runtime_features=torch.zeros(
                batch_size,
                0,
                len(ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES),
            ),
            mask=torch.zeros(batch_size, 0, dtype=torch.bool),
        ),
        events=events,
        relation_edges=relation_edges,
        explicit_spatial_planes=torch.zeros(
            batch_size,
            config.explicit_spatial_channels,
            config.board_height,
            config.board_width,
        ),
        previous_action=previous,
        candidates=candidates,
    )


def _permuted_candidates(
    candidates: ActionCandidatesV4,
    permutation: torch.Tensor,
) -> ActionCandidatesV4:
    values = {}
    count = candidates.mask.shape[1]
    for item in fields(candidates):
        value = getattr(candidates, item.name)
        values[item.name] = (
            value.index_select(1, permutation)
            if isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[1] == count
            else value
        )
    return ActionCandidatesV4(**values)


def _with_candidate_mask(
    candidates: ActionCandidatesV4,
    mask: torch.Tensor,
) -> ActionCandidatesV4:
    result = replace(candidates, mask=mask)
    padding = ~mask
    for name in (
        "uid",
        "variant",
        "visible_card_vocab_id",
        "effective_card_vocab_id",
        "effective_form",
        "ability_vocab_id",
        "cost",
        "runtime_features",
        "target_mode",
        "placement",
        "is_building",
        "building_half_width",
        "building_half_height",
        "building_offset_x",
        "building_offset_y",
    ):
        value = getattr(result, name).clone()
        value[padding] = 0
        setattr(result, name, value)
    for name in (
        "own_card_row",
        "source_group_index",
        "source_child_index",
        "exclusion_group_id",
        "native_hand_slot",
        "native_source_entity",
        "native_visible_card_id",
    ):
        value = getattr(result, name).clone()
        value[padding] = -1
        setattr(result, name, value)
    return result


def test_v4_forward_has_one_recurrent_state_and_exact_ppo_replay() -> None:
    torch.manual_seed(17)
    batch = _batch()
    catalog = _catalog()
    model = _model(catalog).eval()
    state = model.initial_state(batch.batch_size)
    with torch.no_grad():
        gate = model.gate_head[-1]
        assert isinstance(gate, torch.nn.Linear)
        gate.bias.copy_(torch.tensor([-20.0, 20.0]))

    sampled = model.sample_for_ppo_rollout(batch, state)
    evaluated = model.evaluate_actions(batch, state, sampled.actions)

    assert {item.name for item in fields(sampled.next_state)} == {"hidden", "cell"}
    assert not hasattr(model, "wait_duration_head")
    assert not hasattr(model, "tactical_gru")
    assert not hasattr(model, "strategic_lstm")
    assert not hasattr(model, "pending_encoder")
    assert "pending_actions" not in {item.name for item in fields(batch)}
    recurrent_modules = [
        module
        for module in model.modules()
        if isinstance(
            module,
            (
                torch.nn.GRU,
                torch.nn.GRUCell,
                torch.nn.LSTM,
                torch.nn.LSTMCell,
            ),
        )
    ]
    assert recurrent_modules == [model.lstm_core]
    assert isinstance(recurrent_modules[0], torch.nn.LSTMCell)
    assert not any("gru" in name.lower() for name in model.state_dict())
    assert not any(
        "catalog_static" in name
        or "catalog_mechanics" in name
        or "ability_features" in name
        for name in model.state_dict()
    )
    assert 8_000_000 <= model.parameter_count() <= 13_000_000
    assert sampled.actions.gate.tolist() == [GATE_ACT, 0]
    assert sampled.actions.micro_action_count[0].item() in {1, 2}
    assert sampled.actions.micro_action_count[1].item() == 0
    torch.testing.assert_close(sampled.log_prob, evaluated.log_prob)
    torch.testing.assert_close(sampled.entropy, evaluated.entropy)
    torch.testing.assert_close(sampled.value, evaluated.value)


def test_production_perception_encoders_cover_expanded_semantic_vocabularies() -> None:
    model = build_production_model_v4().eval()
    entity_vocab_size = model.entity_archetype_catalog.vocab_size
    effect_vocab_size = model.effect_catalog.vocab_size

    assert entity_vocab_size == model.config.child_archetype_count == 429
    assert model.child_encoder.archetype_embedding.num_embeddings == 429
    assert model.child_encoder.archetype_profile_ids.shape == (429,)
    assert effect_vocab_size == 172
    assert model.effect_encoder.id_embedding.num_embeddings == 172
    assert model.effect_encoder.catalog_features.shape[0] == 172

    batch = _batch(batch_size=1)
    batch.groups.child_archetype_id[0, 0] = entity_vocab_size - 1
    batch = replace(
        batch,
        active_effects=ActiveEffectSetV4(
            effect_vocab_id=torch.tensor([[effect_vocab_size - 1]]),
            parent_type=torch.tensor([[0]]),
            parent_index=torch.tensor([[0]]),
            source_owner_type=torch.tensor([[1]]),
            runtime_features=torch.zeros(
                1,
                1,
                len(ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES),
            ),
            mask=torch.tensor([[True]]),
        ),
    )

    with torch.no_grad():
        encoded = model.encode_observation(batch)

    assert torch.isfinite(encoded.child_memory).all()
    assert torch.count_nonzero(encoded.child_memory[0, 0]) > 0


def test_v4_coordinate_contract_and_fixed_wait() -> None:
    x = torch.tensor([0, 17, 5])
    y = torch.tensor([0, 31, 9])
    cells = target_cell_from_xy(x, y)
    assert cells.tolist() == [0, 575, 167]
    actual_x, actual_y = target_xy_from_cell(cells)
    torch.testing.assert_close(actual_x, x)
    torch.testing.assert_close(actual_y, y)

    model = _model().eval()
    batch = _batch()
    with torch.no_grad():
        gate = model.gate_head[-1]
        assert isinstance(gate, torch.nn.Linear)
        gate.bias.copy_(torch.tensor([20.0, -20.0]))
    output = model.act(batch, model.initial_state(batch.batch_size))
    assert output.actions.micro_action_count.tolist() == [0, 0]
    assert wait_ticks(output.actions, model.config).tolist() == [5, 5]
    assert not hasattr(output.actions, "wait_duration_bin")


def test_v4_frontend_projection_widths_are_frozen() -> None:
    model = _model()

    def linear_shapes(module: torch.nn.Module) -> list[tuple[int, int]]:
        return [
            (layer.in_features, layer.out_features)
            for layer in module.modules()
            if isinstance(layer, torch.nn.Linear)
        ]

    assert linear_shapes(model.card_token_encoder.runtime) == [(16, 128), (128, 256)]
    assert linear_shapes(model.tower_encoder.feature) == [(21, 128), (128, 256)]
    assert isinstance(model.tower_encoder.position, torch.nn.Linear)
    assert linear_shapes(model.tower_encoder.position) == [(6, 256)]
    assert linear_shapes(model.scalar_encoder.network) == [(18, 128), (128, 128)]
    assert isinstance(model.candidate_encoder.ability_static, torch.nn.Linear)
    assert linear_shapes(model.candidate_encoder.ability_static) == [(10, 128)]
    assert isinstance(model.candidate_encoder.runtime, torch.nn.Linear)
    assert linear_shapes(model.candidate_encoder.runtime) == [(2, 128)]


def test_masked_rows_reject_illegal_ids_before_embedding_lookup() -> None:
    batch = _batch(batch_size=1)
    card_ids = batch.opponent_cards.card_vocab_id.clone()
    forms = batch.opponent_cards.runtime_form.clone()
    padding_row = int(
        (~batch.opponent_cards.mask[0]).nonzero(as_tuple=False)[0]
    )
    card_ids[0, padding_row] = 99_999
    forms[0, padding_row] = 99_999
    dirty = replace(
        batch,
        opponent_cards=replace(
            batch.opponent_cards,
            card_vocab_id=card_ids,
            runtime_form=forms,
        ),
    )

    with pytest.raises(ValueError, match="out-of-range vocabulary ID"):
        _model().encode_observation(dirty, validate=True)


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_nonfinite_numeric_inputs_fail_before_encoding(value: float) -> None:
    batch = _batch(batch_size=1)
    scalars = batch.match_scalars.clone()
    scalars[0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        _model().encode_observation(
            replace(batch, match_scalars=scalars),
            validate=True,
        )


def test_previous_action_encoder_consumes_effective_form() -> None:
    torch.manual_seed(23)
    batch = _batch(batch_size=1)
    model = _model().eval()
    previous = replace(
        batch.previous_action,
        gate=torch.tensor([GATE_ACT]),
        micro_action_count=torch.tensor([1]),
        visible_card_vocab_id=torch.tensor(
            [[int(batch.own_cards.card_vocab_id[0, 0]), 0]]
        ),
        effective_card_vocab_id=torch.tensor(
            [[int(batch.own_cards.card_vocab_id[0, 0]), 0]]
        ),
        effective_form=torch.tensor([[1, 0]]),
        target_cell=torch.tensor([[0, -1]]),
        delay_offset_bin=torch.tensor([[0, -1]]),
        action_mask=torch.tensor([[True, False]]),
    )
    evolved = replace(batch, previous_action=previous)
    normal = replace(
        evolved,
        previous_action=replace(
            previous,
            effective_form=torch.zeros_like(previous.effective_form),
        ),
    )

    with torch.no_grad():
        evolved_encoded = model.encode_observation(evolved)
        normal_encoded = model.encode_observation(normal)
    assert not torch.allclose(
        evolved_encoded.previous_action_summary,
        normal_encoded.previous_action_summary,
    )


def test_previous_action_encoder_preserves_micro_action_order() -> None:
    torch.manual_seed(29)
    model = _model().eval()
    batch = _batch(batch_size=1)
    own = batch.own_cards.card_vocab_id[0, :2]
    previous = replace(
        batch.previous_action,
        gate=torch.tensor([GATE_ACT]),
        micro_action_count=torch.tensor([2]),
        visible_card_vocab_id=own[None].clone(),
        effective_card_vocab_id=own[None].clone(),
        target_cell=torch.tensor([[10, 20]]),
        delay_offset_bin=torch.tensor([[0, 1]]),
        action_mask=torch.tensor([[True, True]]),
    )
    swapped = replace(
        previous,
        visible_card_vocab_id=previous.visible_card_vocab_id.flip(1),
        effective_card_vocab_id=previous.effective_card_vocab_id.flip(1),
        target_cell=previous.target_cell.flip(1),
    )

    with torch.no_grad():
        effect_memory = model.effect_encoder.all_effects()
        profile_memory = model.mechanic_profile_encoder(effect_memory)
        ordered = model.previous_action_encoder(
            previous,
            profile_memory=profile_memory,
        )
        reversed_order = model.previous_action_encoder(
            swapped,
            profile_memory=profile_memory,
        )
    assert not torch.allclose(ordered, reversed_order)


def test_gate_uses_initial_shadow_legality_not_raw_candidate_presence() -> None:
    model = _model().eval()
    batch = _batch(batch_size=1)
    mask = torch.zeros_like(batch.candidates.mask)
    mask[0, 0] = True
    candidates = replace(
        _with_candidate_mask(batch.candidates, mask),
        elixir=torch.tensor([0.0]),
    )
    batch = replace(batch, candidates=candidates)
    with torch.no_grad():
        gate = model.gate_head[-1]
        assert isinstance(gate, torch.nn.Linear)
        gate.bias.copy_(torch.tensor([-40.0, 40.0]))

    output = model.act(batch, model.initial_state(1))

    assert output.actions.gate.tolist() == [0]
    assert output.actions.micro_action_count.tolist() == [0]


def test_live_inference_masks_delay_offsets_above_fifty_ms() -> None:
    model = _model().eval()
    batch = _batch(batch_size=1)
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        model.continue_head[-1].bias.copy_(torch.tensor([20.0, -20.0]))
        delay_head = model.delay_offset_head[-1]
        assert isinstance(delay_head, torch.nn.Linear)
        delay_head.weight.zero_()
        delay_head.bias.copy_(torch.arange(5, dtype=delay_head.bias.dtype))
    model.set_inference_delay_offset_max_ms(50)

    output = model.act(batch, model.initial_state(1))

    assert output.actions.gate.tolist() == [GATE_ACT]
    assert output.actions.delay_offset_bin.tolist() == [[1, -1]]
    assert output.action_components is not None
    assert output.action_components.delay_offset_legal_mask[0, 0].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_group_type_and_owner_are_direct_group_encoder_inputs() -> None:
    torch.manual_seed(31)
    model = _model().eval()
    batch = _batch(batch_size=1)
    with torch.no_grad():
        effect_memory = model.effect_encoder.all_effects()
        profile_memory = model.mechanic_profile_encoder(effect_memory)
        child_effects, _ = model.active_effect_encoder(
            batch.active_effects,
            effect_memory=effect_memory,
            child_count=batch.groups.child_mask.shape[1],
            tower_count=batch.towers.mask.shape[1],
        )
        _, local_slots = model.child_encoder(
            batch.groups,
            profile_memory=profile_memory,
            active_effect_memory=child_effects,
        )
        original = model.group_encoder(
            batch.groups,
            local_slots,
            profile_memory=profile_memory,
        )
        changed_groups = replace(
            batch.groups,
            group_type=(batch.groups.group_type + 1).remainder(
                model.config.group_type_count
            ),
            owner_type=(batch.groups.owner_type + 1).remainder(
                model.config.owner_type_count
            ),
        )
        changed = model.group_encoder(
            changed_groups,
            local_slots,
            profile_memory=profile_memory,
        )

    assert not torch.allclose(original, changed)


def test_ppo_log_prob_parity_covers_wait_ability_one_and_two_actions() -> None:
    torch.manual_seed(18)
    model = _model().eval()
    state = model.initial_state(1)
    base = _batch(batch_size=1)
    gate = model.gate_head[-1]
    continuation = model.continue_head[-1]
    assert isinstance(gate, torch.nn.Linear)
    assert isinstance(continuation, torch.nn.Linear)

    def sample_and_replay(batch: UniversalSemanticBatchV4) -> tuple[int, int]:
        sampled = model.sample_for_ppo_rollout(batch, state)
        replayed = model.evaluate_actions(batch, state, sampled.actions)
        torch.testing.assert_close(sampled.log_prob, replayed.log_prob)
        torch.testing.assert_close(sampled.entropy, replayed.entropy)
        return (
            int(sampled.actions.gate.item()),
            int(sampled.actions.micro_action_count.item()),
        )

    with torch.no_grad():
        gate.bias.copy_(torch.tensor([40.0, -40.0]))
    assert sample_and_replay(base) == (0, 0)

    ability_mask = torch.zeros_like(base.candidates.mask)
    ability_mask[:, 2] = True
    ability_batch = replace(
        base,
        candidates=_with_candidate_mask(base.candidates, ability_mask),
    )
    with torch.no_grad():
        gate.bias.copy_(torch.tensor([-40.0, 40.0]))
        continuation.bias.copy_(torch.tensor([40.0, -40.0]))
    assert sample_and_replay(ability_batch) == (GATE_ACT, 1)

    two_mask = torch.zeros_like(base.candidates.mask)
    two_mask[:, :2] = True
    two_batch = replace(
        base,
        candidates=_with_candidate_mask(base.candidates, two_mask),
    )
    with torch.no_grad():
        continuation.bias.copy_(torch.tensor([-40.0, 40.0]))
    assert sample_and_replay(two_batch) == (GATE_ACT, 2)


def test_argmax_after_preselected_act_overrides_only_wait_gate() -> None:
    torch.manual_seed(19)
    model = _model().eval()
    batch = _batch(batch_size=1)
    state = model.initial_state(1)
    gate = model.gate_head[-1]
    assert isinstance(gate, torch.nn.Linear)
    with torch.no_grad():
        gate.bias.copy_(torch.tensor([40.0, -40.0]))

    regular = model.act(batch, state)
    context = model.forward(batch, state)
    forced = model.act_after_preselected_act(batch, context)

    assert regular.actions.gate.item() == 0
    assert forced.actions.gate.item() == GATE_ACT
    assert forced.actions.micro_action_count.item() in {1, 2}


def test_candidate_permutation_is_equivariant_and_uid_replays() -> None:
    torch.manual_seed(19)
    model = _model().eval()
    batch = _batch(batch_size=1)
    state = model.initial_state(1)
    with torch.no_grad():
        gate = model.gate_head[-1]
        continuation = model.continue_head[-1]
        assert isinstance(gate, torch.nn.Linear)
        assert isinstance(continuation, torch.nn.Linear)
        gate.bias.copy_(torch.tensor([-20.0, 20.0]))
        continuation.bias.copy_(torch.tensor([20.0, -20.0]))

    original_context = model(batch, state)
    original_decoder = model._decoder_context(
        model.decoder_initial(original_context.policy_context),
        original_context.encoded,
    )
    original_logits = model.candidate_logits(
        original_decoder,
        original_context.encoded.candidate_memory,
    )
    action = model.act(batch, state)

    permutation = torch.tensor([2, 0, 3, 1, 4])
    permuted_batch = replace(
        batch,
        candidates=_permuted_candidates(batch.candidates, permutation),
    )
    permuted_context = model(permuted_batch, state)
    permuted_decoder = model._decoder_context(
        model.decoder_initial(permuted_context.policy_context),
        permuted_context.encoded,
    )
    permuted_logits = model.candidate_logits(
        permuted_decoder,
        permuted_context.encoded.candidate_memory,
    )
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        original_context.policy_context,
        permuted_context.policy_context,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        original_logits,
        permuted_logits.index_select(1, inverse),
        atol=2e-5,
        rtol=2e-5,
    )
    replayed = model.evaluate_actions(permuted_batch, state, action.actions)
    torch.testing.assert_close(action.log_prob, replayed.log_prob, atol=2e-5, rtol=2e-5)


def test_own_card_rows_are_addresses_not_embeddings() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    batch = _batch(batch_size=1)
    state = model.initial_state(1)
    original = model(batch, state)

    permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    inverse = torch.argsort(permutation)
    own_values = {
        item.name: getattr(batch.own_cards, item.name).index_select(1, permutation)
        for item in fields(batch.own_cards)
    }
    old_rows = batch.candidates.own_card_row
    new_rows = torch.where(
        old_rows >= 0,
        inverse[old_rows.clamp_min(0)],
        old_rows,
    )
    permuted_candidates = replace(batch.candidates, own_card_row=new_rows)
    permuted = replace(
        batch,
        own_cards=CardSetV4(**own_values),
        candidates=permuted_candidates,
    )
    actual = model(permuted, state)

    torch.testing.assert_close(original.encoded.scene_state, actual.encoded.scene_state)
    torch.testing.assert_close(original.policy_context, actual.policy_context)
    torch.testing.assert_close(original.value, actual.value)
    torch.testing.assert_close(
        original.encoded.candidate_memory,
        actual.encoded.candidate_memory,
    )


def test_shadow_legality_consumes_hand_exclusion_and_orders_absolute_offsets() -> None:
    batch = _batch(batch_size=1)
    shadow = ShadowCandidateLegality(batch.candidates, ModelConfigV4())
    initial = shadow.candidate_mask()
    assert initial[0, :4].tolist() == [True, True, True, True]
    shadow.apply(
        torch.tensor([True]),
        torch.tensor([1]),
        torch.tensor([10 * 18 + 9]),
        torch.tensor([2]),
        step=0,
    )
    after = shadow.candidate_mask()
    assert not after[0, 1]
    assert shadow.remaining_elixir.item() == pytest.approx(6.0)
    assert shadow.previous_delay_offset_ms.item() == pytest.approx(100.0)
    assert shadow.delay_offset_mask(step=1).tolist() == [
        [False, False, True, True, True]
    ]

    model = _model().eval()
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        model.continue_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
    output = model.act(batch, model.initial_state(1))
    offsets = delay_offset_ms(output.actions, model.config)
    if output.actions.micro_action_count.item() == 2:
        assert offsets[0, 1] >= offsets[0, 0]


def test_action_sequence_rejects_decreasing_absolute_offsets() -> None:
    batch = _batch(batch_size=1)
    actions = ActionSequenceV4(
        gate=torch.tensor([GATE_ACT]),
        micro_action_count=torch.tensor([2]),
        candidate_index=torch.tensor([[0, 1]]),
        candidate_uid=batch.candidates.uid[:, :2].clone(),
        target_cell=torch.tensor([[0, 1]]),
        delay_offset_bin=torch.tensor([[4, 0]]),
    )

    with pytest.raises(ValueError, match="nondecreasing"):
        actions.validate(
            ModelConfigV4(),
            candidate_count=batch.candidates.mask.shape[1],
        )


def test_learned_spatial_scatter_preserves_multiplicity_by_sum() -> None:
    torch.manual_seed(29)
    config = ModelConfigV4()
    encoder = SpatialScatterEncoder(config)
    child = torch.randn(1, 2, config.child_dim)
    child[:, 1] = child[:, 0]
    position = torch.tensor([[[4.25, 8.5], [4.25, 8.5]]])
    extent = torch.cat((position, position), dim=-1)
    radius = torch.zeros(1, 2)
    one = encoder.scatter(
        child,
        position,
        extent,
        radius,
        torch.tensor([[True, False]]),
    )
    two = encoder.scatter(
        child,
        position,
        extent,
        radius,
        torch.tensor([[True, True]]),
    )
    torch.testing.assert_close(two, 2.0 * one)


def test_learned_spatial_scatter_rasterizes_rectangle_and_circle_coverage() -> None:
    config = ModelConfigV4()
    encoder = SpatialScatterEncoder(config)
    with torch.no_grad():
        encoder.child_value.weight.zero_()
        encoder.child_value.bias.fill_(1.0)
    child = torch.zeros(1, 2, config.child_dim)
    position = torch.tensor([[[10.5, 12.5], [9.0, 16.0]]])
    extent = torch.tensor(
        [[[9.0, 11.0, 12.0, 14.0], [5.5, 12.5, 12.5, 19.5]]]
    )
    radius = torch.tensor([[0.0, 3.5]])

    building = encoder.scatter(
        child,
        position,
        extent,
        radius,
        torch.tensor([[True, False]]),
    )[0, 0]
    expected_building = torch.zeros(config.board_height, config.board_width)
    expected_building[11:14, 9:12] = 1.0
    torch.testing.assert_close(building, expected_building)

    area = encoder.scatter(
        child,
        position,
        extent,
        radius,
        torch.tensor([[False, True]]),
    )[0, 0]
    cell_y, cell_x = torch.meshgrid(
        torch.arange(config.board_height, dtype=torch.float32) + 0.5,
        torch.arange(config.board_width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    expected_area = (
        (cell_x - 9.0).square() + (cell_y - 16.0).square() <= 3.5**2
    ).to(area.dtype)
    torch.testing.assert_close(area, expected_area)


def test_catalog_order_is_global_and_independent_of_input_order() -> None:
    first = CardCatalogV1.from_raw_ids((90, 10, 50))
    second = CardCatalogV1.from_raw_ids((50, 90, 10))
    assert first.raw_card_ids == second.raw_card_ids == (10, 50, 90)
    assert first.vocab_id(10) == 2
    assert first.vocab_id(50) == 3
    assert first.vocab_id(999) == 1
    model = _model(first)
    gate = model.card_encoder.id_gate[-1]
    assert isinstance(gate, torch.nn.Linear)
    torch.testing.assert_close(gate.bias, torch.full_like(gate.bias, -2.0))
    child_gate = model.child_encoder.id_gate
    assert isinstance(child_gate, torch.nn.Linear)
    torch.testing.assert_close(
        child_gate.weight,
        torch.zeros_like(child_gate.weight),
    )
    torch.testing.assert_close(
        child_gate.bias,
        torch.full_like(child_gate.bias, -2.0),
    )
    abilities = AbilityCatalogV1(
        first.raw_card_ids,
        ("ability-a", "ability-z"),
        (
            (0.0,) * ModelConfigV4().ability_feature_dim,
            (0.0,) * ModelConfigV4().ability_feature_dim,
        ),
    )
    assert abilities.vocab_id("ability-a") == 1
    assert abilities.ability_id(2) == "ability-z"


def test_child_archetype_identity_is_a_gated_residual() -> None:
    model = _model().eval()
    batch = _batch(batch_size=1)

    with torch.no_grad():
        effect_memory = model.effect_encoder.all_effects()
        profile_memory = model.mechanic_profile_encoder(effect_memory)
        child_effects, _ = model.active_effect_encoder(
            batch.active_effects,
            effect_memory=effect_memory,
            child_count=batch.groups.child_mask.shape[1],
            tower_count=batch.towers.mask.shape[1],
        )
        model.child_encoder.id_gate.weight.zero_()
        model.child_encoder.id_gate.bias.fill_(-100.0)
        before, _ = model.child_encoder(
            batch.groups,
            profile_memory=profile_memory,
            active_effect_memory=child_effects,
        )
        model.child_encoder.archetype_embedding.weight.normal_(
            mean=1000.0,
            std=100.0,
        )
        after, _ = model.child_encoder(
            batch.groups,
            profile_memory=profile_memory,
            active_effect_memory=child_effects,
        )

    torch.testing.assert_close(before, after)
