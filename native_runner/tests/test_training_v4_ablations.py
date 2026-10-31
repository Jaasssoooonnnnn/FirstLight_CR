"""Model-side precision/geometry fixes and architecture ablation switches."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from native_runner.tests.test_training_v4_model import _ability_catalog, _batch, _catalog
from native_runner.training.v4.catalog import CardCatalogV1, EntityArchetypeCatalogV1
from native_runner.training.v4.components import SpatialScatterEncoder
from native_runner.training.v4.config import ABLATION_FIELD_NAMES, ModelConfigV4
from native_runner.training.v4.factory import build_production_model_v4
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.mechanics import EffectSemanticCatalogV1, MechanicProfileCatalogV1
from native_runner.training.v4.model import UniversalCardPolicyV4
from native_runner.training.v4.tensors import (
    GATE_ACT,
    GATE_WAIT,
    ActionSequenceV4,
    UniversalSemanticBatchV4,
    concatenate_padded_tensor_records,
    concatenate_tensor_records,
)


# Parameter count of the small fixture model before any ablation switch
# existed.  Default switches must not add or remove a single parameter.
SMALL_MODEL_BASELINE_PARAMETERS = 11_666_221

ABLATIONS: dict[str, dict[str, object]] = {
    "spatial_memory": {"ablate_spatial_memory": True},
    "spatial_summary": {"ablate_spatial_summary": True},
    "event_summary": {"ablate_event_summary": True},
    "previous_action_summary": {"ablate_previous_action_summary": True},
    "relation_bias": {"ablate_relation_bias": True},
    "spatial_pair_bias": {"ablate_spatial_pair_bias": True},
    "mechanic_profiles": {"ablate_mechanic_profiles": True},
    "card_id_gate_zero": {"card_id_gate": "zero"},
    "card_id_gate_one": {"card_id_gate": "one"},
    "lstm": {"ablate_lstm": True},
    "decoder_cross_attention": {"ablate_decoder_cross_attention": True},
    "no_transformer": {"transformer_layers": 0},
    "everything": {
        "ablate_spatial_memory": True,
        "ablate_spatial_summary": True,
        "ablate_event_summary": True,
        "ablate_previous_action_summary": True,
        "ablate_relation_bias": True,
        "ablate_spatial_pair_bias": True,
        "ablate_mechanic_profiles": True,
        "card_id_gate": "zero",
        "ablate_lstm": True,
        "ablate_decoder_cross_attention": True,
    },
}


def _build(config: ModelConfigV4 | None = None, *, seed: int = 7) -> UniversalCardPolicyV4:
    torch.manual_seed(seed)
    catalog: CardCatalogV1 = _catalog()
    abilities = _ability_catalog(catalog)
    archetypes = EntityArchetypeCatalogV1(catalog.raw_card_ids, ("form:fixture_a", "form:fixture_b"))
    return UniversalCardPolicyV4(
        catalog,
        ability_catalog=abilities,
        entity_archetype_catalog=archetypes,
        effect_catalog=EffectSemanticCatalogV1.empty(catalog.raw_card_ids),
        mechanic_profile_catalog=MechanicProfileCatalogV1.empty(
            card_catalog=catalog, ability_catalog=abilities, entity_archetype_catalog=archetypes
        ),
        config=config,
    )


def _force_two_actions(model: UniversalCardPolicyV4) -> UniversalCardPolicyV4:
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
        model.continue_head[-1].bias.copy_(torch.tensor([-20.0, 20.0]))
    return model


def _acting_model(config: ModelConfigV4 | None = None, *, seed: int = 7) -> UniversalCardPolicyV4:
    return _force_two_actions(_build(config, seed=seed)).eval()


def _outputs(
    model: UniversalCardPolicyV4, batch: UniversalSemanticBatchV4, actions: ActionSequenceV4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output = model.evaluate_actions(batch, model.initial_state(batch.batch_size), actions)
    return output.log_prob, output.entropy, output.value


def _assert_same_outputs(
    first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...], *, atol: float = 1e-5
) -> None:
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-5)


def _none_gradient_names(model: UniversalCardPolicyV4) -> set[str]:
    batch = _batch(batch_size=2)
    state = model.initial_state(2)
    with torch.no_grad():
        actions = model.act(batch, state).actions
    model.zero_grad(set_to_none=True)
    evaluated = model.evaluate_actions(batch, state, actions)
    (evaluated.log_prob.sum() + evaluated.value.sum()).backward()
    missing = set()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.add(name)
        else:
            assert torch.isfinite(parameter.grad).all(), name
    return missing


# --------------------------------------------------------------------- M2/F1


def test_bf16_autocast_log_probs_match_fp32_within_tolerance() -> None:
    for seed in (5, 11, 23):
        model = _acting_model(seed=seed)
        batch = _batch(batch_size=2)
        state = model.initial_state(2)
        with torch.no_grad():
            actions = model.act(batch, state).actions
            reference = model.evaluate_actions(batch, state, actions)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                low = model.evaluate_actions(batch, state, actions)
        assert actions.gate.tolist() == [GATE_ACT, GATE_WAIT]
        assert low.log_prob.dtype == torch.float32
        components = low.action_components
        assert components is not None
        for name in ("gate_log_prob", "candidate_log_prob", "target_log_prob", "continue_log_prob"):
            assert getattr(components, name).dtype == torch.float32, name
        assert reference.action_components is not None
        # The 576-way location term was ~1.3e-2 off before fp32 normalization.
        torch.testing.assert_close(
            components.target_log_prob, reference.action_components.target_log_prob, atol=2e-3, rtol=0.0
        )
        # The remaining total drift is bf16 rounding of encoder activations
        # feeding the heads (<= ~6e-3 on the candidate term), not normalization.
        torch.testing.assert_close(low.log_prob, reference.log_prob, atol=1e-2, rtol=0.0)


def test_location_head_fp32_toggle_is_numerically_inert() -> None:
    model = _acting_model()
    batch = _batch(batch_size=2)
    state = model.initial_state(2)
    with torch.no_grad():
        actions = model.act(batch, state).actions
        with torch.autocast("cpu", dtype=torch.bfloat16):
            default = model.evaluate_actions(batch, state, actions).log_prob
            model.set_ppo_location_head_fp32(True)
            toggled = model.evaluate_actions(batch, state, actions).log_prob
    torch.testing.assert_close(default, toggled, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- S1/F2


def _unit_scatter_encoder() -> SpatialScatterEncoder:
    encoder = SpatialScatterEncoder(ModelConfigV4())
    with torch.no_grad():
        encoder.child_value.weight.zero_()
        encoder.child_value.bias.fill_(1.0)
    return encoder


def _scatter_one(
    encoder: SpatialScatterEncoder, position: tuple[float, float], *, radius: float = 0.0, extent: tuple[float, ...] | None = None
) -> torch.Tensor:
    config = encoder.config
    child = torch.zeros(1, 1, config.child_dim)
    child_position = torch.tensor([[position]])
    child_extent = torch.tensor([[extent]]) if extent is not None else torch.cat((child_position, child_position), dim=-1)
    with torch.no_grad():
        return encoder.scatter(child, child_position, child_extent, torch.tensor([[radius]]), torch.ones(1, 1, dtype=torch.bool))[0, 0]


def test_point_scatter_at_cell_centre_lands_in_that_cell() -> None:
    plane = _scatter_one(_unit_scatter_encoder(), (4.5, 8.5))
    expected = torch.zeros_like(plane)
    expected[8, 4] = 1.0
    torch.testing.assert_close(plane, expected)


def test_point_scatter_near_border_keeps_all_weight_on_board() -> None:
    plane = _scatter_one(_unit_scatter_encoder(), (0.3, 0.3))
    assert plane.sum().item() == pytest.approx(1.0)
    assert plane[0, 0].item() == pytest.approx(1.0)
    between = _scatter_one(_unit_scatter_encoder(), (5.0, 8.5))
    assert between[8, 4].item() == pytest.approx(0.5)
    assert between[8, 5].item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("radius", "extent"),
    ((0.5, None), (0.0, (3.6, 7.6, 3.9, 7.9))),
    ids=("small-circle", "sub-cell-rectangle"),
)
def test_shaped_entity_without_covered_cells_falls_back_to_point(radius: float, extent: tuple[float, ...] | None) -> None:
    encoder = _unit_scatter_encoder()
    position = (4.0, 8.0) if extent is None else (3.75, 7.75)
    plane = _scatter_one(encoder, position, radius=radius, extent=extent)
    point = _scatter_one(encoder, position)
    assert plane.sum().item() == pytest.approx(1.0)
    torch.testing.assert_close(plane, point)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph required")
def test_cuda_graph_scatter_matches_eager_including_fallback() -> None:
    config = ModelConfigV4()
    torch.manual_seed(3)
    encoder = SpatialScatterEncoder(config).cuda().eval()
    child = torch.randn(1, 4, config.child_dim, device="cuda")
    position = torch.tensor([[[4.5, 8.5], [4.0, 8.0], [9.0, 16.0], [0.3, 31.9]]], device="cuda")
    extent = torch.cat((position, position), dim=-1)
    radius = torch.tensor([[0.0, 0.5, 3.5, 0.0]], device="cuda")
    mask = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        eager = encoder.scatter(child, position, extent, radius, mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            encoder.scatter(child, position, extent, radius, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = encoder.scatter(child, position, extent, radius, mask)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- S2/F4


def test_tower_geometry_and_categorical_embeddings_start_small() -> None:
    model = _build()
    batch = _batch(batch_size=2)
    towers = batch.towers
    geometry = torch.cat((towers.position, towers.extent), dim=-1) / model.tower_encoder.geometry_scale
    assert float(geometry.abs().max()) <= 1.0
    with torch.no_grad():
        position_term = model.tower_encoder.position(geometry)
    assert position_term.square().mean().sqrt().item() < 0.3
    form = model.card_encoder.form_embedding.weight
    assert form.square().mean().sqrt().item() < 0.3
    for embedding in (
        model.child_encoder.archetype_embedding,
        model.child_encoder.child_type_embedding,
        model.child_encoder.owner_embedding,
        model.card_token_encoder.owner,
        model.tower_encoder.tower_type,
        model.tower_encoder.owner_type,
        model.event_encoder.event_type,
        model.event_encoder.owner_type,
        model.group_encoder.group_type,
        model.group_encoder.owner_type,
    ):
        assert embedding.weight.square().mean().sqrt().item() < 0.1
    troop = model.tower_encoder.tower_troop_type
    assert troop.padding_idx == 0
    torch.testing.assert_close(troop.weight[0], torch.zeros_like(troop.weight[0]))
    assert troop.weight[1:].square().mean().sqrt().item() < 0.1


# --------------------------------------------------------------------- S3/F3/F5


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
    ),
)
def test_recurrent_core_runs_fp32_under_bf16_autocast(device: str) -> None:
    model = _build().to(device).eval()
    batch = _batch(batch_size=2).to_model_input(torch.device(device))
    with torch.no_grad():
        encoded = model.encode_observation(batch)
        reference = model.initial_state(2, device=device)
        low = model.initial_state(2, device=device)
        for _step in range(12):
            reference = model.advance_core(encoded, reference).next_state
            with torch.autocast(device, dtype=torch.bfloat16):
                low = model.advance_core(encoded, low).next_state
            assert low.hidden.dtype == torch.float32
            assert low.cell.dtype == torch.float32
    torch.testing.assert_close(low.hidden, reference.hidden, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(low.cell, reference.cell, atol=1e-3, rtol=0.0)


def _wait_actions(batch_size: int) -> ActionSequenceV4:
    return ActionSequenceV4(
        gate=torch.full((batch_size,), GATE_WAIT, dtype=torch.long),
        micro_action_count=torch.zeros(batch_size, dtype=torch.long),
        candidate_index=torch.full((batch_size, 2), -1, dtype=torch.long),
        candidate_uid=torch.full((batch_size, 2), -1, dtype=torch.long),
        target_cell=torch.full((batch_size, 2), -1, dtype=torch.long),
        delay_offset_bin=torch.full((batch_size, 2), -1, dtype=torch.long),
    )


def test_fused_sequence_rejects_mid_lane_reset_without_validation() -> None:
    model = _build().eval()
    batch = _batch(batch_size=2)
    flat_batch = concatenate_padded_tensor_records((batch, batch))
    flat_actions = concatenate_tensor_records((_wait_actions(2), _wait_actions(2)))
    starts = torch.tensor([[True, True], [False, True]])
    with torch.no_grad():
        encoded = model.encode_observation(flat_batch, validate=False)
        with pytest.raises(ValueError, match="mid-lane resets"):
            model.evaluate_encoded_action_sequence(
                flat_batch, encoded, model.initial_state(2), flat_actions, starts, time_steps=2, validate=False
            )


def test_fused_sequence_state_is_fp32_under_autocast() -> None:
    model = _build().eval()
    batch = _batch(batch_size=2)
    flat_batch = concatenate_padded_tensor_records((batch, batch))
    flat_actions = concatenate_tensor_records((_wait_actions(2), _wait_actions(2)))
    starts = torch.tensor([[True, True], [False, False]])
    with torch.no_grad():
        encoded = model.encode_observation(flat_batch)
        *_, reference = model.evaluate_encoded_action_sequence(
            flat_batch, encoded, model.initial_state(2), flat_actions, starts, time_steps=2
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            *_, low = model.evaluate_encoded_action_sequence(
                flat_batch, encoded, model.initial_state(2), flat_actions, starts, time_steps=2
            )
    assert low.hidden.dtype == low.cell.dtype == torch.float32
    torch.testing.assert_close(low.hidden, reference.hidden, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(low.cell, reference.cell, atol=1e-3, rtol=0.0)


# --------------------------------------------------------------------- M4


@pytest.mark.parametrize(
    "overrides",
    ({"token_dim": 192, "group_dim": 192}, {"lstm_hidden_dim": 256}, {"lstm_input_dim": 384, "decoder_hidden_dim": 256}),
    ids=("token-192", "lstm-256", "core-384-decoder-256"),
)
def test_core_width_is_derived_from_config(overrides: dict[str, int]) -> None:
    model = _acting_model(ModelConfigV4(**overrides))
    batch = _batch(batch_size=2)
    output = model.act(batch, model.initial_state(2))
    assert output.actions.gate.tolist() == [GATE_ACT, GATE_WAIT]
    assert tuple(output.next_state.hidden.shape) == (2, model.config.lstm_hidden_dim)
    assert torch.isfinite(output.log_prob).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"spatial_hidden_channels": 60}, "spatial_hidden_channels"),
        ({"decoder_hidden_dim": 516}, "decoder_hidden_dim"),
        ({"token_dim": 260, "group_dim": 260}, "token_dim"),
        ({"transformer_layers": -1}, "transformer_layers"),
        ({"lstm_hidden_dim": 0}, "positive"),
        ({"card_id_gate": "half"}, "card_id_gate"),
        ({"ablate_lstm": 1}, "bool"),
    ),
)
def test_config_rejects_inconsistent_widths_and_switches(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfigV4(**overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------- M3 / 7


def test_default_config_keeps_parameter_count_and_reports_no_ablation() -> None:
    model = _build()
    assert model.config.active_ablations() == {}
    assert model.parameter_count() == SMALL_MODEL_BASELINE_PARAMETERS
    assert model.ablation_summary() == {"ablations": {}, "parameter_count": SMALL_MODEL_BASELINE_PARAMETERS}
    assert model.lstm_core is not None and model.core_mlp is None
    assert model.constant_spatial_memory is None
    assert model.relation_transformer.pool_projection is None
    model.register_buffer("large_constant_table", torch.zeros(1000, 1000), persistent=False)
    assert model.parameter_count() == SMALL_MODEL_BASELINE_PARAMETERS


def test_ablation_summary_lists_active_switches() -> None:
    config = ModelConfigV4(**ABLATIONS["everything"], transformer_layers=0)  # type: ignore[arg-type]
    model = _build(config)
    summary = model.ablation_summary()
    assert summary["parameter_count"] == model.parameter_count()
    assert summary["ablations"] == {**ABLATIONS["everything"], "transformer_layers": 0}
    assert set(ABLATION_FIELD_NAMES) == set(ABLATIONS["everything"])


@pytest.fixture(scope="module")
def default_none_gradients() -> set[str]:
    return _none_gradient_names(_acting_model())


@pytest.mark.parametrize("name", tuple(ABLATIONS))
def test_ablation_act_matches_evaluate_and_keeps_every_parameter_in_graph(
    name: str, default_none_gradients: set[str]
) -> None:
    model = _acting_model(ModelConfigV4(**ABLATIONS[name]))  # type: ignore[arg-type]
    batch = _batch(batch_size=2)
    state = model.initial_state(2)
    with torch.no_grad():
        acted = model.act(batch, state)
        evaluated = model.evaluate_actions(batch, state, acted.actions)
    assert acted.actions.gate.tolist() == [GATE_ACT, GATE_WAIT]
    torch.testing.assert_close(acted.log_prob, evaluated.log_prob)
    torch.testing.assert_close(acted.value, evaluated.value)
    assert torch.isfinite(acted.log_prob).all()

    missing = _none_gradient_names(model)
    # DDP (find_unused_parameters=False) needs exactly the baseline's
    # participation: an ablation may zero a gradient but never drop one.
    assert missing <= default_none_gradients, sorted(missing - default_none_gradients)
    for added in ("constant_spatial_memory", "core_mlp.0.weight", "relation_transformer.pool_projection.weight"):
        if added in dict(model.named_parameters()):
            assert added not in missing


@pytest.mark.parametrize("name", ("lstm", "spatial_memory", "no_transformer", "everything"))
def test_fused_sequence_matches_stepwise_for_ablation(name: str) -> None:
    model = _build(ModelConfigV4(**ABLATIONS[name])).eval()  # type: ignore[arg-type]
    batch = _batch(batch_size=2)
    actions = _wait_actions(2)
    starts = torch.tensor([[True, True], [False, False], [False, False]])
    flat_batch = concatenate_padded_tensor_records((batch, batch, batch))
    flat_actions = concatenate_tensor_records((actions, actions, actions))
    with torch.no_grad():
        encoded = model.encode_observation(flat_batch)
        log_prob, entropy, value, state = model.evaluate_encoded_action_sequence(
            flat_batch, encoded, model.initial_state(2), flat_actions, starts, time_steps=3
        )
        stepwise = evaluate_recurrent_sequence(
            model,
            (batch, batch, batch),
            (actions, actions, actions),
            starts,
            encoded_observations=tuple(encoded.narrow_batch(step * 2, 2) for step in range(3)),
        )
    torch.testing.assert_close(log_prob, stepwise.log_prob)
    torch.testing.assert_close(entropy, stepwise.entropy)
    torch.testing.assert_close(value, stepwise.value)
    torch.testing.assert_close(state.hidden, stepwise.final_state.hidden)
    torch.testing.assert_close(state.cell, stepwise.final_state.cell)
    assert tuple(state.hidden.shape) == tuple(state.cell.shape) == (2, model.config.lstm_hidden_dim)


def test_all_wait_fused_backward_reaches_constant_spatial_map() -> None:
    model = _build(ModelConfigV4(ablate_spatial_memory=True)).train()
    batch = _batch(batch_size=2)
    flat_batch = concatenate_padded_tensor_records((batch, batch))
    flat_actions = concatenate_tensor_records((_wait_actions(2), _wait_actions(2)))
    encoded = model.encode_observation(flat_batch)
    log_prob, _entropy, value, _state = model.evaluate_encoded_action_sequence(
        flat_batch, encoded, model.initial_state(2), flat_actions, torch.tensor([[True, True], [False, False]]), time_steps=2
    )
    (log_prob.sum() + value.sum()).backward()
    assert model.constant_spatial_memory is not None
    assert model.constant_spatial_memory.grad is not None


def _changed_previous_action(batch: UniversalSemanticBatchV4) -> UniversalSemanticBatchV4:
    own = batch.own_cards.card_vocab_id[:, :1]
    previous = replace(
        batch.previous_action,
        gate=torch.full_like(batch.previous_action.gate, GATE_ACT),
        micro_action_count=torch.ones_like(batch.previous_action.micro_action_count),
        visible_card_vocab_id=torch.cat((own, torch.zeros_like(own)), dim=1),
        effective_card_vocab_id=torch.cat((own, torch.zeros_like(own)), dim=1),
        effective_form=torch.tensor([[1, 0]] * batch.batch_size),
        target_cell=torch.tensor([[40, -1]] * batch.batch_size),
        delay_offset_bin=torch.tensor([[2, -1]] * batch.batch_size),
        action_mask=torch.tensor([[True, False]] * batch.batch_size),
    )
    return replace(batch, previous_action=previous)


def _changed_input(name: str, batch: UniversalSemanticBatchV4) -> UniversalSemanticBatchV4:
    generator = torch.Generator().manual_seed(99)
    if name == "explicit_spatial_planes":
        return replace(batch, explicit_spatial_planes=torch.rand(batch.explicit_spatial_planes.shape, generator=generator))
    if name == "events":
        features = torch.rand(batch.events.features.shape, generator=generator)
        return replace(batch, events=replace(batch.events, features=features))
    if name == "previous_action":
        return _changed_previous_action(batch)
    if name == "relation_type":
        relation_type = batch.relation_edges.relation_type % 13 + 1
        return replace(batch, relation_edges=replace(batch.relation_edges, relation_type=relation_type))
    if name == "group_geometry":
        return replace(
            batch, groups=replace(batch.groups, position=batch.groups.position + 1.0, extent=batch.groups.extent + 1.0)
        )
    raise AssertionError(name)


@pytest.mark.parametrize(
    ("switches", "changed"),
    (
        ({"ablate_spatial_summary": True, "ablate_spatial_memory": True}, "explicit_spatial_planes"),
        ({"ablate_event_summary": True}, "events"),
        ({"ablate_previous_action_summary": True}, "previous_action"),
        ({"ablate_relation_bias": True}, "relation_type"),
        ({"ablate_spatial_pair_bias": True}, "group_geometry"),
        ({"transformer_layers": 0}, "relation_type"),
        ({"transformer_layers": 0}, "group_geometry"),
    ),
)
def test_ablated_input_cannot_change_outputs(switches: dict[str, object], changed: str) -> None:
    batch = _batch(batch_size=2)
    other = _changed_input(changed, batch)
    baseline = _acting_model()
    with torch.no_grad():
        actions = baseline.act(batch, baseline.initial_state(2)).actions
    # The perturbation is real: the baseline reacts to it.
    with pytest.raises(AssertionError):
        _assert_same_outputs(_outputs(baseline, batch, actions), _outputs(baseline, other, actions), atol=1e-6)
    model = _acting_model(ModelConfigV4(**switches))  # type: ignore[arg-type]
    _assert_same_outputs(_outputs(model, batch, actions), _outputs(model, other, actions))


def test_spatial_summary_ablation_isolates_policy_and_value_from_spatial_input() -> None:
    model = _acting_model(ModelConfigV4(ablate_spatial_summary=True))
    batch = _batch(batch_size=2)
    other = _changed_input("explicit_spatial_planes", batch)
    state = model.initial_state(2)
    with torch.no_grad():
        first = model(batch, state)
        second = model(other, state)
    torch.testing.assert_close(first.policy_context, second.policy_context)
    torch.testing.assert_close(first.value_context, second.value_context)
    assert not torch.allclose(first.encoded.spatial_memory, second.encoded.spatial_memory)


def test_spatial_memory_ablation_location_logits_ignore_spatial_memory() -> None:
    model = _build(ModelConfigV4(ablate_spatial_memory=True)).eval()
    config = model.config
    decoder_context = torch.randn(2, config.decoder_hidden_dim)
    candidate = torch.randn(2, config.candidate_dim)
    shape = (2, config.spatial_hidden_channels, config.board_height, config.board_width)
    with torch.no_grad():
        first_logits, first_keys = model._location_logits(decoder_context, candidate, torch.randn(shape))
        second_logits, second_keys = model._location_logits(decoder_context, candidate, torch.randn(shape))
    torch.testing.assert_close(first_logits, second_logits)
    torch.testing.assert_close(first_keys, second_keys)


def _perturb(parameters: list[torch.nn.Parameter]) -> None:
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(torch.randn_like(parameter))


@pytest.mark.parametrize(
    ("switches", "parameters"),
    (
        ({"card_id_gate": "zero"}, lambda model: [model.card_encoder.id_embedding.weight]),
        ({"card_id_gate": "one"}, lambda model: list(model.card_encoder.id_gate.parameters())),
        ({"ablate_decoder_cross_attention": True}, lambda model: list(model.cross_attention.parameters())),
    ),
    ids=("card-id-gate-zero", "card-id-gate-one", "decoder-cross-attention"),
)
def test_ablated_parameters_cannot_change_outputs(switches: dict[str, object], parameters: object) -> None:
    batch = _batch(batch_size=2)
    model = _acting_model(ModelConfigV4(**switches))  # type: ignore[arg-type]
    with torch.no_grad():
        actions = model.act(batch, model.initial_state(2)).actions
    before = _outputs(model, batch, actions)
    _perturb(parameters(model))  # type: ignore[operator]
    _assert_same_outputs(before, _outputs(model, batch, actions))

    baseline = _acting_model()
    baseline_before = _outputs(baseline, batch, actions)
    _perturb(parameters(baseline))  # type: ignore[operator]
    with pytest.raises(AssertionError):
        _assert_same_outputs(baseline_before, _outputs(baseline, batch, actions), atol=1e-6)


def test_lstm_ablation_ignores_incoming_state_but_returns_state_shapes() -> None:
    model = _acting_model(ModelConfigV4(ablate_lstm=True))
    assert model.lstm_core is None and model.core_mlp is not None
    assert not any(isinstance(module, (torch.nn.LSTM, torch.nn.LSTMCell)) for module in model.modules())
    batch = _batch(batch_size=2)
    zero = model.initial_state(2)
    random = type(zero)(hidden=torch.randn_like(zero.hidden), cell=torch.randn_like(zero.cell))
    with torch.no_grad():
        first = model.act(batch, zero)
        second = model.act(batch, random)
    torch.testing.assert_close(first.log_prob, second.log_prob)
    torch.testing.assert_close(first.value, second.value)
    assert tuple(first.next_state.hidden.shape) == (2, model.config.lstm_hidden_dim)
    torch.testing.assert_close(first.next_state.cell, torch.zeros_like(first.next_state.cell))


@pytest.fixture(scope="module")
def production_model() -> UniversalCardPolicyV4:
    torch.manual_seed(0)
    return build_production_model_v4().eval()


def _production_variant(model: UniversalCardPolicyV4, config: ModelConfigV4) -> UniversalCardPolicyV4:
    torch.manual_seed(1)
    return UniversalCardPolicyV4(
        model.catalog,
        ability_catalog=model.ability_catalog,
        entity_archetype_catalog=model.entity_archetype_catalog,
        effect_catalog=model.effect_catalog,
        mechanic_profile_catalog=model.mechanic_profile_catalog,
        config=config,
    ).eval()


@pytest.mark.parametrize("name", ("mechanic_profiles", "card_id_gate_one", "everything"))
def test_production_catalog_ablation_keeps_every_parameter_in_graph(
    name: str, production_model: UniversalCardPolicyV4
) -> None:
    # The fixture catalogs are empty, so effect/profile encoders only run
    # (and only matter for DDP) with the production catalogs.
    baseline = _force_two_actions(_production_variant(production_model, ModelConfigV4()))
    ablated = _force_two_actions(_production_variant(production_model, ModelConfigV4(**ABLATIONS[name])))  # type: ignore[arg-type]
    baseline_missing = _none_gradient_names(baseline)
    assert not any(parameter.startswith("mechanic_profile_encoder.") for parameter in baseline_missing)
    missing = _none_gradient_names(ablated)
    assert missing <= baseline_missing, sorted(missing - baseline_missing)


def test_mechanic_profile_ablation_on_production_catalog(production_model: UniversalCardPolicyV4) -> None:
    model = _production_variant(production_model, ModelConfigV4())
    assert model.mechanic_profile_encoder.profile_id.numel() > 0
    ablated = _production_variant(production_model, ModelConfigV4(ablate_mechanic_profiles=True))
    ablated.load_state_dict(model.state_dict())
    assert ablated.parameter_count() == model.parameter_count()
    batch = _batch(batch_size=1)

    def encode(policy: UniversalCardPolicyV4) -> torch.Tensor:
        with torch.no_grad():
            encoded = policy.encode_observation(batch)
        return torch.cat((encoded.scene_state, encoded.candidate_summary, encoded.previous_action_summary), dim=-1)

    ablated_before = encode(ablated)
    baseline_before = encode(model)
    noise = list(model.mechanic_profile_encoder.parameters())
    _perturb(noise)
    ablated.load_state_dict(model.state_dict())
    torch.testing.assert_close(encode(ablated), ablated_before)
    assert not torch.allclose(encode(model), baseline_before)
    assert math.isfinite(float(ablated_before.abs().sum()))
