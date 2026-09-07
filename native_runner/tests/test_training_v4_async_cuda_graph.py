from __future__ import annotations

from dataclasses import fields, is_dataclass
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import Tensor

from native_runner.training.v4.async_cluster_self_play import (
    _ActorIdentityV4,
    _PackedCudaGraphBatchV4,
    _StochasticCudaGraphV4,
    _elixir_overflow_penalty_transition_v4,
    _frozen_opponent_opening_force_owner_v4,
    _inference_model_key,
    _persistent_rollout_graph_v4,
    _ppo_cuda_graph_row_capacity_v4,
)
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.decoding import ShadowCandidateLegality
from native_runner.training.v4.policy_session import _fixed_cuda_graph_batch_v4


def test_frozen_anchor_keeps_its_dedicated_inference_model() -> None:
    current = _ActorIdentityV4(0, 0, "current", True)
    anchor = _ActorIdentityV4(0, 1, "anchor", False)

    assert _inference_model_key(current) == "current"
    assert _inference_model_key(anchor) == "anchor"


def _model_test_batch() -> object:
    source = Path(__file__).with_name("test_training_v4_model.py")
    spec = importlib.util.spec_from_file_location("v4_model_test_fixture", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load the V4 model test fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._batch()  # type: ignore[attr-defined]


def _tensor_rows(value: object, path: str = "batch") -> list[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        return [(path, value)]
    if is_dataclass(value):
        return [
            row
            for item in fields(value)
            for row in _tensor_rows(
                getattr(value, item.name),
                f"{path}.{item.name}",
            )
        ]
    return []


@pytest.mark.parametrize(
    ("rows", "capacity"),
    (
        (1, 1),
        (2, 2),
        (3, 4),
        (8, 8),
        (9, 16),
        (16, 16),
        (17, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 96),
    ),
)
def test_ppo_cuda_graph_row_capacity(rows: int, capacity: int) -> None:
    assert _ppo_cuda_graph_row_capacity_v4(rows) == capacity


def test_ppo_cuda_graph_row_capacity_rejects_empty() -> None:
    with pytest.raises(ValueError, match="row count"):
        _ppo_cuda_graph_row_capacity_v4(0)


def test_eval_opening_fallback_only_forces_a_passive_frozen_opponent() -> None:
    assert _frozen_opponent_opening_force_owner_v4(
        current_owners=(0,),
        active_actions=(3, 0),
        tick=15 * 20 - 1,
        after_seconds=15,
    ) is None
    assert _frozen_opponent_opening_force_owner_v4(
        current_owners=(0,),
        active_actions=(3, 0),
        tick=15 * 20,
        after_seconds=15,
    ) == 1
    assert _frozen_opponent_opening_force_owner_v4(
        current_owners=(0,),
        active_actions=(3, 1),
        tick=60 * 20,
        after_seconds=15,
    ) is None
    assert _frozen_opponent_opening_force_owner_v4(
        current_owners=(0, 1),
        active_actions=(0, 0),
        tick=60 * 20,
        after_seconds=15,
    ) is None
    assert _frozen_opponent_opening_force_owner_v4(
        current_owners=(0,),
        active_actions=(0, 0),
        tick=60 * 20,
        after_seconds=None,
    ) is None


def test_personal_elixir_overflow_has_grace_then_is_linear() -> None:
    first = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=0.0,
        personal_accrued_cost=0.0,
        unilateral_wasted_elixir=0.0,
        unilateral_accrued_cost=0.0,
        before_elixir=10.0,
        after_elixir=10.0,
        generated_elixir=1.0,
        both_full=True,
        personal_coefficient=0.01,
        personal_grace_elixir=4.0,
        unilateral_coefficient=0.025,
        step_penalty_cap=0.1,
    )
    assert first == pytest.approx((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    grace_edge = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=first[0],
        personal_accrued_cost=first[1],
        unilateral_wasted_elixir=first[2],
        unilateral_accrued_cost=first[3],
        before_elixir=10.0,
        after_elixir=10.0,
        generated_elixir=3.0,
        both_full=True,
        personal_coefficient=0.01,
        personal_grace_elixir=4.0,
        unilateral_coefficient=0.025,
        step_penalty_cap=0.1,
    )
    assert grace_edge == pytest.approx((4.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    after_grace = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=grace_edge[0],
        personal_accrued_cost=grace_edge[1],
        unilateral_wasted_elixir=grace_edge[2],
        unilateral_accrued_cost=grace_edge[3],
        before_elixir=10.0,
        after_elixir=10.0,
        generated_elixir=1.0,
        both_full=True,
        personal_coefficient=0.01,
        personal_grace_elixir=4.0,
        unilateral_coefficient=0.025,
        step_penalty_cap=0.1,
    )
    assert after_grace == pytest.approx((5.0, 0.01, 0.0, 0.0, 0.01, 0.0))


def test_personal_and_unilateral_overflow_reset_after_spending() -> None:
    reset = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=5.0,
        personal_accrued_cost=0.01,
        unilateral_wasted_elixir=2.0,
        unilateral_accrued_cost=0.1,
        before_elixir=10.0,
        after_elixir=7.0,
        generated_elixir=0.1,
        both_full=False,
        personal_coefficient=0.01,
        personal_grace_elixir=4.0,
        unilateral_coefficient=0.025,
        step_penalty_cap=0.1,
    )
    assert reset == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_unilateral_overflow_starts_fresh_while_personal_clock_persists() -> None:
    transition = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=4.0,
        personal_accrued_cost=0.0,
        unilateral_wasted_elixir=0.0,
        unilateral_accrued_cost=0.0,
        before_elixir=10.0,
        after_elixir=10.0,
        generated_elixir=0.0,
        both_full=False,
        personal_coefficient=0.01,
        personal_grace_elixir=4.0,
        unilateral_coefficient=0.025,
        step_penalty_cap=0.1,
    )
    assert transition == pytest.approx((4.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_unilateral_overflow_is_quadratic_with_incremental_cost_and_cap() -> None:
    first = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=0.0,
        personal_accrued_cost=0.0,
        unilateral_wasted_elixir=0.0,
        unilateral_accrued_cost=0.0,
        before_elixir=10.0,
        after_elixir=10.0,
        generated_elixir=1.0,
        both_full=False,
        personal_coefficient=0.01,
        personal_grace_elixir=4.0,
        unilateral_coefficient=0.025,
        step_penalty_cap=0.1,
    )
    assert first == pytest.approx((1.0, 0.0, 1.0, 0.025, 0.0, 0.025))
    second = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=first[0],
        personal_accrued_cost=first[1],
        unilateral_wasted_elixir=first[2],
        unilateral_accrued_cost=first[3],
        before_elixir=10.0,
        after_elixir=10.0,
        generated_elixir=1.0,
        both_full=False,
        personal_coefficient=0.01,
        personal_grace_elixir=4.0,
        unilateral_coefficient=0.025,
        step_penalty_cap=0.1,
    )
    assert second == pytest.approx((2.0, 0.0, 2.0, 0.1, 0.0, 0.075))
    capped = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=4.0,
        personal_accrued_cost=0.0,
        unilateral_wasted_elixir=4.0,
        unilateral_accrued_cost=0.04,
        before_elixir=10.0,
        after_elixir=10.0,
        generated_elixir=0.0,
        both_full=False,
        personal_coefficient=0.01,
        personal_grace_elixir=4.0,
        unilateral_coefficient=0.025,
        step_penalty_cap=0.1,
    )
    assert capped == pytest.approx((4.0, 0.0, 4.0, 0.14, 0.0, 0.1))


def test_personal_and_unilateral_overflow_penalties_stack_after_grace() -> None:
    combined = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=2.0,
        personal_accrued_cost=0.0,
        unilateral_wasted_elixir=0.0,
        unilateral_accrued_cost=0.0,
        before_elixir=10.0,
        after_elixir=10.0,
        generated_elixir=1.0,
        both_full=False,
        personal_coefficient=0.0025,
        personal_grace_elixir=2.0,
        unilateral_coefficient=0.00625,
        step_penalty_cap=0.1,
    )

    assert combined == pytest.approx(
        (3.0, 0.0025, 1.0, 0.00625, 0.0025, 0.00625)
    )
    assert combined[4] + combined[5] == pytest.approx(
        0.0025 * (3.0 - 2.0) + 0.00625 * 1.0**2
    )


def test_storage_preserves_exact_candidate_elixir_for_shadow_legality() -> None:
    batch = _model_test_batch()
    candidates = batch.candidates
    candidates.mask.zero_()
    candidates.mask[:, :2] = True
    candidates.placement[:, :2] = True
    candidates.elixir.fill_(4.999)
    candidates.cost[:, 0] = 3.0
    candidates.cost[:, 1] = 2.0
    candidates.own_card_row[:, 0] = 0
    candidates.own_card_row[:, 1] = 1
    candidates.exclusion_group_id[:, 0] = 1000
    candidates.exclusion_group_id[:, 1] = 1001
    assert float(candidates.elixir[0].to(torch.float16)) == 5.0

    stored = batch.to_storage("cpu", float_dtype=torch.float16)
    assert stored.match_scalars.dtype == torch.float16
    assert stored.candidates.elixir.dtype == torch.float32
    assert stored.candidates.cost.dtype == torch.float32
    assert stored.candidates.building_half_width.dtype == torch.float16
    torch.testing.assert_close(
        stored.candidates.elixir,
        candidates.elixir,
        rtol=0.0,
        atol=0.0,
    )

    shadow = ShadowCandidateLegality(stored.candidates, ModelConfigV4())
    assert bool(torch.all(shadow.candidate_mask()[:, :2]))
    active = torch.ones(stored.batch_size, dtype=torch.bool)
    stored.candidates.is_building[:, 0] = True
    shadow.apply(
        active,
        torch.zeros(stored.batch_size, dtype=torch.long),
        torch.zeros(stored.batch_size, dtype=torch.long),
        torch.zeros(stored.batch_size, dtype=torch.long),
        step=0,
    )
    assert shadow.planned_half_width.dtype == torch.float32
    assert bool(torch.all(shadow.planned_count == 1))
    assert not bool(torch.any(shadow.candidate_mask()[:, 1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_cuda_graph_batch_is_field_exact() -> None:
    device = torch.device("cuda", 0)
    storage = _model_test_batch().to_storage("cpu", float_dtype=torch.float16)
    fixed = _fixed_cuda_graph_batch_v4(
        storage,
        effect_capacity=max(1, int(storage.active_effects.mask.shape[1])),
        relation_capacity=max(1, int(storage.relation_edges.mask.shape[1])),
    )
    packed = _PackedCudaGraphBatchV4(fixed, device)

    for _ in range(4):
        packed.stage(fixed)
        torch.cuda.synchronize(device)
        expected = fixed.to_model_input(device)
        expected_rows = _tensor_rows(expected)
        actual_rows = _tensor_rows(packed.device_batch)
        assert [path for path, _tensor in actual_rows] == [
            path for path, _tensor in expected_rows
        ]
        for (path, actual), (_expected_path, reference) in zip(
            actual_rows,
            expected_rows,
            strict=True,
        ):
            assert actual.dtype == reference.dtype, path
            assert actual.shape == reference.shape, path
            assert torch.equal(actual, reference), path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_persistent_rollout_graph_variants_share_one_safe_pool() -> None:
    device = torch.device("cuda", 0)
    fixture = _model_test_batch()
    source = Path(__file__).with_name("test_training_v4_model.py")
    spec = importlib.util.spec_from_file_location("v4_model_pool_fixture", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load the V4 model graph fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module._model().to(device).eval()  # type: ignore[attr-defined]
    variants: list[tuple[_StochasticCudaGraphV4, object, Tensor, Tensor]] = []

    for capacity in (32, 64):
        rows = torch.tensor(
            [0, 1, *([0] * (capacity - 2))],
            dtype=torch.long,
        )
        storage = fixture.index_select(rows).to_storage(
            "cpu",
            float_dtype=torch.float16,
        )
        graph = _persistent_rollout_graph_v4(
            model,
            device,
            capacity,
            effect_capacity=max(1, int(storage.active_effects.mask.shape[1])),
            relation_capacity=max(1, int(storage.relation_edges.mask.shape[1])),
        )
        assert graph is not None
        state = model.initial_state(capacity, device=device)
        episode_start = torch.zeros(capacity, dtype=torch.bool, device=device)
        variants.append((graph, storage, state, episode_start))

    assert variants[0][0].graph_pool == variants[1][0].graph_pool
    for graph, storage, state, episode_start in (*variants, *reversed(variants)):
        output = graph.run(model, storage, state, episode_start)
        assert output.actions.gate.shape[0] == storage.batch_size
    torch.cuda.synchronize(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_frozen_rollout_graph_replays_argmax_deterministically() -> None:
    device = torch.device("cuda", 0)
    fixture = _model_test_batch()
    source = Path(__file__).with_name("test_training_v4_model.py")
    spec = importlib.util.spec_from_file_location(
        "v4_model_frozen_graph_fixture",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load the V4 frozen graph fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module._model().to(device).eval()  # type: ignore[attr-defined]
    model.set_ppo_rollout_sampling(False)
    storage = fixture.to_storage("cpu", float_dtype=torch.float16)
    graph = _persistent_rollout_graph_v4(
        model,
        device,
        storage.batch_size,
        effect_capacity=max(1, int(storage.active_effects.mask.shape[1])),
        relation_capacity=max(1, int(storage.relation_edges.mask.shape[1])),
    )
    assert graph is not None
    state = model.initial_state(storage.batch_size, device=device)
    episode_start = torch.ones(
        storage.batch_size,
        dtype=torch.bool,
        device=device,
    )

    first = graph.run(model, storage, state, episode_start)
    first_gate = first.actions.gate.clone()
    first_count = first.actions.micro_action_count.clone()
    first_candidate = first.actions.candidate_index.clone()
    first_target = first.actions.target_cell.clone()
    first_delay = first.actions.delay_offset_bin.clone()
    torch.manual_seed(1_337)
    second = graph.run(model, storage, state, episode_start)
    torch.cuda.synchronize(device)

    assert torch.equal(first_gate, second.actions.gate)
    assert torch.equal(
        first_count,
        second.actions.micro_action_count,
    )
    assert torch.equal(
        first_candidate,
        second.actions.candidate_index,
    )
    assert torch.equal(first_target, second.actions.target_cell)
    assert torch.equal(
        first_delay,
        second.actions.delay_offset_bin,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_distinct_policy_graphs_replay_on_parallel_streams() -> None:
    device = torch.device("cuda", 0)
    fixture = _model_test_batch()
    rows = torch.tensor([0, 1, *([0] * 30)], dtype=torch.long)
    storage = fixture.index_select(rows).to_storage(
        "cpu",
        float_dtype=torch.float16,
    )
    source = Path(__file__).with_name("test_training_v4_model.py")
    spec = importlib.util.spec_from_file_location(
        "v4_parallel_model_graph_fixture",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load the V4 parallel graph fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    models = [
        module._model().to(device).eval()  # type: ignore[attr-defined]
        for _ in range(10)
    ]
    graphs = [
        _persistent_rollout_graph_v4(
            model,
            device,
            storage.batch_size,
            effect_capacity=max(1, int(storage.active_effects.mask.shape[1])),
            relation_capacity=max(1, int(storage.relation_edges.mask.shape[1])),
        )
        for model in models
    ]
    assert all(graph is not None for graph in graphs)
    states = [
        model.initial_state(storage.batch_size, device=device) for model in models
    ]
    episode_start = torch.zeros(
        storage.batch_size,
        dtype=torch.bool,
        device=device,
    )
    streams = [torch.cuda.Stream(device=device) for _ in models]
    coordinator = torch.cuda.current_stream(device)

    for _ in range(8):
        outputs = []
        for model, graph, state, stream in zip(
            models,
            graphs,
            states,
            streams,
            strict=True,
        ):
            assert graph is not None
            stream.wait_stream(coordinator)
            with torch.cuda.stream(stream):
                outputs.append(graph.run(model, storage, state, episode_start))
        for stream in streams:
            coordinator.wait_stream(stream)
        coordinator.synchronize()
        for output in outputs:
            assert output.actions.gate.shape == (storage.batch_size,)
            assert bool(torch.all(torch.isfinite(output.log_prob)))
