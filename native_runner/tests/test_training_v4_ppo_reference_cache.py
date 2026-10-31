from dataclasses import replace

import pytest
import torch

from native_runner.tests.test_training_v4_ppo_device_rollout import _packed_rollout
from native_runner.tests.test_training_v4_ppo_anchor import _clone
from native_runner.training.v4.distributed_ppo import distributed_ppo_update_v4
from native_runner.training.v4.ppo import PPOConfigV4


@pytest.mark.parametrize("device,autocast", [("cpu", None), ("cuda", None), ("cuda", torch.bfloat16)])
def test_reference_replay_cache_preserves_shuffled_recurrent_ppo_and_adam(monkeypatch, device, autocast):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model, reference, rollout = _packed_rollout()
        rollout.valid_mask[3:, 2:] = False
        rollout.terminated[2, 2:] = True
        rollout.rewards[3:, 2:] = 0
        forced = torch.zeros_like(rollout.valid_mask)
        forced[1, 0] = True
        rollout = replace(rollout, forced_gate_mask=forced).select_lanes(torch.tensor([3, 0, 2]))
        model.to(device)
        expected = _clone(model).to(device)
        reference.to(device)
        calls = []
        original = reference.encode_observation
        def counted(*args, **kwargs):
            calls.append(args[0].batch_size)
            return original(*args, **kwargs)
        monkeypatch.setattr(reference, "encode_observation", counted)
        def update(actor):
            return distributed_ppo_update_v4(actor, torch.optim.AdamW(actor.parameters(), lr=1e-4), rollout,
                epochs=2, lane_minibatch_size=2, lane_microbatch_size=2, sequence_chunk_steps=2,
                config=PPOConfigV4(entropy_coefficient=0.0, target_kl=None), autocast_dtype=autocast,
                reference_model=reference, anchor_kl_coefficient=0.05, anchor_continue_kl_coefficient=0.25,
                device_rollout_budget_gib=1 if device == "cuda" else 0)
        monkeypatch.setenv("CR_PPO_CACHE_REFERENCE", "0")
        baseline = update(expected)
        baseline_rows = sum(calls)
        calls.clear()
        monkeypatch.setenv("CR_PPO_CACHE_REFERENCE", "1")
        actual = update(model)
        assert sum(calls) * 2 == baseline_rows
        assert len(actual) == len(baseline) == 4
        for got, want in zip(actual, baseline, strict=True):
            for key in ("policy", "value", "total", "anchor_kl", "approx_kl", "valid_count",
                        "anchor_kl_gate", "anchor_kl_candidate", "anchor_kl_target", "anchor_kl_continue"):
                assert got[key] == pytest.approx(want[key], abs=1e-7, rel=1e-5)
        for (name, got), (_, want) in zip(model.named_parameters(), expected.named_parameters(), strict=True):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7, msg=name)
        assert all(parameter.grad is None for parameter in reference.parameters())
    finally:
        torch.use_deterministic_algorithms(previous)
