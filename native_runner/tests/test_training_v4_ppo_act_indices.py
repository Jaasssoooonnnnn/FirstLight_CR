import pytest
import torch

from native_runner.tests.test_training_v4_ppo_device_rollout import _packed_rollout
from native_runner.tests.test_training_v4_ppo_anchor import _clone, _update


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("failed_filter", [False, True])
def test_host_act_indices_preserve_ppo_losses_gradients_and_adam_step(monkeypatch, failed_filter):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model, reference, rollout = _packed_rollout()
        rollout.valid_mask[3:, 2:] = False
        rollout.terminated[2, 2:] = True
        rollout.rewards[3:, 2:] = 0
        if failed_filter:
            rollout = rollout.select_lanes(torch.tensor([3, 0, 2]))
        expected = _clone(model).cuda()
        model.cuda()
        reference.cuda()
        monkeypatch.setenv("CR_PPO_PRECOMPUTE_ACT_INDICES", "0")
        baseline = _update(expected, reference, rollout, chunk=2, lr=1e-4,
                           device_rollout_budget_gib=1, lane_microbatch_size=2)
        monkeypatch.setenv("CR_PPO_PRECOMPUTE_ACT_INDICES", "1")
        observed_indices = []
        original = model._evaluate_sparse_forced_actions
        def checked(*args, **kwargs):
            actions = args[2]
            indices = kwargs["forced_act_indices"]
            assert indices is not None
            torch.testing.assert_close(indices, (actions.gate == 1).nonzero().flatten(), rtol=0, atol=0)
            observed_indices.append(indices.numel())
            return original(*args, **kwargs)
        monkeypatch.setattr(model, "_evaluate_sparse_forced_actions", checked)
        actual = _update(model, reference, rollout, chunk=2, lr=1e-4,
                         device_rollout_budget_gib=1, lane_microbatch_size=2)
        assert observed_indices
        for got, want in zip(actual, baseline, strict=True):
            for key in ("policy", "value", "total", "anchor_kl", "approx_kl", "valid_count"):
                assert got[key] == pytest.approx(want[key], abs=1e-7, rel=1e-6)
        for (name, got), (_, want) in zip(model.named_parameters(), expected.named_parameters(), strict=True):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7, msg=name)
            if got.grad is not None:
                torch.testing.assert_close(got.grad, want.grad, rtol=1e-6, atol=1e-7, msg=name)
    finally:
        torch.use_deterministic_algorithms(previous)
