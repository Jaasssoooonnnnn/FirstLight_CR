from unittest.mock import Mock

import pytest
import torch

from native_runner.training.v4.distributed_ppo import _ppo_device_rollout_budget_bytes_v4

GIB = 2**30


def test_previous_wave_unused_cache_does_not_force_cpu_fallback(monkeypatch):
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '2')
    memory = {'free': 20 * GIB}
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (memory['free'], 48 * GIB))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: memory.update(free=40 * GIB))
    # The next wave needs 18 GiB. The old driver-only check admitted
    # just 10 GiB because the previous wave retained unused blocks.
    budget = _ppo_device_rollout_budget_bytes_v4(torch.device('cuda:0'), 32, world_size=2)
    assert 18 * GIB <= budget == 20 * GIB


@pytest.mark.parametrize(('local_world', 'expected'), [(2, 20), (4, 10), (8, 5)])
def test_reclaimed_memory_still_preserves_activation_and_shared_rank_headroom(monkeypatch, local_world, expected):
    monkeypatch.setenv('LOCAL_WORLD_SIZE', str(local_world))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (40 * GIB, 48 * GIB))
    assert _ppo_device_rollout_budget_bytes_v4(torch.device('cuda:1'), 32, world_size=2) == expected * GIB
    assert _ppo_device_rollout_budget_bytes_v4(torch.device('cuda:1'), 4, world_size=2) == 4 * GIB


def test_cpu_budget_does_not_touch_cuda(monkeypatch):
    unused = Mock(side_effect=AssertionError('CPU budget must not query CUDA'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', unused)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', unused)
    assert _ppo_device_rollout_budget_bytes_v4(torch.device('cpu'), 2.5, world_size=2) == 2.5 * GIB
