import pytest
import torch

from native_runner.training.v4 import distributed_ppo as trainer
from native_runner.tests.test_training_v4_ppo_anchor import _learner_and_reference, _rollout, _clone, _update
from native_runner.tests.test_training_v4_async_cuda_graph import _tensor_rows


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_chunk_transfer_preserves_all_field_shapes_dtypes_and_values(device_name):
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from native_runner.tests.test_training_v4_model import _batch
    device = torch.device(device_name)
    source = _batch().to_storage('cpu', float_dtype=torch.float16)
    expected = source.to_model_input(device)
    actual = trainer._ppo_transfer_record_v4(source, device, model_input=True)
    for (path, got), (reference_path, reference) in zip(_tensor_rows(actual), _tensor_rows(expected), strict=True):
        assert path == reference_path
        assert got.dtype == reference.dtype
        assert got.shape == reference.shape
        assert torch.equal(got, reference), path


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_ppo_transfer_matches_legacy_loss_and_parameter_update(monkeypatch):
    # Compare arithmetic rather than CUDA's nondeterministic reduction order.
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        _compare_updates(monkeypatch)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)


def _compare_updates(monkeypatch):
    model, reference = _learner_and_reference()
    model.set_delay_policy('earliest')
    reference.set_delay_policy('earliest')
    rollout = _rollout(model, steps=5, lanes=4)
    old_model = _clone(model).to('cuda')
    model.to('cuda')
    reference.to('cuda')
    torch.manual_seed(99)
    actual = _update(model, reference, rollout, chunk=2, lr=1e-4)
    def legacy(record, device, *, model_input=False):
        return record.to_model_input(device) if model_input else record.to(device)
    monkeypatch.setattr(trainer, '_ppo_transfer_record_v4', legacy)
    torch.manual_seed(99)
    expected = _update(old_model, reference, rollout, chunk=2, lr=1e-4)
    for got_row, reference_row in zip(actual, expected, strict=True):
        for key in ('policy', 'value', 'total', 'anchor_kl', 'approx_kl', 'clip_fraction', 'valid_count'):
            assert got_row[key] == pytest.approx(reference_row[key], rel=1e-6, abs=1e-7), key
    for (name, got), (reference_name, parameter) in zip(model.named_parameters(), old_model.named_parameters(), strict=True):
        assert name == reference_name
        torch.testing.assert_close(got, parameter, rtol=1e-6, atol=1e-7,
                                   msg=f'{name}: max delta {(got - parameter).abs().max().item()}')
