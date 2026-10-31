import pytest
import torch
from native_runner.tests.test_training_v4_model import _batch, _model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_static_memory_matches_uncached_refreshes_in_place_and_does_not_cut_gradients():
    model = _model().cuda().eval()
    batch = _batch().to_model_input('cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model.encode_observation(batch, validate=False)
        model.refresh_rollout_static_memory()
        actual = model.encode_observation(batch, validate=False)
        torch.testing.assert_close(actual.scene_state, expected.scene_state, rtol=0, atol=0)
        torch.testing.assert_close(actual.candidate_memory, expected.candidate_memory, rtol=0, atol=0)
        cache = model.__dict__['_rollout_static_memory']
        pointers = (cache[1].data_ptr(), cache[2].data_ptr())
        next(model.effect_encoder.parameters()).add_(0.01)
        model.refresh_rollout_static_memory()
        assert pointers == (cache[1].data_ptr(), cache[2].data_ptr())
        refreshed = model.encode_observation(batch, validate=False)
        del model.__dict__['_rollout_static_memory']
        rebuilt = model.encode_observation(batch, validate=False)
        torch.testing.assert_close(refreshed.scene_state, rebuilt.scene_state, rtol=0, atol=0)
        model.refresh_rollout_static_memory()
    model.train()
    seen = []
    original = model.effect_encoder.all_effects
    def tracked():
        seen.append(torch.is_grad_enabled())
        return original()
    model.effect_encoder.all_effects = tracked
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model.encode_observation(batch, validate=False)
    result.scene_state.square().sum().backward()
    assert seen == [True]
