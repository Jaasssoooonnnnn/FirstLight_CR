from __future__ import annotations

from pathlib import Path

import pytest
import torch

from native_runner.tests.test_training_v4_model import _model
from native_runner.training.v4.checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from native_runner.training.v4.config import CARD_RUNTIME_FEATURE_NAMES
from native_runner.training.v4.migrate_checkpoint_card_runtime import PADDED_TENSOR_KEY, migrate_checkpoint


OLD_DIM = 16
NEW_DIM = len(CARD_RUNTIME_FEATURE_NAMES)


def _write_old_checkpoint(directory: Path, *, mutate=None) -> tuple[Path, dict]:
    """Save a current checkpoint, then rewrite it as a pre-append (16-column) one."""

    torch.manual_seed(7)
    model = _model()
    model.set_ppo_gate_temperature(0.2)
    model.set_ppo_action_temperature(0.7)
    model.set_ppo_continue_temperature(0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter) * 1e-3
    optimizer.step()  # populate optimizer state so we can see it is dropped
    current = directory / "current.pt"
    save_actor_critic_checkpoint(
        current,
        model,
        optimizer=optimizer,
        update_step=1234,
        training_stage="ppo-league-snapshot",
        gamma_per_decision=0.995,
        gae_lambda=0.95,
        extra={"run": "fixture"},
    )
    payload = torch.load(current, map_location="cpu", weights_only=False)
    assert payload["optimizer_state_dict"] is not None
    state = payload["model_state_dict"]
    state[PADDED_TENSOR_KEY] = state[PADDED_TENSOR_KEY][:, :OLD_DIM].clone()
    payload["contract"]["model_config"]["card_runtime_feature_dim"] = OLD_DIM
    if mutate is not None:
        mutate(payload)
    old = directory / "old.pt"
    torch.save(payload, old)
    return old, payload


def test_migration_zero_pads_and_preserves_everything_else(tmp_path: Path) -> None:
    assert NEW_DIM > OLD_DIM
    source, old_payload = _write_old_checkpoint(tmp_path)
    destination = tmp_path / "out" / "migrated.pt"

    torch.manual_seed(99)  # different init: weights must come from the checkpoint
    summary = migrate_checkpoint(source, destination, model=_model())

    assert summary["padded_tensors"] == [PADDED_TENSOR_KEY]
    assert summary["update_step"] == 1234
    assert summary["training_stage"] == "ppo-league-snapshot"
    assert summary["ppo_gate_temperature"] == pytest.approx(0.2)
    assert summary["dropped_state"] == ["optimizer_state_dict"]
    assert summary["source_checkpoint_id"] == old_payload["checkpoint_id"]

    fresh = _model()
    payload = load_actor_critic_checkpoint(destination, fresh)  # strict contract + state dict
    assert payload["checkpoint_id"] == summary["checkpoint_id"] != old_payload["checkpoint_id"]

    old_state = old_payload["model_state_dict"]
    weight = fresh.card_token_encoder.runtime[0].weight.detach()
    assert weight.shape == (128, NEW_DIM)
    assert weight.dtype == old_state[PADDED_TENSOR_KEY].dtype
    assert torch.equal(weight[:, :OLD_DIM], old_state[PADDED_TENSOR_KEY])
    assert torch.count_nonzero(weight[:, OLD_DIM:]) == 0
    for key, value in fresh.state_dict().items():
        if key != PADDED_TENSOR_KEY:
            assert torch.equal(value, old_state[key]), key

    assert payload["contract"]["model_config"]["card_runtime_feature_dim"] == NEW_DIM
    assert payload["update_step"] == 1234
    assert payload["training_stage"] == "ppo-league-snapshot"
    assert payload["gamma_per_decision"] == pytest.approx(0.995)
    assert payload["gae_lambda"] == pytest.approx(0.95)
    assert payload["ppo_gate_temperature"] == pytest.approx(0.2)
    assert payload["ppo_action_temperature"] == pytest.approx(0.7)
    assert payload["ppo_continue_temperature"] == pytest.approx(0.9)
    assert fresh.ppo_gate_temperature == pytest.approx(0.2)
    assert fresh.ppo_action_temperature == pytest.approx(0.7)
    assert fresh.ppo_continue_temperature == pytest.approx(0.9)
    assert payload["optimizer_state_dict"] is None
    assert payload["scheduler_state_dict"] is None
    assert payload["grad_scaler_state_dict"] is None

    extra = payload["extra"]
    assert extra["run"] == "fixture"
    assert extra["migrated_from"] == str(source)
    assert extra["migrated_from_checkpoint_id"] == old_payload["checkpoint_id"]
    assert extra["migration"] == f"card_runtime_feature_dim {OLD_DIM}->{NEW_DIM} zero-pad"
    assert extra["migration_zero_padded_tensors"] == [PADDED_TENSOR_KEY]


def test_migrated_runtime_mlp_matches_old_mlp_when_new_features_are_zero(tmp_path: Path) -> None:
    source, old_payload = _write_old_checkpoint(tmp_path)
    model = _model()
    migrate_checkpoint(source, tmp_path / "migrated.pt", model=model)

    old_state = old_payload["model_state_dict"]
    functional = torch.nn.functional
    features = torch.randn(4, 8, OLD_DIM)
    padded = torch.cat((features, torch.zeros(4, 8, NEW_DIM - OLD_DIM)), dim=-1)
    with torch.no_grad():
        hidden = functional.linear(
            features, old_state[PADDED_TENSOR_KEY], old_state["card_token_encoder.runtime.0.bias"]
        )
        expected = functional.linear(
            functional.silu(hidden),
            old_state["card_token_encoder.runtime.2.weight"],
            old_state["card_token_encoder.runtime.2.bias"],
        )
        actual = model.card_token_encoder.runtime(padded)
        nonzero_new = padded.clone()
        nonzero_new[..., OLD_DIM:] = 5.0
        ignored = model.card_token_encoder.runtime(nonzero_new)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ignored, actual, rtol=0, atol=0)


def test_rejects_additional_shape_difference(tmp_path: Path) -> None:
    def mutate(payload: dict) -> None:
        state = payload["model_state_dict"]
        state["card_token_encoder.role.weight"] = torch.zeros(state["card_token_encoder.role.weight"].shape[0], 3)

    source, _ = _write_old_checkpoint(tmp_path, mutate=mutate)
    destination = tmp_path / "migrated.pt"
    with pytest.raises(ValueError, match="card_token_encoder.role.weight"):
        migrate_checkpoint(source, destination, model=_model())
    assert not destination.exists()


def test_rejects_different_catalog_id(tmp_path: Path) -> None:
    def mutate(payload: dict) -> None:
        payload["contract"]["catalog_ids"]["card"] = "not-the-same-catalog"

    source, _ = _write_old_checkpoint(tmp_path, mutate=mutate)
    destination = tmp_path / "migrated.pt"
    with pytest.raises(ValueError, match="catalog_ids"):
        migrate_checkpoint(source, destination, model=_model())
    assert not destination.exists()


def test_rejects_other_contract_difference(tmp_path: Path) -> None:
    def mutate(payload: dict) -> None:
        payload["contract"]["model_config"]["token_dim"] += 1

    source, _ = _write_old_checkpoint(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="model_config.token_dim"):
        migrate_checkpoint(source, tmp_path / "migrated.pt", model=_model())


def test_rejects_missing_or_extra_state_keys(tmp_path: Path) -> None:
    def mutate(payload: dict) -> None:
        payload["model_state_dict"]["bogus.weight"] = torch.zeros(1)

    source, _ = _write_old_checkpoint(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="bogus.weight"):
        migrate_checkpoint(source, tmp_path / "migrated.pt", model=_model())


def test_rejects_wrong_schema_and_already_current_checkpoint(tmp_path: Path) -> None:
    def wrong_schema(payload: dict) -> None:
        payload["schema"] = "something-else"

    source, _ = _write_old_checkpoint(tmp_path, mutate=wrong_schema)
    with pytest.raises(ValueError, match="schema"):
        migrate_checkpoint(source, tmp_path / "migrated.pt", model=_model())

    with pytest.raises(ValueError, match="not smaller"):
        migrate_checkpoint(tmp_path / "current.pt", tmp_path / "migrated.pt", model=_model())


def test_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    source, _ = _write_old_checkpoint(tmp_path)
    destination = tmp_path / "migrated.pt"
    destination.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="already exists"):
        migrate_checkpoint(source, destination, model=_model())
    assert destination.read_bytes() == b"keep me"
    with pytest.raises(ValueError, match="differ from source"):
        migrate_checkpoint(source, source, model=_model())
