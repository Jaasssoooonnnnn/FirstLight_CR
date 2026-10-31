"""Actor-critic checkpoint contract shared by V4 IL and PPO."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping
import uuid

import torch

from .config import (
    UNIVERSAL_ACTION_VERSION,
    UNIVERSAL_OBSERVATION_VERSION,
    UNIVERSAL_POLICY_VERSION,
    ModelConfigV4,
)
from .model import UniversalCardPolicyV4


CHECKPOINT_SCHEMA_V4 = "universal-card-actor-critic-checkpoint.v4"

# ModelConfigV4 fields whose ABSENCE from a stored contract means "the old
# behaviour", which differs from the current dataclass default.  A checkpoint
# written before ``delay_aware_legality`` existed was trained (and must keep
# being decoded) with the decision-tick elixir rule.
LEGACY_FIELD_DEFAULTS: Mapping[str, Any] = {"delay_aware_legality": False}

# Behavioural ModelConfigV4 fields: no parameter, buffer or input width depends
# on them, so the loader does not require them to match the model it loads
# into.  It adopts the checkpoint's (legacy-filled) value instead, so loading a
# pre-fix checkpoint into ``build_production_model_v4()`` keeps the old rule.
BEHAVIOURAL_FIELD_NAMES: frozenset[str] = frozenset({"delay_aware_legality"})


def _catalog_ids(model: UniversalCardPolicyV4) -> dict[str, str]:
    return {
        "card": model.catalog.catalog_id,
        "ability": model.ability_catalog.catalog_id,
        "entity_archetype": model.entity_archetype_catalog.catalog_id,
        "effect": model.effect_catalog.catalog_id,
        "mechanic_profile": model.mechanic_profile_catalog.catalog_id,
    }


def checkpoint_contract(model: UniversalCardPolicyV4) -> dict[str, Any]:
    """Return architecture/data identities without changing the model."""

    return {
        "policy_version": UNIVERSAL_POLICY_VERSION,
        "observation_version": UNIVERSAL_OBSERVATION_VERSION,
        "action_version": UNIVERSAL_ACTION_VERSION,
        "model_config": asdict(model.config),
        "catalog_ids": _catalog_ids(model),
    }


def _filled_model_config(stored: object) -> dict[str, Any]:
    """Fill keys added to ModelConfigV4 after a checkpoint was written.

    Missing keys take ``LEGACY_FIELD_DEFAULTS`` when listed there (fields whose
    absence means the old behaviour), otherwise the current ``ModelConfigV4()``
    defaults (new architecture fields are added with defaults that reproduce
    the older architecture).  Keys unknown to the current dataclass are
    rejected: silently dropping them could load weights into a different
    architecture.
    """

    if not isinstance(stored, Mapping):
        raise ValueError("checkpoint contract has no model_config mapping")
    defaults = ModelConfigV4()
    known = {item.name for item in fields(ModelConfigV4)}
    unknown = sorted(str(key) for key in set(stored).difference(known))
    if unknown:
        raise ValueError(f"checkpoint model_config has fields unknown to ModelConfigV4: {unknown}")
    filled = asdict(defaults)
    filled.update(LEGACY_FIELD_DEFAULTS)
    for key, value in stored.items():
        if isinstance(filled[key], tuple) and isinstance(value, list):
            value = tuple(value)
        filled[key] = value
    return filled


def model_config_from_contract(contract: Mapping[str, Any]) -> ModelConfigV4:
    """Rebuild the ModelConfigV4 a checkpoint was trained with (missing keys -> defaults)."""

    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint contract must be a mapping")
    return ModelConfigV4(**_filled_model_config(contract.get("model_config")))


def _contract_differences(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    differences: list[str] = []
    for key in sorted(set(stored).union(expected)):
        left, right = stored.get(key), expected.get(key)
        if key == "model_config" and isinstance(left, Mapping) and isinstance(right, Mapping):
            differences.extend(
                f"model_config.{name}"
                for name in sorted(set(left).union(right))
                if left.get(name) != right.get(name)
            )
        elif left != right:
            differences.append(str(key))
    return differences


def save_actor_critic_checkpoint(
    path: str | Path,
    model: UniversalCardPolicyV4,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    grad_scaler: object | None = None,
    update_step: int = 0,
    training_stage: str,
    gamma_per_decision: float,
    gae_lambda: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Save the complete actor, critic, optimizer, and behavior temperatures."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_id = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA_V4,
        "checkpoint_id": checkpoint_id,
        "contract": checkpoint_contract(model),
        "model_state_dict": model.state_dict(),
        **{
            f"{name}_state_dict": None if value is None else value.state_dict()
            for name, value in (("optimizer", optimizer), ("scheduler", scheduler), ("grad_scaler", grad_scaler))
        },
        "update_step": int(update_step),
        "training_stage": str(training_stage),
        "gamma_per_decision": float(gamma_per_decision),
        "gae_lambda": None if gae_lambda is None else float(gae_lambda),
        "ppo_gate_temperature": model.ppo_gate_temperature,
        "ppo_action_temperature": model.ppo_action_temperature,
        "ppo_continue_temperature": model.ppo_continue_temperature,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "extra": dict(extra or {}),
    }
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint_id


def load_actor_critic_checkpoint(
    path: str | Path,
    model: UniversalCardPolicyV4,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    grad_scaler: object | None = None,
    map_location: torch.device | str | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load a strict checkpoint after verifying V4 and catalog identities."""

    payload = read_actor_critic_checkpoint(path, map_location=map_location)
    return restore_actor_critic_checkpoint(
        payload,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
        restore_rng=restore_rng,
    )


def read_actor_critic_checkpoint(
    path: str | Path, *, map_location: torch.device | str | None = None
) -> dict[str, Any]:
    """Read a V4 checkpoint payload without applying it (schema checked only)."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA_V4:
        raise ValueError("unsupported V4 actor-critic checkpoint")
    return payload


def restore_actor_critic_checkpoint(
    payload: dict[str, Any],
    model: UniversalCardPolicyV4,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    grad_scaler: object | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Apply a payload from ``read_actor_critic_checkpoint`` after the contract check.

    ``model_config`` keys absent from the stored contract are compared at
    ``LEGACY_FIELD_DEFAULTS`` or their current ``ModelConfigV4`` defaults.
    ``BEHAVIOURAL_FIELD_NAMES`` are not compared: the model adopts the
    checkpoint's value (so a pre-fix checkpoint keeps the decision-tick elixir
    rule).  Every other contract field, and the state dict, must match exactly.
    """

    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA_V4:
        raise ValueError("unsupported V4 actor-critic checkpoint")
    stored_contract = payload.get("contract")
    if not isinstance(stored_contract, Mapping):
        raise ValueError("checkpoint has no model/catalog contract")
    stored_config = _filled_model_config(stored_contract.get("model_config"))
    behaviour = {name: stored_config[name] for name in BEHAVIOURAL_FIELD_NAMES}
    model_config = asdict(model.config)
    stored = {
        **stored_contract,
        "model_config": {**stored_config, **{name: model_config[name] for name in BEHAVIOURAL_FIELD_NAMES}},
    }
    expected = checkpoint_contract(model)
    if stored != expected:
        differences = _contract_differences(stored, expected)
        raise ValueError(f"checkpoint model/catalog contract does not match: {differences}")
    # Validate the adopted values through the frozen dataclass before use.
    adopted = replace(model.config, **behaviour)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.config = adopted
    model.set_ppo_gate_temperature(float(payload["ppo_gate_temperature"]))
    model.set_ppo_action_temperature(float(payload["ppo_action_temperature"]))
    model.set_ppo_continue_temperature(float(payload.get("ppo_continue_temperature", 1.0)))

    for name, value in (("optimizer", optimizer), ("scheduler", scheduler), ("grad_scaler", grad_scaler)):
        state = payload.get(f"{name}_state_dict")
        if value is not None and state is not None:
            value.load_state_dict(state)
    if restore_rng:
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_state = payload.get("cuda_rng_state_all")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    return payload
