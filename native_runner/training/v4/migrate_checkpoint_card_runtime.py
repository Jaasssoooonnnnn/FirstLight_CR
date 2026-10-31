"""Migrate V4 actor-critic checkpoints across a card-runtime feature append.

Older checkpoints were trained with a smaller ``card_runtime_feature_dim``.
The current own-card runtime features only APPEND new columns to
``CARD_RUNTIME_FEATURE_NAMES``, so the first Linear layer of the card-token
runtime MLP can be migrated by appending zero input columns: the migrated model
computes the same function as the old one and ignores the appended features
until it is retrained.

Every other contract field, catalog identity, state-dict key, and tensor shape
must match the current model exactly; anything else raises ``ValueError``.
Optimizer, scheduler, and grad-scaler state are intentionally dropped: migrated
checkpoints are evaluation references, not resume points.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import (
    CHECKPOINT_SCHEMA_V4,
    checkpoint_contract,
    load_actor_critic_checkpoint,
    save_actor_critic_checkpoint,
)
from .model import UniversalCardPolicyV4


PADDED_TENSOR_KEY = "card_token_encoder.runtime.0.weight"
MIGRATED_CONFIG_FIELD = "card_runtime_feature_dim"


def _contract_differences(old: Any, new: Any, path: str = "") -> list[str]:
    """Return human-readable paths at which two contract trees differ."""

    if isinstance(old, Mapping) and isinstance(new, Mapping):
        differences: list[str] = []
        for key in sorted(set(old) | set(new), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in old:
                differences.append(f"{child}: missing in source (current={new[key]!r})")
            elif key not in new:
                differences.append(f"{child}: absent in current model (source={old[key]!r})")
            else:
                differences.extend(_contract_differences(old[key], new[key], child))
        return differences
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        if type(old) is not type(new):
            return [f"{path}: container type {type(old).__name__} != {type(new).__name__}"]
        if len(old) != len(new):
            return [f"{path}: length {len(old)} != {len(new)} (source={old!r}, current={new!r})"]
        differences = []
        for index, (left, right) in enumerate(zip(old, new)):
            differences.extend(_contract_differences(left, right, f"{path}[{index}]"))
        return differences
    if type(old) is not type(new) or old != new:
        return [f"{path}: source={old!r} current={new!r}"]
    return []


def _validate_payload(payload: Any, model: UniversalCardPolicyV4) -> tuple[int, int]:
    """Validate the source payload and return (old_dim, new_dim)."""

    if not isinstance(payload, dict):
        raise ValueError(f"source checkpoint payload is {type(payload).__name__}, expected dict")
    if payload.get("schema") != CHECKPOINT_SCHEMA_V4:
        raise ValueError(
            f"unsupported checkpoint schema {payload.get('schema')!r}; expected {CHECKPOINT_SCHEMA_V4!r}"
        )
    old_contract = payload.get("contract")
    if not isinstance(old_contract, dict):
        raise ValueError("source checkpoint has no contract mapping")
    new_contract = checkpoint_contract(model)

    old_catalogs = old_contract.get("catalog_ids")
    if old_catalogs != new_contract["catalog_ids"]:
        raise ValueError(
            "catalog_ids differ between source checkpoint and current model: "
            + "; ".join(_contract_differences(old_catalogs, new_contract["catalog_ids"], "catalog_ids"))
        )

    old_config = old_contract.get("model_config")
    if not isinstance(old_config, dict) or MIGRATED_CONFIG_FIELD not in old_config:
        raise ValueError(f"source contract has no model_config.{MIGRATED_CONFIG_FIELD}")
    old_dim = old_config[MIGRATED_CONFIG_FIELD]
    new_dim = new_contract["model_config"][MIGRATED_CONFIG_FIELD]
    if not isinstance(old_dim, int) or isinstance(old_dim, bool):
        raise ValueError(f"source model_config.{MIGRATED_CONFIG_FIELD} is not an int: {old_dim!r}")
    if old_dim >= new_dim:
        raise ValueError(
            f"source {MIGRATED_CONFIG_FIELD}={old_dim} is not smaller than current {new_dim}; "
            "nothing to zero-pad (load the checkpoint directly instead)"
        )

    patched_contract = copy.deepcopy(old_contract)
    patched_contract["model_config"][MIGRATED_CONFIG_FIELD] = new_dim
    differences = _contract_differences(patched_contract, new_contract)
    if differences or patched_contract != new_contract:
        raise ValueError(
            f"source contract differs from current model beyond model_config.{MIGRATED_CONFIG_FIELD}: "
            + ("; ".join(differences) or "(equality check failed)")
        )

    old_state = payload.get("model_state_dict")
    if not isinstance(old_state, Mapping):
        raise ValueError("source checkpoint has no model_state_dict mapping")
    new_state = model.state_dict()
    missing = sorted(set(new_state) - set(old_state))
    unexpected = sorted(set(old_state) - set(new_state))
    if missing or unexpected:
        raise ValueError(f"state_dict key sets differ: missing={missing} unexpected={unexpected}")

    shape_differences = []
    for key, current in new_state.items():
        source = old_state[key]
        if not isinstance(source, torch.Tensor):
            raise ValueError(f"source state_dict[{key!r}] is {type(source).__name__}, expected Tensor")
        if key == PADDED_TENSOR_KEY:
            continue
        if tuple(source.shape) != tuple(current.shape):
            shape_differences.append(f"{key}: source={tuple(source.shape)} current={tuple(current.shape)}")
    if shape_differences:
        raise ValueError("unexpected tensor shape differences: " + "; ".join(shape_differences))

    source_shape = tuple(old_state[PADDED_TENSOR_KEY].shape)
    current_shape = tuple(new_state[PADDED_TENSOR_KEY].shape)
    if current_shape != (current_shape[0], new_dim) or source_shape != (current_shape[0], old_dim):
        raise ValueError(
            f"{PADDED_TENSOR_KEY}: source shape {source_shape} / current shape {current_shape} "
            f"do not match a {old_dim}->{new_dim} input-column append"
        )
    return old_dim, new_dim


def migrate_checkpoint(
    source: Path,
    destination: Path,
    *,
    model: UniversalCardPolicyV4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Zero-pad an old checkpoint's card-runtime input weight and save it anew.

    ``model`` must be a current-config model with the checkpoint's catalogs; its
    weights and temperatures are replaced by the migrated ones.
    """

    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("destination must differ from source")
    if destination.exists() and not overwrite:
        raise ValueError(f"destination already exists: {destination}")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    old_dim, new_dim = _validate_payload(payload, model)

    old_state = payload["model_state_dict"]
    old_weight = old_state[PADDED_TENSOR_KEY]
    padding = torch.zeros((old_weight.shape[0], new_dim - old_dim), dtype=old_weight.dtype)
    padded_weight = torch.cat((old_weight, padding), dim=1).contiguous()
    migrated_state = dict(old_state)
    migrated_state[PADDED_TENSOR_KEY] = padded_weight
    model.load_state_dict(migrated_state, strict=True)

    gate = float(payload["ppo_gate_temperature"])
    action = float(payload["ppo_action_temperature"])
    continue_temperature = float(payload.get("ppo_continue_temperature", 1.0))
    model.set_ppo_gate_temperature(gate)
    model.set_ppo_action_temperature(action)
    model.set_ppo_continue_temperature(continue_temperature)

    padded_tensors = [PADDED_TENSOR_KEY]
    extra = dict(payload.get("extra") or {})
    extra.update(
        {
            "migrated_from": str(source),
            "migrated_from_checkpoint_id": payload.get("checkpoint_id"),
            "migration": f"{MIGRATED_CONFIG_FIELD} {old_dim}->{new_dim} zero-pad",
            "migration_zero_padded_tensors": list(padded_tensors),
        }
    )
    gae_lambda = payload.get("gae_lambda")
    checkpoint_id = save_actor_critic_checkpoint(
        destination,
        model,
        optimizer=None,
        scheduler=None,
        grad_scaler=None,
        update_step=int(payload["update_step"]),
        training_stage=str(payload["training_stage"]),
        gamma_per_decision=float(payload["gamma_per_decision"]),
        gae_lambda=None if gae_lambda is None else float(gae_lambda),
        extra=extra,
    )

    # Verify: strict reload through the production loader, then compare tensors.
    reloaded = load_actor_critic_checkpoint(destination, model, map_location="cpu")
    if reloaded["checkpoint_id"] != checkpoint_id:
        raise ValueError("reloaded checkpoint id does not match the written one")
    loaded_weight = model.card_token_encoder.runtime[0].weight.detach().cpu()
    if loaded_weight.dtype != old_weight.dtype:
        raise ValueError(f"migrated weight dtype {loaded_weight.dtype} != source dtype {old_weight.dtype}")
    if not torch.equal(loaded_weight[:, :old_dim], old_weight):
        raise ValueError("migrated weight's original columns differ from the source weight")
    if torch.count_nonzero(loaded_weight[:, old_dim:]).item() != 0:
        raise ValueError("migrated weight's appended columns are not all zero")
    loaded_state = model.state_dict()
    for key, value in old_state.items():
        if key != PADDED_TENSOR_KEY and not torch.equal(loaded_state[key].detach().cpu(), value):
            raise ValueError(f"migrated tensor {key} differs from the source")
    if (
        model.ppo_gate_temperature != gate
        or model.ppo_action_temperature != action
        or model.ppo_continue_temperature != continue_temperature
    ):
        raise ValueError("reloaded temperatures differ from the source checkpoint")

    return {
        "source": str(source),
        "destination": str(destination),
        "source_checkpoint_id": payload.get("checkpoint_id"),
        "checkpoint_id": checkpoint_id,
        "migration": extra["migration"],
        "padded_tensors": padded_tensors,
        "padded_weight_shape": {"source": list(old_weight.shape), "migrated": list(padded_weight.shape)},
        "dtype": str(old_weight.dtype),
        "update_step": int(payload["update_step"]),
        "training_stage": str(payload["training_stage"]),
        "gamma_per_decision": float(payload["gamma_per_decision"]),
        "gae_lambda": None if gae_lambda is None else float(gae_lambda),
        "ppo_gate_temperature": gate,
        "ppo_action_temperature": action,
        "ppo_continue_temperature": continue_temperature,
        "dropped_state": [
            name
            for name in ("optimizer_state_dict", "scheduler_state_dict", "grad_scaler_state_dict")
            if payload.get(name) is not None
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zero-pad a V4 checkpoint's card_runtime_feature_dim.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing destination file")
    args = parser.parse_args(argv)

    from .factory import build_production_model_v4

    model = build_production_model_v4(device="cpu")
    summary = migrate_checkpoint(args.source, args.destination, model=model, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
