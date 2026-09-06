"""Export a model without optimizer state, training paths or experiment metadata."""

import argparse
from pathlib import Path

import torch


def export(source: Path, destination: Path):
    payload = torch.load(source, map_location="cpu", weights_only=True)
    required = (
        "schema", "checkpoint_id", "contract", "model_state_dict", "update_step", "training_stage",
        "gamma_per_decision", "gae_lambda", "ppo_gate_temperature", "ppo_action_temperature",
        "torch_rng_state",
    )
    result = {key: payload[key] for key in required}
    result.update(
        ppo_continue_temperature=payload.get("ppo_continue_temperature", 1.0),
        optimizer_state_dict=None, scheduler_state_dict=None, grad_scaler_state_dict=None,
        cuda_rng_state_all=None, extra={},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(result, temporary)
    restored = torch.load(temporary, map_location="cpu", weights_only=True)
    assert restored["contract"] == payload["contract"]
    assert restored["model_state_dict"].keys() == payload["model_state_dict"].keys()
    for key, value in payload["model_state_dict"].items():
        assert torch.equal(restored["model_state_dict"][key], value), key
    temporary.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
