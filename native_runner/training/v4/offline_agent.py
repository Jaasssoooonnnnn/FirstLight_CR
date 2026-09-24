"""Run one selected V4 model against a human in a controlled offline renderer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
from typing import Callable

import torch

from ...battle_env import BattleEnvV1
from ...contracts import ActionKind, ActionV1
from ...cr_native_env import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    NATIVE_RENDER_SPEEDS,
    NativeClashEnv,
    RunnerError,
)
from ...match_factory import MatchConfig
from ...rich_telemetry_adapter import RuntimeEffectCatalog
from .expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
from .factory import production_semantic_bundle
from .policy_session import build_policy_session_v4, load_policy_v4


def _rendered_action(action: ActionV1) -> ActionV1:
    if action.kind != ActionKind.PLAY_CARD:
        return action
    # The native queue backdates command-age fields to apply on the next tick.
    # Preserve the model's predicted delay only for diagnosis.
    metadata = dict(action.metadata)
    metadata.pop("native_command_age_ticks", None)
    metadata.update(execute_offset_ticks=1, model_delay_ignored=True)
    return replace(
        action,
        execute_offset_ticks=1,
        metadata=metadata,
    )


def run_offline_match(
    checkpoint: str | Path,
    match: MatchConfig,
    *,
    actor_owner: int,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    device: str = "cpu",
    sample: bool = True,
    speed: float = 1.0,
    artifact_dir: Path,
    should_stop: Callable[[], bool] = lambda: False,
    on_ready: Callable[[dict], None] = lambda _info: None,
) -> dict:
    if actor_owner not in (0, 1) or speed not in NATIVE_RENDER_SPEEDS:
        raise ValueError("model owner must be 0 or 1 and speed must be a supported native render speed")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_policy_v4(checkpoint, device=device)
    if should_stop():
        result = {"stopped": True, "terminated": False, "decisions": 0}
        (artifact_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return result
    bundle = production_semantic_bundle()
    native = NativeClashEnv(host, port, timeout=30.0)
    native.public_card_play_events_from_combat_ring = True
    environment = BattleEnvV1(
        native=native,
        card_catalog=bundle.native_card_catalog,
        semantic_subset_contract=bundle.semantic_subset_contract,
        effect_catalog=RuntimeEffectCatalog.from_catalog(bundle.native_effect_catalog),
    )
    session = None
    decisions = actions = rejected = 0
    terminated = truncated = False
    result = {}
    rejected_events = set()
    try:
        observations, info = environment.reset(
            match_config=match,
            options={
                "render_mode": "native-render",
                "pause_native_render": True,
                "allow_initial_remaining_runtime_rejections": True,
                "decision_ticks": POLICY_DECISION_TICKS,
                "event_driven_decisions": False,
            },
        )
        session = build_policy_session_v4(
            environment, loaded.model, actor_owner=actor_owner, device=device, sample=sample
        )
        initial_elixir = {
            owner: next(p.elixir_exact for p in obs.players if p.owner == owner) for owner, obs in observations.items()
        }
        session.start_episode(observations[actor_owner], initial_elixir=initial_elixir)
        identity = {
            "mode": "native-render",
            "checkpoint": str(loaded.checkpoint_path),
            "checkpoint_id": loaded.checkpoint_id,
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "actor_owner": actor_owner,
            "human_owner": 1 - actor_owner,
            "match": asdict(match),
        }
        (artifact_dir / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        native.set_speed(speed)
        native.resume()
        on_ready(identity)
        with (artifact_dir / "decisions.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            while not should_stop() and not terminated and not truncated:
                observation = observations[actor_owner]
                commands = {owner: (ActionV1.wait(owner, ticks=POLICY_DECISION_TICKS),) for owner in (0, 1)}
                if observation.tick >= FIRST_POLICY_DECISION_TICK:
                    decision = session.decide(observation)
                    commands[actor_owner] = (
                        tuple(_rendered_action(a) for a in decision.decoded.actions) or commands[actor_owner]
                    )
                    decisions += 1
                    actions += sum(a.kind != ActionKind.WAIT for a in decision.decoded.actions)
                    log.write(
                        json.dumps(
                            {
                                "tick": observation.tick,
                                "owner": actor_owner,
                                "inference_ms": decision.inference_ms,
                                "actions": [a.to_dict() for a in commands[actor_owner]],
                            }
                        )
                        + "\n"
                    )
                else:
                    session.tensorizer.tensorize(observation, validate=False)
                observations, rewards, terms, truncs, info = environment.step(commands, record_trace=False)
                terminated = bool(terms.get("__all__", all(terms.values())))
                truncated = bool(truncs.get("__all__", all(truncs.values())))
                for event in observations[actor_owner].events:
                    if event.event_type == "action_rejected" and event.owner == actor_owner:
                        rejected_events.add(json.dumps(event.to_dict(), sort_keys=True))
                rejected = len(rejected_events)
                result = {"tick": observations[actor_owner].tick, "rewards": rewards, "info": info}
        result.update(
            terminated=terminated,
            truncated=truncated,
            stopped=should_stop(),
            decisions=decisions,
            model_actions=actions,
            rejected_actions=rejected,
        )
        (artifact_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if session is not None:
            session.end_episode()
        try:
            native.pause()
        except RunnerError:
            pass
        finally:
            native.close_transport()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--match-config", type=Path, required=True)
    parser.add_argument("--actor-owner", type=int, choices=(0, 1), default=0)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.match_config.read_text(encoding="utf-8"))
    for key in ("deck0", "deck1", "deck0_form_availability", "deck1_form_availability"):
        config[key] = tuple(config[key])

    def ready(identity):
        (args.artifact_dir / "ready.json").write_text(json.dumps(identity), encoding="utf-8")

    result = run_offline_match(
        args.checkpoint,
        MatchConfig(**{f.name: config[f.name] for f in fields(MatchConfig) if f.init and f.name in config}),
        actor_owner=args.actor_owner,
        host=args.host,
        port=args.port,
        device=args.device,
        sample=not args.deterministic,
        speed=args.speed,
        artifact_dir=args.artifact_dir,
        should_stop=lambda: (args.artifact_dir / "stop").exists(),
        on_ready=ready,
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
