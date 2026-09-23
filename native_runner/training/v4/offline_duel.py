"""Run two V4 policies against each other in one offline native renderer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path

import torch

from ...battle_env import BattleEnvV1
from ...contracts import ActionKind, ActionV1
from ...cr_native_env import DEFAULT_HOST, DEFAULT_PORT, NATIVE_RENDER_SPEEDS, NativeClashEnv, RunnerError
from ...match_factory import MatchConfig
from ...rich_telemetry_adapter import RuntimeEffectCatalog
from .expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
from .factory import production_semantic_bundle
from .offline_agent import _rendered_action
from .policy_session import build_policy_session_v4, load_policy_v4


def run_ai_duel(
    checkpoints: tuple[str | Path, str | Path],
    match: MatchConfig,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    device: str = "cpu",
    speed: float = 1.0,
    artifact_dir: Path,
) -> dict:
    if speed not in NATIVE_RENDER_SPEEDS:
        raise ValueError("unsupported native render speed")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    first = load_policy_v4(checkpoints[0], device=device)
    second = (
        first
        if Path(checkpoints[0]).resolve() == Path(checkpoints[1]).resolve()
        else load_policy_v4(checkpoints[1], device=device)
    )
    loaded = (first, second)
    if (artifact_dir / "stop").exists():
        result = {"stopped": True, "terminated": False, "decisions": [0, 0]}
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
    sessions = []
    decisions = [0, 0]
    actions = [0, 0]
    forced_actions = [0, 0]
    terminated = truncated = False
    result: dict = {}
    try:
        observations, _info = environment.reset(
            match_config=match,
            options={
                "render_mode": "native-render",
                "pause_native_render": True,
                "allow_initial_remaining_runtime_rejections": True,
                "decision_ticks": POLICY_DECISION_TICKS,
                "event_driven_decisions": False,
            },
        )
        initial_elixir = {
            owner: next(p.elixir_exact for p in observations[owner].players if p.owner == owner) for owner in (0, 1)
        }
        for owner in (0, 1):
            session = build_policy_session_v4(
                environment, loaded[owner].model, actor_owner=owner, device=device, sample=False
            )
            sessions.append(session)
            session.start_episode(observations[owner], initial_elixir=initial_elixir)
        identity = {
            "mode": "native-render-ai-duel",
            "checkpoints": [
                {"path": str(policy.checkpoint_path), "id": policy.checkpoint_id, "sha256": policy.checkpoint_sha256}
                for policy in loaded
            ],
            "match": asdict(match),
        }
        (artifact_dir / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        native.set_speed(speed)
        native.resume()
        (artifact_dir / "ready.json").write_text(json.dumps(identity), encoding="utf-8")
        with (artifact_dir / "decisions.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            while not (artifact_dir / "stop").exists() and not terminated and not truncated:
                commands = {owner: (ActionV1.wait(owner, ticks=POLICY_DECISION_TICKS),) for owner in (0, 1)}
                for owner in (0, 1):
                    observation = observations[owner]
                    force_path = artifact_dir / f"force-{owner}"
                    if observation.tick < FIRST_POLICY_DECISION_TICK:
                        sessions[owner].tensorizer.tensorize(observation, validate=False)
                        continue
                    forced = force_path.exists()
                    if forced:
                        force_path.unlink(missing_ok=True)
                    decision = sessions[owner].decide(observation, force_act=forced)
                    selected = tuple(_rendered_action(action) for action in decision.decoded.actions)
                    commands[owner] = selected or commands[owner]
                    count = sum(action.kind == ActionKind.PLAY_CARD for action in selected)
                    decisions[owner] += 1
                    actions[owner] += count
                    if forced:
                        forced_actions[owner] += count
                    log.write(
                        json.dumps(
                            {
                                "tick": observation.tick,
                                "owner": owner,
                                "forced": forced,
                                "inference_ms": decision.inference_ms,
                                "actions": [action.to_dict() for action in commands[owner]],
                            }
                        )
                        + "\n"
                    )
                observations, rewards, terms, truncs, info = environment.step(commands, record_trace=False)
                terminated = bool(terms.get("__all__", all(terms.values())))
                truncated = bool(truncs.get("__all__", all(truncs.values())))
                result = {"tick": observations[0].tick, "rewards": rewards, "info": info}
        result.update(
            terminated=terminated,
            truncated=truncated,
            stopped=(artifact_dir / "stop").exists(),
            decisions=decisions,
            model_actions=actions,
            forced_plays=forced_actions,
        )
        (artifact_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        for session in sessions:
            session.end_episode()
        try:
            native.pause()
        except RunnerError:
            pass
        finally:
            native.close_transport()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-0", required=True)
    parser.add_argument("--checkpoint-1", required=True)
    parser.add_argument("--match-config", type=Path, required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.match_config.read_text(encoding="utf-8"))
    for key in ("deck0", "deck1", "deck0_form_availability", "deck1_form_availability"):
        config[key] = tuple(config[key])
    result = run_ai_duel(
        (args.checkpoint_0, args.checkpoint_1),
        MatchConfig(
            **{field.name: config[field.name] for field in fields(MatchConfig) if field.init and field.name in config}
        ),
        host=args.host,
        port=args.port,
        device=args.device,
        speed=args.speed,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
