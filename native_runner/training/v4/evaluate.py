"""Evaluate two checkpoints in complete, sequential offline native matches."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from ...battle_env import BattleEnvV1
from ...contracts import ActionKind, ActionV1
from ...cr_native_env import DEFAULT_HOST, DEFAULT_PORT, NativeClashEnv
from ...match_factory import MatchConfig
from ...rich_telemetry_adapter import RuntimeEffectCatalog
from .expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
from .factory import production_semantic_bundle
from .policy_session import build_policy_session_v4, load_policy_v4


def evaluate(
    checkpoint_a,
    checkpoint_b,
    *,
    games,
    match,
    output,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    device="cpu",
    sample=False,
):
    if games < 1:
        raise ValueError("games must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policies = [load_policy_v4(path, device=device) for path in (checkpoint_a, checkpoint_b)]
    bundle = production_semantic_bundle()
    native = NativeClashEnv(host, port, timeout=30.0)
    native.public_card_play_events_from_combat_ring = True
    environment = BattleEnvV1(
        native=native,
        card_catalog=bundle.native_card_catalog,
        semantic_subset_contract=bundle.semantic_subset_contract,
        effect_catalog=RuntimeEffectCatalog.from_catalog(bundle.native_effect_catalog),
    )
    identity = {
        "games": games,
        "sample": sample,
        "match": asdict(match),
        "checkpoints": [
            {"path": str(p.checkpoint_path), "sha256": p.checkpoint_sha256, "checkpoint_id": p.checkpoint_id}
            for p in policies
        ],
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    rows = []
    sessions = []
    try:
        with (output / "matches.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for index in range(games):
                a_owner = index % 2
                current = replace(match, seed=match.seed + index)
                if a_owner:
                    current = replace(
                        current,
                        deck0=match.deck1,
                        deck1=match.deck0,
                        deck0_form_availability=match.deck1_form_availability,
                        deck1_form_availability=match.deck0_form_availability,
                        tower_troop0_id=match.tower_troop1_id,
                        tower_troop1_id=match.tower_troop0_id,
                    )
                torch.manual_seed(current.seed)
                observations, _ = environment.reset(
                    match_config=current,
                    options={
                        "render_mode": "headless",
                        "decision_ticks": POLICY_DECISION_TICKS,
                        "event_driven_decisions": False,
                        "allow_initial_remaining_runtime_rejections": True,
                    },
                )
                initial_elixir = {
                    owner: next(p.elixir_exact for p in obs.players if p.owner == owner)
                    for owner, obs in observations.items()
                }
                sessions = [
                    build_policy_session_v4(
                        environment,
                        policies[0 if owner == a_owner else 1].model,
                        actor_owner=owner,
                        device=device,
                        sample=sample,
                    )
                    for owner in (0, 1)
                ]
                for owner, session in enumerate(sessions):
                    session.start_episode(observations[owner], initial_elixir=initial_elixir)
                actions = [0, 0]
                rejected = set()
                terminated = truncated = False
                while not terminated and not truncated:
                    commands = {}
                    for owner, session in enumerate(sessions):
                        observation = observations[owner]
                        chosen = ()
                        if observation.tick >= FIRST_POLICY_DECISION_TICK:
                            chosen = session.decide(observation).decoded.actions
                            actions[owner] += sum(a.kind != ActionKind.WAIT for a in chosen)
                        else:
                            session.tensorizer.tensorize(observation, validate=False)
                        commands[owner] = tuple(chosen) or (ActionV1.wait(owner, ticks=POLICY_DECISION_TICKS),)
                    observations, _, terms, truncs, _ = environment.step(commands, record_trace=False)
                    terminated, truncated = bool(terms["__all__"]), bool(truncs["__all__"])
                    for observation in observations.values():
                        for event in observation.events:
                            if event.event_type == "action_rejected":
                                rejected.add(json.dumps(event.to_dict(), sort_keys=True))
                terminal = observations[0].terminal
                winner = None if terminal.winner is None else ("a" if terminal.winner == a_owner else "b")
                row = {
                    "game": index + 1,
                    "seed": current.seed,
                    "a_owner": a_owner,
                    "winner": winner,
                    "terminated": terminated,
                    "truncated": truncated,
                    "tick": observations[0].tick,
                    "terminal": terminal.to_dict(),
                    "actions_by_owner": actions,
                    "rejected_actions": len(rejected),
                }
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                for session in sessions:
                    session.end_episode()
                sessions = []
        result = {
            "requested": games,
            "completed": len(rows),
            "a_wins": sum(r["winner"] == "a" for r in rows),
            "b_wins": sum(r["winner"] == "b" for r in rows),
            "draws": sum(r["terminated"] and r["winner"] is None for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        for session in sessions:
            session.end_episode()
        native.close_transport()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--match-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    config = json.loads(args.match_config.read_text(encoding="utf-8")) if args.match_config else {}
    for key in ("owner0_account_id", "owner1_account_id"):
        config.pop(key, None)
    config["seed"] = args.seed
    torch.set_num_threads(1)
    evaluate(
        args.checkpoint_a,
        args.checkpoint_b,
        games=args.games,
        match=MatchConfig(**config),
        output=args.output,
        host=args.host,
        port=args.port,
        device=args.device,
        sample=args.sample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
