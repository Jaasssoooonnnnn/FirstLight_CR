"""Create a synthetic replay, Parquet dataset and deck pool for workflow checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def demo_payload():
    keys = ("hog-rider", "musketeer", "cannon", "fireball", "the-log", "skeletons", "ice-golem", "ice-spirit")
    battle = {"requested_player_tag": "0000", "result": "draw"}
    for side, tag in (("team", "0000"), ("opponent", "0001")):
        battle[side] = {
            "crowns": 0,
            "players": [{
                "name": "Synthetic " + side, "tag": tag,
                "deck": [{"card_key": key, "name": key.replace("-", " ").title(), "level": 11} for key in keys],
                "tower_card": {"card_key": "tower-princess", "level": 11},
            }],
        }
    events = []
    for index, (tick, side, card) in enumerate(((140, "team", "hog-rider"), (180, "opponent", "hog-rider"),
                                               (400, "team", "skeletons"), (440, "opponent", "skeletons"))):
        events.append({
            "source_index": index, "replay_tick_20hz": tick, "side": side,
            "kind": "play_card", "card_key": card, "source_fields": {"data_i": 1},
            "coordinates": {
                "inside_18x32_arena": True,
                "grid_cell_floor": {"x": 3, "y": 10 if side == "team" else 21},
                "subcell_offset_from_floor_center": {"x": 0.0, "y": 0.0},
            },
        })
    return {
        "schema_version": "royaleapi-battle-replay.v1",
        "source": {"replay_tag": "0000", "synthetic": True},
        "battle": battle, "events": events,
        "replay": {"duration": {"timeline_seconds": 45}},
    }


def record_native_replay(directory):
    from native_runner.battle_env import BattleEnvV1
    from native_runner.contracts import ActionV1
    from native_runner.match_factory import MatchConfig
    from native_runner.training.replay_archive import capture_training_replay, save_training_replay

    environment = BattleEnvV1()
    try:
        observations, _ = environment.reset(match_config=MatchConfig(seed=1), options={
            "render_mode": "headless", "decision_ticks": 20, "event_driven_decisions": False,
            "allow_initial_remaining_runtime_rejections": True,
        })
        rounds = 0
        while True:
            commands = {owner: (ActionV1.wait(owner, ticks=20),) for owner in (0, 1)}
            if rounds in (3, 18):
                for owner, observation in observations.items():
                    player = next(p for p in environment.raw_observation["players"] if p["owner"] == owner)
                    card = next(c for c in player["hand"] if observation.action_mask.hand_slots[c["handIndex"]]
                                and 26000000 <= c["cardId"] < 27000000)
                    mask = observation.action_mask.placement_masks[str(card["handIndex"])]["row_major"]
                    cells = ((x, y) for y, row in enumerate(mask) for x, legal in enumerate(row) if legal)
                    target = min(cells, key=lambda cell: (cell[0] - 3) ** 2 + (cell[1] - 16) ** 2)
                    commands[owner] = (ActionV1.play(owner, card["handIndex"], target, card_id=card["cardId"]),)
            observations, _, terms, truncs, _ = environment.step(commands)
            if any(event.event_type == "action_rejected" for obs in observations.values() for event in obs.events):
                raise RuntimeError("Synthetic native action was rejected")
            rounds += 1
            if terms["__all__"] or truncs["__all__"]:
                break
        replay = capture_training_replay(environment, completed_match_index=1, worker_id=0, global_slot=0,
                                        local_slot=0, final_observation=observations[0])
        path = save_training_replay(replay, directory)
        print(f"Recorded native replay: {path}; final tick {replay.end_native_tick}", flush=True)
    finally:
        environment.native.close_transport()


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from native_runner.training.v4.deck_pool import build_deck_pool_manifest_from_counts, deck_counts_from_payloads

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--native", action="store_true", help="also record a complete native match; requires an idle offline engine")
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise FileExistsError(f"Choose a new demo output directory: {root}")
    replay_dir = root / "dataset/replays"
    replay_dir.mkdir(parents=True)
    payload = demo_payload()
    (root / "payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{
        "replay_tag": "0000", "requested_player_tag": "0000",
        "payload_json": json.dumps(payload),
    }]), replay_dir / "part-00000.parquet")
    (root / "train-tags.txt").write_text("0000\n", encoding="utf-8")
    if args.native:
        record_native_replay(root / "training_replays")
    counts, _ = deck_counts_from_payloads((("0000", payload),))
    build_deck_pool_manifest_from_counts(counts, minimum_uses=1, metadata={"synthetic": True}).save(root / "deck-pool.json")
    print(f"Synthetic dataset written to {root}. For workflow verification, not model-quality evaluation.")


if __name__ == "__main__":
    main()
