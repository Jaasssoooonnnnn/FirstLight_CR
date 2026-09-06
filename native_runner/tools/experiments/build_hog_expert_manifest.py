"""Select exact Hog-side experts and resolve the original IL cache policy.

Run inside the original CR_AI Python environment. Source outcome is never
replaced by a mismatching reconstructed outcome. Loss quality uses a symmetric
5% final-total-tower-HP gap; unknown HP and draws are not used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

EXACT_DECK = frozenset(
    ("cannon-ev1", "skeletons-ev1", "musketeer-hero", "hog-rider", "ice-golem", "ice-spirit", "fireball", "the-log")
)


def select_sides(payload, tag, *, max_loss_hp_gap=0.05):
    battle = payload.get("battle", {})
    result = battle.get("result")
    if result not in ("victory", "defeat"):
        return []
    players = {}
    for side in ("team", "opponent"):
        p = battle.get(side, {}).get("players", [])
        if len(p) != 1:
            return []
        players[side] = p[0]
    # Cache owners preserve native True Red=0 / True Blue=1, NOT the source
    # team/opponent ordering. Mirror royaleapi_replay.py owner_for_side exactly.
    markers = {
        e.get("source_fields", {}).get("data_i")
        for e in payload.get("events", [])
        if "data_i" in e.get("source_fields", {})
    }
    if len(markers) != 1 or next(iter(markers)) not in (0, 1):
        raise ValueError(f"missing or mixed source data_i: {tag}")
    data_i = next(iter(markers))
    owner_for_side = {"team": 1 - data_i, "opponent": data_i}
    rows = []
    for side in ("team", "opponent"):
        owner = owner_for_side[side]
        player = players[side]
        deck = player.get("deck", [])
        if len(deck) != 8 or frozenset(c.get("card_key") for c in deck) != EXACT_DECK:
            continue
        if player.get("tower_card", {}).get("card_key") != "tower-princess":
            continue
        other = "opponent" if side == "team" else "team"
        won = (result == "victory") == (side == "team")
        hp = player.get("final_tower_hitpoints", {})
        other_hp = players[other].get("final_tower_hitpoints", {})
        own_total, other_total = hp.get("total"), other_hp.get("total")
        gap = None
        if isinstance(own_total, (int, float)) and isinstance(other_total, (int, float)):
            if max(own_total, other_total) > 0:
                gap = abs(own_total - other_total) / max(own_total, other_total)
        if not won and (gap is None or gap > max_loss_hp_gap):
            continue
        ticks = sorted(
            int(e["replay_tick_20hz"])
            for e in payload.get("events", [])
            if e.get("kind") == "play_card" and e.get("side") == side and e.get("card_key") == "hog-rider"
        )
        if not ticks:
            continue
        rows.append(
            dict(
                replay_tag=tag,
                owner=owner,
                source_side=side,
                outcome="win" if won else "close_loss",
                weight=1.0 if won else 0.25,
                final_tower_hp=own_total,
                opponent_final_tower_hp=other_total,
                relative_hp_gap=gap,
                source_hog_ticks=ticks,
                source_duration_seconds=payload.get("replay", {}).get("duration", {}).get("timeline_seconds"),
            )
        )
    return rows


def choose_cache_result(result):
    """Mirror scan_cache_results.py, not every partial replay is trustworthy."""
    if result.get("completed") is True:
        return "completed"
    reason = str(result.get("failure_reason") or "")
    if int(result.get("decision_frame_count") or 0) == 0 or int(result.get("simulated_end_tick") or 0) < 800:
        return "excluded"
    if reason == "battle ended before all expert actions executed" or reason.startswith("terminal fidelity mismatch:"):
        if result.get("first_untrusted_tick") is None:
            raise ValueError("trusted prefix has no failure tick")
        return "trusted_prefix"
    return "excluded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-selection", type=Path)
    args = parser.parse_args()
    import orjson
    import pyarrow.parquet as pq
    from native_runner.training.v4.cache import load_il_cache_index, load_il_cache_manifest

    started = time.monotonic()
    artifacts = args.root / "artifacts"
    audit_dir = artifacts / "v4-il-failure-audit-20260819"
    policy = json.loads((audit_dir / "training-selection.json").read_text())
    excluded = set(Path(policy["excluded_replay_tags"]["path"]).read_text().split())
    replacements = set(Path(policy["replacement_replay_tags"]["path"]).read_text().split())
    split = artifacts / "v4-il-split-240000-10818-20260819"
    tags = {}
    for label, file in (("train", "train-replay-tags.txt"), ("calibration", "calibration-replay-tags.txt")):
        for tag in (split / file).read_text().split():
            if tag in tags:
                raise ValueError("source splits overlap")
            tags[tag] = label
    if set(tags) & excluded:
        raise ValueError("original split unexpectedly contains excluded replays")
    source_rows = []
    if args.source_selection:
        source_rows = json.loads(args.source_selection.read_text())["rows"]
    else:
        seen = set()
        files = sorted((args.root / "datasets").glob("*/replays/*.parquet"))
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=512, columns=["replay_tag", "payload_json"]):
                for tag, raw in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
                    if tag in seen:
                        raise ValueError(f"duplicate source replay: {tag}")
                    seen.add(tag)
                    if tag not in tags:
                        continue
                    for row in select_sides(orjson.loads(raw), tag):
                        source_rows.append(dict(row, source_split=tags[tag]))
            print(json.dumps(dict(event="source_scan", file=str(file), selected_sides=len(source_rows))), flush=True)
        source_path = args.output.with_suffix(".sources.json")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            raise FileExistsError(source_path)
        source_path.write_text(
            json.dumps(dict(rows=source_rows, unique_source_replays=len(seen)), sort_keys=True) + "\n"
        )
    recovery_index = Path(policy["replacement_cache_root"]) / "index.json"
    missing_recovery_rows = []
    if not recovery_index.is_file():
        # A missing corrected copy NEVER permits falling back to its known-bad
        # original. This is an explicit quality exclusion, recorded in output.
        missing_recovery_rows = [r for r in source_rows if r["replay_tag"] in replacements]
        source_rows = [r for r in source_rows if r["replay_tag"] not in replacements]
        print(
            json.dumps(
                dict(
                    event="missing_recovery_excluded",
                    index=str(recovery_index),
                    excluded_sides=len(missing_recovery_rows),
                )
            ),
            flush=True,
        )
    wanted = {(r["replay_tag"], r["owner"]): r for r in source_rows}
    if len(wanted) != len(source_rows):
        raise ValueError("duplicate source-side selection")
    indexes = sorted(Path(policy["primary_cache_root"]).glob("part-*/index.json"))
    if recovery_index.is_file():
        indexes.append(recovery_index)
    tasks = []
    for index in indexes:
        for summary in load_il_cache_index(index)["shards"]:
            tasks.append((index.parent / summary["summary_file"], index == recovery_index))

    def inspect(task):
        path, replacement = task
        manifest = load_il_cache_manifest(path)
        results = {r["replay_tag"]: r for r in manifest["results"]}
        found = []
        for seq_index, seq in enumerate(manifest["sequences"]):
            tag, owner = seq["replay_tag"], seq["owner"]
            if (tag, owner) not in wanted or ((tag in replacements) != replacement):
                continue
            result = results[tag]
            status = choose_cache_result(result)
            if status == "excluded":
                raise ValueError(f"selected expert violates original cache policy: {tag}")
            found.append(
                dict(
                    wanted[(tag, owner)],
                    summary_path=str(path),
                    sequence_index=seq_index,
                    sequence_length=seq["length"],
                    cache_status=status,
                    replacement_cache=replacement,
                    first_untrusted_tick=result.get("first_untrusted_tick"),
                    simulated_end_tick=result.get("simulated_end_tick"),
                )
            )
        return found

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, found in enumerate(executor.map(inspect, tasks)):
            rows.extend(found)
            if i % 2000 == 0:
                print(
                    json.dumps(dict(event="cache_index", scanned=i, total=len(tasks), selected_sides=len(rows))),
                    flush=True,
                )
    keys = [(r["replay_tag"], r["owner"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(wanted):
        raise ValueError(f"cache resolution mismatch: {len(keys)} != {len(wanted)}")
    rows.sort(key=lambda r: (r["replay_tag"], r["owner"]))
    summary = dict(
        selected_sides=len(rows),
        selected_games=len({r["replay_tag"] for r in rows}),
        outcomes=dict(Counter(r["outcome"] for r in rows)),
        splits=dict(Counter(r["source_split"] for r in rows)),
        cache_status=dict(Counter(r["cache_status"] for r in rows)),
        replacements=sum(r["replacement_cache"] for r in rows),
        missing_recovery_excluded_sides=len(missing_recovery_rows),
        source_mean_hogs=statistics.mean(len(r["source_hog_ticks"]) for r in rows),
        source_first_hog_median_seconds=statistics.median(r["source_hog_ticks"][0] / 20 for r in rows),
        seconds=time.monotonic() - started,
    )
    payload = dict(
        schema="v4-hog-expert-selection.v1",
        rows=rows,
        summary=summary,
        exact_deck=sorted(EXACT_DECK),
        tower="tower-princess",
        close_loss_hp_gap=0.05,
        original_selection_policy=policy,
        missing_recovery_excluded_keys=[(r["replay_tag"], r["owner"]) for r in missing_recovery_rows],
        calibration_used_for_training=True,
        discard_tail_ticks=200,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(payload, sort_keys=True) + "\n")
    print(
        json.dumps(
            dict(
                event="selection_complete",
                output=str(args.output),
                sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                **summary,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
