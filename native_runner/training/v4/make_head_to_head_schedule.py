"""Generate a model-blind, seat-swapped head-to-head schedule from a deck pool.

Every configuration (seed, deck0, deck1, battle level, category) is emitted
twice.  Decks stay with seats; the two models swap seats: row ``2k`` puts
model slot ``A`` in seat 0 and row ``2k + 1`` puts slot ``B`` in seat 0.  The
schedule never names a model, so every model pair evaluated against it plays
the identical configurations (common random numbers).

Deck categories come from ``DeckMatchmakerV4`` with the adversarial ``hard``
bucket disabled (weight 0); its 10% share moves to ``focus-tail-weighted`` by
default.  ``--battle-level mixed`` (default) reproduces the matchmaker's exact
shuffled 50/50 split of levels 11 and 16; ``--battle-level 11`` or ``16`` pins
every configuration to one level.  The output is a pure function of the deck
pool and the arguments (``--seed`` drives the matchmaker RNG).

Usage:
    python -m native_runner.training.v4.make_head_to_head_schedule \
        --deck-pool v4-ppo-deck-pool-use5.json --output h2h/schedule.jsonl \
        --configurations 200 --seed 20261023 [--battle-level mixed|11|16] \
        [--deck-mix 30,30,30,10] [--allow-mirror]
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .deck_pool import DeckPoolManifestV4
from .matchmaking import BATTLE_LEVELS, DeckMatchmakerV4, MatchmakerConfigV4


HEAD_TO_HEAD_SCHEDULE_SCHEMA_V4 = "v4-head-to-head-schedule.v1"
SEAT_SLOTS = ("A", "B")
_MAX_MIRROR_REDRAWS = 1_000


@dataclass(frozen=True, slots=True)
class HeadToHeadScheduleConfigV4:
    configurations: int
    seed: int
    battle_level: str = "mixed"
    deck_mix: tuple[int, int, int, int] = (30, 30, 30, 10)
    allow_mirror: bool = False

    def __post_init__(self) -> None:
        if self.configurations <= 0:
            raise ValueError("schedule needs at least one configuration")
        if self.seed <= 0:
            raise ValueError("schedule seed must be positive")
        if self.battle_level not in {"mixed", *(str(level) for level in BATTLE_LEVELS)}:
            raise ValueError("battle level must be 'mixed', '11' or '16'")
        if len(self.deck_mix) != 4 or any(value < 0 for value in self.deck_mix) or sum(self.deck_mix) != 100:
            raise ValueError("deck mix needs four nonnegative weights summing to 100")

    def matchmaker_config(self) -> MatchmakerConfigV4:
        focus_focus, coverage, weighted, tail_tail = self.deck_mix
        return MatchmakerConfigV4(
            focus_focus=focus_focus,
            focus_tail_coverage=coverage,
            focus_tail_weighted=weighted,
            tail_tail=tail_tail,
            focus_tail_hard=0,
        )


def generate_head_to_head_schedule_v4(
    manifest: DeckPoolManifestV4, config: HeadToHeadScheduleConfigV4
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return schedule rows plus generation statistics."""

    matchmaker = DeckMatchmakerV4(manifest, seed=config.seed, config=config.matchmaker_config())
    if config.battle_level == "mixed":
        levels = matchmaker.battle_levels(config.configurations)
    else:
        levels = (int(config.battle_level),) * config.configurations
    rows: list[dict[str, Any]] = []
    mirror_redraws = 0
    for config_id, level in enumerate(levels):
        matchup = matchmaker.next_match(has_history=False, battle_level=level)
        redraws = 0
        while not config.allow_mirror and matchup.deck0.deck_id == matchup.deck1.deck_id:
            redraws += 1
            if redraws > _MAX_MIRROR_REDRAWS:
                raise RuntimeError("deck pool cannot produce a non-mirror configuration")
            matchup = matchmaker.next_match(has_history=False, battle_level=level)
        mirror_redraws += redraws
        layout0, layout1 = matchup.runtime_layouts()
        common = {
            "schema": HEAD_TO_HEAD_SCHEDULE_SCHEMA_V4,
            "config_id": config_id,
            "category": matchup.category,
            "battle_level": matchup.battle_level,
            "seed": matchup.seed,
            "deck0_id": matchup.deck0.deck_id,
            "deck1_id": matchup.deck1.deck_id,
            "deck0_is_focus": matchup.deck0.is_focus,
            "deck1_is_focus": matchup.deck1.is_focus,
            "deck0": list(layout0.card_ids),
            "deck1": list(layout1.card_ids),
            "deck0_form_availability": list(layout0.form_availability),
            "deck1_form_availability": list(layout1.form_availability),
            "tower_troop0_id": layout0.tower_troop_id,
            "tower_troop1_id": layout1.tower_troop_id,
        }
        for swap, owner0_slot in enumerate(SEAT_SLOTS):
            rows.append(
                {
                    **common,
                    "schedule_row_id": 2 * config_id + swap,
                    "owner0_slot": owner0_slot,
                    "owner1_slot": SEAT_SLOTS[1 - swap],
                }
            )
    validate_head_to_head_schedule_v4(rows)
    statistics = {
        "rows": len(rows),
        "configurations": config.configurations,
        "levels": {str(key): value for key, value in sorted(Counter(row["battle_level"] for row in rows).items())},
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "mirror_redraws": mirror_redraws,
        "mirror_rows": sum(row["deck0_id"] == row["deck1_id"] for row in rows),
        "unique_decks": len({row[key] for row in rows for key in ("deck0_id", "deck1_id")}),
    }
    return rows, statistics


def validate_head_to_head_schedule_v4(rows: Sequence[dict[str, Any]]) -> None:
    """Enforce the pairing/swap invariants every consumer relies on."""

    if not rows or len(rows) % 2:
        raise ValueError("head-to-head schedule must contain complete seat-swapped pairs")
    shared = ("seed", "deck0_id", "deck1_id", "battle_level", "category", "deck0", "deck1")
    seen_configs: set[int] = set()
    for index in range(0, len(rows), 2):
        first, second = rows[index], rows[index + 1]
        config_id = int(first["config_id"])
        if int(second["config_id"]) != config_id or config_id in seen_configs or config_id != index // 2:
            raise ValueError(f"schedule configuration ids are not consecutive pairs at row {index}")
        seen_configs.add(config_id)
        if int(first["schedule_row_id"]) != index or int(second["schedule_row_id"]) != index + 1:
            raise ValueError(f"schedule row ids are not consecutive at row {index}")
        if any(first[key] != second[key] for key in shared):
            raise ValueError(f"schedule configuration {config_id} rows disagree on seed/decks/level")
        if (first["owner0_slot"], first["owner1_slot"]) != ("A", "B") or (
            second["owner0_slot"],
            second["owner1_slot"],
        ) != ("B", "A"):
            raise ValueError(f"schedule configuration {config_id} does not swap model seats")
        if int(first["battle_level"]) not in BATTLE_LEVELS or int(first["seed"]) <= 0:
            raise ValueError(f"schedule configuration {config_id} has an invalid level or seed")


def load_head_to_head_schedule_v4(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("schema") != HEAD_TO_HEAD_SCHEDULE_SCHEMA_V4 for row in rows):
        raise ValueError("schedule rows are not a V4 head-to-head schedule")
    validate_head_to_head_schedule_v4(rows)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deck_mix(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item) for item in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("deck mix needs four comma-separated integers")
    return parts  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="schedule JSONL path")
    parser.add_argument("--configurations", type=int, required=True, help="configurations (2 rows each)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--battle-level",
        choices=("mixed", *(str(level) for level in BATTLE_LEVELS)),
        default="mixed",
        help="'mixed' = exact shuffled 50/50 of levels 11/16 (matchmaker default); or pin one level",
    )
    parser.add_argument(
        "--deck-mix",
        type=_deck_mix,
        default=(30, 30, 30, 10),
        help="focus-focus,focus-tail-coverage,focus-tail-weighted,tail-tail weights (sum 100; hard is always 0)",
    )
    parser.add_argument("--allow-mirror", action="store_true", help="keep configurations where both decks match")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = HeadToHeadScheduleConfigV4(
        configurations=args.configurations,
        seed=args.seed,
        battle_level=args.battle_level,
        deck_mix=args.deck_mix,
        allow_mirror=args.allow_mirror,
    )
    sidecar = args.output.with_name(args.output.name + ".meta.json")
    if not args.overwrite and (args.output.exists() or sidecar.exists()):
        raise ValueError(f"refusing to overwrite {args.output} (pass --overwrite)")
    manifest = DeckPoolManifestV4.load(args.deck_pool)
    rows, statistics = generate_head_to_head_schedule_v4(manifest, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    meta = {
        "schema": HEAD_TO_HEAD_SCHEDULE_SCHEMA_V4 + ".meta",
        "generator": "native_runner.training.v4.make_head_to_head_schedule",
        "parameters": {
            "configurations": config.configurations,
            "seed": config.seed,
            "battle_level": config.battle_level,
            "deck_mix": {
                "focus_focus": config.deck_mix[0],
                "focus_tail_coverage": config.deck_mix[1],
                "focus_tail_weighted": config.deck_mix[2],
                "tail_tail": config.deck_mix[3],
                "focus_tail_hard": 0,
            },
            "allow_mirror": config.allow_mirror,
            "hard_matchups": [],
        },
        "deck_pool": {"path": str(args.deck_pool), "sha256": _sha256(args.deck_pool)},
        "schedule": {"path": str(args.output), "sha256": _sha256(args.output)},
        "statistics": statistics,
    }
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "head_to_head_schedule", **statistics, "schedule": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
