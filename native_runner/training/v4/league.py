"""Small frozen-policy league and PFSP sampling state for V4 PPO."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


LEAGUE_SCHEMA_V4 = "v4-ppo-policy-league.v1"


@dataclass(frozen=True, slots=True)
class LeagueEntryV4:
    checkpoint_id: str
    checkpoint_path: str
    update_step: int
    wins_against: int = 0
    draws_against: int = 0
    losses_against: int = 0

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.checkpoint_path or self.update_step < 0:
            raise ValueError("league entry identity is invalid")
        if min(self.wins_against, self.draws_against, self.losses_against) < 0:
            raise ValueError("league result counts cannot be negative")

    @property
    def games(self) -> int:
        return self.wins_against + self.draws_against + self.losses_against

    @property
    def learner_win_rate(self) -> float:
        if self.games == 0:
            return 0.5
        return (self.wins_against + 0.5 * self.draws_against) / self.games

    @property
    def pfsp_weight(self) -> float:
        win_rate = self.learner_win_rate
        near_even = win_rate * (1.0 - win_rate)
        weakness = (1.0 - win_rate) ** 2
        return 0.05 + 0.7 * near_even + 0.3 * weakness

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "update_step": self.update_step,
            "wins_against": self.wins_against,
            "draws_against": self.draws_against,
            "losses_against": self.losses_against,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeagueEntryV4":
        return cls(
            checkpoint_id=str(value["checkpoint_id"]),
            checkpoint_path=str(value["checkpoint_path"]),
            update_step=int(value["update_step"]),
            wins_against=int(value.get("wins_against", 0)),
            draws_against=int(value.get("draws_against", 0)),
            losses_against=int(value.get("losses_against", 0)),
        )


class PolicyLeagueV4:
    def __init__(
        self, *, anchor_checkpoint_id: str, anchor_checkpoint_path: str | Path, entries: tuple[LeagueEntryV4, ...] = ()
    ) -> None:
        if not anchor_checkpoint_id:
            raise ValueError("policy league requires the IL anchor checkpoint ID")
        self.anchor_checkpoint_id = str(anchor_checkpoint_id)
        self.anchor_checkpoint_path = str(Path(anchor_checkpoint_path).resolve())
        self.entries = list(entries)
        if len({item.checkpoint_id for item in self.entries}) != len(self.entries):
            raise ValueError("policy league contains duplicate checkpoint IDs")

    @property
    def has_history(self) -> bool:
        return bool(self.entries)

    def add_snapshot(self, checkpoint_id: str, checkpoint_path: str | Path, *, update_step: int) -> None:
        if checkpoint_id == self.anchor_checkpoint_id or any(
            item.checkpoint_id == checkpoint_id for item in self.entries
        ):
            raise ValueError("policy league snapshot identity already exists")
        self.entries.append(
            LeagueEntryV4(
                checkpoint_id=checkpoint_id,
                checkpoint_path=str(Path(checkpoint_path).resolve()),
                update_step=int(update_step),
            )
        )

    def sample_history(self, rng: random.Random) -> LeagueEntryV4:
        return self.sample_history_from(rng, self.entries)

    def sample_history_candidates(self, rng: random.Random, *, limit: int) -> tuple[LeagueEntryV4, ...]:
        """Select one batch's PFSP candidate pool without replacement."""

        if limit <= 0:
            raise ValueError("historical policy candidate limit must be positive")
        if not self.entries:
            return ()
        if len(self.entries) <= limit:
            return tuple(self.entries)
        remaining = list(self.entries)
        selected: list[LeagueEntryV4] = []
        for _ in range(limit):
            index = rng.choices(range(len(remaining)), weights=tuple(item.pfsp_weight for item in remaining), k=1)[0]
            selected.append(remaining.pop(index))
        return tuple(selected)

    @staticmethod
    def sample_history_from(rng: random.Random, candidates: Sequence[LeagueEntryV4]) -> LeagueEntryV4:
        pool = tuple(candidates)
        if not pool:
            raise ValueError("cannot sample an empty historical policy league")
        return rng.choices(pool, weights=tuple(item.pfsp_weight for item in pool), k=1)[0]

    def record_result(self, checkpoint_id: str, learner_result: float) -> None:
        if not math.isfinite(learner_result):
            raise ValueError("league learner result must be finite")
        index = next(
            (position for position, item in enumerate(self.entries) if item.checkpoint_id == checkpoint_id), None
        )
        if index is None:
            raise ValueError("league result references an unknown checkpoint")
        entry = self.entries[index]
        if learner_result > 0.0:
            entry = replace(entry, wins_against=entry.wins_against + 1)
        elif learner_result < 0.0:
            entry = replace(entry, losses_against=entry.losses_against + 1)
        else:
            entry = replace(entry, draws_against=entry.draws_against + 1)
        self.entries[index] = entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEAGUE_SCHEMA_V4,
            "anchor_checkpoint_id": self.anchor_checkpoint_id,
            "anchor_checkpoint_path": self.anchor_checkpoint_path,
            "entries": [item.to_dict() for item in self.entries],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PolicyLeagueV4":
        if value.get("schema") != LEAGUE_SCHEMA_V4:
            raise ValueError("unsupported V4 PPO policy league")
        rows = value.get("entries")
        if not isinstance(rows, list):
            raise ValueError("policy league entries must be a list")
        return cls(
            anchor_checkpoint_id=str(value["anchor_checkpoint_id"]),
            anchor_checkpoint_path=str(value["anchor_checkpoint_path"]),
            entries=tuple(LeagueEntryV4.from_dict(item) for item in rows),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PolicyLeagueV4":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("PPO policy league must be a JSON object")
        return cls.from_dict(raw)
