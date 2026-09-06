"""Canonical exact-form deck identities for V4 PPO matchmaking.

A deck identity contains eight order-invariant RoyaleAPI card keys, including
their selected ``-evN``/``-hero`` form, plus the exact Tower Troop.  Runtime
slot order is deliberately generated later by the matchmaker and is not part
of the identity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from ...card_specs import build_card_catalog
from ...contracts import content_hash
from ...normal_form_evidence import NATIVE_CHAMPION_ABILITY_BINDINGS
from ...royaleapi_replay import ROYALAPI_TOWER_TROOP_IDS, ROYALAPI_CARD_KEY_IDS, base_card_key, resolve_native_card
from .catalog import normal_mode_card_specs


DECK_POOL_SCHEMA_V4 = "v4-ppo-deck-pool.v1"
DECK_IDENTITY_SCHEMA_V4 = "v4-exact-deck-with-tower.v1"
_EVOLUTION_SUFFIX = re.compile(r"-ev\d+$", re.IGNORECASE)


def _normalized_card_key(value: object) -> str:
    key = str(value or "").strip().casefold()
    if not key:
        raise ValueError("deck card key cannot be empty")
    return key


def _normalized_alias(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


@lru_cache(maxsize=1)
def _runtime_card_aliases() -> tuple[dict[int, object], dict[str, tuple[int, ...]]]:
    catalog = build_card_catalog()
    eligible = normal_mode_card_specs(catalog.by_id, require_frozen_baseline=True)
    aliases: dict[str, set[int]] = {}
    for spec in eligible.values():
        attributes = spec.attributes
        raw = {
            spec.name,
            attributes.get("Name"),
            attributes.get("IconFile"),
            attributes.get("Stats"),
            attributes.get("TID"),
            Path(str(attributes.get("HighresImageFilename") or "")).stem,
        }
        for value in raw:
            alias = _normalized_alias(value)
            if alias:
                aliases.setdefault(alias, set()).add(spec.card_id)
    return (eligible, {key: tuple(sorted(values)) for key, values in aliases.items()})


def _resolve_card_id(card_key: str) -> int:
    """Resolve a public card key even when immutable published artifacts are absent."""

    try:
        return int(resolve_native_card(card_key).card_id)
    except FileNotFoundError:
        pass
    public_key = base_card_key(card_key)
    explicit = ROYALAPI_CARD_KEY_IDS.get(public_key)
    by_id, aliases = _runtime_card_aliases()
    if explicit is not None:
        if explicit not in by_id:
            raise ValueError(f"runtime catalog lacks explicit card ID {explicit}")
        return int(explicit)
    probe = _normalized_alias(public_key)
    candidates = aliases.get(probe, ())
    if len(candidates) == 1:
        return int(candidates[0])
    fallback = {probe + "s"}
    if probe.endswith("s"):
        fallback.add(probe[:-1])
    matches = {card_id for key in fallback for card_id in aliases.get(key, ())}
    if len(matches) == 1:
        return int(next(iter(matches)))
    raise ValueError(f"cannot resolve RoyaleAPI card key {card_key!r}")


def selected_form_mask(card_key: str, card_id: int) -> int:
    """Return the exact native form bit selected by one RoyaleAPI key."""

    key = _normalized_card_key(card_key)
    if _EVOLUTION_SUFFIX.search(key):
        return 1
    if key.endswith("-hero") or int(card_id) in NATIVE_CHAMPION_ABILITY_BINDINGS:
        return 2
    return 0


@dataclass(frozen=True, slots=True)
class DeckIdentityV4:
    deck_id: str
    card_keys: tuple[str, ...]
    card_ids: tuple[int, ...]
    form_availability: tuple[int, ...]
    tower_key: str
    tower_troop_id: int
    uses: int
    focus_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.deck_id:
            raise ValueError("deck identity requires a stable ID")
        if len(self.card_keys) != 8 or len(self.card_ids) != 8 or len(self.form_availability) != 8:
            raise ValueError("deck identity requires exactly eight card rows")
        if tuple(sorted(self.card_keys)) != self.card_keys:
            raise ValueError("deck identity card keys must be canonical and sorted")
        if len(set(self.card_ids)) != 8:
            raise ValueError("deck identity cannot repeat a native base card")
        if any(value not in (0, 1, 2) for value in self.form_availability):
            raise ValueError("deck selected form masks must be 0, 1, or 2")
        if self.tower_key not in ROYALAPI_TOWER_TROOP_IDS:
            raise ValueError("deck identity has an unsupported Tower Troop")
        if ROYALAPI_TOWER_TROOP_IDS[self.tower_key] != self.tower_troop_id:
            raise ValueError("deck Tower Troop key and ID disagree")
        if self.uses <= 0:
            raise ValueError("deck usage count must be positive")
        if self.focus_rank is not None and self.focus_rank <= 0:
            raise ValueError("focus rank must be positive or None")

    @property
    def is_focus(self) -> bool:
        return self.focus_rank is not None

    @property
    def base_card_keys(self) -> tuple[str, ...]:
        return tuple(sorted(base_card_key(item) for item in self.card_keys))

    @property
    def average_elixir(self) -> float:
        """Return the exact eight-card average used by the runtime tensorizer."""

        specs, _aliases = _runtime_card_aliases()
        costs: list[float] = []
        for card_id in self.card_ids:
            try:
                cost = specs[card_id].elixir_cost
            except KeyError as error:
                raise ValueError(f"deck {self.deck_id} card {card_id} has no runtime spec") from error
            if cost is None:
                raise ValueError(f"deck {self.deck_id} card {card_id} has no exact elixir cost")
            costs.append(float(cost))
        return sum(costs) / len(costs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deck_id": self.deck_id,
            "card_keys": list(self.card_keys),
            "card_ids": list(self.card_ids),
            "form_availability": list(self.form_availability),
            "tower_key": self.tower_key,
            "tower_troop_id": self.tower_troop_id,
            "uses": self.uses,
            "focus_rank": self.focus_rank,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeckIdentityV4":
        return cls(
            deck_id=str(value["deck_id"]),
            card_keys=tuple(str(item) for item in value["card_keys"]),
            card_ids=tuple(int(item) for item in value["card_ids"]),
            form_availability=tuple(int(item) for item in value["form_availability"]),
            tower_key=str(value["tower_key"]),
            tower_troop_id=int(value["tower_troop_id"]),
            uses=int(value["uses"]),
            focus_rank=(None if value.get("focus_rank") is None else int(value["focus_rank"])),
        )


def deck_identity_from_keys(
    card_keys: Sequence[str], tower_key: str, *, uses: int, focus_rank: int | None = None
) -> DeckIdentityV4:
    normalized = tuple(sorted(_normalized_card_key(item) for item in card_keys))
    if len(normalized) != 8:
        raise ValueError("standard deck identity requires eight card keys")
    tower = str(tower_key or "").strip().casefold()
    try:
        tower_id = ROYALAPI_TOWER_TROOP_IDS[tower]
    except KeyError as error:
        raise ValueError(f"unsupported Tower Troop key: {tower!r}") from error
    rows = tuple((key, _resolve_card_id(key)) for key in normalized)
    card_ids = tuple(int(card_id) for _key, card_id in rows)
    forms = tuple(selected_form_mask(key, card_id) for key, card_id in rows)
    deck_id = content_hash({"schema": DECK_IDENTITY_SCHEMA_V4, "card_keys": normalized, "tower_key": tower})
    return DeckIdentityV4(
        deck_id=deck_id,
        card_keys=normalized,
        card_ids=card_ids,
        form_availability=forms,
        tower_key=tower,
        tower_troop_id=tower_id,
        uses=int(uses),
        focus_rank=focus_rank,
    )


@dataclass(frozen=True, slots=True)
class DeckPoolManifestV4:
    decks: tuple[DeckIdentityV4, ...]
    minimum_uses: int
    metadata: Mapping[str, Any]
    schema: str = DECK_POOL_SCHEMA_V4

    def __post_init__(self) -> None:
        if self.schema != DECK_POOL_SCHEMA_V4:
            raise ValueError("unsupported V4 PPO deck-pool schema")
        if self.minimum_uses <= 0:
            raise ValueError("deck-pool minimum usage must be positive")
        if not self.decks:
            raise ValueError("deck pool cannot be empty")
        ids = tuple(item.deck_id for item in self.decks)
        if len(set(ids)) != len(ids):
            raise ValueError("deck pool contains duplicate identities")
        focus = sorted(item.focus_rank for item in self.decks if item.focus_rank is not None)
        if focus and focus != list(range(1, len(focus) + 1)):
            raise ValueError("focus ranks must be contiguous from one")

    @property
    def focus_decks(self) -> tuple[DeckIdentityV4, ...]:
        return tuple(sorted((item for item in self.decks if item.is_focus), key=lambda item: int(item.focus_rank or 0)))

    @property
    def tail_decks(self) -> tuple[DeckIdentityV4, ...]:
        return tuple(item for item in self.decks if not item.is_focus)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "minimum_uses": self.minimum_uses,
            "metadata": dict(self.metadata),
            "decks": [item.to_dict() for item in self.decks],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DeckPoolManifestV4":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("deck-pool manifest root must be a mapping")
        decks = raw.get("decks")
        if not isinstance(decks, list):
            raise ValueError("deck-pool manifest has no deck list")
        return cls(
            schema=str(raw.get("schema")),
            minimum_uses=int(raw["minimum_uses"]),
            metadata=dict(raw.get("metadata") or {}),
            decks=tuple(DeckIdentityV4.from_dict(item) for item in decks),
        )


def _player_deck_key(player: Mapping[str, Any]) -> tuple[tuple[str, ...], str] | None:
    raw_deck = player.get("deck")
    if not isinstance(raw_deck, Sequence) or isinstance(raw_deck, (str, bytes)) or len(raw_deck) != 8:
        return None
    card_keys: list[str] = []
    for card in raw_deck:
        if not isinstance(card, Mapping) or not card.get("card_key"):
            return None
        card_keys.append(_normalized_card_key(card["card_key"]))
    tower_card = player.get("tower_card")
    tower_key = (
        "tower-princess"
        if not isinstance(tower_card, Mapping)
        else str(tower_card.get("card_key") or "tower-princess").strip().casefold()
    )
    tower_key = {"princess-tower": "tower-princess", "chef-tower": "royal-chef"}.get(tower_key, tower_key)
    if tower_key not in ROYALAPI_TOWER_TROOP_IDS:
        return None
    return tuple(sorted(card_keys)), tower_key


def deck_counts_from_payloads(
    rows: Iterable[tuple[str, Mapping[str, Any]]],
) -> tuple[Counter[tuple[tuple[str, ...], str]], dict[str, int]]:
    """Count player-side deck uses after globally deduplicating replay tags."""

    counts: Counter[tuple[tuple[str, ...], str]] = Counter()
    seen_replays: set[str] = set()
    duplicate_rows = 0
    invalid_player_decks = 0
    for replay_tag, payload in rows:
        label = str(replay_tag)
        if label in seen_replays:
            duplicate_rows += 1
            continue
        seen_replays.add(label)
        battle = payload.get("battle")
        if not isinstance(battle, Mapping):
            invalid_player_decks += 2
            continue
        for side in ("team", "opponent"):
            raw_side = battle.get(side)
            players = raw_side.get("players") if isinstance(raw_side, Mapping) else None
            if (
                not isinstance(players, Sequence)
                or isinstance(players, (str, bytes))
                or len(players) != 1
                or not isinstance(players[0], Mapping)
            ):
                invalid_player_decks += 1
                continue
            key = _player_deck_key(players[0])
            if key is None:
                invalid_player_decks += 1
            else:
                counts[key] += 1
    return counts, {
        "unique_replays": len(seen_replays),
        "duplicate_replay_rows_skipped": duplicate_rows,
        "invalid_player_decks": invalid_player_decks,
        "valid_player_deck_uses": sum(counts.values()),
    }


def build_deck_pool_manifest_from_counts(
    counts: Mapping[tuple[tuple[str, ...], str], int],
    *,
    minimum_uses: int = 5,
    metadata: Mapping[str, Any] | None = None,
) -> DeckPoolManifestV4:
    if minimum_uses <= 0:
        raise ValueError("minimum deck uses must be positive")
    decks = tuple(
        sorted(
            (
                deck_identity_from_keys(card_keys, tower_key, uses=uses)
                for (card_keys, tower_key), uses in counts.items()
                if int(uses) >= minimum_uses
            ),
            key=lambda item: item.deck_id,
        )
    )
    combined_metadata = dict(metadata or {})
    combined_metadata.update({"pool_size": len(decks), "pool_player_uses": sum(item.uses for item in decks)})
    return DeckPoolManifestV4(decks=decks, minimum_uses=minimum_uses, metadata=combined_metadata)


def mark_focus_decks(
    manifest: DeckPoolManifestV4, focus_identities: Sequence[tuple[Sequence[str], str]]
) -> DeckPoolManifestV4:
    requested: dict[str, int] = {}
    for rank, (card_keys, tower_key) in enumerate(focus_identities, start=1):
        identity = deck_identity_from_keys(card_keys, tower_key, uses=1)
        if identity.deck_id in requested:
            raise ValueError("focus deck list contains a duplicate identity")
        requested[identity.deck_id] = rank
    available = {item.deck_id for item in manifest.decks}
    missing = tuple(deck_id for deck_id in requested if deck_id not in available)
    if missing:
        raise ValueError(f"focus decks are absent from the usage-filtered pool: {missing}")
    decks = tuple(replace(item, focus_rank=requested.get(item.deck_id)) for item in manifest.decks)
    metadata = {**dict(manifest.metadata), "focus_deck_count": len(requested)}
    return DeckPoolManifestV4(decks=decks, minimum_uses=manifest.minimum_uses, metadata=metadata)
