"""Configurable native-match schema for the 15.535.13 battle engine.

The JSON emitted here is consumed by the stock client's replay/battle parser,
but recorded commands and events are always removed.  It therefore describes
a fresh match configuration rather than replaying the source fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


from .paths import PACKAGE_ROOT

DEFAULT_TEMPLATE_PATH = PACKAGE_ROOT / "data" / "native_match.json"
# Synthetic local identities; these never refer to server accounts.
NATIVE_OWNER0_ACCOUNT_ID = 1
NATIVE_OWNER1_ACCOUNT_ID = 2
# The Ladder timeline contains 180 seconds of regulation plus 120 seconds of
# overtime at 20 Hz.  No policy decision is legal after this boundary.
NATIVE_GAMEPLAY_END_TICK = 6000
# ``endTick`` keeps the replay scene alive while the native manager performs
# its tiebreak/final-result transition.  This is an infrastructure watchdog,
# not an extra minute of gameplay.
NATIVE_FINALIZATION_DEADLINE_TICK = 7200
# Compatibility name for persisted match configs.  It refers to the native
# scene/deadline boundary, not to the policy decision horizon.
NATIVE_MATCH_END_TICK = NATIVE_FINALIZATION_DEADLINE_TICK
PRINCESS_TOWER_TROOP_ID = 159_000_000
MAX_CARD_LEVEL = 16
# Lowest absolute level of each tower troop's rarity.  The native support-card
# level index is ``absolute - minimum`` exactly like deck cards: Tower Princess
# is Common (1), Cannoneer Epic (6), Dagger Duchess and Royal Chef Legendary (9).
# Measured on the standalone 15.535 engine from tower maxHp (2026-10-24).
TOWER_TROOP_RARITY_MIN_LEVEL: Mapping[int, int] = {
    159_000_000: 1,
    159_000_001: 6,
    159_000_002: 9,
    159_000_004: 9,
}
# Episode tags of faithful per-card levels (see ``match_level_tags``).
CARD_LEVELS_VERSION = "per-card-absolute-levels.v1"

STANDARD_DECK = (
    26000000,  # Knight
    26000001,  # Archers
    26000005,  # Minions
    28000001,  # Arrows
    28000000,  # Fireball
    26000003,  # Giant
    26000014,  # Musketeer
    26000018,  # Mini P.E.K.K.A
)

ROYAL_GIANT_DECK = (
    26000024,  # Royal Giant
    26000044,  # Hunter
    26000084,  # Electro Spirit
    28000011,  # The Log
    28000000,  # Fireball
    26000061,  # Fisherman
    26000002,  # Goblins
    26000050,  # Royal Ghost
)


def _deck(value: Iterable[int], label: str) -> tuple[int, ...]:
    cards = tuple(int(card_id) for card_id in value)
    if len(cards) != 8:
        raise ValueError(f"{label} must contain exactly eight cards")
    if any(card_id <= 0 or card_id > 0x7FFFFFFF for card_id in cards):
        raise ValueError(f"{label} contains an invalid card ID")
    if len(set(cards)) != len(cards):
        raise ValueError(f"{label} must not contain duplicate cards")
    return cards


@lru_cache(maxsize=1)
def _card_rarity_min_levels() -> Mapping[int, int]:
    from .competitive_data import load_competitive_data

    logic = load_competitive_data("card_logic")
    minimum = {str(name): int(curve["card_levels"][0]) for name, curve in logic["rarity_curves"].items()}
    return {int(card["card_id"]): minimum[str(card["rarity"])] for card in logic["cards"]}


def rarity_min_level(card_id: int) -> int:
    """Lowest absolute level of the card's rarity (``rarity_curves[R].card_levels[0]``).

    The native replay schema stores rarity-relative level indices: absolute
    level = ``sp[i].l`` (or ``sc[0].l``) + this value.
    """

    card_id = int(card_id)
    if card_id in TOWER_TROOP_RARITY_MIN_LEVEL:
        return TOWER_TROOP_RARITY_MIN_LEVEL[card_id]
    try:
        return _card_rarity_min_levels()[card_id]
    except KeyError as error:
        raise ValueError(f"card {card_id} has no rarity curve in card_logic.json") from error


def _card_levels(value: Iterable[int] | None, deck: tuple[int, ...], label: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    levels = tuple(value)
    if len(levels) != len(deck):
        raise ValueError(f"{label} must contain exactly eight levels")
    for card_id, level in zip(deck, levels):
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError(f"{label} values must be integers")
        if not rarity_min_level(card_id) <= level <= MAX_CARD_LEVEL:
            raise ValueError(
                f"{label}: card {card_id} level {level} is outside {rarity_min_level(card_id)}..{MAX_CARD_LEVEL}"
            )
    return levels


def _form_availability(value: Iterable[int], label: str) -> tuple[int, ...]:
    """Validate per-deck-slot native form-availability bit masks.

    This field is the decoded ``battle.deckN.sp[i].el`` mask.  Bit 0 declares
    that the spell has an Evo form and bit 1 declares that it has a Hero form.
    It does not select an active evolution slot or force either form to be
    played.
    """

    masks = tuple(value)
    if len(masks) != 8:
        raise ValueError(f"{label} must contain exactly eight masks")
    if any(isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask > 3 for mask in masks):
        raise ValueError(f"{label} values must be integer bit masks in 0..3")
    return masks


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Inputs used to construct one fresh native 1v1 match.

    Owners are engine owners, not screen sides. ``deck0`` belongs to owner 0
    and ``deck1`` belongs to owner 1 in every observation and action.
    """

    deck0: tuple[int, ...] = STANDARD_DECK
    deck1: tuple[int, ...] = STANDARD_DECK
    # Exact native availability metadata for each corresponding deck slot:
    # 1 = EvoForm, 2 = HeroForm, 3 = both.  These masks only make forms
    # available to the native deck decoder; they are not active-slot selectors.
    deck0_form_availability: tuple[int, ...] = (0,) * 8
    deck1_form_availability: tuple[int, ...] = (0,) * 8
    tower_troop0_id: int = PRINCESS_TOWER_TROOP_ID
    tower_troop1_id: int = PRINCESS_TOWER_TROOP_ID
    seed: int = 1
    game_mode: int = 72000006
    arena: int = 54000001
    location: int = 15000199
    level_cap: int = 0
    minimum_card_level: int = 0
    # King Tower level is an independent native replay input. ``lvlcap`` and
    # ``cardlvlmin`` normalize cards (including the Princess Tower troop), but
    # do not replace ``battle.hbd[owner].kt``. ``None`` preserves the template
    # value for backwards-compatible fixtures.
    king_tower_level: int | None = None
    owner0_name: str = "Native-0"
    owner1_name: str = "Native-1"
    # These are internal v15 player keys, not policy identities. The warmed
    # native command context is keyed to them; agents address sides with owner
    # 0/1 through the public API instead.
    owner0_account_id: int = field(default=NATIVE_OWNER0_ACCOUNT_ID, init=False)
    owner1_account_id: int = field(default=NATIVE_OWNER1_ACCOUNT_ID, init=False)
    # The stock ReplayBattleController uses endTick only as its scene timeline
    # boundary. A negative value falls back to 1200 ticks and pauses rendering
    # after one minute even though the live manager is still valid. Keep the
    # empty-command scene alive beyond a complete regulation/overtime match.
    end_tick: int = NATIVE_MATCH_END_TICK
    # Faithful per-card levels, as absolute levels 1..16 (the in-game display
    # level). Each tuple is aligned with the corresponding deck slot. When set,
    # ``battle.deckN.sp[i].l`` = level - rarity_min_level(card); ``None`` keeps
    # the template (no ``l``: the lowest level of the rarity). ``level_cap``
    # (0 = none) and ``minimum_card_level`` still clamp these natively, so
    # faithful levels need both at 0.
    deck0_card_levels: tuple[int, ...] | None = None
    deck1_card_levels: tuple[int, ...] | None = None
    # Absolute tower-troop levels -> ``battle.deckN.sc[0].l`` (same encoding);
    # ``None`` keeps the template's support-card level index.
    tower_troop0_level: int | None = None
    tower_troop1_level: int | None = None
    # Per-owner King Tower levels; ``None`` falls back to ``king_tower_level``.
    king_tower0_level: int | None = None
    king_tower1_level: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "deck0", _deck(self.deck0, "deck0"))
        object.__setattr__(self, "deck1", _deck(self.deck1, "deck1"))
        object.__setattr__(
            self, "deck0_form_availability", _form_availability(self.deck0_form_availability, "deck0_form_availability")
        )
        object.__setattr__(
            self, "deck1_form_availability", _form_availability(self.deck1_form_availability, "deck1_form_availability")
        )
        for label, value in (("tower_troop0_id", self.tower_troop0_id), ("tower_troop1_id", self.tower_troop1_id)):
            if isinstance(value, bool) or not isinstance(value, int) or value // 1_000_000 != 159:
                raise ValueError(f"{label} must be a native support-card ID in class 159")
        if not -(1 << 31) <= int(self.seed) < (1 << 32):
            raise ValueError("seed must fit the native 32-bit replay field")
        for label, value in (("game_mode", self.game_mode), ("arena", self.arena), ("location", self.location)):
            if int(value) <= 0:
                raise ValueError(f"{label} must be positive")
        for label, value in (
            ("owner0_account_id", self.owner0_account_id),
            ("owner1_account_id", self.owner1_account_id),
        ):
            if not 0 < int(value) < (1 << 64):
                raise ValueError(f"{label} must fit an unsigned 64-bit value")
        if self.owner0_account_id == self.owner1_account_id:
            raise ValueError("the two owners must have distinct account IDs")
        if self.owner0_account_id != NATIVE_OWNER0_ACCOUNT_ID or self.owner1_account_id != NATIVE_OWNER1_ACCOUNT_ID:
            raise ValueError(
                "v15 native account IDs are fixed internal keys; identify agents "
                "with owner 0/1 and customize owner names instead"
            )
        if not self.owner0_name or not self.owner1_name:
            raise ValueError("player names must not be empty")
        if self.king_tower_level is not None and (
            isinstance(self.king_tower_level, bool)
            or not isinstance(self.king_tower_level, int)
            or not 1 <= int(self.king_tower_level) <= 16
        ):
            raise ValueError("king_tower_level must be an integer in 1..16")
        if not 1 <= int(self.end_tick) <= 0x7FFFFFFF:
            raise ValueError("end_tick must be a positive signed 32-bit timeline bound")
        object.__setattr__(
            self, "deck0_card_levels", _card_levels(self.deck0_card_levels, self.deck0, "deck0_card_levels")
        )
        object.__setattr__(
            self, "deck1_card_levels", _card_levels(self.deck1_card_levels, self.deck1, "deck1_card_levels")
        )
        for label, troop, level in (
            ("tower_troop0_level", self.tower_troop0_id, self.tower_troop0_level),
            ("tower_troop1_level", self.tower_troop1_id, self.tower_troop1_level),
        ):
            if level is not None and (
                isinstance(level, bool)
                or not isinstance(level, int)
                or not rarity_min_level(troop) <= level <= MAX_CARD_LEVEL
            ):
                raise ValueError(f"{label} must be an integer in {rarity_min_level(troop)}..{MAX_CARD_LEVEL}")
        for label, level in (
            ("king_tower0_level", self.king_tower0_level),
            ("king_tower1_level", self.king_tower1_level),
        ):
            if level is not None and (isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 16):
                raise ValueError(f"{label} must be an integer in 1..16")

    @classmethod
    def from_decks(cls, deck0: Iterable[int], deck1: Iterable[int], **kwargs: Any) -> "MatchConfig":
        return cls(deck0=_deck(deck0, "deck0"), deck1=_deck(deck1, "deck1"), **kwargs)

    def to_replay_dict(self, template_path: str | Path = DEFAULT_TEMPLATE_PATH) -> dict[str, Any]:
        path = Path(template_path)
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load native match schema template: {path}") from error

        battle = replay.get("battle")
        if not isinstance(battle, dict):
            raise ValueError(f"native match template has no battle object: {path}")
        for owner, deck, form_availability, tower_troop_id, card_levels, tower_troop_level in (
            (
                0,
                self.deck0,
                self.deck0_form_availability,
                self.tower_troop0_id,
                self.deck0_card_levels,
                self.tower_troop0_level,
            ),
            (
                1,
                self.deck1,
                self.deck1_form_availability,
                self.tower_troop1_id,
                self.deck1_card_levels,
                self.tower_troop1_level,
            ),
        ):
            deck_object = battle.get(f"deck{owner}")
            if not isinstance(deck_object, dict):
                raise ValueError(f"native match template has no deck{owner} object")
            deck_object["sp"] = [
                ({"d": card_id, "el": form_availability[index]} if form_availability[index] else {"d": card_id})
                for index, card_id in enumerate(deck)
            ]
            if card_levels is not None:
                for entry, card_id, level in zip(deck_object["sp"], deck, card_levels):
                    entry["l"] = int(level) - rarity_min_level(card_id)
            support_cards = deck_object.get("sc")
            if not isinstance(support_cards, list) or len(support_cards) != 1 or not isinstance(support_cards[0], dict):
                raise ValueError(f"native match template deck{owner}.sc must contain one support card")
            support_cards[0]["d"] = int(tower_troop_id)
            if tower_troop_level is not None:
                support_cards[0]["l"] = int(tower_troop_level) - rarity_min_level(tower_troop_id)

        battle["gamemode"] = int(self.game_mode)
        battle["arena"] = int(self.arena)
        battle["location"] = int(self.location)
        battle["lvlcap"] = int(self.level_cap)
        battle["cardlvlmin"] = int(self.minimum_card_level)

        king_levels = tuple(
            owner_level if owner_level is not None else self.king_tower_level
            for owner_level in (self.king_tower0_level, self.king_tower1_level)
        )
        for owner, name, account_id in (
            (0, self.owner0_name, self.owner0_account_id),
            (1, self.owner1_name, self.owner1_account_id),
        ):
            avatar = battle.get(f"avatar{owner}")
            if not isinstance(avatar, dict):
                raise ValueError(f"native match template has no avatar{owner} object")
            avatar["name"] = name
            avatar["accountID.lo"] = int(account_id) & 0xFFFFFFFF
            avatar["accountID.hi"] = int(account_id) >> 32
            if king_levels[owner] is not None:
                # ``expLevel`` drives the stock render's displayed player
                # level, while ``hbd.kt`` drives the native King Tower stats.
                # Keep both representations synchronized.
                avatar["expLevel"] = int(king_levels[owner])

        if any(level is not None for level in king_levels):
            home_battle_data = battle.get("hbd")
            if not isinstance(home_battle_data, list) or len(home_battle_data) != 2:
                raise ValueError("native match template must contain two battle.hbd owner rows")
            for owner, owner_data in enumerate(home_battle_data):
                if not isinstance(owner_data, dict):
                    raise ValueError(f"native match template battle.hbd[{owner}] is not an object")
                if king_levels[owner] is not None:
                    owner_data["kt"] = int(king_levels[owner])

        # These are the only historical streams in the schema. The native
        # manager receives an empty live queue and all subsequent decisions
        # enter through the control API.
        replay["cmd"] = []
        replay["evt"] = []
        replay["rndSeed"] = int(self.seed)
        replay["time"] = -1
        replay["endTick"] = int(self.end_tick)
        return replay

    def to_json(self, template_path: str | Path = DEFAULT_TEMPLATE_PATH) -> str:
        payload = json.dumps(self.to_replay_dict(template_path), ensure_ascii=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) >= 64 * 1024:
            raise ValueError("native match payload exceeds the probe's 64 KiB limit")
        if "\n" in payload or "\r" in payload:
            raise ValueError("native match payload must be a single wire line")
        return payload


def match_level_tags(match: MatchConfig) -> dict[str, Any]:
    """Episode tags that carry ``match``'s faithful per-card levels ({} without them).

    Card levels are keyed by card ID, ``((card_id, absolute_level), ...)``
    sorted by card ID, so deal calibration may permute deck slots freely. The
    legacy ``level_cap`` / ``minimum_card_level`` / ``king_tower_level`` tags
    stay alongside as the uniform fallback for readers that predate them.
    """

    if match.deck0_card_levels is None or match.deck1_card_levels is None:
        return {}
    if None in (match.tower_troop0_level, match.tower_troop1_level, match.king_tower0_level, match.king_tower1_level):
        raise ValueError("per-card level tags need both tower-troop and both King Tower levels")
    return {
        "card_levels_version": CARD_LEVELS_VERSION,
        "deck0_card_levels": tuple(sorted(zip(match.deck0, match.deck0_card_levels))),
        "deck1_card_levels": tuple(sorted(zip(match.deck1, match.deck1_card_levels))),
        "tower_troop0_level": int(match.tower_troop0_level),
        "tower_troop1_level": int(match.tower_troop1_level),
        "king_tower0_level": int(match.king_tower0_level),
        "king_tower1_level": int(match.king_tower1_level),
    }


def match_level_kwargs(deck0: Iterable[int], deck1: Iterable[int], tags: Mapping[str, Any]) -> dict[str, Any]:
    """``MatchConfig`` level keywords for an episode's decks and tags.

    Without ``card_levels_version`` this is the legacy uniform encoding
    (``level_cap`` / ``minimum_card_level`` / ``king_tower_level``). With it,
    every card, both tower troops and both King Towers get their own level and
    the native clamps are disabled (``lvlcap = cardlvlmin = 0``).
    """

    version = tags.get("card_levels_version")
    if version is None:
        return {
            "level_cap": int(tags.get("level_cap", 0)),
            "minimum_card_level": int(tags.get("minimum_card_level", 0)),
            "king_tower_level": (int(tags["king_tower_level"]) if tags.get("king_tower_level") is not None else None),
        }
    if version != CARD_LEVELS_VERSION:
        raise ValueError(f"unsupported card_levels_version {version!r}")
    kwargs: dict[str, Any] = {"level_cap": 0, "minimum_card_level": 0, "king_tower_level": None}
    decks = (tuple(int(card_id) for card_id in deck0), tuple(int(card_id) for card_id in deck1))
    for owner, deck in enumerate(decks):
        pairs = tuple((int(card_id), int(level)) for card_id, level in tags[f"deck{owner}_card_levels"])
        by_card = dict(pairs)
        if len(by_card) != len(pairs) or set(by_card) != set(deck):
            raise ValueError(f"deck{owner}_card_levels does not cover deck{owner} exactly")
        kwargs[f"deck{owner}_card_levels"] = tuple(by_card[card_id] for card_id in deck)
        kwargs[f"tower_troop{owner}_level"] = int(tags[f"tower_troop{owner}_level"])
        kwargs[f"king_tower{owner}_level"] = int(tags[f"king_tower{owner}_level"])
    return kwargs
