"""Faithful per-card levels: native l-encoding, clamps, King Tower levels and the converter tags."""

from __future__ import annotations

from dataclasses import replace

import pytest

from native_runner.match_factory import (
    CARD_LEVELS_VERSION,
    MatchConfig,
    match_level_kwargs,
    match_level_tags,
    rarity_min_level,
)
from native_runner.royaleapi_replay import (
    KING_TOWER_HITPOINTS_BY_LEVEL,
    _episode_match_config,
    prepare_collected_replay,
    resolve_native_card,
)
from native_runner.tests.test_royaleapi_replay import OPPONENT_KEYS, TEAM_KEYS, _payload

KNIGHT, MUSKETEER, BABY_DRAGON, PRINCESS, GOLDEN_KNIGHT = 26000000, 26000014, 26000015, 26000026, 26000074
ARCHERS, GIANT, MINI_PEKKA = 26000001, 26000003, 26000018
RARITY_DECK = (KNIGHT, MUSKETEER, BABY_DRAGON, PRINCESS, GOLDEN_KNIGHT, ARCHERS, GIANT, MINI_PEKKA)
PRINCESS_TOWER, CANNONEER, DAGGER_DUCHESS, ROYAL_CHEF = 159_000_000, 159_000_001, 159_000_002, 159_000_004


def _battle(match: MatchConfig) -> dict:
    return match.to_replay_dict()["battle"]


@pytest.mark.parametrize(
    ("card_id", "minimum"),
    [
        (KNIGHT, 1),  # Common
        (MUSKETEER, 3),  # Rare
        (BABY_DRAGON, 6),  # Epic
        (PRINCESS, 9),  # Legendary
        (GOLDEN_KNIGHT, 11),  # Champion
        (PRINCESS_TOWER, 1),
        (CANNONEER, 6),
        (DAGGER_DUCHESS, 9),
        (ROYAL_CHEF, 9),
    ],
)
def test_rarity_min_level(card_id: int, minimum: int) -> None:
    assert rarity_min_level(card_id) == minimum


def test_unknown_card_has_no_rarity() -> None:
    with pytest.raises(ValueError, match="no rarity curve"):
        rarity_min_level(26_999_999)


def test_card_levels_encode_level_minus_rarity_minimum_for_every_card() -> None:
    levels = (11, 14, 16, 9, 12, 16, 3, 13)
    battle = _battle(MatchConfig(deck0=RARITY_DECK, deck1=RARITY_DECK, deck0_card_levels=levels, deck1_card_levels=(16,) * 8))
    assert [entry["l"] for entry in battle["deck0"]["sp"]] == [10, 11, 10, 0, 1, 15, 0, 10]
    assert [entry["l"] for entry in battle["deck1"]["sp"]] == [15, 13, 10, 7, 5, 15, 13, 13]
    for owner, expected in ((0, levels), (1, (16,) * 8)):
        sp = battle[f"deck{owner}"]["sp"]
        assert [entry["d"] for entry in sp] == list(RARITY_DECK)
        assert [entry["l"] + rarity_min_level(entry["d"]) for entry in sp] == list(expected)


def test_card_levels_keep_form_availability() -> None:
    battle = _battle(
        MatchConfig(
            deck0_form_availability=(1, 0, 0, 0, 0, 2, 0, 3),
            deck0_card_levels=(16,) * 8,
            deck1_card_levels=(16,) * 8,
        )
    )
    assert battle["deck0"]["sp"][0] == {"d": KNIGHT, "el": 1, "l": 15}
    assert battle["deck0"]["sp"][1] == {"d": ARCHERS, "l": 15}


def test_without_card_levels_the_template_is_unchanged() -> None:
    battle = _battle(MatchConfig(level_cap=11, minimum_card_level=11, king_tower_level=11))
    assert all("l" not in entry for owner in (0, 1) for entry in battle[f"deck{owner}"]["sp"])
    assert (battle["lvlcap"], battle["cardlvlmin"]) == (11, 11)
    assert [row["kt"] for row in battle["hbd"]] == [11, 11]


@pytest.mark.parametrize(
    ("troop", "level", "index"),
    [(PRINCESS_TOWER, 16, 15), (PRINCESS_TOWER, 11, 10), (CANNONEER, 16, 10), (DAGGER_DUCHESS, 11, 2), (ROYAL_CHEF, 9, 0)],
)
def test_tower_troop_level_encoding(troop: int, level: int, index: int) -> None:
    battle = _battle(MatchConfig(tower_troop0_id=troop, tower_troop1_id=troop, tower_troop0_level=level, tower_troop1_level=16))
    assert battle["deck0"]["sc"][0]["d"] == troop
    assert battle["deck0"]["sc"][0]["l"] == index
    assert battle["deck1"]["sc"][0]["l"] == 16 - rarity_min_level(troop)


def test_king_tower_levels_per_owner() -> None:
    battle = _battle(MatchConfig(king_tower0_level=16, king_tower1_level=14))
    assert [row["kt"] for row in battle["hbd"]] == [16, 14]
    assert (battle["avatar0"]["expLevel"], battle["avatar1"]["expLevel"]) == (16, 14)
    # A per-owner level overrides the shared one only for that owner.
    battle = _battle(MatchConfig(king_tower_level=11, king_tower1_level=15))
    assert [row["kt"] for row in battle["hbd"]] == [11, 15]
    assert (battle["avatar0"]["expLevel"], battle["avatar1"]["expLevel"]) == (11, 15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"deck0_card_levels": (16,) * 7},
        {"deck0": RARITY_DECK, "deck0_card_levels": (16, 16, 16, 8, 16, 16, 16, 16)},  # Legendary below 9
        {"deck0": RARITY_DECK, "deck0_card_levels": (16, 16, 16, 16, 10, 16, 16, 16)},  # Champion below 11
        {"deck0_card_levels": (17,) + (16,) * 7},
        {"deck0_card_levels": (True,) + (16,) * 7},
        {"tower_troop0_id": CANNONEER, "tower_troop0_level": 5},
        {"tower_troop0_level": 17},
        {"king_tower0_level": 0},
        {"king_tower1_level": 17},
    ],
)
def test_invalid_levels_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MatchConfig(**kwargs)


def _per_card_match() -> MatchConfig:
    return MatchConfig(
        deck0=RARITY_DECK,
        deck1=RARITY_DECK[::-1],
        deck0_card_levels=(11, 14, 16, 9, 12, 16, 3, 13),
        deck1_card_levels=(15, 16, 15, 16, 15, 16, 15, 16),
        tower_troop0_id=CANNONEER,
        tower_troop0_level=15,
        tower_troop1_level=16,
        king_tower0_level=15,
        king_tower1_level=16,
    )


def test_level_tags_round_trip_through_any_deck_permutation() -> None:
    match = _per_card_match()
    tags = {"level_cap": 16, "minimum_card_level": 16, "king_tower_level": 16, **match_level_tags(match)}
    assert tags["card_levels_version"] == CARD_LEVELS_VERSION
    assert tags["deck0_card_levels"] == tuple(sorted(zip(match.deck0, match.deck0_card_levels)))
    kwargs = match_level_kwargs(match.deck0, match.deck1, tags)
    assert (kwargs["level_cap"], kwargs["minimum_card_level"], kwargs["king_tower_level"]) == (0, 0, None)
    assert kwargs["deck0_card_levels"] == match.deck0_card_levels
    assert kwargs["deck1_card_levels"] == match.deck1_card_levels
    assert (kwargs["king_tower0_level"], kwargs["king_tower1_level"]) == (15, 16)
    # Deal calibration permutes deck slots: levels follow their cards.
    permuted = match.deck0[3:] + match.deck0[:3]
    by_card = dict(zip(match.deck0, match.deck0_card_levels))
    assert match_level_kwargs(permuted, match.deck1, tags)["deck0_card_levels"] == tuple(by_card[c] for c in permuted)
    with pytest.raises(ValueError, match="cover"):
        match_level_kwargs((26000010,) + match.deck0[1:], match.deck1, tags)


def test_legacy_tags_keep_the_uniform_encoding() -> None:
    tags = {"level_cap": 11, "minimum_card_level": 11, "king_tower_level": 11}
    assert match_level_kwargs(RARITY_DECK, RARITY_DECK, tags) == {
        "level_cap": 11,
        "minimum_card_level": 11,
        "king_tower_level": 11,
    }
    assert match_level_tags(MatchConfig()) == {}
    with pytest.raises(ValueError, match="card_levels_version"):
        match_level_kwargs(RARITY_DECK, RARITY_DECK, {**tags, "card_levels_version": "other"})


# --------------------------------------------------------------------------------------- converter


def _players(payload: dict) -> dict[str, dict]:
    return {side: payload["battle"][side]["players"][0] for side in ("team", "opponent")}


def _mixed_payload() -> dict:
    payload = _payload()
    players = _players(payload)
    # data_i=1: team = True Red = owner 0, opponent = True Blue = owner 1.
    for index, item in enumerate(players["team"]["deck"]):
        item["level"] = 16 - (index % 3)  # 16, 15, 14, ...
    players["opponent"]["deck"][4]["level"] = 13  # the-log (Legendary)
    players["team"]["tower_card"] = {"card_key": "cannoneer", "level": 15}
    players["opponent"]["tower_card"] = {"card_key": "tower-princess", "level": 14}
    players["team"]["final_tower_hitpoints"] = {"king": 7728}  # undamaged King Tower 16
    players["opponent"]["final_tower_hitpoints"] = {"king": 1234}  # damaged
    return payload


def _expected_levels(payload: dict, side: str) -> tuple[tuple[int, int], ...]:
    deck = _players(payload)[side]["deck"]
    return tuple(sorted((resolve_native_card(item["card_key"]).card_id, item["level"]) for item in deck))


def test_converter_writes_true_levels_for_every_card_tower_and_king() -> None:
    payload = _mixed_payload()
    prepared = prepare_collected_replay(payload)
    episode = prepared.replay.episode_config
    tags = episode.tags
    assert tags["card_levels_version"] == CARD_LEVELS_VERSION
    assert tags["deck0_card_levels"] == _expected_levels(payload, "team")
    assert tags["deck1_card_levels"] == _expected_levels(payload, "opponent")
    assert (tags["tower_troop0_level"], tags["tower_troop1_level"]) == (15, 14)
    assert (tags["king_tower0_level"], tags["king_tower0_level_source"]) == (16, "final-king-hitpoints")
    assert (tags["king_tower1_level"], tags["king_tower1_level_source"]) == (14, "princess-level")
    # The uniform fallback (modal level) stays for readers without per-card support.
    assert (tags["level_cap"], tags["minimum_card_level"], tags["king_tower_level"]) == (16, 16, 16)
    assert not any("众数" in warning for warning in prepared.warnings)
    assert any("逐卡等级" in warning for warning in prepared.warnings)

    battle = _episode_match_config(episode).to_replay_dict()["battle"]
    assert (battle["lvlcap"], battle["cardlvlmin"]) == (0, 0)
    for owner, side in ((0, "team"), (1, "opponent")):
        expected = dict(_expected_levels(payload, side))
        sp = battle[f"deck{owner}"]["sp"]
        assert [entry["d"] for entry in sp] == list(getattr(episode, f"deck{owner}"))
        assert {entry["d"]: entry["l"] + rarity_min_level(entry["d"]) for entry in sp} == expected
    assert (battle["deck0"]["sc"][0]["d"], battle["deck0"]["sc"][0]["l"]) == (CANNONEER, 15 - 6)
    assert (battle["deck1"]["sc"][0]["d"], battle["deck1"]["sc"][0]["l"]) == (PRINCESS_TOWER, 14 - 1)
    assert [row["kt"] for row in battle["hbd"]] == [16, 14]
    assert (battle["avatar0"]["expLevel"], battle["avatar1"]["expLevel"]) == (16, 14)


def test_converter_uniform_game_is_effectively_the_legacy_uniform_match() -> None:
    payload = _payload()  # every card and both Tower Princesses at 16
    prepared = prepare_collected_replay(payload)
    episode = prepared.replay.episode_config
    assert not any("等级" in warning for warning in prepared.warnings)
    new = _episode_match_config(episode).to_replay_dict()["battle"]
    legacy_tags = {k: episode.tags[k] for k in ("level_cap", "minimum_card_level", "king_tower_level")}
    legacy = MatchConfig(
        deck0=episode.deck0, deck1=episode.deck1, **match_level_kwargs(episode.deck0, episode.deck1, legacy_tags)
    ).to_replay_dict()["battle"]
    assert (legacy["lvlcap"], legacy["cardlvlmin"], new["lvlcap"], new["cardlvlmin"]) == (16, 16, 0, 0)
    for owner in (0, 1):
        assert all(entry["l"] + rarity_min_level(entry["d"]) == 16 for entry in new[f"deck{owner}"]["sp"])
        assert new[f"deck{owner}"]["sc"][0]["l"] == 15
        assert new["hbd"][owner]["kt"] == legacy["hbd"][owner]["kt"] == 16
        assert new[f"avatar{owner}"]["expLevel"] == legacy[f"avatar{owner}"]["expLevel"] == 16


def test_converter_missing_levels_fall_back_to_the_modal_level_with_a_warning() -> None:
    payload = _payload()
    players = _players(payload)
    del players["team"]["deck"][0]["level"]  # giant-hero
    players["opponent"]["deck"][1]["level"] = None  # archers
    players["opponent"]["tower_card"]["level"] = 0  # seen in tournament payloads
    for item in players["team"]["deck"][1:]:
        item["level"] = 11
    players["team"]["tower_card"]["level"] = 11
    for item in players["opponent"]["deck"]:
        if item.get("level") is not None:
            item["level"] = 11
    prepared = prepare_collected_replay(payload)
    tags = prepared.replay.episode_config.tags
    giant, archers = resolve_native_card("giant-hero").card_id, resolve_native_card("archers").card_id
    assert dict(tags["deck0_card_levels"])[giant] == 11
    assert dict(tags["deck1_card_levels"])[archers] == 11
    assert tags["tower_troop1_level"] == 11
    # The princess level is missing for owner 1, so its King Tower follows the highest card level.
    assert (tags["king_tower1_level"], tags["king_tower1_level_source"]) == (11, "max-card-level")
    assert any("3 个卡牌/塔兵等级字段缺失" in warning and "众数等级 11" in warning for warning in prepared.warnings)


def test_converter_clamps_levels_into_the_rarity_range() -> None:
    payload = _payload()
    players = _players(payload)
    the_log = next(item for item in players["opponent"]["deck"] if item["card_key"] == "the-log")
    the_log["level"] = 3  # impossible for a Legendary
    players["team"]["deck"][1]["level"] = 18  # skeletons above the native maximum
    prepared = prepare_collected_replay(payload)
    tags = prepared.replay.episode_config.tags
    assert dict(tags["deck1_card_levels"])[resolve_native_card("the-log").card_id] == 9
    assert dict(tags["deck0_card_levels"])[resolve_native_card("skeletons").card_id] == 16
    assert any("截到 16" in warning for warning in prepared.warnings)
    assert any("抬到最低等级" in warning for warning in prepared.warnings)


@pytest.mark.parametrize(
    ("tower", "king_hitpoints", "expected"),
    [
        ({"card_key": "tower-princess", "level": 15}, KING_TOWER_HITPOINTS_BY_LEVEL[15], (15, "final-king-hitpoints")),
        ({"card_key": "tower-princess", "level": 15}, KING_TOWER_HITPOINTS_BY_LEVEL[16], (16, "final-king-hitpoints")),
        # A damaged King Tower that lands on a far table value is not evidence.
        ({"card_key": "tower-princess", "level": 16}, KING_TOWER_HITPOINTS_BY_LEVEL[6], (16, "princess-level")),
        ({"card_key": "tower-princess", "level": 15}, 5000, (15, "princess-level")),
        ({"card_key": "dagger-duchess", "level": 15}, 0, (16, "max-card-level")),
        ({"card_key": "royal-chef", "level": 15}, KING_TOWER_HITPOINTS_BY_LEVEL[15], (15, "final-king-hitpoints")),
    ],
)
def test_converter_king_tower_level(tower: dict, king_hitpoints: int, expected: tuple[int, str]) -> None:
    payload = _payload()
    team = _players(payload)["team"]
    team["tower_card"] = tower
    team["final_tower_hitpoints"] = {"king": king_hitpoints}
    tags = prepare_collected_replay(payload).replay.episode_config.tags
    assert (tags["king_tower0_level"], tags["king_tower0_level_source"]) == expected


def test_converter_level_tags_survive_deal_calibration_permutations() -> None:
    payload = _mixed_payload()
    episode = prepare_collected_replay(payload).replay.episode_config
    permuted = replace(episode, deck0=episode.deck0[::-1], deck1=episode.deck1[2:] + episode.deck1[:2])
    battle = _episode_match_config(permuted).to_replay_dict()["battle"]
    for owner, side in ((0, "team"), (1, "opponent")):
        expected = dict(_expected_levels(payload, side))
        assert {e["d"]: e["l"] + rarity_min_level(e["d"]) for e in battle[f"deck{owner}"]["sp"]} == expected


def test_fixture_decks_cover_the_expected_rarities() -> None:
    # Guards the converter tests above: the opponent deck carries a Legendary (the-log).
    minimums = {key: rarity_min_level(resolve_native_card(key).card_id) for key in (*TEAM_KEYS, *OPPONENT_KEYS)}
    assert minimums["the-log"] == 9 and minimums["giant-hero"] == 3 and minimums["skeletons"] == 1
