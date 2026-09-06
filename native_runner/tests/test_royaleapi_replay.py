from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

import native_runner.royaleapi_replay as royaleapi_replay
from native_runner.contracts import ActionKind
from native_runner.cr_native_env import RunnerError
from native_runner.royaleapi_replay import (
    CLASSIC_CHALLENGE_ARENA_ID,
    CLASSIC_CHALLENGE_GAME_MODE_ID,
    CLASSIC_CHALLENGE_LOCATION_ID,
    CONTROL_DATABASE_RELATIVE,
    NATIVE_DEAL_LAYOUTS,
    NativeRenderConfigurationLoadError,
    POLICY_DEAL_INVARIANCE_VERSION,
    POLICY_EPISODE_CONFIG_VERSION,
    ROYALAPI_CARD_KEY_IDS,
    ROYALAPI_COMMAND_TO_NATIVE_OBSERVABLE_TICKS,
    ROYALAPI_TOWER_TROOP_IDS,
    RoyaleAPIReplayError,
    _episode_match_config,
    calibrate_collected_replay_deal,
    list_collected_replays,
    load_collected_replay_payload,
    prepare_collected_replay,
    policy_episode_config_id,
    resolve_native_card,
)


TEAM_KEYS = (
    "giant-hero",
    "skeletons",
    "mini-pekka",
    "fireball",
    "zap-ev1",
    "musketeer",
    "cannon",
    "ice-spirit",
)
OPPONENT_KEYS = (
    "knight",
    "archers",
    "hog-rider",
    "rocket",
    "the-log",
    "tesla",
    "goblins",
    "fire-spirit",
)


def test_anonymized_replay_id_requires_marker_and_remains_playable() -> None:
    payload = _payload(replay_tag="84a066ec0b0a4bef801ae287e4d45960")
    with pytest.raises(RoyaleAPIReplayError, match="replay tag"):
        prepare_collected_replay(payload)
    payload["source"] = {"replay_tag": payload["source"]["replay_tag"], "anonymized": True}
    for key in ("requested_player_tag", "played_at_utc", "played_at_local"):
        payload["battle"].pop(key)
    for side in ("team", "opponent"):
        for player in payload["battle"][side]["players"]:
            player.pop("name")
            player.pop("tag")
    prepared = prepare_collected_replay(payload)
    assert prepared.event_count == len(payload["events"])
    assert prepared.replay.created_at_ns == 1
    payload["source"]["replay_tag"] = "../" + "a" * 32
    with pytest.raises(RoyaleAPIReplayError, match="replay tag"):
        prepare_collected_replay(payload)


def test_tower_troop_ids_preserve_hidden_native_support_card_slot() -> None:
    assert ROYALAPI_TOWER_TROOP_IDS == {
        "tower-princess": 159_000_000,
        "cannoneer": 159_000_001,
        "dagger-duchess": 159_000_002,
        "royal-chef": 159_000_004,
    }
    assert 159_000_003 not in ROYALAPI_TOWER_TROOP_IDS.values()


def _display_name(key: str) -> str:
    return key.removesuffix("-hero").removesuffix("-ev1").replace("-", " ").title()


def _deck_card(card_key: str) -> royaleapi_replay._DeckCard:
    info = resolve_native_card(card_key)
    return royaleapi_replay._DeckCard(
        key=card_key,
        name=_display_name(card_key),
        info=info,
        form_mask=royaleapi_replay._deck_form_mask(card_key, info),
        level=16,
    )


def _event(
    index: int,
    *,
    tick: int,
    side: str,
    card_key: str,
    data_i: int,
) -> dict[str, object]:
    y = 10 if side == "team" else 21
    return {
        "source_index": index,
        "replay_tick_20hz": tick,
        "side": side,
        "kind": "play_card",
        "card_key": card_key.removesuffix("-hero").removesuffix("-ev1"),
        "source_fields": {"data_i": data_i},
        "coordinates": {
            "inside_18x32_arena": True,
            "grid_cell_floor": {"x": 8, "y": y},
            "subcell_offset_from_floor_center": {"x": 0.0, "y": 0.0},
        },
    }


def _payload(
    replay_tag: str = "02PYPPJGVQ80",
    *,
    data_i: int = 1,
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    tick = 140
    for index in range(8):
        events.append(
            _event(
                index * 2,
                tick=tick,
                side="team",
                card_key=TEAM_KEYS[index],
                data_i=data_i,
            )
        )
        tick += 35
        events.append(
            _event(
                index * 2 + 1,
                tick=tick,
                side="opponent",
                card_key=OPPONENT_KEYS[index],
                data_i=data_i,
            )
        )
        tick += 35
    return {
        "schema_version": "royaleapi-battle-replay.v1",
        "source": {
            "replay_tag": replay_tag,
            "fetched_at": "2026-07-30T12:00:00+00:00",
        },
        "battle": {
            "requested_player_tag": "JUC2UJC",
            "played_at_utc": "2026-07-29T01:02:03+00:00",
            "played_at_local": "2026-07-29T09:02:03+08:00",
            "result": "win",
            "team": {
                "crowns": 3,
                "players": [
                    {
                        "name": "Team",
                        "tag": "JUC2UJC",
                        "deck": [
                            {
                                "card_key": key,
                                "name": _display_name(key),
                                "level": 16,
                            }
                            for key in TEAM_KEYS
                        ],
                        "tower_card": {
                            "card_key": "tower-princess",
                            "level": 16,
                        },
                    }
                ],
            },
            "opponent": {
                "crowns": 0,
                "players": [
                    {
                        "name": "Opponent",
                        "tag": "YG0PCL80L",
                        "deck": [
                            {
                                "card_key": key,
                                "name": _display_name(key),
                                "level": 16,
                            }
                            for key in OPPONENT_KEYS
                        ],
                        "tower_card": {
                            "card_key": "tower-princess",
                            "level": 16,
                        },
                    }
                ],
            },
        },
        "events": events,
    }


def _assert_recorded_slots_follow_cycle(
    deck: tuple[int, ...],
    owner: int,
    actions: list[object],
) -> None:
    opening_slots, queue_slots = NATIVE_DEAL_LAYOUTS[owner]
    hand = {
        deck[deck_slot]: hand_index
        for hand_index, deck_slot in enumerate(opening_slots)
    }
    queue = [deck[deck_slot] for deck_slot in queue_slots]
    for action in actions:
        card_id = action.card_id
        assert card_id in hand
        hand_index = hand.pop(card_id)
        assert action.hand_slot == hand_index
        drawn = queue.pop(0)
        hand[drawn] = hand_index
        queue.append(card_id)


def test_prepare_replay_selects_deterministic_compatible_native_orders() -> None:
    first = prepare_collected_replay(_payload())
    second = prepare_collected_replay(_payload())

    assert first.opening_cards == second.opening_cards
    assert first.queue_cards == second.queue_cards
    assert first.replay.episode_config.deck0 == second.replay.episode_config.deck0
    assert first.event_count == 16
    assert first.replay.terminal["winner"] == 0

    actions_by_owner = {0: [], 1: []}
    for operation in first.replay.operations:
        for action in operation.actions:
            if action.kind is ActionKind.PLAY_CARD:
                actions_by_owner[action.owner].append(action)
    _assert_recorded_slots_follow_cycle(
        first.replay.episode_config.deck0,
        0,
        actions_by_owner[0],
    )
    _assert_recorded_slots_follow_cycle(
        first.replay.episode_config.deck1,
        1,
        actions_by_owner[1],
    )


def test_prepare_replay_attests_selected_deal_from_opening_boundary() -> None:
    prepared = prepare_collected_replay(_payload())
    owner0, owner1 = prepared.policy_deal_invariance

    assert POLICY_DEAL_INVARIANCE_VERSION == ("royaleapi-policy-selected-deal.v2")
    assert owner0.candidate_count == owner1.candidate_count == 1
    assert owner0.state_count_by_play_count == (1,) * 9
    assert owner1.state_count_by_play_count == owner0.state_count_by_play_count
    assert owner0.first_invariant_play_count == 0
    assert owner1.first_invariant_play_count == 0
    assert owner0.first_output_command_group_index == 0
    assert owner0.first_output_command_tick == 140
    assert owner1.first_output_command_group_index == 0
    assert owner1.first_output_command_tick == 140
    assert owner0.first_output_prior_play_count == 0
    assert owner1.first_output_prior_play_count == 0
    for evidence in (owner0, owner1):
        assert len(evidence.first_invariant_hand) == 4
        assert len(evidence.first_invariant_cycle) == 4
        assert set(
            (*evidence.first_invariant_hand, *evidence.first_invariant_cycle)
        ) == set(evidence.source_deck)
        assert evidence.first_output_hand == evidence.first_invariant_hand
        assert evidence.first_output_cycle == evidence.first_invariant_cycle

    tags = prepared.replay.episode_config.tags
    assert tags["policy_deal_invariance_version"] == (POLICY_DEAL_INVARIANCE_VERSION)
    assert tags["policy_deal_invariance_id"] == (prepared.policy_deal_invariance_id)
    assert tuple(item["evidence_id"] for item in tags["policy_deal_invariance"]) == (
        owner0.evidence_id,
        owner1.evidence_id,
    )


def test_selected_deal_is_available_to_both_owners_from_the_first_boundary() -> None:
    payload = _payload()
    payload["events"] = payload["events"][:8]

    prepared = prepare_collected_replay(payload)
    owner0, owner1 = prepared.policy_deal_invariance

    for evidence in (owner0, owner1):
        assert evidence.candidate_count == 1
        assert evidence.first_invariant_play_count == 0
        assert evidence.first_output_command_group_index == 0
        assert evidence.first_output_command_tick == 140
        assert evidence.first_output_prior_play_count == 0
        assert evidence.first_output_hand == evidence.first_invariant_hand
        assert evidence.first_output_cycle == evidence.first_invariant_cycle


def test_short_replay_still_uses_one_selected_deal_per_owner() -> None:
    payload = _payload()
    payload["events"] = payload["events"][:2]

    prepared = prepare_collected_replay(payload)

    for evidence in prepared.policy_deal_invariance:
        assert evidence.candidate_count == 1
        assert evidence.state_count_by_play_count == (1, 1)
        assert evidence.first_invariant_play_count == 0
        assert evidence.first_output_command_group_index == 0
        assert evidence.first_output_command_tick == 140
        assert evidence.first_invariant_hand
        assert evidence.first_invariant_cycle


def test_prepare_replay_maps_source_command_ticks_to_native_observable_ticks() -> None:
    payload = _payload()

    prepared = prepare_collected_replay(payload)
    actions = {
        int(action.metadata["source_index"]): (operation, action)
        for operation in prepared.replay.operations
        for action in operation.actions
        if action.kind is ActionKind.PLAY_CARD
    }

    assert ROYALAPI_COMMAND_TO_NATIVE_OBSERVABLE_TICKS == 1
    for event in payload["events"]:
        operation, action = actions[int(event["source_index"])]
        source_tick = int(event["replay_tick_20hz"])
        native_tick = operation.start_native_tick + action.execute_offset_ticks
        assert native_tick == source_tick + 1
        assert operation.end_native_tick == native_tick
        assert action.metadata["source_command_tick"] == source_tick
        assert action.metadata["native_observable_tick"] == native_tick

    tags = prepared.replay.episode_config.tags
    assert prepared.replay.ruleset_id == "royaleapi-native-render-adapter.v3"
    assert tags["reconstruction_mode"] == "royaleapi-public-actions.v3"
    assert tags["source_tick_semantics"] == "royaleapi-command-boundary.v1"
    assert tags["native_observable_tick_offset"] == 1


@pytest.mark.parametrize("card_key", ("elixir-collector", "mirror"))
def test_native_card_index_exposes_omit_from_starting_hand(card_key: str) -> None:
    assert resolve_native_card(card_key).omit_from_starting_hand is True


def _payload_with_team_elixir_collector() -> dict[str, object]:
    payload = _payload("09C9JLPL8PVU")
    payload["battle"]["team"]["players"][0]["deck"][7].update(
        card_key="elixir-collector",
        name="Elixir Collector",
    )
    for event in payload["events"]:
        if event["side"] == "team" and event["card_key"] == "ice-spirit":
            event["card_key"] = "elixir-collector"
    return payload


def _payload_with_team_multi_omitted() -> dict[str, object]:
    payload = _payload("09C9JLPL8PVV")
    replacements = {
        "cannon": ("mirror", "Mirror"),
        "ice-spirit": ("elixir-collector", "Elixir Collector"),
    }
    for card in payload["battle"]["team"]["players"][0]["deck"]:
        replacement = replacements.get(card["card_key"])
        if replacement is not None:
            card["card_key"], card["name"] = replacement
    for event in payload["events"]:
        if event["side"] != "team":
            continue
        replacement = replacements.get(event["card_key"])
        if replacement is not None:
            event["card_key"] = replacement[0]
    return payload


def test_prepare_replay_excludes_elixir_collector_from_opening_candidates() -> None:
    prepared = prepare_collected_replay(_payload_with_team_elixir_collector())
    collector_id = resolve_native_card("elixir-collector").card_id

    assert collector_id not in prepared.opening_cards[0]
    assert collector_id in prepared.queue_cards[0]
    assert prepared.replay.episode_config.tags["deck0_omit_from_starting_hand_ids"] == (
        collector_id,
    )
    assert any(
        "Elixir Collector 不会出现在初始手牌" in warning
        for warning in prepared.warnings
    )


def test_prepare_replay_accepts_two_omitted_starting_hand_cards() -> None:
    prepared = prepare_collected_replay(_payload_with_team_multi_omitted())
    mirror_id = resolve_native_card("mirror").card_id
    collector_id = resolve_native_card("elixir-collector").card_id
    omitted = (mirror_id, collector_id)

    assert not set(omitted).intersection(prepared.opening_cards[0])
    assert set(omitted).issubset(prepared.queue_cards[0])
    assert (
        prepared.replay.episode_config.tags["deck0_omit_from_starting_hand_ids"]
        == omitted
    )


def test_multi_omit_slot_probe_is_deterministic_unique_and_stable() -> None:
    deck = tuple(range(1, 9))
    omitted = (2, 7)
    initial_positions = tuple(deck.index(card_id) for card_id in omitted)

    def enumerate_prefix() -> tuple[tuple[int, ...], ...]:
        tested = {initial_positions}
        outputs: list[tuple[int, ...]] = []
        for _ in range(12):
            relocated = royaleapi_replay._relocate_omitted_card_for_probe(
                replay_tag="09C9JLPL8PVV",
                owner=0,
                deck=deck,
                omitted_ids=omitted,
                tested_positions=tested,
            )
            positions = tuple(relocated.index(card_id) for card_id in omitted)
            assert positions not in tested
            assert set(relocated) == set(deck)
            assert tuple(
                card_id for card_id in relocated if card_id not in omitted
            ) == tuple(card_id for card_id in deck if card_id not in omitted)
            tested.add(positions)
            outputs.append(relocated)
        return tuple(outputs)

    assert enumerate_prefix() == enumerate_prefix()

    all_two_card_positions = {
        (first, second)
        for first in range(len(deck))
        for second in range(len(deck))
        if first != second
    }
    with pytest.raises(RoyaleAPIReplayError, match="全部多禁止初手牌"):
        royaleapi_replay._relocate_omitted_card_for_probe(
            replay_tag="09C9JLPL8PVV",
            owner=0,
            deck=deck,
            omitted_ids=omitted,
            tested_positions=all_two_card_positions,
        )

    relocated_all = royaleapi_replay._relocate_omitted_card_for_probe(
        replay_tag="09C9JLPL8PVV",
        owner=0,
        deck=deck,
        omitted_ids=deck,
        tested_positions={tuple(range(len(deck)))},
    )
    assert len(relocated_all) == len(deck)
    assert set(relocated_all) == set(deck)


def test_prepare_replay_uses_legendary_arena_for_classic_challenge() -> None:
    payload = _payload()
    payload["battle"]["game_mode"] = "Classic Challenge"

    prepared = prepare_collected_replay(payload)
    episode = prepared.replay.episode_config
    native_battle = _episode_match_config(episode).to_replay_dict()["battle"]

    assert CLASSIC_CHALLENGE_ARENA_ID == 54_000_036
    assert CLASSIC_CHALLENGE_LOCATION_ID == 15_000_013
    assert episode.game_mode == CLASSIC_CHALLENGE_GAME_MODE_ID
    assert episode.arena == CLASSIC_CHALLENGE_ARENA_ID
    assert episode.tags["location"] == CLASSIC_CHALLENGE_LOCATION_ID
    assert episode.tags["source_game_mode"] == "Classic Challenge"
    assert episode.tags["native_arena_profile"] == "classic-legendary-arena.v1"
    assert native_battle["gamemode"] == CLASSIC_CHALLENGE_GAME_MODE_ID
    assert native_battle["arena"] == CLASSIC_CHALLENGE_ARENA_ID
    assert native_battle["location"] == CLASSIC_CHALLENGE_LOCATION_ID
    assert any("Arena_Legendary / PvP_champion" in item for item in prepared.warnings)


def test_prepare_replay_preserves_data_i_zero_true_blue_team_at_bottom() -> None:
    payload = _payload(data_i=0)
    payload["battle"]["team"]["players"][0]["tower_card"]["card_key"] = "royal-chef"
    payload["battle"]["opponent"]["players"][0]["tower_card"]["card_key"] = (
        "dagger-duchess"
    )
    payload["events"][0]["coordinates"]["subcell_offset_from_floor_center"] = {
        "x": 0.25,
        "y": -0.125,
    }
    payload["events"][0]["coordinates"]["grid_cell_floor"] = {"x": 2, "y": 10}

    prepared = prepare_collected_replay(
        payload,
        bottom_player_tag="#JUC2UJC",
    )
    episode = prepared.replay.episode_config
    actions = {
        int(action.metadata["source_index"]): action
        for operation in prepared.replay.operations
        for action in operation.actions
        if action.kind is ActionKind.PLAY_CARD
    }

    assert episode.tags["owner0_name"] == "Opponent"
    assert episode.tags["owner1_name"] == "Team"
    assert episode.tags["source_team_owner"] == 1
    assert episode.tags["royaleapi_data_i"] == 0
    assert episode.tags["source_team_true_side"] == "blue"
    assert episode.tags["source_true_blue_side"] == "team"
    assert episode.tags["source_true_red_side"] == "opponent"
    assert episode.tags["arena_horizontal_flip"] is False
    assert episode.tags["arena_vertical_flip"] is True
    assert episode.tags["bottom_player_tag"] == "JUC2UJC"
    assert episode.tags["tower_troop0_id"] == ROYALAPI_TOWER_TROOP_IDS["dagger-duchess"]
    assert episode.tags["tower_troop1_id"] == ROYALAPI_TOWER_TROOP_IDS["royal-chef"]
    assert actions[0].owner == 1
    assert actions[0].target_grid == (2, 21)
    assert actions[0].subcell_offset == (0.25, 0.125)
    assert actions[1].owner == 0
    assert actions[1].target_grid == (8, 10)
    assert prepared.replay.terminal["winner"] == 1
    assert prepared.replay.terminal["players"] == [
        {"owner": 0, "crowns": 0},
        {"owner": 1, "crowns": 3},
    ]
    assert any("data_i=0" in warning for warning in prepared.warnings)


def test_prepare_replay_preserves_data_i_one_true_red_team_at_top() -> None:
    payload = _payload(data_i=1)
    payload["events"][0]["coordinates"]["grid_cell_floor"] = {"x": 2, "y": 10}
    payload["events"][0]["coordinates"]["subcell_offset_from_floor_center"] = {
        "x": 0.25,
        "y": -0.125,
    }

    prepared = prepare_collected_replay(payload)
    episode = prepared.replay.episode_config
    action = next(
        action
        for operation in prepared.replay.operations
        for action in operation.actions
        if action.kind is ActionKind.PLAY_CARD
        and int(action.metadata["source_index"]) == 0
    )

    assert episode.tags["owner0_name"] == "Team"
    assert episode.tags["owner1_name"] == "Opponent"
    assert episode.tags["source_team_owner"] == 0
    assert episode.tags["royaleapi_data_i"] == 1
    assert episode.tags["source_team_true_side"] == "red"
    assert episode.tags["source_true_blue_side"] == "opponent"
    assert episode.tags["source_true_red_side"] == "team"
    assert episode.tags["arena_horizontal_flip"] is True
    assert episode.tags["arena_vertical_flip"] is False
    assert episode.tags["bottom_player_tag"] == "YG0PCL80L"
    assert action.owner == 0
    assert action.target_grid == (15, 10)
    assert action.subcell_offset == (-0.25, -0.125)
    assert prepared.replay.terminal["winner"] == 0


def test_prepare_replay_rejects_mixed_true_side_markers() -> None:
    payload = _payload(data_i=0)
    payload["events"][1]["source_fields"]["data_i"] = 1

    with pytest.raises(RoyaleAPIReplayError, match="mixes RoyaleAPI data_i"):
        prepare_collected_replay(payload)


def test_prepare_replay_rejects_missing_true_side_marker() -> None:
    payload = _payload()
    for event in payload["events"]:
        event.pop("source_fields")

    with pytest.raises(RoyaleAPIReplayError, match="has no RoyaleAPI data_i marker"):
        prepare_collected_replay(payload)


def test_prepare_replay_rejects_bottom_override_that_swaps_true_sides() -> None:
    with pytest.raises(RoyaleAPIReplayError, match="conflicts with replay"):
        prepare_collected_replay(
            _payload(data_i=1),
            bottom_player_tag="JUC2UJC",
        )


def test_prepare_replay_rejects_bottom_player_not_in_battle() -> None:
    with pytest.raises(RoyaleAPIReplayError, match="在对局双方中出现 0 次"):
        prepare_collected_replay(
            _payload(),
            bottom_player_tag="P89P2JYUP",
        )


def test_prepare_replay_keeps_selected_hero_and_evolution_masks() -> None:
    prepared = prepare_collected_replay(_payload())
    episode = prepared.replay.episode_config
    masks = dict(
        zip(
            episode.deck0,
            episode.tags["deck0_form_availability"],
            strict=True,
        )
    )

    assert masks[resolve_native_card("giant").card_id] == 2
    assert masks[resolve_native_card("zap").card_id] == 1
    assert masks[resolve_native_card("skeletons").card_id] == 0


@pytest.mark.parametrize(
    ("team_tower", "opponent_tower"),
    (
        ("royal-chef", "dagger-duchess"),
        ("cannoneer", "tower-princess"),
    ),
)
def test_prepare_replay_configures_collected_tower_troops(
    team_tower: str,
    opponent_tower: str,
) -> None:
    payload = _payload()
    payload["battle"]["team"]["players"][0]["tower_card"]["card_key"] = team_tower
    payload["battle"]["opponent"]["players"][0]["tower_card"]["card_key"] = (
        opponent_tower
    )

    prepared = prepare_collected_replay(payload)
    tags = prepared.replay.episode_config.tags
    match = _episode_match_config(prepared.replay.episode_config)
    native_payload = match.to_replay_dict()

    assert tags["tower_troop0_id"] == ROYALAPI_TOWER_TROOP_IDS[team_tower]
    assert tags["tower_troop1_id"] == ROYALAPI_TOWER_TROOP_IDS[opponent_tower]
    assert native_payload["battle"]["deck0"]["sc"][0]["d"] == tags["tower_troop0_id"]
    assert native_payload["battle"]["deck1"]["sc"][0]["d"] == tags["tower_troop1_id"]
    assert any("已按原局配置双方塔兵" in warning for warning in prepared.warnings)


def test_prepare_replay_rejects_unknown_tower_troop() -> None:
    payload = _payload()
    payload["battle"]["team"]["players"][0]["tower_card"]["card_key"] = "future-tower"

    with pytest.raises(RoyaleAPIReplayError, match="拒绝静默替换成公主塔"):
        prepare_collected_replay(payload)


def test_prepare_replay_binds_hero_events_and_keeps_source_duration() -> None:
    payload = _payload()
    payload["events"].append(
        {
            "source_index": 16,
            "replay_tick_20hz": 700,
            "side": "team",
            "kind": "activate_ability",
            "card_key": None,
            # Exercise the adapter fallback used by older collector rows.
            "ability_source_candidates": [],
            "coordinates": None,
        }
    )
    payload["replay"] = {
        "duration": {
            "display_label": "0:40",
            "display_seconds": 40,
            "timeline_seconds": 40.0,
        }
    }

    prepared = prepare_collected_replay(payload)
    abilities = [
        action
        for operation in prepared.replay.operations
        for action in operation.actions
        if action.kind is ActionKind.ACTIVATE_ABILITY
    ]

    assert prepared.replay.end_native_tick == 800
    assert prepared.replay.terminal["last_source_event_tick"] == 700
    assert prepared.replay.terminal["last_native_event_tick"] == 701
    assert prepared.replay.terminal["source_end_tick"] == 800
    assert len(abilities) == 1
    ability_operation = next(
        operation
        for operation in prepared.replay.operations
        if abilities[0] in operation.actions
    )
    assert (
        ability_operation.start_native_tick + abilities[0].execute_offset_ticks == 701
    )
    assert abilities[0].metadata["source_command_tick"] == 700
    assert abilities[0].metadata["native_observable_tick"] == 701
    assert abilities[0].metadata["ability_source_keys"] == ("giant-hero",)
    assert "giant" in abilities[0].metadata["ability_runtime_hints"]
    assert any("已按牌组唯一绑定" in warning for warning in prepared.warnings)


def test_prepare_replay_adds_verified_mini_pekka_native_ability_alias() -> None:
    payload = _payload()
    team_deck = payload["battle"]["team"]["players"][0]["deck"]
    team_deck[0].update(card_key="giant", name="Giant")
    team_deck[2].update(card_key="mini-pekka-hero", name="Mini P.E.K.K.A")
    payload["events"].append(
        {
            "source_index": 16,
            "replay_tick_20hz": 700,
            "side": "team",
            "kind": "activate_ability",
            "card_key": None,
            "ability_source_candidates": ["mini-pekka"],
            "coordinates": None,
        }
    )

    prepared = prepare_collected_replay(payload)
    ability = next(
        action
        for operation in prepared.replay.operations
        for action in operation.actions
        if action.kind is ActionKind.ACTIVATE_ABILITY
    )

    assert ability.metadata["ability_runtime_hints"] == (
        "minipekk",
        "minipekka",
    )


def test_all_playable_hero_forms_have_a_first_try_native_runtime_hint() -> None:
    expected_keys = {
        "balloon",
        "barbarian-barrel",
        "berserker",
        "bowler",
        "dark-prince",
        "giant",
        "goblins",
        "ice-golem",
        "knight",
        "magic-archer",
        "mega-minion",
        "mini-pekka",
        "musketeer",
        "tombstone",
        "valkyrie",
        "wizard",
    }

    assert set(royaleapi_replay.NATIVE_HERO_FORM_ABILITY_BINDINGS) == expected_keys
    for card_key, (
        runtime_name,
        expected_hint,
    ) in royaleapi_replay.NATIVE_HERO_FORM_ABILITY_BINDINGS.items():
        hints = royaleapi_replay._ability_runtime_hints(
            (_deck_card(f"{card_key}-hero"),)
        )

        assert expected_hint.casefold() in runtime_name.casefold()
        assert hints[0] == expected_hint


def test_all_playable_champions_have_a_first_try_native_runtime_hint() -> None:
    champion_keys = {
        26_000_065: "mighty-miner",
        26_000_069: "skeleton-king",
        26_000_072: "archer-queen",
        26_000_074: "golden-knight",
        26_000_077: "monk",
        26_000_093: "little-prince",
        26_000_099: "goblinstein",
        26_000_103: "boss-bandit",
    }

    assert set(royaleapi_replay.NATIVE_CHAMPION_ABILITY_BINDINGS) == set(champion_keys)
    for card_id, card_key in champion_keys.items():
        runtime_name, expected_hint = royaleapi_replay.NATIVE_CHAMPION_ABILITY_BINDINGS[
            card_id
        ]
        card = _deck_card(card_key)
        hints = royaleapi_replay._ability_runtime_hints((card,))

        assert card.info.card_id == card_id
        assert royaleapi_replay._active_ability_deck_cards((card,)) == (card,)
        assert expected_hint.casefold() in runtime_name.casefold()
        assert hints[0] == expected_hint


def test_ability_source_fallback_includes_goblinstein_not_rune_giant() -> None:
    deck = (_deck_card("rune-giant"), _deck_card("goblinstein"))

    sources = royaleapi_replay._ability_source_cards(
        {"ability_source_candidates": []},
        deck,
    )

    assert tuple(card.key for card in sources) == ("goblinstein",)


def test_non_authoritative_legacy_candidates_do_not_hide_a_champion() -> None:
    hero = _deck_card("mini-pekka-hero")
    champion = _deck_card("monk")
    deck = (hero, champion)

    inferred = royaleapi_replay._ability_source_cards(
        {
            "ability_source_candidates": ["mini-pekka-hero"],
            "ability_source_authoritative": False,
        },
        deck,
    )
    authoritative = royaleapi_replay._ability_source_cards(
        {
            "ability_source_candidates": ["mini-pekka-hero"],
            "ability_source_authoritative": True,
        },
        deck,
    )

    assert inferred == deck
    assert authoritative == (hero,)


def test_native_render_deal_is_probed_remapped_and_verified() -> None:
    prepared = prepare_collected_replay(_payload())
    render_layouts = {
        0: ((7, 2, 0, 5), (6, 1, 3, 4)),
        1: ((7, 1, 4, 3), (6, 0, 2, 5)),
    }

    class FakeNative:
        def __init__(self) -> None:
            self.calls = 0

        def create_native_match(self, config, *, wait_timeout):
            assert wait_timeout == 30.0
            self.calls += 1
            players = []
            for owner, deck in enumerate((config.deck0, config.deck1)):
                opening_slots, queue_slots = render_layouts[owner]
                players.append(
                    {
                        "owner": owner,
                        "hand": [
                            {
                                "handIndex": hand_index,
                                "deckSlot": deck_slot,
                                "cardId": deck[deck_slot],
                            }
                            for hand_index, deck_slot in enumerate(opening_slots)
                        ],
                        "cycle": [
                            {
                                "cycleIndex": cycle_index,
                                "deckSlot": deck_slot,
                                "cardId": deck[deck_slot],
                            }
                            for cycle_index, deck_slot in enumerate(queue_slots)
                        ],
                    }
                )
            return {"players": players}

        def pause(self) -> None:
            return None

    native = FakeNative()
    calibrated = calibrate_collected_replay_deal(prepared, native)

    assert 1 <= native.calls <= 2
    episode = calibrated.replay.episode_config
    assert calibrated.policy_episode_config_id == policy_episode_config_id(
        calibrated.replay_tag,
        episode,
    )
    assert episode.tags["policy_episode_config_version"] == (
        POLICY_EPISODE_CONFIG_VERSION
    )
    assert episode.tags["policy_episode_config_id"] == (
        calibrated.policy_episode_config_id
    )
    for owner, deck in enumerate((episode.deck0, episode.deck1)):
        opening_slots, queue_slots = render_layouts[owner]
        assert (
            tuple(deck[index] for index in opening_slots)
            == (calibrated.opening_cards[owner])
        )
        assert (
            tuple(deck[index] for index in queue_slots)
            == (calibrated.queue_cards[owner])
        )
    assert f"{native.calls} 次配置" in calibrated.warnings[-1]


def test_native_render_calibration_bubbles_non_load_runner_failure() -> None:
    prepared = prepare_collected_replay(_payload())
    attempt_messages: list[str] = []

    class TransientNative:
        def __init__(self) -> None:
            self.calls = 0
            self.pause_calls = 0

        def create_native_match(self, config, *, wait_timeout):
            del config
            assert wait_timeout == 30.0
            self.calls += 1
            raise RunnerError("temporary native control connection failure")

        def pause(self) -> None:
            self.pause_calls += 1

    native = TransientNative()
    with pytest.raises(
        RunnerError,
        match="temporary native control connection failure",
    ):
        calibrate_collected_replay_deal(
            prepared,
            native,
            max_attempts=10,
            on_attempt=attempt_messages.append,
        )

    assert native.calls == 1
    assert native.pause_calls == 0
    assert attempt_messages == []


def test_native_render_load_timeout_surfaces_immediately_with_status() -> None:
    prepared = prepare_collected_replay(_payload())
    attempt_messages: list[str] = []

    class TimedOutNative:
        def __init__(self) -> None:
            self.calls = 0
            self.status_calls = 0
            self.pause_calls = 0

        def create_native_match(self, _config, *, wait_timeout):
            assert wait_timeout == 30.0
            self.calls += 1
            raise RunnerError("native renderer did not load configuration 6")

        def status(self):
            self.status_calls += 1
            return {
                "ok": True,
                "mode": "native-render",
                "nativeRenderReady": False,
                "nativeRenderSequence": 6,
                "nativeRenderProcessed": 1,
                "nativeRenderLoaded": 0,
            }

        def pause(self) -> None:
            self.pause_calls += 1

    native = TimedOutNative()
    with pytest.raises(NativeRenderConfigurationLoadError) as raised:
        calibrate_collected_replay_deal(
            prepared,
            native,
            max_attempts=10,
            on_attempt=attempt_messages.append,
        )

    error = raised.value
    assert native.calls == 1
    assert native.status_calls == 1
    assert native.pause_calls == 0
    assert attempt_messages == []
    assert error.requested_sequence == 6
    assert error.status_sequence == 6
    assert error.status_processed == 1
    assert error.status_loaded == 0
    assert "status sequence=6/processed=1/loaded=0" in str(error)


def test_native_render_calibration_does_not_remap_without_complete_4_plus_4() -> None:
    prepared = prepare_collected_replay(_payload())

    class IncompleteNative:
        def __init__(self) -> None:
            self.configured_decks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def create_native_match(self, config, *, wait_timeout):
            assert wait_timeout == 30.0
            self.configured_decks.append((config.deck0, config.deck1))
            return {
                "players": [
                    {
                        "owner": owner,
                        "hand": [],
                        "cycle": [],
                    }
                    for owner in (0, 1)
                ]
            }

        def pause(self) -> None:
            return None

    native = IncompleteNative()
    with pytest.raises(RoyaleAPIReplayError, match=r"4\+4"):
        calibrate_collected_replay_deal(prepared, native, max_attempts=10)

    assert native.configured_decks == [
        (
            prepared.replay.episode_config.deck0,
            prepared.replay.episode_config.deck1,
        )
    ]


def test_native_render_calibration_relocates_two_omitted_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_collected_replay(_payload_with_team_multi_omitted())
    mirror_id = resolve_native_card("mirror").card_id
    collector_id = resolve_native_card("elixir-collector").card_id
    omitted = (mirror_id, collector_id)
    plays = royaleapi_replay._replay_card_plays(prepared.replay)

    class FakeMultiOmittedNative:
        def __init__(self) -> None:
            self.calls = 0
            self.decks: list[tuple[int, ...]] = []
            self.forms: list[tuple[int, ...]] = []

        def create_native_match(self, config, *, wait_timeout):
            assert wait_timeout == 30.0
            self.calls += 1
            self.decks.append(tuple(config.deck0))
            self.forms.append(tuple(config.deck0_form_availability))
            players = []
            for owner, deck_value in enumerate((config.deck0, config.deck1)):
                deck = tuple(deck_value)
                if owner == 0 and self.calls == 1:
                    omitted_slots = {deck.index(card_id) for card_id in omitted}
                    first_play_slot = deck.index(plays[owner][0])
                    stable_candidates = tuple(
                        slot
                        for slot in range(len(deck))
                        if slot not in omitted_slots and slot != first_play_slot
                    )
                    opening_slots = stable_candidates[:4]
                    queue_slots = tuple(
                        slot for slot in range(len(deck)) if slot not in opening_slots
                    )
                    assert first_play_slot in queue_slots
                    assert omitted_slots.issubset(queue_slots)
                else:
                    opening_slots = tuple(
                        deck.index(card_id) for card_id in prepared.opening_cards[owner]
                    )
                    queue_slots = tuple(
                        deck.index(card_id) for card_id in prepared.queue_cards[owner]
                    )
                players.append(
                    {
                        "owner": owner,
                        "hand": [
                            {
                                "handIndex": hand_index,
                                "deckSlot": deck_slot,
                                "cardId": deck[deck_slot],
                            }
                            for hand_index, deck_slot in enumerate(opening_slots)
                        ],
                        "cycle": [
                            {
                                "cycleIndex": cycle_index,
                                "deckSlot": deck_slot,
                                "cardId": deck[deck_slot],
                            }
                            for cycle_index, deck_slot in enumerate(queue_slots)
                        ],
                    }
                )
            return {"players": players}

        def pause(self) -> None:
            return None

    monkeypatch.setattr(
        royaleapi_replay,
        "_special_deal_target",
        lambda **_kwargs: None,
    )
    native = FakeMultiOmittedNative()
    calibrated = calibrate_collected_replay_deal(prepared, native)

    assert native.calls == 2
    assert tuple(native.decks[0].index(card_id) for card_id in omitted) != tuple(
        native.decks[1].index(card_id) for card_id in omitted
    )
    assert tuple(
        card_id for card_id in native.decks[0] if card_id not in omitted
    ) == tuple(card_id for card_id in native.decks[1] if card_id not in omitted)
    assert dict(zip(native.decks[0], native.forms[0], strict=True)) == dict(
        zip(native.decks[1], native.forms[1], strict=True)
    )
    assert not set(omitted).intersection(calibrated.opening_cards[0])
    assert set(omitted).issubset(calibrated.queue_cards[0])
    assert (
        calibrated.replay.episode_config.tags["deck0_omit_from_starting_hand_ids"]
        == omitted
    )


def test_native_render_calibration_pins_omitted_card_and_retargets_actions() -> None:
    prepared = prepare_collected_replay(_payload_with_team_elixir_collector())
    collector_id = resolve_native_card("elixir-collector").card_id
    plays = royaleapi_replay._replay_card_plays(prepared.replay)

    class FakeOmittedNative:
        def __init__(self) -> None:
            self.calls = 0
            self.decks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
            self.layouts: dict[
                tuple[int, int],
                tuple[tuple[int, ...], tuple[int, ...]],
            ] = {}

        def create_native_match(self, config, *, wait_timeout):
            assert wait_timeout == 30.0
            self.calls += 1
            decks = (tuple(config.deck0), tuple(config.deck1))
            self.decks.append(decks)
            players = []
            for owner, deck in enumerate(decks):
                if collector_id not in deck:
                    opening_slots, queue_slots = NATIVE_DEAL_LAYOUTS[owner]
                else:
                    omitted_slot = deck.index(collector_id)
                    key = (owner, omitted_slot)
                    if key not in self.layouts:
                        first_play_slot = deck.index(plays[owner][0])
                        opening_slots = tuple(
                            slot
                            for slot in range(8)
                            if slot not in {omitted_slot, first_play_slot}
                        )[:4]
                        remainder = tuple(
                            slot
                            for slot in range(8)
                            if slot
                            not in {
                                omitted_slot,
                                first_play_slot,
                                *opening_slots,
                            }
                        )
                        queue_slots = (
                            first_play_slot,
                            *remainder,
                            omitted_slot,
                        )
                        self.layouts[key] = (opening_slots, queue_slots)
                    opening_slots, queue_slots = self.layouts[key]
                players.append(
                    {
                        "owner": owner,
                        "hand": [
                            {
                                "handIndex": hand_index,
                                "deckSlot": deck_slot,
                                "cardId": deck[deck_slot],
                            }
                            for hand_index, deck_slot in enumerate(opening_slots)
                        ],
                        "cycle": [
                            {
                                "cycleIndex": cycle_index,
                                "deckSlot": deck_slot,
                                "cardId": deck[deck_slot],
                            }
                            for cycle_index, deck_slot in enumerate(queue_slots)
                        ],
                    }
                )
            return {"players": players}

        def pause(self) -> None:
            return None

    native = FakeOmittedNative()
    calibrated = calibrate_collected_replay_deal(prepared, native)

    assert native.calls == 2
    collector_slots = [deck0.index(collector_id) for deck0, _deck1 in native.decks]
    assert collector_slots[0] == collector_slots[1]
    assert collector_id not in calibrated.opening_cards[0]
    assert calibrated.queue_cards[0][3] == collector_id

    final_deck = calibrated.replay.episode_config.deck0
    final_layout = native.layouts[(0, collector_slots[-1])]
    assert (
        tuple(final_deck[slot] for slot in final_layout[0])
        == (calibrated.opening_cards[0])
    )
    assert (
        tuple(final_deck[slot] for slot in final_layout[1])
        == (calibrated.queue_cards[0])
    )
    final_actions = [
        action
        for operation in calibrated.replay.operations
        for action in operation.actions
        if action.owner == 0 and action.kind is ActionKind.PLAY_CARD
    ]
    expected_slots = royaleapi_replay._cycle_hand_slots(
        calibrated.opening_cards[0],
        calibrated.queue_cards[0],
        tuple(action.card_id for action in final_actions),
    )
    assert tuple(action.hand_slot for action in final_actions) == expected_slots


def test_native_render_calibration_probes_another_omitted_card_slot() -> None:
    payload = _payload_with_team_elixir_collector()
    team_events = [event for event in payload["events"] if event["side"] == "team"]
    team_events[1]["card_key"], team_events[-1]["card_key"] = (
        team_events[-1]["card_key"],
        team_events[1]["card_key"],
    )
    prepared = prepare_collected_replay(payload)
    collector_id = resolve_native_card("elixir-collector").card_id

    class FakeSlotProbeNative:
        def __init__(self) -> None:
            self.calls = 0
            self.collector_slots: list[int] = []
            self.first_collector_slot: int | None = None

        def create_native_match(self, config, *, wait_timeout):
            assert wait_timeout == 30.0
            self.calls += 1
            players = []
            for owner, deck_value in enumerate((config.deck0, config.deck1)):
                deck = tuple(deck_value)
                if collector_id not in deck:
                    opening_slots, queue_slots = NATIVE_DEAL_LAYOUTS[owner]
                else:
                    collector_slot = deck.index(collector_id)
                    self.collector_slots.append(collector_slot)
                    if self.first_collector_slot is None:
                        self.first_collector_slot = collector_slot
                    opening = prepared.opening_cards[owner]
                    queue = list(prepared.queue_cards[owner])
                    if collector_slot == self.first_collector_slot:
                        collector_queue_index = queue.index(collector_id)
                        queue[collector_queue_index], queue[3] = (
                            queue[3],
                            queue[collector_queue_index],
                        )
                    opening_slots = tuple(deck.index(card_id) for card_id in opening)
                    queue_slots = tuple(deck.index(card_id) for card_id in queue)
                players.append(
                    {
                        "owner": owner,
                        "hand": [
                            {
                                "handIndex": hand_index,
                                "deckSlot": deck_slot,
                                "cardId": deck[deck_slot],
                            }
                            for hand_index, deck_slot in enumerate(opening_slots)
                        ],
                        "cycle": [
                            {
                                "cycleIndex": cycle_index,
                                "deckSlot": deck_slot,
                                "cardId": deck[deck_slot],
                            }
                            for cycle_index, deck_slot in enumerate(queue_slots)
                        ],
                    }
                )
            return {"players": players}

        def pause(self) -> None:
            return None

    native = FakeSlotProbeNative()
    calibrated = calibrate_collected_replay_deal(prepared, native)

    assert native.calls == 2
    assert len(set(native.collector_slots)) == 2
    assert calibrated.queue_cards[0][0] == collector_id
    assert collector_id not in calibrated.opening_cards[0]


@pytest.mark.parametrize(
    ("key", "expected"),
    tuple(ROYALAPI_CARD_KEY_IDS.items()),
)
def test_historical_public_card_names_resolve_to_native_ids(
    key: str,
    expected: int,
) -> None:
    assert resolve_native_card(key).card_id == expected


def test_prepare_replay_rejects_event_card_not_in_the_recorded_deck() -> None:
    payload = _payload()
    payload["events"][0]["card_key"] = "pekka"

    with pytest.raises(RoyaleAPIReplayError, match="不在 team 牌组"):
        prepare_collected_replay(payload)


def test_stage_listing_and_loading_are_read_only(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    database = root / CONTROL_DATABASE_RELATIVE
    database.parent.mkdir(parents=True)
    payload = _payload()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE replay_stage(
                replay_tag TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                inserted_at TEXT NOT NULL
            );
            CREATE TABLE replay_catalog(
                replay_tag TEXT PRIMARY KEY,
                part_number INTEGER NOT NULL,
                requested_player_tag TEXT NOT NULL,
                played_at_utc TEXT,
                inserted_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO replay_stage VALUES (?, ?, ?)",
            (
                payload["source"]["replay_tag"],
                json.dumps(payload),
                "2026-07-30T12:00:00+00:00",
            ),
        )

    entries = list_collected_replays(root, query="#YG0PCL80L")

    assert len(entries) == 1
    assert entries[0].storage == "SQLite 待合并"
    assert load_collected_replay_payload(entries[0]) == payload
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT count(*) FROM replay_stage").fetchone()[0] == 1
        )


def test_catalog_listing_searches_parquet_participant_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    database = root / CONTROL_DATABASE_RELATIVE
    database.parent.mkdir(parents=True)
    replay_tag = "02PYPPJGVQ80"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE replay_stage(
                replay_tag TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                inserted_at TEXT NOT NULL
            );
            CREATE TABLE replay_catalog(
                replay_tag TEXT PRIMARY KEY,
                part_number INTEGER NOT NULL,
                requested_player_tag TEXT NOT NULL,
                played_at_utc TEXT,
                inserted_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO replay_catalog VALUES (?, ?, ?, ?, ?)",
            (
                replay_tag,
                7,
                "COLLECTOR",
                "2026-07-29T01:02:03+00:00",
                "2026-07-30T12:00:00+00:00",
            ),
        )
    monkeypatch.setattr(
        royaleapi_replay,
        "_search_parquet_participant_replay_tags",
        lambda _root, query: (replay_tag,) if "YG0PCL80L" in query else (),
    )

    entries = list_collected_replays(root, query="YG0PCL80L")

    assert [entry.replay_tag for entry in entries] == [replay_tag]
    assert entries[0].requested_player_tag == "COLLECTOR"
    assert entries[0].part_number == 7


def test_personal_parquet_listing_and_loading_are_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    central_root = tmp_path / "central"
    database = central_root / CONTROL_DATABASE_RELATIVE
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE replay_stage(
                replay_tag TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                inserted_at TEXT NOT NULL
            );
            CREATE TABLE replay_catalog(
                replay_tag TEXT PRIMARY KEY,
                part_number INTEGER NOT NULL,
                requested_player_tag TEXT NOT NULL,
                played_at_utc TEXT,
                inserted_at TEXT NOT NULL
            );
            """
        )

    personal_root = tmp_path / "datasets" / "royaleapi_player_P89P2JYUP_since_test"
    personal_root.mkdir(parents=True)
    parquet_path = personal_root / "replays.parquet"
    parquet_path.write_bytes(b"read-only fixture")
    (personal_root / "manifest.json").write_text(
        json.dumps(
            {
                "player_tag": "P89P2JYUP",
                "collector_status": "complete",
                "replays_file": "replays.parquet",
            }
        ),
        encoding="utf-8",
    )
    replay_tag = "000YYU90VL0P"
    payload = _payload(replay_tag=replay_tag)
    payload["battle"]["requested_player_tag"] = "P89P2JYUP"

    def fake_listing(
        path_text: str,
        query: str,
        _file_size: int,
        _modified_ns: int,
    ) -> tuple[tuple[str, str, str, str], ...]:
        assert Path(path_text) == parquet_path.resolve()
        assert query == "P89P2JYUP"
        return (
            (
                replay_tag,
                "P89P2JYUP",
                "2026-07-31T06:41:37+00:00",
                "2026-07-31T14:41:37+08:00",
            ),
        )

    monkeypatch.setattr(
        royaleapi_replay,
        "_list_personal_parquet_cached",
        fake_listing,
    )
    monkeypatch.setattr(
        royaleapi_replay,
        "_read_parquet_payload",
        lambda path, tag: (
            json.dumps(payload)
            if path == parquet_path.resolve() and tag == replay_tag
            else pytest.fail("unexpected personal Parquet lookup")
        ),
    )

    entries = list_collected_replays(
        central_root,
        query="#P89P2JYUP",
        personal_dataset_roots=(personal_root,),
    )

    assert [entry.replay_tag for entry in entries] == [replay_tag]
    assert entries[0].storage == "个人 Parquet · P89P2JYUP"
    assert entries[0].parquet_path == parquet_path.resolve()
    assert load_collected_replay_payload(entries[0]) == payload
    assert parquet_path.read_bytes() == b"read-only fixture"


def test_standalone_parquet_directory_listing_and_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "merged"
    parquet_path = dataset_root / "replays" / "part-000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"standalone fixture")
    replay_tag = "09J9JLQ0CUYC"
    payload = _payload(replay_tag=replay_tag)
    payload["battle"]["requested_player_tag"] = "LVCLC8RL"

    def fake_listing(
        path_text: str,
        query: str,
        _file_size: int,
        _modified_ns: int,
    ) -> tuple[tuple[str, str, str, str | None], ...]:
        assert Path(path_text) == parquet_path.resolve()
        assert query == "LVCLC8RL"
        return (
            (
                replay_tag,
                "LVCLC8RL",
                "2026-07-29T01:57:30+00:00",
                None,
            ),
        )

    monkeypatch.setattr(
        royaleapi_replay,
        "_list_personal_parquet_cached",
        fake_listing,
    )
    monkeypatch.setattr(
        royaleapi_replay,
        "_read_parquet_payload",
        lambda path, tag: (
            json.dumps(payload)
            if path == parquet_path.resolve() and tag == replay_tag
            else pytest.fail("unexpected standalone Parquet lookup")
        ),
    )

    entries = list_collected_replays(
        dataset_root,
        query="#LVCLC8RL",
        personal_dataset_roots=(),
    )

    assert [entry.replay_tag for entry in entries] == [replay_tag]
    assert entries[0].storage == "目录 Parquet · part-000000.parquet"
    assert entries[0].parquet_path == parquet_path.resolve()
    assert load_collected_replay_payload(entries[0]) == payload
    assert parquet_path.read_bytes() == b"standalone fixture"


def test_personal_dataset_discovery_prefers_canonical_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets_root = tmp_path / "datasets"
    alias = datasets_root / "royaleapi_player_P89P2JYUP_24h_test"
    canonical = datasets_root / "royaleapi_player_P89P2JYUP_since_test"
    alias.mkdir(parents=True)
    canonical.mkdir(parents=True)
    for root in (alias, canonical):
        (root / "replays.parquet").write_bytes(b"same export")
    (alias / "manifest.json").write_text(
        json.dumps(
            {
                "replays_sha256": "ABC123",
                "canonical_output_directory": "../royaleapi_player_P89P2JYUP_since_test",
            }
        ),
        encoding="utf-8",
    )
    (canonical / "manifest.json").write_text(
        json.dumps({"replays_sha256": "ABC123"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(royaleapi_replay, "WORKSPACE_ROOT", tmp_path)

    assert royaleapi_replay._default_personal_dataset_roots() == (canonical.resolve(),)
