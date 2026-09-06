from collections import Counter
import math
import random

from native_runner.training.v4.deck_pool import (
    DeckIdentityV4,
    DeckPoolManifestV4,
    deck_counts_from_payloads,
)
from native_runner.training.v4.matchmaking import (
    MATCH_FOCUS_COVERAGE,
    MATCH_FOCUS_FOCUS,
    MATCH_FOCUS_WEIGHTED,
    MATCH_HARD,
    MATCH_LEARNER_FOCUS,
    MATCH_LEARNER_HARD,
    MATCH_LEARNER_WEIGHTED,
    MATCH_TAIL_TAIL,
    POLICY_CURRENT,
    POLICY_HISTORY,
    POLICY_IL,
    DeckMatchmakerV4,
    FixedLearnerOpponentDeckSamplerV4,
    MatchmakerConfigV4,
)
from native_runner.training.v4.league import PolicyLeagueV4
from native_runner.training.v4.ppo_runtime import (
    episode_for_matchup_v4,
    ppo_environment_config_v4,
)
from native_runner.training.v4.train_ppo_self_play_cluster import (
    HISTORY_CANDIDATES_PER_BATCH,
    _balanced_assignments,
)


def _deck(index: int, *, focus_rank: int | None = None) -> DeckIdentityV4:
    keys = tuple(sorted(f"card-{index:03d}-{slot}" for slot in range(8)))
    return DeckIdentityV4(
        deck_id=f"deck-{index:03d}",
        card_keys=keys,
        card_ids=tuple(index * 100 + slot + 1 for slot in range(8)),
        form_availability=(0,) * 8,
        tower_key="tower-princess",
        tower_troop_id=159_000_000,
        uses=index + 5,
        focus_rank=focus_rank,
    )


def _manifest() -> DeckPoolManifestV4:
    return DeckPoolManifestV4(
        decks=tuple(
            [_deck(index, focus_rank=index + 1) for index in range(20)]
            + [_deck(index) for index in range(20, 60)]
        ),
        minimum_uses=5,
        metadata={},
    )


def _payload(*, tower: str | None = "tower-princess") -> dict:
    def player(prefix: str) -> dict:
        result = {
            "deck": [
                {"card_key": f"{prefix}-{index}"}
                for index in range(8)
            ]
        }
        if tower is not None:
            result["tower_card"] = {"card_key": tower}
        return result

    return {
        "battle": {
            "team": {"players": [player("team")]},
            "opponent": {"players": [player("opponent")]},
        }
    }


def test_deck_counts_deduplicate_replays_and_default_old_tower_rows() -> None:
    counts, metadata = deck_counts_from_payloads(
        (
            ("A", _payload(tower=None)),
            ("A", _payload(tower="dagger-duchess")),
            ("B", _payload(tower="royal-chef")),
        )
    )

    assert sum(counts.values()) == 4
    assert Counter(tower for _cards, tower in counts for _ in range(counts[(_cards, tower)])) == {
        "tower-princess": 2,
        "royal-chef": 2,
    }
    assert metadata == {
        "unique_replays": 2,
        "duplicate_replay_rows_skipped": 1,
        "invalid_player_decks": 0,
        "valid_player_deck_uses": 4,
    }


def test_matchmaker_emits_exact_deck_and_policy_mix() -> None:
    matchmaker = DeckMatchmakerV4(
        _manifest(),
        seed=17,
        config=MatchmakerConfigV4(
            current_current=35,
            current_il=25,
            current_history=40,
        ),
    )
    matches = [
        matchmaker.next_match(has_history=True, battle_level=11)
        for _ in range(100)
    ]

    assert Counter(item.category for item in matches) == {
        MATCH_FOCUS_FOCUS: 30,
        MATCH_FOCUS_COVERAGE: 30,
        MATCH_FOCUS_WEIGHTED: 20,
        MATCH_TAIL_TAIL: 10,
        MATCH_HARD: 10,
    }
    assert Counter(item.policy_matchup for item in matches) == {
        POLICY_CURRENT: 35,
        POLICY_IL: 25,
        POLICY_HISTORY: 40,
    }
    coverage_tail_ids = {
        (item.deck1 if item.deck0.is_focus else item.deck0).deck_id
        for item in matches
        if item.category == MATCH_FOCUS_COVERAGE
    }
    assert len(coverage_tail_ids) == 30
    assert all(
        item.current_owners == (0, 1)
        if item.policy_matchup == POLICY_CURRENT
        else len(item.current_owners) == 1
        for item in matches
    )


def test_matchmaker_reassigns_empty_history_share_to_current_policy() -> None:
    matchmaker = DeckMatchmakerV4(
        _manifest(),
        seed=19,
        config=MatchmakerConfigV4(
            current_current=35,
            current_il=25,
            current_history=40,
        ),
    )
    matches = [
        matchmaker.next_match(has_history=False, battle_level=11)
        for _ in range(100)
    ]

    assert Counter(item.policy_matchup for item in matches) == {
        POLICY_CURRENT: 75,
        POLICY_IL: 25,
    }


def test_runtime_layout_randomizes_slots_without_detaching_forms() -> None:
    match = DeckMatchmakerV4(_manifest(), seed=23).next_match(
        has_history=False,
        battle_level=11,
    )
    layouts = match.runtime_layouts()

    for identity, layout in zip((match.deck0, match.deck1), layouts, strict=True):
        expected = sorted(zip(identity.card_ids, identity.form_availability, strict=True))
        actual = sorted(zip(layout.card_ids, layout.form_availability, strict=True))
        assert actual == expected
        assert layout.tower_troop_id == identity.tower_troop_id


def test_matchmaker_state_round_trip_preserves_next_match() -> None:
    first = DeckMatchmakerV4(_manifest(), seed=29)
    for _ in range(37):
        first.next_match(has_history=False, battle_level=11)
    state = first.state_dict()
    second = DeckMatchmakerV4(_manifest(), seed=999)
    second.load_state_dict(state)

    assert first.next_match(
        has_history=False,
        battle_level=11,
    ) == second.next_match(
        has_history=False,
        battle_level=11,
    )


def test_ppo_environment_discount_matches_five_tick_decisions() -> None:
    environment = ppo_environment_config_v4(
        gamma_per_decision=0.9997,
        shaping_beta=0.05,
    )

    assert abs(math.exp(-5.0 / environment.shaping_tau_ticks) - 0.9997) < 1e-12
    assert environment.shaping_beta == 0.05


def test_episode_identity_contains_exact_forms_and_towers() -> None:
    match = DeckMatchmakerV4(_manifest(), seed=31).next_match(
        has_history=False,
        battle_level=11,
    )
    environment = ppo_environment_config_v4()
    episode = episode_for_matchup_v4(match, environment)

    expected = match.runtime_layouts()
    assert episode.deck0 == expected[0].card_ids
    assert episode.deck1 == expected[1].card_ids
    assert episode.tags["deck0_form_availability"] == expected[0].form_availability
    assert episode.tags["tower_troop1_id"] == expected[1].tower_troop_id


def test_policy_league_records_results_and_round_trips(tmp_path) -> None:
    anchor = tmp_path / "anchor.pt"
    snapshot = tmp_path / "snapshot.pt"
    league = PolicyLeagueV4(
        anchor_checkpoint_id="anchor-id",
        anchor_checkpoint_path=anchor,
    )
    league.add_snapshot("snapshot-id", snapshot, update_step=10)
    league.record_result("snapshot-id", 1.0)
    league.record_result("snapshot-id", 0.0)
    league.record_result("snapshot-id", -1.0)
    path = tmp_path / "league.json"
    league.save(path)

    loaded = PolicyLeagueV4.load(path)
    entry = loaded.sample_history(random.Random(1))
    assert entry.checkpoint_id == "snapshot-id"
    assert (entry.wins_against, entry.draws_against, entry.losses_against) == (1, 1, 1)


def test_policy_league_limits_each_batch_to_eight_candidates(tmp_path) -> None:
    league = PolicyLeagueV4(
        anchor_checkpoint_id="anchor-id",
        anchor_checkpoint_path=tmp_path / "anchor.pt",
    )
    for index in range(12):
        league.add_snapshot(
            f"snapshot-{index}",
            tmp_path / f"snapshot-{index}.pt",
            update_step=10 * (index + 1),
        )

    candidates = league.sample_history_candidates(random.Random(41), limit=8)
    repeated = league.sample_history_candidates(random.Random(41), limit=8)

    assert candidates == repeated
    assert len(candidates) == 8
    assert len({entry.checkpoint_id for entry in candidates}) == 8
    candidate_ids = {entry.checkpoint_id for entry in candidates}
    match_rng = random.Random(43)
    assert {
        league.sample_history_from(match_rng, candidates).checkpoint_id
        for _ in range(200)
    } <= candidate_ids


def test_full_batch_uses_at_most_eight_history_checkpoints(tmp_path) -> None:
    league = PolicyLeagueV4(
        anchor_checkpoint_id="anchor-id",
        anchor_checkpoint_path=tmp_path / "anchor.pt",
    )
    for index in range(12):
        league.add_snapshot(
            f"snapshot-{index}",
            tmp_path / f"snapshot-{index}.pt",
            update_step=10 * (index + 1),
        )

    shards = _balanced_assignments(
        DeckMatchmakerV4(
            _manifest(),
            seed=47,
            config=MatchmakerConfigV4(
                current_current=35,
                current_il=25,
                current_history=40,
            ),
        ),
        league,
        random.Random(53),
        match_capacities=(191,) * 8,
        rich_capacities=(0,) * 8,
        special_towers_require_rich=False,
    )
    history_ids = {
        assignment.opponent.checkpoint_id
        for shard in shards
        for assignment in shard
        if assignment.opponent is not None
    }

    assert len(history_ids) == HISTORY_CANDIDATES_PER_BATCH
    assert all(
        len(
            {
                assignment.opponent.checkpoint_id
                for assignment in shard
                if assignment.opponent is not None
            }
        )
        <= HISTORY_CANDIDATES_PER_BATCH
        for shard in shards
    )


def test_fixed_opponent_model_is_selected_from_its_assigned_deck_average(
    tmp_path,
) -> None:
    manifest = _manifest()
    league = PolicyLeagueV4(
        anchor_checkpoint_id="anchor-id",
        anchor_checkpoint_path=tmp_path / "anchor.pt",
    )
    league.add_snapshot(
        "proactive-id",
        tmp_path / "proactive.pt",
        update_step=30,
    )
    averages = {
        deck.deck_id: (2.9 if int(deck.deck_id[-3:]) % 2 == 0 else 3.1)
        for deck in manifest.decks
    }
    shards = _balanced_assignments(
        DeckMatchmakerV4(
            manifest,
            seed=59,
            config=MatchmakerConfigV4(
                current_current=0,
                current_il=0,
                current_history=100,
            ),
        ),
        league,
        random.Random(61),
        match_capacities=(100,),
        rich_capacities=(0,),
        special_towers_require_rich=False,
        uniform_history_sampling=True,
        fixed_opponent_il_max_average_elixir=3.0,
        deck_average_elixir_by_id=averages,
    )

    assignments = shards[0]
    assert {row.matchup.policy_matchup for row in assignments} == {
        POLICY_IL,
        POLICY_HISTORY,
    }
    for assignment in assignments:
        current_owner = assignment.matchup.current_owners[0]
        opponent_deck = (
            assignment.matchup.deck0,
            assignment.matchup.deck1,
        )[1 - current_owner]
        if averages[opponent_deck.deck_id] <= 3.0:
            assert assignment.matchup.policy_matchup == POLICY_IL
            assert assignment.opponent is None
        else:
            assert assignment.matchup.policy_matchup == POLICY_HISTORY
            assert assignment.opponent is not None
            assert assignment.opponent.checkpoint_id == "proactive-id"


def test_fixed_learner_deck_replaces_only_current_policy_owners(tmp_path) -> None:
    manifest = _manifest()
    learner_deck = manifest.decks[-1]
    league = PolicyLeagueV4(
        anchor_checkpoint_id="anchor-id",
        anchor_checkpoint_path=tmp_path / "anchor.pt",
    )
    shards = _balanced_assignments(
        DeckMatchmakerV4(manifest, seed=67),
        league,
        random.Random(71),
        match_capacities=(100,),
        rich_capacities=(0,),
        special_towers_require_rich=False,
        learner_deck=learner_deck,
    )

    for assignment in shards[0]:
        matchup = assignment.matchup
        assert matchup.policy_matchup == POLICY_IL
        assert len(matchup.current_owners) == 1
        current_owner = matchup.current_owners[0]
        assert (matchup.deck0, matchup.deck1)[current_owner] == learner_deck


def test_fixed_learner_opponent_sampler_emits_exact_stratified_mix() -> None:
    sampler = FixedLearnerOpponentDeckSamplerV4(_manifest(), seed=73)
    policies = (POLICY_HISTORY,) * 60 + (POLICY_IL,) * 40
    rows = sampler.sample_batch(policies)

    assert Counter(category for category, _deck in rows) == {
        MATCH_LEARNER_FOCUS: 30,
        MATCH_LEARNER_WEIGHTED: 50,
        MATCH_LEARNER_HARD: 20,
    }
    for policy, expected in (
        (
            POLICY_HISTORY,
            {
                MATCH_LEARNER_FOCUS: 18,
                MATCH_LEARNER_WEIGHTED: 30,
                MATCH_LEARNER_HARD: 12,
            },
        ),
        (
            POLICY_IL,
            {
                MATCH_LEARNER_FOCUS: 12,
                MATCH_LEARNER_WEIGHTED: 20,
                MATCH_LEARNER_HARD: 8,
            },
        ),
    ):
        assert Counter(
            rows[index][0]
            for index, item in enumerate(policies)
            if item == policy
        ) == expected


def test_fixed_learner_hard_sampler_learns_and_forgets_bad_decks() -> None:
    manifest = _manifest()
    sampler = FixedLearnerOpponentDeckSamplerV4(
        manifest,
        seed=79,
        decay_per_batch=0.8,
    )
    hard_deck = manifest.decks[23]
    easy_deck = manifest.decks[24]
    sampler.record_completed_batch(
        ((POLICY_IL, hard_deck.deck_id, -1.0),) * 8
        + ((POLICY_IL, easy_deck.deck_id, 1.0),) * 8
        + ((POLICY_HISTORY, hard_deck.deck_id, 1.0),) * 8
    )

    candidates = sampler.hard_candidates(POLICY_IL)
    assert candidates
    assert candidates[0][0] == hard_deck
    assert easy_deck.deck_id not in {deck.deck_id for deck, _priority in candidates}
    assert hard_deck.deck_id not in {
        deck.deck_id
        for deck, _priority in sampler.hard_candidates(POLICY_HISTORY)
    }
    hard_rows = [
        deck
        for category, deck in sampler.sample_batch((POLICY_IL,) * 100)
        if category == MATCH_LEARNER_HARD
    ]
    assert {deck.deck_id for deck in hard_rows} == {hard_deck.deck_id}

    for _ in range(3):
        sampler.record_completed_batch(
            ((POLICY_IL, hard_deck.deck_id, 1.0),) * 8
        )
    assert hard_deck.deck_id not in {
        deck.deck_id
        for deck, _priority in sampler.hard_candidates(POLICY_IL)
    }


def test_fixed_learner_opponent_sampler_state_preserves_next_batch() -> None:
    manifest = _manifest()
    first = FixedLearnerOpponentDeckSamplerV4(manifest, seed=83)
    first.record_completed_batch(
        ((POLICY_HISTORY, manifest.decks[25].deck_id, -1.0),) * 4
    )
    first.sample_batch((POLICY_HISTORY,) * 60 + (POLICY_IL,) * 40)
    state = first.state_dict()
    second = FixedLearnerOpponentDeckSamplerV4(manifest, seed=89)
    second.load_state_dict(state)

    policies = (POLICY_HISTORY,) * 61 + (POLICY_IL,) * 39
    assert first.sample_batch(policies) == second.sample_batch(policies)


def test_fixed_learner_opponent_sampler_supports_30_40_30_mix() -> None:
    sampler = FixedLearnerOpponentDeckSamplerV4(
        _manifest(),
        seed=91,
        focus_percent=30,
        weighted_percent=40,
        hard_percent=30,
    )
    policies = (POLICY_HISTORY,) * 60 + (POLICY_IL,) * 40
    rows = sampler.sample_batch(policies)

    assert Counter(category for category, _deck in rows) == {
        MATCH_LEARNER_FOCUS: 30,
        MATCH_LEARNER_WEIGHTED: 40,
        MATCH_LEARNER_HARD: 30,
    }
    for policy, expected in (
        (
            POLICY_HISTORY,
            {
                MATCH_LEARNER_FOCUS: 18,
                MATCH_LEARNER_WEIGHTED: 24,
                MATCH_LEARNER_HARD: 18,
            },
        ),
        (
            POLICY_IL,
            {
                MATCH_LEARNER_FOCUS: 12,
                MATCH_LEARNER_WEIGHTED: 16,
                MATCH_LEARNER_HARD: 12,
            },
        ),
    ):
        assert Counter(
            rows[index][0]
            for index, item in enumerate(policies)
            if item == policy
        ) == expected


def test_balanced_assignments_use_fixed_learner_and_sampled_opponents(
    tmp_path,
) -> None:
    manifest = _manifest()
    learner_deck = manifest.decks[-1]
    league = PolicyLeagueV4(
        anchor_checkpoint_id="anchor-id",
        anchor_checkpoint_path=tmp_path / "anchor.pt",
    )
    shards = _balanced_assignments(
        DeckMatchmakerV4(manifest, seed=97),
        league,
        random.Random(101),
        match_capacities=(100,),
        rich_capacities=(0,),
        special_towers_require_rich=False,
        learner_deck=learner_deck,
        learner_opponent_sampler=FixedLearnerOpponentDeckSamplerV4(
            manifest,
            seed=103,
        ),
    )

    assignments = shards[0]
    assert Counter(row.matchup.category for row in assignments) == {
        MATCH_LEARNER_FOCUS: 30,
        MATCH_LEARNER_WEIGHTED: 50,
        MATCH_LEARNER_HARD: 20,
    }
    for assignment in assignments:
        matchup = assignment.matchup
        current_owner = matchup.current_owners[0]
        assert (matchup.deck0, matchup.deck1)[current_owner] == learner_deck
        assert (matchup.deck0, matchup.deck1)[1 - current_owner] in manifest.decks
