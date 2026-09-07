"""Deterministic stratified deck and policy matchmaking for V4 PPO."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal, Mapping, Sequence

from .deck_pool import DeckIdentityV4, DeckPoolManifestV4


MATCH_FOCUS_FOCUS = "focus-focus"
MATCH_FOCUS_COVERAGE = "focus-tail-coverage"
MATCH_FOCUS_WEIGHTED = "focus-tail-weighted"
MATCH_TAIL_TAIL = "tail-tail"
MATCH_HARD = "focus-tail-hard"
MATCH_LEARNER_FOCUS = "learner-opponent-focus"
MATCH_LEARNER_WEIGHTED = "learner-opponent-weighted"
MATCH_LEARNER_HARD = "learner-opponent-hard"

POLICY_CURRENT = "current-current"
POLICY_IL = "current-il"
POLICY_HISTORY = "current-history"
BATTLE_LEVELS = (11, 16)

MatchCategoryV4 = Literal[
    "focus-focus",
    "focus-tail-coverage",
    "focus-tail-weighted",
    "tail-tail",
    "focus-tail-hard",
    "learner-opponent-focus",
    "learner-opponent-weighted",
    "learner-opponent-hard",
]
PolicyMatchupV4 = Literal["current-current", "current-il", "current-history"]


@dataclass(frozen=True, slots=True)
class MatchmakerConfigV4:
    focus_focus: int = 30
    focus_tail_coverage: int = 30
    focus_tail_weighted: int = 20
    tail_tail: int = 10
    focus_tail_hard: int = 10
    current_current: int = 0
    current_il: int = 100
    current_history: int = 0
    focus_tail_current_focus_percent: int = 75

    def __post_init__(self) -> None:
        deck_mix = (
            self.focus_focus,
            self.focus_tail_coverage,
            self.focus_tail_weighted,
            self.tail_tail,
            self.focus_tail_hard,
        )
        policy_mix = (self.current_current, self.current_il, self.current_history)
        if any(value < 0 for value in (*deck_mix, *policy_mix)):
            raise ValueError("matchmaker weights cannot be negative")
        if sum(deck_mix) != 100 or sum(policy_mix) != 100:
            raise ValueError("deck and policy matchmaking weights must each sum to 100")
        if not 0 <= self.focus_tail_current_focus_percent <= 100:
            raise ValueError("focus-tail current focus percent must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class HardMatchupV4:
    focus_deck_id: str
    tail_deck_id: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.focus_deck_id or not self.tail_deck_id:
            raise ValueError("hard matchup requires two deck IDs")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("hard matchup weight must be positive")


@dataclass(frozen=True, slots=True)
class RuntimeDeckLayoutV4:
    card_ids: tuple[int, ...]
    form_availability: tuple[int, ...]
    tower_troop_id: int


class FixedLearnerOpponentDeckSamplerV4:
    """Sample configurable deck buckets with hard stats split by opponent policy."""

    _DEFAULT_CATEGORY_WEIGHTS = ((MATCH_LEARNER_FOCUS, 30), (MATCH_LEARNER_WEIGHTED, 50), (MATCH_LEARNER_HARD, 20))

    def __init__(
        self,
        manifest: DeckPoolManifestV4,
        *,
        seed: int,
        decay_per_batch: float = 0.8,
        hard_win_rate_threshold: float = 0.55,
        hard_top_k: int = 128,
        focus_percent: int = 30,
        weighted_percent: int = 50,
        hard_percent: int = 20,
    ) -> None:
        if seed <= 0:
            raise ValueError("fixed-learner opponent sampler seed must be positive")
        if not 0.0 < decay_per_batch <= 1.0:
            raise ValueError("hard-stat decay must be in (0, 1]")
        if not 0.0 < hard_win_rate_threshold < 1.0:
            raise ValueError("hard win-rate threshold must be in (0, 1)")
        if hard_top_k <= 0:
            raise ValueError("hard top-k must be positive")
        category_weights = (
            (MATCH_LEARNER_FOCUS, int(focus_percent)),
            (MATCH_LEARNER_WEIGHTED, int(weighted_percent)),
            (MATCH_LEARNER_HARD, int(hard_percent)),
        )
        if any(weight < 0 for _category, weight in category_weights):
            raise ValueError("fixed-learner opponent percentages cannot be negative")
        if sum(weight for _category, weight in category_weights) != 100:
            raise ValueError("fixed-learner opponent percentages must sum to 100")
        if hard_percent <= 0:
            raise ValueError("fixed-learner hard opponent percentage must be positive")
        if len(manifest.focus_decks) != 20:
            raise ValueError("fixed-learner sampler requires exactly 20 focus decks")
        if not manifest.decks:
            raise ValueError("fixed-learner sampler requires a non-empty deck pool")
        self.manifest = manifest
        self.rng = random.Random(int(seed))
        self.decay_per_batch = float(decay_per_batch)
        self.hard_win_rate_threshold = float(hard_win_rate_threshold)
        self.hard_top_k = int(hard_top_k)
        self.category_weights = category_weights
        self.focus = manifest.focus_decks
        self.decks = manifest.decks
        self.by_id = {item.deck_id: item for item in self.decks}
        self._weighted_deck_weights = tuple(math.sqrt(item.uses) for item in self.decks)
        self._focus_queue: list[DeckIdentityV4] = []
        self._learner_score_by_opponent_deck: dict[tuple[str, str], float] = {}
        self._effective_games_by_opponent_deck: dict[tuple[str, str], float] = {}
        self._completed_batches = 0

    @staticmethod
    def _largest_remainder_counts(total: int, weighted_labels: Sequence[tuple[str, int]]) -> dict[str, int]:
        if total < 0:
            raise ValueError("opponent sampler total cannot be negative")
        weight_sum = sum(weight for _label, weight in weighted_labels)
        if weight_sum <= 0:
            raise ValueError("opponent sampler weights must be positive")
        counts = {label: total * weight // weight_sum for label, weight in weighted_labels}
        remaining = total - sum(counts.values())
        ranked = sorted(weighted_labels, key=lambda item: (-(total * item[1] % weight_sum), item[0]))
        for label, _weight in ranked[:remaining]:
            counts[label] += 1
        return counts

    def _stratified_categories(self, policy_matchups: Sequence[PolicyMatchupV4]) -> tuple[MatchCategoryV4, ...]:
        total = len(policy_matchups)
        remaining_by_category = self._largest_remainder_counts(total, self.category_weights)
        indices_by_policy: dict[PolicyMatchupV4, list[int]] = {}
        for index, policy in enumerate(policy_matchups):
            if policy == POLICY_CURRENT:
                raise ValueError("fixed-learner opponent sampler does not support self-play")
            indices_by_policy.setdefault(policy, []).append(index)
        categories: list[MatchCategoryV4 | None] = [None] * total
        policies = sorted(indices_by_policy)
        remaining_total = total
        for policy_index, policy in enumerate(policies):
            indices = indices_by_policy[policy]
            if policy_index == len(policies) - 1:
                counts = dict(remaining_by_category)
            else:
                counts = self._largest_remainder_counts(
                    len(indices), tuple((category, count) for category, count in remaining_by_category.items())
                )
            if sum(counts.values()) != len(indices):
                raise RuntimeError("stratified opponent category count drifted")
            policy_categories: list[MatchCategoryV4] = []
            for category, count in counts.items():
                if count > remaining_by_category[category]:
                    raise RuntimeError("stratified opponent category over-allocated")
                policy_categories.extend([category] * count)  # type: ignore[list-item]
                remaining_by_category[category] -= count
            self.rng.shuffle(policy_categories)
            for index, category in zip(indices, policy_categories, strict=True):
                categories[index] = category
            remaining_total -= len(indices)
        if remaining_total != 0 or any(remaining_by_category.values()):
            raise RuntimeError("stratified opponent categories did not balance")
        if any(category is None for category in categories):
            raise RuntimeError("stratified opponent category was left empty")
        return tuple(category for category in categories if category is not None)

    def _focus_deck(self) -> DeckIdentityV4:
        if not self._focus_queue:
            self._focus_queue = list(self.focus)
            self.rng.shuffle(self._focus_queue)
        return self._focus_queue.pop()

    def _weighted_deck(self) -> DeckIdentityV4:
        return self.rng.choices(self.decks, weights=self._weighted_deck_weights, k=1)[0]

    @staticmethod
    def _hard_key(policy_matchup: str, deck_id: str) -> tuple[str, str]:
        if policy_matchup not in (POLICY_IL, POLICY_HISTORY):
            raise ValueError("fixed-learner hard statistics require a frozen opponent")
        if not deck_id:
            raise ValueError("fixed-learner hard statistics require a deck ID")
        return policy_matchup, deck_id

    def _hard_priority(self, policy_matchup: str, deck_id: str) -> float:
        key = self._hard_key(policy_matchup, deck_id)
        games = self._effective_games_by_opponent_deck.get(key, 0.0)
        if games <= 0.0:
            return 0.0
        learner_score = self._learner_score_by_opponent_deck.get(key, 0.0)
        smoothed_win_rate = (learner_score + 2.0) / (games + 4.0)
        reliability = games / (games + 4.0)
        return max(0.0, self.hard_win_rate_threshold - smoothed_win_rate) * reliability

    def hard_candidates(self, policy_matchup: PolicyMatchupV4) -> tuple[tuple[DeckIdentityV4, float], ...]:
        rows = tuple(
            (self.by_id[deck_id], self._hard_priority(policy_matchup, deck_id))
            for observed_policy, deck_id in self._effective_games_by_opponent_deck
            if observed_policy == policy_matchup
            and deck_id in self.by_id
            and self._hard_priority(policy_matchup, deck_id) > 0.0
        )
        return tuple(sorted(rows, key=lambda item: (-item[1], item[0].deck_id))[: self.hard_top_k])

    def _hard_deck(self, candidates: Sequence[tuple[DeckIdentityV4, float]]) -> DeckIdentityV4:
        if not candidates:
            return self._weighted_deck()
        return self.rng.choices(
            tuple(item[0] for item in candidates), weights=tuple(item[1] * item[1] for item in candidates), k=1
        )[0]

    def sample_batch(
        self, policy_matchups: Sequence[PolicyMatchupV4]
    ) -> tuple[tuple[MatchCategoryV4, DeckIdentityV4], ...]:
        categories = self._stratified_categories(policy_matchups)
        hard_candidates_by_policy = {policy: self.hard_candidates(policy) for policy in set(policy_matchups)}
        result = []
        for category, policy_matchup in zip(categories, policy_matchups, strict=True):
            if category == MATCH_LEARNER_FOCUS:
                deck = self._focus_deck()
            elif category == MATCH_LEARNER_WEIGHTED:
                deck = self._weighted_deck()
            elif category == MATCH_LEARNER_HARD:
                deck = self._hard_deck(hard_candidates_by_policy[policy_matchup])
            else:
                raise RuntimeError("unsupported fixed-learner opponent category")
            result.append((category, deck))
        return tuple(result)

    def record_completed_batch(
        self, learner_results_by_opponent_deck: Sequence[tuple[PolicyMatchupV4, str, float]]
    ) -> None:
        for key in tuple(self._effective_games_by_opponent_deck):
            games = self._effective_games_by_opponent_deck[key] * self.decay_per_batch
            score = self._learner_score_by_opponent_deck[key] * self.decay_per_batch
            if games < 1e-6:
                self._effective_games_by_opponent_deck.pop(key, None)
                self._learner_score_by_opponent_deck.pop(key, None)
            else:
                self._effective_games_by_opponent_deck[key] = games
                self._learner_score_by_opponent_deck[key] = score
        for policy_matchup, deck_id, result in learner_results_by_opponent_deck:
            if deck_id not in self.by_id:
                raise ValueError("hard-stat result references an unknown deck")
            if not math.isfinite(result) or result not in (-1.0, 0.0, 1.0):
                raise ValueError("hard-stat result must be -1, 0, or 1")
            key = self._hard_key(policy_matchup, deck_id)
            learner_score = 1.0 if result > 0.0 else 0.5 if result == 0.0 else 0.0
            self._effective_games_by_opponent_deck[key] = self._effective_games_by_opponent_deck.get(key, 0.0) + 1.0
            self._learner_score_by_opponent_deck[key] = (
                self._learner_score_by_opponent_deck.get(key, 0.0) + learner_score
            )
        self._completed_batches += 1

    def summary(self, *, limit: int = 20) -> dict[str, object]:
        if limit <= 0:
            raise ValueError("hard summary limit must be positive")
        candidates_by_policy = {policy: self.hard_candidates(policy) for policy in (POLICY_IL, POLICY_HISTORY)}
        candidates = tuple(
            sorted(
                ((policy, deck, priority) for policy, rows in candidates_by_policy.items() for deck, priority in rows),
                key=lambda item: (-item[2], item[0], item[1].deck_id),
            )
        )
        return {
            "completed_batches": self._completed_batches,
            "tracked_decks": len({deck_id for _policy, deck_id in self._effective_games_by_opponent_deck}),
            "tracked_opponent_decks": len(self._effective_games_by_opponent_deck),
            "hard_candidate_count": len(candidates),
            "hard_candidate_count_by_opponent": {policy: len(rows) for policy, rows in candidates_by_policy.items()},
            "top_hard_decks": tuple(
                {
                    "opponent_policy": policy,
                    "deck_id": deck.deck_id,
                    "priority": priority,
                    "effective_games": self._effective_games_by_opponent_deck[(policy, deck.deck_id)],
                    "smoothed_win_rate": (self._learner_score_by_opponent_deck[(policy, deck.deck_id)] + 2.0)
                    / (self._effective_games_by_opponent_deck[(policy, deck.deck_id)] + 4.0),
                }
                for policy, deck, priority in candidates[:limit]
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "rng_state": self.rng.getstate(),
            "focus_queue": tuple(deck.deck_id for deck in self._focus_queue),
            "category_weights": tuple(self.category_weights),
            "learner_score_by_opponent_deck": dict(self._learner_score_by_opponent_deck),
            "effective_games_by_opponent_deck": dict(self._effective_games_by_opponent_deck),
            "completed_batches": self._completed_batches,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        saved_weights = value["category_weights"]
        if tuple(saved_weights) != tuple(self.category_weights):  # type: ignore[arg-type]
            raise ValueError("opponent sampler category percentages changed on resume")
        self.rng.setstate(value["rng_state"])  # type: ignore[arg-type]
        try:
            self._focus_queue = [
                self.by_id[str(deck_id)]
                for deck_id in value.get("focus_queue", ())  # type: ignore[union-attr]
            ]
        except KeyError as error:
            raise ValueError("opponent sampler state references an unknown deck") from error

        score_rows = dict(value["learner_score_by_opponent_deck"])  # type: ignore[arg-type]
        game_rows = dict(value["effective_games_by_opponent_deck"])  # type: ignore[arg-type]
        if set(score_rows) != set(game_rows):
            raise ValueError("opponent sampler hard-stat keys are inconsistent")
        normalized_score: dict[tuple[str, str], float] = {}
        normalized_games: dict[tuple[str, str], float] = {}
        for raw_key in score_rows:
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError("opponent sampler hard-stat key changed shape")
            key = self._hard_key(str(raw_key[0]), str(raw_key[1]))
            if key[1] not in self.by_id:
                raise ValueError("opponent sampler state references an unknown deck")
            normalized_score[key] = float(score_rows[raw_key])
            normalized_games[key] = float(game_rows[raw_key])
        if any(
            not math.isfinite(item) or item < 0.0 for item in (*normalized_score.values(), *normalized_games.values())
        ):
            raise ValueError("opponent sampler hard stats must be finite and nonnegative")
        if any(normalized_score[key] > normalized_games[key] for key in normalized_score):
            raise ValueError("opponent sampler score exceeds its effective games")
        self._learner_score_by_opponent_deck = normalized_score
        self._effective_games_by_opponent_deck = normalized_games
        self._completed_batches = int(value.get("completed_batches", 0))
        if self._completed_batches < 0:
            raise ValueError("opponent sampler completed batch count cannot be negative")


@dataclass(frozen=True, slots=True)
class ScheduledMatchupV4:
    sequence: int
    category: MatchCategoryV4
    policy_matchup: PolicyMatchupV4
    deck0: DeckIdentityV4
    deck1: DeckIdentityV4
    current_owners: tuple[int, ...]
    battle_level: int
    seed: int

    def __post_init__(self) -> None:
        if self.sequence <= 0 or self.seed <= 0:
            raise ValueError("scheduled matchup sequence and seed must be positive")
        if not self.current_owners or any(owner not in (0, 1) for owner in self.current_owners):
            raise ValueError("scheduled matchup has invalid current-policy owners")
        if len(set(self.current_owners)) != len(self.current_owners):
            raise ValueError("scheduled matchup repeats a current-policy owner")
        if self.battle_level not in BATTLE_LEVELS:
            raise ValueError("scheduled matchup battle level must be 11 or 16")
        if self.policy_matchup == POLICY_CURRENT and self.current_owners != (0, 1):
            raise ValueError("current-current matchup must train both owners")
        if self.policy_matchup != POLICY_CURRENT and len(self.current_owners) != 1:
            raise ValueError("frozen-opponent matchup must train exactly one owner")

    def runtime_layouts(self) -> tuple[RuntimeDeckLayoutV4, RuntimeDeckLayoutV4]:
        rng = random.Random(self.seed ^ 0x5A17_D3C9)
        result: list[RuntimeDeckLayoutV4] = []
        for deck in (self.deck0, self.deck1):
            order = list(range(8))
            rng.shuffle(order)
            result.append(
                RuntimeDeckLayoutV4(
                    card_ids=tuple(deck.card_ids[index] for index in order),
                    form_availability=tuple(deck.form_availability[index] for index in order),
                    tower_troop_id=deck.tower_troop_id,
                )
            )
        return result[0], result[1]


class DeckMatchmakerV4:
    """Emit the frozen 30/30/20/10/10 deck mixture without replacement drift."""

    def __init__(self, manifest: DeckPoolManifestV4, *, seed: int, config: MatchmakerConfigV4 | None = None) -> None:
        if seed <= 0:
            raise ValueError("matchmaker seed must be positive")
        self.manifest = manifest
        self.config = config or MatchmakerConfigV4()
        self.rng = random.Random(int(seed))
        self.focus = manifest.focus_decks
        self.tail = manifest.tail_decks
        if len(self.focus) != 20:
            raise ValueError("V4 PPO matchmaker requires exactly 20 focus decks")
        if not self.tail:
            raise ValueError("V4 PPO matchmaker requires a non-empty tail pool")
        self.by_id = {item.deck_id: item for item in manifest.decks}
        self._deck_cycle: list[MatchCategoryV4] = []
        self._policy_cycle: list[PolicyMatchupV4] = []
        self._coverage_queue: list[DeckIdentityV4] = []
        self._coverage_draw_count = 0
        self._hard_matchups: tuple[HardMatchupV4, ...] = ()
        self._sequence = 0

    def set_hard_matchups(self, rows: Sequence[HardMatchupV4]) -> None:
        normalized = tuple(rows)
        for row in normalized:
            focus = self.by_id.get(row.focus_deck_id)
            tail = self.by_id.get(row.tail_deck_id)
            if focus is None or not focus.is_focus:
                raise ValueError("hard matchup focus ID is not in the focus pool")
            if tail is None or tail.is_focus:
                raise ValueError("hard matchup tail ID is not in the tail pool")
        self._hard_matchups = normalized

    def state_dict(self) -> dict[str, object]:
        return {
            "rng_state": self.rng.getstate(),
            "deck_cycle": tuple(self._deck_cycle),
            "policy_cycle": tuple(self._policy_cycle),
            "coverage_queue": tuple(item.deck_id for item in self._coverage_queue),
            "coverage_draw_count": self._coverage_draw_count,
            "hard_matchups": tuple(
                (item.focus_deck_id, item.tail_deck_id, item.weight) for item in self._hard_matchups
            ),
            "sequence": self._sequence,
        }

    def load_state_dict(self, value: dict[str, object]) -> None:
        self.rng.setstate(value["rng_state"])  # type: ignore[arg-type]
        self._deck_cycle = list(value.get("deck_cycle", ()))  # type: ignore[arg-type]
        self._policy_cycle = list(value.get("policy_cycle", ()))  # type: ignore[arg-type]
        coverage_ids = tuple(value.get("coverage_queue", ()))
        try:
            self._coverage_queue = [self.by_id[str(item)] for item in coverage_ids]
        except KeyError as error:
            raise ValueError("matchmaker state references an unknown coverage deck") from error
        self._coverage_draw_count = int(value.get("coverage_draw_count", 0))
        self._sequence = int(value.get("sequence", 0))
        self.set_hard_matchups(
            tuple(
                HardMatchupV4(str(focus), str(tail), float(weight))
                for focus, tail, weight in value.get("hard_matchups", ())  # type: ignore[misc]
            )
        )

    def _refill_deck_cycle(self) -> None:
        config = self.config
        self._deck_cycle = (
            [MATCH_FOCUS_FOCUS] * config.focus_focus
            + [MATCH_FOCUS_COVERAGE] * config.focus_tail_coverage
            + [MATCH_FOCUS_WEIGHTED] * config.focus_tail_weighted
            + [MATCH_TAIL_TAIL] * config.tail_tail
            + [MATCH_HARD] * config.focus_tail_hard
        )
        self.rng.shuffle(self._deck_cycle)

    def _refill_policy_cycle(self, *, has_history: bool) -> None:
        config = self.config
        history = POLICY_HISTORY if has_history else POLICY_CURRENT
        self._policy_cycle = (
            [POLICY_CURRENT] * config.current_current
            + [POLICY_IL] * config.current_il
            + [history] * config.current_history
        )
        self.rng.shuffle(self._policy_cycle)

    def _coverage_pair(self) -> tuple[DeckIdentityV4, DeckIdentityV4]:
        if not self._coverage_queue:
            self._coverage_queue = list(self.tail)
            self.rng.shuffle(self._coverage_queue)
        tail = self._coverage_queue.pop()
        focus = self.focus[self._coverage_draw_count % len(self.focus)]
        self._coverage_draw_count += 1
        return focus, tail

    def _weighted_tail(self) -> DeckIdentityV4:
        return self.rng.choices(self.tail, weights=tuple(math.sqrt(item.uses) for item in self.tail), k=1)[0]

    def _hard_pair(self) -> tuple[DeckIdentityV4, DeckIdentityV4]:
        if not self._hard_matchups:
            return self.rng.choice(self.focus), self._weighted_tail()
        row = self.rng.choices(self._hard_matchups, weights=tuple(item.weight for item in self._hard_matchups), k=1)[0]
        return self.by_id[row.focus_deck_id], self.by_id[row.tail_deck_id]

    def _decks(self, category: MatchCategoryV4) -> tuple[DeckIdentityV4, DeckIdentityV4]:
        if category == MATCH_FOCUS_FOCUS:
            return self.rng.choice(self.focus), self.rng.choice(self.focus)
        if category == MATCH_FOCUS_COVERAGE:
            return self._coverage_pair()
        if category == MATCH_FOCUS_WEIGHTED:
            return self.rng.choice(self.focus), self._weighted_tail()
        if category == MATCH_HARD:
            return self._hard_pair()
        if category != MATCH_TAIL_TAIL:
            raise RuntimeError(f"unsupported matchup category: {category}")
        first = self.rng.choice(self.tail)
        second = self.rng.choice(self.tail)
        while second.deck_id == first.deck_id and len(self.tail) > 1:
            second = self.rng.choice(self.tail)
        return first, second

    def battle_levels(self, count: int) -> tuple[int, ...]:
        """Return a shuffled exact 50/50 level split for one match batch."""

        if count <= 0:
            raise ValueError("battle level batch must contain at least one match")
        half = count // 2
        levels = [BATTLE_LEVELS[0]] * half + [BATTLE_LEVELS[1]] * half
        if count % 2:
            levels.append(self.rng.choice(BATTLE_LEVELS))
        self.rng.shuffle(levels)
        return tuple(levels)

    def next_match(self, *, has_history: bool, battle_level: int) -> ScheduledMatchupV4:
        if battle_level not in BATTLE_LEVELS:
            raise ValueError("battle level must be 11 or 16")
        if not self._deck_cycle:
            self._refill_deck_cycle()
        if not self._policy_cycle:
            self._refill_policy_cycle(has_history=has_history)
        category = self._deck_cycle.pop()
        policy_matchup = self._policy_cycle.pop()
        first, second = self._decks(category)
        if self.rng.random() < 0.5:
            deck0, deck1 = first, second
        else:
            deck0, deck1 = second, first
        if policy_matchup == POLICY_CURRENT:
            current_owners = (0, 1)
        else:
            focus_owners = tuple(owner for owner, deck in enumerate((deck0, deck1)) if deck.is_focus)
            if len(focus_owners) == 1:
                focus_owner = focus_owners[0]
                current_owner = (
                    focus_owner
                    if self.rng.randrange(100) < self.config.focus_tail_current_focus_percent
                    else 1 - focus_owner
                )
            elif focus_owners:
                current_owner = self.rng.choice(focus_owners)
            else:
                current_owner = self.rng.randrange(2)
            current_owners = (current_owner,)
        self._sequence += 1
        return ScheduledMatchupV4(
            sequence=self._sequence,
            category=category,
            policy_matchup=policy_matchup,
            deck0=deck0,
            deck1=deck1,
            current_owners=current_owners,
            battle_level=battle_level,
            seed=self.rng.randrange(1, 2**31),
        )
