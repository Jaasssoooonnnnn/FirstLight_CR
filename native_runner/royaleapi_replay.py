"""Read collected RoyaleAPI replays and adapt them for native rendering.

The collector intentionally stores normalized battle metadata and sparse
20 Hz actions, not private opening hands.  This module selects a reproducible
opening hand and draw queue that is compatible with every observed play, then
maps that logical order onto one empirically verified native deal layout.

All dataset access is read-only.  Unflushed rows come from ``replay_stage`` in
the collector SQLite database; flushed rows are located through
``replay_catalog`` and read from the corresponding Parquet part.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .contracts import ActionKind, ActionV1, EpisodeConfigV1, TargetKind, content_hash
from .normal_form_evidence import NATIVE_CHAMPION_ABILITY_BINDINGS, NATIVE_HERO_FORM_ABILITY_BINDINGS
from .snapshot import SnapshotOperationV1
from .training.replay_archive import TrainingReplayV1
from .paths import WORKSPACE_ROOT
from .local_config import setting


DEFAULT_DATASET_ROOT = Path(setting("CR_REPLAY_DATASET_ROOT", str(WORKSPACE_ROOT / "datasets" / "royaleapi")))
PERSONAL_DATASET_DIRECTORY_GLOB = "royaleapi_player_*"
PERSONAL_DATASET_MANIFEST = "manifest.json"
PERSONAL_REPLAYS_FILENAME = "replays.parquet"
CONTROL_DATABASE_RELATIVE = Path("_control") / "progress.sqlite3"
REPLAY_PART_DIRECTORY = "replays"
ARENA_GRID_WIDTH = 18
ARENA_GRID_HEIGHT = 32
NATIVE_DEAL_SEED = 1_784_463_263
NATIVE_RENDER_RULESET_ID = "royaleapi-native-render-adapter.v3"
TRUE_SIDE_MAPPING_VERSION = "royaleapi-true-side-data-i.v1"
# RoyaleAPI's public ``data-t`` marker identifies the source command boundary.
# ``NativeClashEnv.queue_hand_action_at`` and the absolute ability APIs instead
# accept the first observable state tick that must include the command. The
# native engine consumes a command while advancing from boundary N to state
# N + 1, so collected actions need this explicit one-tick conversion.
ROYALAPI_COMMAND_TO_NATIVE_OBSERVABLE_TICKS = 1
POLICY_DEAL_INVARIANCE_VERSION = "royaleapi-policy-selected-deal.v2"
POLICY_DEAL_COMMAND_BOUNDARY_SEMANTICS = "royaleapi-command-boundary.v1"
POLICY_ACTION_STREAM_VERSION = "royaleapi-policy-action-stream.v1"
POLICY_EPISODE_CONFIG_VERSION = "royaleapi-policy-episode-config.v1"
DEFAULT_NATIVE_GAME_MODE_ID = 72_000_006
DEFAULT_NATIVE_ARENA_ID = 54_000_001
DEFAULT_NATIVE_LOCATION_ID = 15_000_199
# Verified in the v15.535.13 stock renderer against the Classic Challenge
# capture: Arena_Legendary resolves to the legacy gray-stone PvP_champion scene.
CLASSIC_CHALLENGE_GAME_MODE_ID = 72_000_009
CLASSIC_CHALLENGE_ARENA_ID = 54_000_036
CLASSIC_CHALLENGE_LOCATION_ID = 15_000_013
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_FORM_SUFFIX = re.compile(r"-(?:ev\d+|hero)$", re.IGNORECASE)
_SAFE_REPLAY_TAG = re.compile(r"^[0289PYLQGRJCUV]+$", re.IGNORECASE)

# Verified against the v15.535.13 native engine with NATIVE_DEAL_SEED.  The
# opening tuple is ordered by native handIndex 0..3; the queue tuple is ordered
# by native cycleIndex 0..3.  Values are configured deck slots.
NATIVE_DEAL_LAYOUTS: Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]] = {
    0: ((2, 0, 4, 7), (6, 1, 3, 5)),
    1: ((7, 1, 4, 3), (6, 0, 2, 5)),
}

# RoyaleAPI uses public names while the v15 card catalog retains several
# historical internal names.  All other normal-mode cards resolve from the
# catalog's Name/IconFile/Stats/TID/image aliases.
ROYALAPI_CARD_KEY_IDS: Mapping[str, int] = {
    "bandit": 26_000_046,
    "barbarian-barrel": 28_000_015,
    "dart-goblin": 26_000_040,
    "elite-barbarians": 26_000_043,
    "giant-snowball": 28_000_017,
    "guards": 26_000_025,
    "lumberjack": 26_000_035,
    "royal-ghost": 26_000_050,
    "rune-giant": 26_000_101,
    "skeleton-barrel": 26_000_056,
    "sparky": 26_000_033,
    "spirit-empress": 28_000_025,
    "void": 28_000_023,
}
# Native support-card IDs follow physical record order.  The hidden,
# not-in-use GoblinQueen_SpawnAbility still occupies 159_000_003, so the
# visible Royal Chef record after it is 159_000_004; hidden rows must not be
# compacted out when deriving these IDs.
ROYALAPI_TOWER_TROOP_IDS: Mapping[str, int] = {
    "tower-princess": 159_000_000,
    "cannoneer": 159_000_001,
    "dagger-duchess": 159_000_002,
    "royal-chef": 159_000_004,
}


# ``kind == "hero"`` is not a reliable active-ability flag in the shipped
# catalog.  The imported mapping is the shared, content-addressed normal-form
# contract; this module retains the same public export and lookup behaviour.
def _native_battle_presentation(source_game_mode: object) -> tuple[int, int, int, str]:
    normalized = re.sub(r"[^a-z0-9]+", "", str(source_game_mode or "").lower())
    if normalized in {"classicchallenge", "grandchallenge"}:
        return (
            CLASSIC_CHALLENGE_GAME_MODE_ID,
            CLASSIC_CHALLENGE_ARENA_ID,
            CLASSIC_CHALLENGE_LOCATION_ID,
            "classic-legendary-arena.v1",
        )
    return (DEFAULT_NATIVE_GAME_MODE_ID, DEFAULT_NATIVE_ARENA_ID, DEFAULT_NATIVE_LOCATION_ID, "default-ladder-arena.v1")


class RoyaleAPIReplayError(RuntimeError):
    """Raised when a collected row cannot be rendered faithfully enough."""


class NativeRenderConfigurationLoadError(RoyaleAPIReplayError):
    """Raised when stock rendering accepted but did not load a configuration.

    Deal calibration must not interpret this lifecycle failure as evidence that
    the observed 4+4 deal is incompatible.  The structured status fields let
    the UI restart its dedicated runner without matching an error string.
    """

    def __init__(
        self,
        replay_tag: str,
        requested_sequence: int,
        *,
        status: Mapping[str, Any] | None = None,
        status_error: str | None = None,
    ) -> None:
        self.replay_tag = str(replay_tag)
        self.requested_sequence = int(requested_sequence)
        self.status = dict(status or {})
        self.status_error = status_error
        self.status_sequence = self.status.get("nativeRenderSequence")
        self.status_processed = self.status.get("nativeRenderProcessed")
        self.status_loaded = self.status.get("nativeRenderLoaded")
        self.status_ready = self.status.get("nativeRenderReady")
        self.status_mode = self.status.get("mode")
        if self.status:
            detail = (
                f"status sequence={self.status_sequence}/"
                f"processed={self.status_processed}/loaded={self.status_loaded}/"
                f"ready={self.status_ready}/mode={self.status_mode}"
            )
        else:
            detail = f"status unavailable ({status_error or 'unknown error'})"
        super().__init__(
            f"replay {self.replay_tag} native renderer configuration load "
            f"timed out: requested={self.requested_sequence}; {detail}"
        )


@dataclass(frozen=True, slots=True)
class CollectedReplayEntry:
    dataset_root: Path
    replay_tag: str
    requested_player_tag: str
    played_at_utc: str | None
    played_at_local: str | None
    storage: str
    part_number: int | None = None
    parquet_path: Path | None = None

    @property
    def played_at_display(self) -> str:
        value = self.played_at_local or self.played_at_utc
        if not value:
            return "未知"
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
            return parsed.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value


@dataclass(frozen=True, slots=True)
class NativeCardInfo:
    card_id: int
    name: str
    kind: str
    aliases: tuple[str, ...]
    omit_from_starting_hand: bool = False


@dataclass(frozen=True, slots=True)
class _DeckCard:
    key: str
    name: str
    info: NativeCardInfo
    form_mask: int
    level: int | None


@dataclass(frozen=True, slots=True)
class _CompatibleDealInvariantTrace:
    state_count_by_play_count: tuple[int, ...]
    unique_state_by_play_count: tuple[tuple[tuple[int, ...], tuple[int, ...]] | None, ...]


@dataclass(frozen=True, slots=True)
class _SelectedCycle:
    source_deck: tuple[int, ...]
    observed_plays: tuple[int, ...]
    omitted_starting_hand_ids: tuple[int, ...]
    deck: tuple[int, ...]
    forms: tuple[int, ...]
    hand_slots: tuple[int, ...]
    opening: tuple[int, ...]
    queue: tuple[int, ...]
    candidate_count: int
    invariant_trace: _CompatibleDealInvariantTrace


@dataclass(frozen=True, slots=True)
class PolicyDealInvariantBoundaryV1:
    """Owner-private state for the one replay-tag-seeded selected deal.

    The legacy class name is retained for artifact plumbing.  Current evidence
    always contains one selected legal deal, so it is known from play count 0
    and can be checked against the first native pre-action observation.
    """

    owner: int
    candidate_count: int
    source_deck: tuple[int, ...]
    observed_plays: tuple[int, ...]
    omitted_starting_hand_ids: tuple[int, ...]
    state_count_by_play_count: tuple[int, ...]
    first_invariant_play_count: int | None
    first_invariant_hand: tuple[int, ...] = ()
    first_invariant_cycle: tuple[int, ...] = ()
    first_output_command_group_index: int | None = None
    first_output_command_tick: int | None = None
    first_output_prior_play_count: int | None = None
    first_output_hand: tuple[int, ...] = ()
    first_output_cycle: tuple[int, ...] = ()
    version: str = field(default=POLICY_DEAL_INVARIANCE_VERSION, init=False)

    def __post_init__(self) -> None:
        counts = tuple(int(value) for value in self.state_count_by_play_count)
        object.__setattr__(self, "state_count_by_play_count", counts)
        if self.owner not in (0, 1) or self.candidate_count <= 0:
            raise ValueError("deal invariance requires owner 0/1 and candidates")
        if self.candidate_count != 1:
            raise ValueError("selected-deal evidence requires exactly one deal")
        if len(self.source_deck) != 8 or len(set(self.source_deck)) != 8:
            raise ValueError("deal invariance requires eight distinct source cards")
        if not set(self.observed_plays).issubset(self.source_deck):
            raise ValueError("deal invariance plays are outside the source deck")
        if not set(self.omitted_starting_hand_ids).issubset(self.source_deck):
            raise ValueError("deal invariance omitted cards are outside the source deck")
        if len(counts) != len(self.observed_plays) + 1:
            raise ValueError("deal invariance trace length disagrees with plays")
        if not counts or counts[0] != self.candidate_count:
            raise ValueError("deal invariance trace does not start at candidate_count")
        if any(value <= 0 for value in counts) or any(later > earlier for earlier, later in zip(counts, counts[1:])):
            raise ValueError("deal invariance state counts must stay positive/nonincreasing")
        first = self.first_invariant_play_count
        if first is None:
            if 1 in counts or any(
                value is not None
                for value in (
                    self.first_output_command_group_index,
                    self.first_output_command_tick,
                    self.first_output_prior_play_count,
                )
            ):
                raise ValueError("non-converged deal invariance carries a boundary")
            if any(
                (self.first_invariant_hand, self.first_invariant_cycle, self.first_output_hand, self.first_output_cycle)
            ):
                raise ValueError("non-converged deal invariance carries a card state")
            return
        if not 0 <= first < len(counts) or counts[first] != 1:
            raise ValueError("first invariant play count is outside the trace")
        if first > 0 and counts[first - 1] == 1:
            raise ValueError("first invariant play count is not minimal")
        if any(value != 1 for value in counts[first:]):
            raise ValueError("deal state diverged after becoming invariant")
        self._validate_state(self.first_invariant_hand, self.first_invariant_cycle, label="first invariant")
        boundary = (
            self.first_output_command_group_index,
            self.first_output_command_tick,
            self.first_output_prior_play_count,
        )
        if all(value is None for value in boundary):
            if self.first_output_hand or self.first_output_cycle:
                raise ValueError("deal state has no output boundary")
            return
        if any(value is None for value in boundary):
            raise ValueError("deal output boundary is only partially defined")
        assert self.first_output_prior_play_count is not None
        if self.first_output_prior_play_count < first:
            raise ValueError("deal output boundary precedes convergence")
        self._validate_state(self.first_output_hand, self.first_output_cycle, label="first output")

    def _validate_state(self, hand: Sequence[int], cycle: Sequence[int], *, label: str) -> None:
        if (
            len(hand) != 4
            or len(cycle) != 4
            or len(set(hand)) != 4
            or len(set(cycle)) != 4
            or set(hand).intersection(cycle)
            or set((*hand, *cycle)) != set(self.source_deck)
        ):
            raise ValueError(f"{label} deal state is not an exact 4+4 partition")

    @property
    def evidence_id(self) -> str:
        return content_hash(self)

    def binding(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "owner": self.owner,
            "evidence_id": self.evidence_id,
            "candidate_count": self.candidate_count,
            "first_invariant_play_count": self.first_invariant_play_count,
            "first_output_command_group_index": (self.first_output_command_group_index),
            "first_output_command_tick": self.first_output_command_tick,
            "first_output_prior_play_count": self.first_output_prior_play_count,
        }


@dataclass(frozen=True, slots=True)
class PreparedCollectedReplay:
    replay: TrainingReplayV1
    replay_tag: str
    event_count: int
    warnings: tuple[str, ...]
    opening_cards: tuple[tuple[int, ...], tuple[int, ...]]
    queue_cards: tuple[tuple[int, ...], tuple[int, ...]]
    policy_deal_invariance: tuple[PolicyDealInvariantBoundaryV1, PolicyDealInvariantBoundaryV1]
    policy_deal_invariance_id: str
    policy_action_stream_id: str
    policy_episode_config_id: str


def policy_episode_config_id(replay_tag: str, episode: EpisodeConfigV1) -> str:
    """Bind every native-facing episode field without a circular self-ID."""

    episode_payload = episode.to_dict()
    tags = dict(episode_payload.get("tags", {}))
    tags.pop("policy_episode_config_version", None)
    tags.pop("policy_episode_config_id", None)
    episode_payload["tags"] = tags
    return content_hash(
        {"version": POLICY_EPISODE_CONFIG_VERSION, "replay_tag": replay_tag, "episode_config": episode_payload}
    )


def _bind_policy_episode_config(replay_tag: str, episode: EpisodeConfigV1) -> tuple[EpisodeConfigV1, str]:
    tags = dict(episode.tags)
    tags.pop("policy_episode_config_version", None)
    tags.pop("policy_episode_config_id", None)
    unbound = replace(episode, tags=tags)
    identifier = policy_episode_config_id(replay_tag, unbound)
    tags.update(
        {"policy_episode_config_version": POLICY_EPISODE_CONFIG_VERSION, "policy_episode_config_id": identifier}
    )
    return replace(unbound, tags=tags), identifier


def _observed_deal_layout(
    observation: Mapping[str, Any], owner: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    try:
        player = next(item for item in observation["players"] if int(item["owner"]) == owner)
        hand = sorted(player["hand"], key=lambda item: int(item["handIndex"]))
        cycle = sorted(player["cycle"], key=lambda item: int(item["cycleIndex"]))
    except (KeyError, StopIteration, TypeError) as error:
        raise RoyaleAPIReplayError(f"native renderer 没有暴露 owner {owner} 的完整发牌状态") from error
    if len(hand) != 4 or len(cycle) != 4:
        raise RoyaleAPIReplayError(f"native renderer owner {owner} 的初手/队列不是 4+4")
    opening_slots = tuple(int(item["deckSlot"]) for item in hand)
    queue_slots = tuple(int(item["deckSlot"]) for item in cycle)
    if set((*opening_slots, *queue_slots)) != set(range(8)):
        raise RoyaleAPIReplayError(f"native renderer owner {owner} 的 deckSlot 发牌布局无效")
    return (
        opening_slots,
        queue_slots,
        tuple(int(item["cardId"]) for item in hand),
        tuple(int(item["cardId"]) for item in cycle),
    )


def _episode_match_config(episode: EpisodeConfigV1) -> Any:
    from .match_factory import MatchConfig, PRINCESS_TOWER_TROOP_ID

    tags = episode.tags
    return MatchConfig(
        deck0=episode.deck0,
        deck1=episode.deck1,
        seed=episode.seed,
        game_mode=episode.game_mode,
        arena=episode.arena,
        location=int(tags.get("location", DEFAULT_NATIVE_LOCATION_ID)),
        level_cap=int(tags.get("level_cap", 0)),
        minimum_card_level=int(tags.get("minimum_card_level", 0)),
        deck0_form_availability=tuple(int(value) for value in tags.get("deck0_form_availability", (0,) * 8)),
        deck1_form_availability=tuple(int(value) for value in tags.get("deck1_form_availability", (0,) * 8)),
        tower_troop0_id=int(tags.get("tower_troop0_id", PRINCESS_TOWER_TROOP_ID)),
        tower_troop1_id=int(tags.get("tower_troop1_id", PRINCESS_TOWER_TROOP_ID)),
        king_tower_level=(int(tags["king_tower_level"]) if tags.get("king_tower_level") is not None else None),
        owner0_name=str(tags.get("owner0_name", "RoyaleAPI-Team")),
        owner1_name=str(tags.get("owner1_name", "RoyaleAPI-Opponent")),
        end_tick=int(tags.get("end_tick", 7200)),
    )


def _replay_card_plays(replay: TrainingReplayV1) -> tuple[tuple[int, ...], ...]:
    plays: list[list[int]] = [[], []]
    for operation in replay.operations:
        for action in operation.actions:
            if action.kind is not ActionKind.PLAY_CARD:
                continue
            if action.card_id is None:
                raise RoyaleAPIReplayError("RoyaleAPI 出牌动作缺少 card_id")
            plays[action.owner].append(int(action.card_id))
    return tuple(tuple(owner_plays) for owner_plays in plays)


def _omitted_starting_hand_ids(episode: EpisodeConfigV1, owner: int) -> tuple[int, ...]:
    raw = episode.tags.get(f"deck{owner}_omit_from_starting_hand_ids", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RoyaleAPIReplayError(f"owner {owner} 的禁止初手牌标记不是数组")
    result = tuple(int(card_id) for card_id in raw)
    deck = episode.deck0 if owner == 0 else episode.deck1
    if len(set(result)) != len(result) or not set(result).issubset(deck):
        raise RoyaleAPIReplayError(f"owner {owner} 的禁止初手牌标记与牌组不一致")
    return result


def _retarget_replay_to_observed_deal(
    prepared: PreparedCollectedReplay,
    replay: TrainingReplayV1,
    observed: Mapping[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    *,
    attempt: int,
) -> PreparedCollectedReplay:
    openings = tuple(observed[owner][2] for owner in (0, 1))
    queues = tuple(observed[owner][3] for owner in (0, 1))
    plays = _replay_card_plays(replay)
    slots_by_owner: list[tuple[int, ...]] = []
    for owner in (0, 1):
        hand_slots = _cycle_hand_slots(openings[owner], queues[owner], plays[owner])
        if hand_slots is None:
            raise AssertionError("observed native deal was not play-compatible")
        slots_by_owner.append(hand_slots)

    slot_offsets = [0, 0]
    operations: list[SnapshotOperationV1] = []
    for operation in replay.operations:
        actions: list[ActionV1] = []
        for action in operation.actions:
            if action.kind is not ActionKind.PLAY_CARD:
                actions.append(action)
                continue
            owner = action.owner
            offset = slot_offsets[owner]
            actions.append(replace(action, hand_slot=slots_by_owner[owner][offset]))
            slot_offsets[owner] += 1
        operations.append(replace(operation, actions=tuple(actions)))
    if tuple(slot_offsets) != tuple(len(owner_plays) for owner_plays in plays):
        raise AssertionError("not all RoyaleAPI play actions were retargeted")

    note = (
        f"已用暂停的 native renderer 校准并验证双方合法初手/队列，并按实际 handIndex 重写出牌槽位（{attempt} 次配置）。"
    )
    source = dict(replay.source)
    source["warnings"] = [*source.get("warnings", ()), note]
    selected_deal_evidence, selected_deal_evidence_id = _retargeted_selected_deal_evidence(
        prepared, openings, queues, plays
    )
    tags = dict(replay.episode_config.tags)
    tags.update(
        {
            "policy_deal_invariance_version": POLICY_DEAL_INVARIANCE_VERSION,
            "policy_deal_invariance_id": selected_deal_evidence_id,
            "policy_deal_invariance": tuple(item.binding() for item in selected_deal_evidence),
        }
    )
    calibrated_episode, episode_config_id = _bind_policy_episode_config(
        prepared.replay_tag, replace(replay.episode_config, tags=tags)
    )
    calibrated_replay = replace(replay, episode_config=calibrated_episode, operations=tuple(operations), source=source)
    return replace(
        prepared,
        replay=calibrated_replay,
        warnings=(*prepared.warnings, note),
        opening_cards=openings,
        queue_cards=queues,
        policy_deal_invariance=selected_deal_evidence,
        policy_deal_invariance_id=selected_deal_evidence_id,
        policy_episode_config_id=episode_config_id,
    )


def _special_deal_target(
    *,
    replay_tag: str,
    owner: int,
    deck: Sequence[int],
    plays: Sequence[int],
    omitted_ids: Sequence[int],
    observed_opening: Sequence[int],
    observed_queue: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    omitted = frozenset(int(card_id) for card_id in omitted_ids)
    leaked = omitted.intersection(observed_opening)
    if leaked:
        raise RoyaleAPIReplayError(f"owner {owner} 的 native 初手包含禁止初手牌 {tuple(sorted(leaked))}")
    fixed_queue = {index: int(card_id) for index, card_id in enumerate(observed_queue) if int(card_id) in omitted}
    if set(fixed_queue.values()) != omitted:
        raise RoyaleAPIReplayError(f"owner {owner} 的 native 队列没有完整包含禁止初手牌")
    constraint = ",".join(f"{index}:{card_id}" for index, card_id in sorted(fixed_queue.items()))
    return _random_compatible_cycle(
        deck,
        plays,
        replay_tag=replay_tag,
        owner=owner,
        salt=f"native-deal-v4-random-single-omit-queue={constraint}",
        omitted_starting_hand_ids=omitted,
        fixed_queue=fixed_queue,
    )


def _relocate_omitted_card_for_probe(
    *,
    replay_tag: str,
    owner: int,
    deck: Sequence[int],
    omitted_ids: Sequence[int],
    tested_positions: set[tuple[int, ...]],
) -> tuple[int, ...]:
    actual_deck = tuple(int(card_id) for card_id in deck)
    omitted = tuple(int(card_id) for card_id in omitted_ids)
    if (
        not omitted
        or len(set(actual_deck)) != len(actual_deck)
        or len(set(omitted)) != len(omitted)
        or not set(omitted).issubset(actual_deck)
    ):
        raise RoyaleAPIReplayError(f"owner {owner} 的多禁止初手牌位置探测输入无效")
    digest = hashlib.sha256(
        (
            f"{replay_tag}|owner={owner}|native-deal-v3-multi-omit-slot-probe|"
            f"deck={','.join(map(str, actual_deck))}|"
            f"omitted={','.join(map(str, omitted))}"
        ).encode("utf-8")
    ).digest()
    position_candidates = list(itertools.permutations(range(len(actual_deck)), len(omitted)))
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(position_candidates)
    omitted_set = set(omitted)
    stable_other_cards = tuple(card_id for card_id in actual_deck if card_id not in omitted_set)
    for target_positions in position_candidates:
        if target_positions in tested_positions:
            continue
        omitted_by_slot = dict(zip(target_positions, omitted, strict=True))
        other_cards = iter(stable_other_cards)
        relocated_items: list[int | None] = []
        for slot in range(len(actual_deck)):
            if slot in omitted_by_slot:
                relocated_items.append(omitted_by_slot[slot])
            else:
                relocated_items.append(next(other_cards, None))
        relocated = tuple(relocated_items)
        if (
            any(card_id is None for card_id in relocated)
            or len(set(relocated)) != len(actual_deck)
            or set(relocated) != set(actual_deck)
        ):
            raise AssertionError("multi-omit relocation corrupted the deck")
        return tuple(int(card_id) for card_id in relocated)
    raise RoyaleAPIReplayError(
        f"owner {owner} 已检查全部多禁止初手牌 native 槽位组合，仍找不到与公开出牌序列相容的发牌"
    )


def calibrate_collected_replay_deal(
    prepared: PreparedCollectedReplay,
    native: Any,
    *,
    max_attempts: int = 10,
    on_attempt: Callable[[str], None] | None = None,
) -> PreparedCollectedReplay:
    """Calibrate a public-action-compatible order against stock dealing.

    A paused native-render probe is authoritative for the actual hand and draw
    queue.  Ordinary decks can be remapped through that observed permutation.
    Cards marked ``OmitFromStartingHand`` stay pinned to their observed native
    slot while the other cards are remapped around them.
    """

    if not 1 <= int(max_attempts) <= 12:
        raise ValueError("max_attempts must be in 1..12")
    from .cr_native_env import RunnerError

    replay = prepared.replay
    plays = _replay_card_plays(replay)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    tested_omitted_positions: dict[int, set[tuple[int, ...]]] = {0: set(), 1: set()}
    for attempt in range(1, int(max_attempts) + 1):
        episode = replay.episode_config
        signature = (episode.deck0, episode.deck1)
        if signature in seen:
            raise RoyaleAPIReplayError("native 发牌校准进入循环，无法构造与出牌序列相容的初手")
        try:
            observation = native.create_native_match(_episode_match_config(episode), wait_timeout=30.0)
            native.pause()
        except RunnerError as error:
            load_timeout = re.fullmatch(r"native renderer did not load configuration ([1-9][0-9]*)", str(error).strip())
            if load_timeout is not None:
                status: Mapping[str, Any] | None = None
                status_error: str | None = None
                try:
                    reported = native.status()
                    if isinstance(reported, Mapping):
                        status = reported
                    else:
                        status_error = f"native status response is not a mapping: {type(reported).__name__}"
                except Exception as diagnostic_error:
                    status_error = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                raise NativeRenderConfigurationLoadError(
                    prepared.replay_tag, int(load_timeout.group(1)), status=status, status_error=status_error
                ) from error
            raise
        observed = {owner: _observed_deal_layout(observation, owner) for owner in (0, 1)}
        if on_attempt is not None:
            on_attempt(f"已成功读取 native 4+4 发牌（第 {attempt} 个牌序），正在校验公开出牌序列…")
        seen.add(signature)
        compatible = {
            owner: _cycle_is_compatible(observed[owner][2], observed[owner][3], plays[owner]) for owner in (0, 1)
        }
        for owner in (0, 1):
            deck = episode.deck0 if owner == 0 else episode.deck1
            omitted_ids = _omitted_starting_hand_ids(episode, owner)
            if omitted_ids:
                tested_omitted_positions[owner].add(tuple(tuple(deck).index(card_id) for card_id in omitted_ids))
        if all(compatible.values()):
            return _retarget_replay_to_observed_deal(prepared, replay, observed, attempt=attempt)

        tags = dict(episode.tags)
        new_decks: dict[int, tuple[int, ...]] = {}
        new_forms: dict[int, tuple[int, ...]] = {}
        for owner in (0, 1):
            deck = episode.deck0 if owner == 0 else episode.deck1
            forms = tuple(int(value) for value in tags.get(f"deck{owner}_form_availability", (0,) * 8))
            forms_by_card = dict(zip(deck, forms, strict=True))
            if compatible[owner]:
                new_decks[owner] = tuple(deck)
                new_forms[owner] = forms
                continue
            opening_slots, queue_slots, observed_opening, observed_queue = observed[owner]
            omitted_ids = _omitted_starting_hand_ids(episode, owner)
            if omitted_ids:
                target = _special_deal_target(
                    replay_tag=prepared.replay_tag,
                    owner=owner,
                    deck=deck,
                    plays=plays[owner],
                    omitted_ids=omitted_ids,
                    observed_opening=observed_opening,
                    observed_queue=observed_queue,
                )
                if target is None:
                    new_deck = _relocate_omitted_card_for_probe(
                        replay_tag=prepared.replay_tag,
                        owner=owner,
                        deck=deck,
                        omitted_ids=omitted_ids,
                        tested_positions=tested_omitted_positions[owner],
                    )
                    new_decks[owner] = new_deck
                    new_forms[owner] = tuple(forms_by_card[card_id] for card_id in new_deck)
                    tags[f"deck{owner}_form_availability"] = new_forms[owner]
                    continue
                target_opening, target_queue = target
            else:
                target_opening = prepared.opening_cards[owner]
                target_queue = prepared.queue_cards[owner]
            configured: list[int | None] = [None] * 8
            for card_id, deck_slot in zip(target_opening, opening_slots, strict=True):
                configured[deck_slot] = card_id
            for card_id, deck_slot in zip(target_queue, queue_slots, strict=True):
                configured[deck_slot] = card_id
            if any(item is None for item in configured):
                raise AssertionError("deal calibration left an empty deck slot")
            new_deck = tuple(int(item) for item in configured if item is not None)
            for card_id in omitted_ids:
                if new_deck.index(card_id) != tuple(deck).index(card_id):
                    raise AssertionError("deal calibration moved an OmitFromStartingHand card")
            new_decks[owner] = new_deck
            new_forms[owner] = tuple(forms_by_card[card_id] for card_id in new_deck)
            tags[f"deck{owner}_form_availability"] = new_forms[owner]
        episode = replace(episode, deck0=new_decks[0], deck1=new_decks[1], tags=tags)
        replay = replace(replay, episode_config=episode)
    raise RoyaleAPIReplayError(f"经过 {max_attempts} 次配置仍无法稳定校准 native 发牌")


def _flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def _normalized_alias(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def base_card_key(value: object) -> str:
    return _FORM_SUFFIX.sub("", str(value or "").strip()).casefold()


def _sqlite_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection


def _normalized_search_query(query: str) -> str:
    return query.strip().upper().removeprefix("#")


def _search_pattern(query: str) -> str:
    value = _normalized_search_query(query)
    value = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{value}%"


def _parquet_part_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (part.name, part.stat().st_size, part.stat().st_mtime_ns)
        for part in sorted((root / REPLAY_PART_DIRECTORY).glob("part-*.parquet"))
        if part.is_file()
    )


def _search_parquet_participants_direct(replay_directory: Path, query: str) -> tuple[str, ...]:
    import pyarrow.parquet as parquet

    needle = _normalized_search_query(query)
    if not needle:
        return ()
    matches: set[str] = set()
    for part in sorted(replay_directory.glob("part-*.parquet")):
        table = parquet.read_table(part, columns=["replay_tag", "team_tags", "opponent_tags"])
        for replay_tag, team_tags, opponent_tags in zip(
            table.column("replay_tag").to_pylist(),
            table.column("team_tags").to_pylist(),
            table.column("opponent_tags").to_pylist(),
            strict=True,
        ):
            participant_tags = (*tuple(team_tags or ()), *tuple(opponent_tags or ()))
            if any(needle in str(tag).upper().removeprefix("#") for tag in participant_tags):
                matches.add(str(replay_tag))
    return tuple(sorted(matches))


@lru_cache(maxsize=64)
def _search_parquet_participants_cached(
    root_text: str, query: str, _part_signature: tuple[tuple[str, int, int], ...]
) -> tuple[str, ...]:
    if not _part_signature:
        return ()
    return _search_parquet_participants_direct(Path(root_text) / REPLAY_PART_DIRECTORY, query)


def _search_parquet_participant_replay_tags(root: Path, query: str) -> tuple[str, ...]:
    normalized = _normalized_search_query(query)
    if not normalized:
        return ()
    return _search_parquet_participants_cached(os.fspath(root.resolve()), normalized, _parquet_part_signature(root))


def _manifest_payload(root: Path) -> dict[str, Any]:
    path = root / PERSONAL_DATASET_MANIFEST
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _personal_replays_path(root: Path) -> Path:
    manifest = _manifest_payload(root)
    filename = str(manifest.get("replays_file") or PERSONAL_REPLAYS_FILENAME)
    return (root / filename).resolve()


def _default_personal_dataset_roots() -> tuple[Path, ...]:
    datasets_root = WORKSPACE_ROOT / "datasets"
    if not datasets_root.is_dir():
        return ()
    roots: list[Path] = []
    fingerprints: set[tuple[str, str]] = set()
    for candidate in sorted(datasets_root.glob(PERSONAL_DATASET_DIRECTORY_GLOB)):
        if not candidate.is_dir():
            continue
        root = candidate.resolve()
        manifest = _manifest_payload(root)
        canonical_value = str(manifest.get("canonical_output_directory") or "")
        if canonical_value:
            canonical_root = (root / canonical_value).resolve()
            if canonical_root != root and _personal_replays_path(canonical_root).is_file():
                continue
        replays_path = _personal_replays_path(root)
        if not replays_path.is_file():
            continue
        digest = str(manifest.get("replays_sha256") or "").upper()
        fingerprint = ("sha256", digest) if digest else ("path", os.fspath(replays_path).casefold())
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        roots.append(root)
    return tuple(roots)


def _timestamp_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _list_personal_parquet_direct(
    parquet_path: Path, query: str
) -> tuple[tuple[str, str, str | None, str | None], ...]:
    import pyarrow.parquet as parquet

    available = set(parquet.ParquetFile(parquet_path).schema_arrow.names)
    required = {"replay_tag", "requested_player_tag"}
    missing = sorted(required - available)
    if missing:
        raise RoyaleAPIReplayError(f"{parquet_path.name} 缺少字段：{', '.join(missing)}")
    optional = ("played_at_utc", "played_at_local", "team_tags", "opponent_tags")
    columns = [*sorted(required), *(name for name in optional if name in available)]
    rows = parquet.read_table(parquet_path, columns=columns).to_pylist()
    needle = _normalized_search_query(query)
    matches: list[tuple[str, str, str | None, str | None]] = []
    for row in rows:
        replay_tag = str(row.get("replay_tag") or "")
        requested_player_tag = str(row.get("requested_player_tag") or "")
        searchable = (
            replay_tag,
            requested_player_tag,
            *tuple(row.get("team_tags") or ()),
            *tuple(row.get("opponent_tags") or ()),
        )
        if needle and not any(needle in _normalized_search_query(str(value)) for value in searchable):
            continue
        matches.append(
            (
                replay_tag,
                requested_player_tag,
                _timestamp_text(row.get("played_at_utc")),
                _timestamp_text(row.get("played_at_local")),
            )
        )
    return tuple(matches)


@lru_cache(maxsize=64)
def _list_personal_parquet_cached(
    parquet_path_text: str, query: str, _file_size: int, _modified_ns: int
) -> tuple[tuple[str, str, str | None, str | None], ...]:
    return _list_personal_parquet_direct(Path(parquet_path_text), query)


def _list_personal_replay_entries(root: Path, query: str) -> tuple[CollectedReplayEntry, ...]:
    parquet_path = _personal_replays_path(root)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    stat = parquet_path.stat()
    rows = _list_personal_parquet_cached(
        os.fspath(parquet_path), _normalized_search_query(query), stat.st_size, stat.st_mtime_ns
    )
    manifest = _manifest_payload(root)
    dataset_player = str(manifest.get("player_tag") or root.name)
    return tuple(
        CollectedReplayEntry(
            dataset_root=root,
            replay_tag=replay_tag,
            requested_player_tag=requested_player_tag,
            played_at_utc=played_at_utc,
            played_at_local=played_at_local,
            storage=f"个人 Parquet · {dataset_player}",
            parquet_path=parquet_path,
        )
        for replay_tag, requested_player_tag, played_at_utc, played_at_local in rows
    )


def _list_parquet_directory_replay_entries(root: Path, query: str) -> tuple[CollectedReplayEntry, ...]:
    replay_directory = root / REPLAY_PART_DIRECTORY
    parts = tuple(sorted(replay_directory.glob("part-*.parquet")))
    if not parts:
        raise RoyaleAPIReplayError(
            f"所选目录既没有 {CONTROL_DATABASE_RELATIVE}，也没有 {REPLAY_PART_DIRECTORY}/part-*.parquet：{root}"
        )
    entries: dict[str, CollectedReplayEntry] = {}
    normalized_query = _normalized_search_query(query)
    for part in parts:
        resolved = part.resolve()
        stat = resolved.stat()
        rows = _list_personal_parquet_cached(os.fspath(resolved), normalized_query, stat.st_size, stat.st_mtime_ns)
        for replay_tag, requested_tag, played_utc, played_local in rows:
            entries[replay_tag] = CollectedReplayEntry(
                dataset_root=root,
                replay_tag=replay_tag,
                requested_player_tag=requested_tag,
                played_at_utc=played_utc,
                played_at_local=played_local,
                storage=f"目录 Parquet · {part.name}",
                parquet_path=resolved,
            )
    return tuple(entries.values())


def list_collected_replays(
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    *,
    limit: int = 500,
    query: str = "",
    personal_dataset_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> tuple[CollectedReplayEntry, ...]:
    """List centralized and personal replay metadata without mutating either."""

    if not 1 <= int(limit) <= 10_000:
        raise ValueError("limit must be in 1..10000")
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    database = root / CONTROL_DATABASE_RELATIVE
    if not database.is_file():
        standalone_entries = {entry.replay_tag: entry for entry in _list_parquet_directory_replay_entries(root, query)}
        personal_roots = (
            () if personal_dataset_roots is None else tuple(Path(item).resolve() for item in personal_dataset_roots)
        )
        for personal_root in personal_roots:
            for entry in _list_personal_replay_entries(personal_root, query):
                standalone_entries[entry.replay_tag] = entry

        def standalone_sort_key(item: CollectedReplayEntry) -> tuple[str, str]:
            return (item.played_at_utc or item.played_at_local or "", item.replay_tag)

        return tuple(sorted(standalone_entries.values(), key=standalone_sort_key, reverse=True)[: int(limit)])

    pattern = _search_pattern(query)
    participant_replay_tags = _search_parquet_participant_replay_tags(root, query)
    entries: dict[str, CollectedReplayEntry] = {}
    with _sqlite_read_only(database) as connection:
        stage_sql = """
            SELECT
                replay_tag,
                json_extract(payload_json, '$.battle.requested_player_tag')
                    AS requested_player_tag,
                json_extract(payload_json, '$.battle.played_at_utc')
                    AS played_at_utc,
                json_extract(payload_json, '$.battle.played_at_local')
                    AS played_at_local,
                inserted_at
            FROM replay_stage
            WHERE upper(replay_tag) LIKE ? ESCAPE '\\'
               OR upper(
                    coalesce(
                        json_extract(
                            payload_json,
                            '$.battle.requested_player_tag'
                        ),
                        ''
                    )
               ) LIKE ? ESCAPE '\\'
               OR EXISTS (
                    SELECT 1
                    FROM json_each(
                        replay_stage.payload_json,
                        '$.battle.team.players'
                    ) AS player
                    WHERE upper(
                        coalesce(json_extract(player.value, '$.tag'), '')
                    ) LIKE ? ESCAPE '\\'
               )
               OR EXISTS (
                    SELECT 1
                    FROM json_each(
                        replay_stage.payload_json,
                        '$.battle.opponent.players'
                    ) AS player
                    WHERE upper(
                        coalesce(json_extract(player.value, '$.tag'), '')
                    ) LIKE ? ESCAPE '\\'
               )
            ORDER BY coalesce(
                json_extract(payload_json, '$.battle.played_at_utc'),
                inserted_at
            ) DESC
            LIMIT ?
        """
        for row in connection.execute(stage_sql, (pattern, pattern, pattern, pattern, int(limit))):
            tag = str(row["replay_tag"])
            entries[tag] = CollectedReplayEntry(
                dataset_root=root,
                replay_tag=tag,
                requested_player_tag=str(row["requested_player_tag"] or ""),
                played_at_utc=(str(row["played_at_utc"]) if row["played_at_utc"] is not None else None),
                played_at_local=(str(row["played_at_local"]) if row["played_at_local"] is not None else None),
                storage="SQLite 待合并",
            )

        catalog_sql = """
            SELECT
                replay_tag,
                part_number,
                requested_player_tag,
                played_at_utc,
                inserted_at
            FROM replay_catalog
            WHERE upper(replay_tag) LIKE ? ESCAPE '\\'
               OR upper(requested_player_tag) LIKE ? ESCAPE '\\'
               OR replay_tag IN (
                    SELECT value FROM json_each(?)
               )
            ORDER BY coalesce(played_at_utc, inserted_at) DESC
            LIMIT ?
        """
        for row in connection.execute(catalog_sql, (pattern, pattern, json.dumps(participant_replay_tags), int(limit))):
            tag = str(row["replay_tag"])
            if tag in entries:
                continue
            part_number = int(row["part_number"])
            entries[tag] = CollectedReplayEntry(
                dataset_root=root,
                replay_tag=tag,
                requested_player_tag=str(row["requested_player_tag"]),
                played_at_utc=(str(row["played_at_utc"]) if row["played_at_utc"] is not None else None),
                played_at_local=None,
                storage=f"Parquet #{part_number:06d}",
                part_number=part_number,
            )

    if personal_dataset_roots is None:
        personal_roots = _default_personal_dataset_roots() if root == DEFAULT_DATASET_ROOT.resolve() else ()
    else:
        personal_roots = tuple(Path(item).resolve() for item in personal_dataset_roots)
    for personal_root in personal_roots:
        for entry in _list_personal_replay_entries(personal_root, query):
            # Prefer the explicit personal export for exact real-game comparison.
            entries[entry.replay_tag] = entry

    def sort_key(item: CollectedReplayEntry) -> tuple[str, str]:
        return (item.played_at_utc or item.played_at_local or "", item.replay_tag)

    return tuple(sorted(entries.values(), key=sort_key, reverse=True)[: int(limit)])


def _read_parquet_payload_direct(part: Path, replay_tag: str) -> str:
    import pyarrow.parquet as parquet

    table = parquet.read_table(part, columns=["replay_tag", "payload_json"], filters=[("replay_tag", "=", replay_tag)])
    matches = [
        str(payload)
        for tag, payload in zip(table.column("replay_tag").to_pylist(), table.column("payload_json").to_pylist())
        if str(tag) == replay_tag
    ]
    if len(matches) != 1:
        raise RoyaleAPIReplayError(f"Parquet {part.name} 中 replay {replay_tag} 出现 {len(matches)} 次")
    return matches[0]


def _read_parquet_payload(part: Path, replay_tag: str) -> str:
    return _read_parquet_payload_direct(part, replay_tag)


def load_collected_replay_payload(entry: CollectedReplayEntry) -> dict[str, Any]:
    """Load one selected payload, tolerating a concurrent stage flush."""

    root = entry.dataset_root.resolve()
    if entry.parquet_path is not None:
        part = entry.parquet_path.resolve()
        if not part.is_file():
            raise FileNotFoundError(part)
        raw = _read_parquet_payload(part, entry.replay_tag)
    else:
        database = root / CONTROL_DATABASE_RELATIVE
        with _sqlite_read_only(database) as connection:
            staged = connection.execute(
                "SELECT payload_json FROM replay_stage WHERE replay_tag = ?", (entry.replay_tag,)
            ).fetchone()
            if staged is not None:
                raw = str(staged["payload_json"])
            else:
                catalog = connection.execute(
                    """
                    SELECT part_number
                    FROM replay_catalog
                    WHERE replay_tag = ?
                    """,
                    (entry.replay_tag,),
                ).fetchone()
                if catalog is None:
                    raise RoyaleAPIReplayError(f"集中存储中找不到 replay {entry.replay_tag}")
                part_number = int(catalog["part_number"])
                part = root / REPLAY_PART_DIRECTORY / f"part-{part_number:06d}.parquet"
                if not part.is_file():
                    raise FileNotFoundError(part)
                raw = _read_parquet_payload(part, entry.replay_tag)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RoyaleAPIReplayError(f"replay {entry.replay_tag} 的 payload_json 无效") from error
    if not isinstance(payload, Mapping):
        raise RoyaleAPIReplayError("采集 replay 根节点不是对象")
    source = payload.get("source", {})
    observed_tag = str(source.get("replay_tag", "")) if isinstance(source, Mapping) else ""
    if observed_tag and observed_tag != entry.replay_tag:
        raise RoyaleAPIReplayError(f"目录 replay tag {entry.replay_tag} 与 payload {observed_tag} 不一致")
    return dict(payload)


@lru_cache(maxsize=1)
def _load_card_index() -> tuple[dict[int, NativeCardInfo], dict[str, tuple[int, ...]]]:
    from .card_specs import build_card_catalog

    payload = build_card_catalog().to_dict()
    by_id: dict[int, NativeCardInfo] = {}
    aliases: dict[str, set[int]] = defaultdict(set)
    for spec in payload.get("specs", ()):
        if not isinstance(spec, Mapping):
            continue
        attributes = spec.get("attributes", {})
        attributes = attributes if isinstance(attributes, Mapping) else {}
        if _flag(attributes.get("NotVisible")) or _flag(attributes.get("NotInUse")):
            continue
        card_id = int(spec["card_id"])
        raw_aliases: set[object] = {
            spec.get("name"),
            attributes.get("Name"),
            attributes.get("IconFile"),
            attributes.get("Stats"),
        }
        tid = str(attributes.get("TID") or "")
        if tid.startswith("TID_SPELL_"):
            raw_aliases.add(tid.removeprefix("TID_SPELL_"))
        image = str(attributes.get("HighresImageFilename") or "")
        if image:
            raw_aliases.add(Path(image).stem)
        normalized = tuple(sorted({_normalized_alias(value) for value in raw_aliases if value}))
        info = NativeCardInfo(
            card_id=card_id,
            name=str(spec.get("name") or card_id),
            kind=str(spec.get("kind") or ""),
            aliases=normalized,
            omit_from_starting_hand=_flag(attributes.get("OmitFromStartingHand")),
        )
        by_id[card_id] = info
        for alias in normalized:
            aliases[alias].add(card_id)
    return (by_id, {alias: tuple(sorted(card_ids)) for alias, card_ids in aliases.items()})


def resolve_native_card(card_key: str, card_name: str | None = None) -> NativeCardInfo:
    by_id, aliases = _load_card_index()
    public_key = base_card_key(card_key)
    explicit = ROYALAPI_CARD_KEY_IDS.get(public_key)
    if explicit is not None:
        try:
            return by_id[explicit]
        except KeyError as error:
            raise RoyaleAPIReplayError(f"native catalog 缺少 {public_key} 对应卡牌 {explicit}") from error

    probes = (_normalized_alias(public_key), _normalized_alias(card_name))
    for probe in probes:
        if not probe:
            continue
        matches = aliases.get(probe, ())
        if len(matches) == 1:
            return by_id[matches[0]]
        if len(matches) > 1:
            raise RoyaleAPIReplayError(f"卡牌 {card_key!r} 在 native catalog 中不唯一：{matches}")
    fallback_probes: set[str] = set()
    for probe in probes:
        if not probe:
            continue
        fallback_probes.add(probe + "s")
        if probe.endswith("s"):
            fallback_probes.add(probe[:-1])
    matches = {card_id for probe in fallback_probes for card_id in aliases.get(probe, ())}
    if len(matches) == 1:
        return by_id[next(iter(matches))]
    raise RoyaleAPIReplayError(f"无法把 RoyaleAPI 卡牌 {card_key!r} ({card_name or '无名称'}) 映射到 native card ID")


def _deck_form_mask(card_key: str, info: NativeCardInfo) -> int:
    key = str(card_key).casefold()
    mask = 0
    if re.search(r"-ev\d+$", key):
        mask |= 1
    if key.endswith("-hero") or info.card_id in NATIVE_CHAMPION_ABILITY_BINDINGS:
        mask |= 2
    return mask


def _player(side: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    players = side.get("players", ())
    if (
        not isinstance(players, Sequence)
        or isinstance(players, (str, bytes))
        or len(players) != 1
        or not isinstance(players[0], Mapping)
    ):
        raise RoyaleAPIReplayError(f"{label} 不是单人 1v1；当前 native viewer 只支持双方各一名玩家")
    return players[0]


def _bottom_side_for_player(
    team_player: Mapping[str, Any], opponent_player: Mapping[str, Any], bottom_player_tag: str | None
) -> str:
    normalized = _normalized_search_query(bottom_player_tag or "")
    if not normalized:
        return "opponent"
    matches = [
        side
        for side, player in (("team", team_player), ("opponent", opponent_player))
        if _normalized_search_query(str(player.get("tag") or "")) == normalized
    ]
    if len(matches) != 1:
        raise RoyaleAPIReplayError(f"希望固定在下方的玩家 #{normalized} 在对局双方中出现 {len(matches)} 次")
    return matches[0]


def _replay_data_i(raw_events: Sequence[Any]) -> int:
    """Return the replay-constant RoyaleAPI true-side inversion flag."""

    flags: set[int] = set()
    for event_index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            continue
        source_fields = raw_event.get("source_fields")
        if not isinstance(source_fields, Mapping) or "data_i" not in source_fields:
            # Older normalized ability rows may omit source fields. Card markers
            # in every supported collector payload still carry the replay flag.
            continue
        value = source_fields["data_i"]
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise RoyaleAPIReplayError(f"event {event_index} has invalid RoyaleAPI data_i {value!r}")
        flags.add(value)
    if not flags:
        raise RoyaleAPIReplayError(
            "replay has no RoyaleAPI data_i marker; True Red / True Blue cannot be mapped safely"
        )
    if len(flags) != 1:
        raise RoyaleAPIReplayError(f"replay mixes RoyaleAPI data_i values {sorted(flags)}")
    return next(iter(flags))


def _oriented_grid_target(
    grid: Mapping[str, Any], offset: Mapping[str, Any] | None, *, flip_horizontal: bool, flip_vertical: bool
) -> tuple[tuple[int, int], tuple[float, float]]:
    grid_x = int(grid["x"])
    grid_y = int(grid["y"])
    if not 0 <= grid_x < ARENA_GRID_WIDTH or not 0 <= grid_y < ARENA_GRID_HEIGHT:
        raise RoyaleAPIReplayError(f"play_card 的 grid_cell_floor 越界：({grid_x}, {grid_y})")
    offset_x = float(offset.get("x", 0.0)) if offset is not None else 0.0
    offset_y = float(offset.get("y", 0.0)) if offset is not None else 0.0
    return (
        (
            ARENA_GRID_WIDTH - 1 - grid_x if flip_horizontal else grid_x,
            ARENA_GRID_HEIGHT - 1 - grid_y if flip_vertical else grid_y,
        ),
        (-offset_x if flip_horizontal else offset_x, -offset_y if flip_vertical else offset_y),
    )


def _load_deck(player: Mapping[str, Any]) -> tuple[_DeckCard, ...]:
    raw = player.get("deck", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 8:
        raise RoyaleAPIReplayError("采集记录没有完整的八卡牌组")
    output: list[_DeckCard] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise RoyaleAPIReplayError("采集牌组条目不是对象")
        key = str(item.get("card_key") or "")
        name = str(item.get("name") or "")
        if not key:
            raise RoyaleAPIReplayError("采集牌组条目缺少 card_key")
        info = resolve_native_card(key, name)
        level_value = item.get("level")
        level = (
            int(level_value) if isinstance(level_value, (int, float)) and not isinstance(level_value, bool) else None
        )
        output.append(_DeckCard(key=key, name=name, info=info, form_mask=_deck_form_mask(key, info), level=level))
    ids = tuple(item.info.card_id for item in output)
    if len(set(ids)) != 8:
        raise RoyaleAPIReplayError("native 映射后的牌组包含重复卡牌")
    return tuple(output)


def _active_ability_deck_cards(deck: Sequence[_DeckCard]) -> tuple[_DeckCard, ...]:
    return tuple(
        card
        for card in deck
        if str(card.key).casefold().endswith("-hero") or card.info.card_id in NATIVE_CHAMPION_ABILITY_BINDINGS
    )


def _ability_source_cards(raw_event: Mapping[str, Any], deck: Sequence[_DeckCard]) -> tuple[_DeckCard, ...]:
    raw_candidates = raw_event.get("ability_source_candidates", ())
    candidate_keys = (
        {base_card_key(value) for value in raw_candidates if isinstance(value, str) and value.strip()}
        if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, (str, bytes))
        else set()
    )
    explicit = tuple(card for card in deck if base_card_key(card.key) in candidate_keys)
    if explicit and raw_event.get("ability_source_authoritative") is True:
        return explicit
    # Older collector rows only marked ``*-hero`` deck variants. Champions
    # have no suffix, and the catalog's broad ``kind == "hero"`` classification
    # is not an active-ability flag, so use the verified native binding table.
    inferred = _active_ability_deck_cards(deck)
    return inferred or explicit


def _ability_runtime_hints(cards: Sequence[_DeckCard]) -> tuple[str, ...]:
    hints: list[str] = []

    def add(value: object) -> None:
        hint = _normalized_alias(value)
        if hint and hint not in hints:
            hints.append(hint)

    for card in cards:
        public_key = base_card_key(card.key)
        if card.form_mask & 2:
            binding = NATIVE_HERO_FORM_ABILITY_BINDINGS.get(public_key)
            if binding is not None:
                add(binding[1])
        champion_binding = NATIVE_CHAMPION_ABILITY_BINDINGS.get(card.info.card_id)
        if champion_binding is not None:
            add(champion_binding[1])
        add(card.info.name)
        add(public_key)

    for hint in tuple(hints):
        if hint.endswith("s") and len(hint) > 3:
            add(hint[:-1])
    return tuple(hints)


def _cycle_is_compatible(opening: Iterable[int], queue: Sequence[int], plays: Sequence[int]) -> bool:
    return _cycle_hand_slots(opening, queue, plays) is not None


def _cycle_hand_slots(opening: Iterable[int], queue: Sequence[int], plays: Sequence[int]) -> tuple[int, ...] | None:
    ordered_opening = tuple(int(card_id) for card_id in opening)
    if len(ordered_opening) != 4 or len(set(ordered_opening)) != 4:
        return None
    if len(queue) != 4 or len(set(queue)) != 4:
        return None
    hand = {card_id: hand_index for hand_index, card_id in enumerate(ordered_opening)}
    cycle = list(queue)
    hand_slots: list[int] = []
    for card_id in plays:
        if card_id not in hand:
            return None
        hand_index = hand.pop(card_id)
        hand_slots.append(hand_index)
        drawn = cycle.pop(0)
        hand[drawn] = hand_index
        cycle.append(card_id)
    return tuple(hand_slots)


def _random_compatible_cycle(
    card_ids: Sequence[int],
    plays: Sequence[int],
    *,
    replay_tag: str,
    owner: int,
    salt: str,
    omitted_starting_hand_ids: Iterable[int] = (),
    fixed_queue: Mapping[int, int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Pick one reproducible random deal without materializing all deals.

    The first observed play is placed in the opening hand.  Remaining opening
    sets and four-card queues are visited in replay-tag-seeded random order and
    the search stops at the first deal that can execute the public play trace.
    The full-trace check is required because native actions address a current
    hand slot; a first-card-only guess can become illegal later in the replay.
    """

    ids = tuple(int(card_id) for card_id in card_ids)
    observed_plays = tuple(int(card_id) for card_id in plays)
    omitted = frozenset(int(card_id) for card_id in omitted_starting_hand_ids)
    queue_constraints = {int(index): int(card_id) for index, card_id in (fixed_queue or {}).items()}
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("random deal selection requires eight distinct cards")
    if not omitted.issubset(ids):
        raise ValueError("omitted starting-hand cards are outside the deck")
    if any(index not in range(4) for index in queue_constraints):
        raise ValueError("fixed queue indices must be in 0..3")
    if len(set(queue_constraints.values())) != len(queue_constraints) or not set(queue_constraints.values()).issubset(
        ids
    ):
        raise ValueError("fixed queue cards must be distinct deck cards")

    digest = hashlib.sha256(f"{replay_tag}|owner={owner}|{salt}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    opening_eligible = tuple(card_id for card_id in ids if card_id not in omitted)
    first_card = observed_plays[0] if observed_plays else None
    if first_card is not None and first_card not in opening_eligible:
        return None
    if first_card is None:
        opening_sets = list(itertools.combinations(opening_eligible, 4))
    else:
        opening_sets = [
            (first_card, *others)
            for others in itertools.combinations(
                tuple(card_id for card_id in opening_eligible if card_id != first_card), 3
            )
        ]
    rng.shuffle(opening_sets)

    for raw_opening in opening_sets:
        opening = list(raw_opening)
        rng.shuffle(opening)
        opening_set = frozenset(opening)
        remaining = tuple(card_id for card_id in ids if card_id not in opening_set)
        if any(card_id not in remaining for card_id in queue_constraints.values()):
            continue
        free_positions = tuple(index for index in range(4) if index not in queue_constraints)
        constrained_cards = frozenset(queue_constraints.values())
        free_cards = tuple(card_id for card_id in remaining if card_id not in constrained_cards)
        queue_orders = list(itertools.permutations(free_cards))
        rng.shuffle(queue_orders)
        for free_order in queue_orders:
            queue_items: list[int | None] = [None] * 4
            for index, card_id in queue_constraints.items():
                queue_items[index] = card_id
            for index, card_id in zip(free_positions, free_order, strict=True):
                queue_items[index] = card_id
            if any(card_id is None for card_id in queue_items):
                raise AssertionError("random deal left an empty queue position")
            queue = tuple(int(card_id) for card_id in queue_items if card_id is not None)
            if _cycle_hand_slots(opening, queue, observed_plays) is not None:
                return tuple(opening), queue
    return None


def _selected_deal_trace(
    opening: Sequence[int], queue: Sequence[int], plays: Sequence[int]
) -> _CompatibleDealInvariantTrace:
    """Record the states of the single randomly selected legal deal."""

    hand = set(int(card_id) for card_id in opening)
    cycle = [int(card_id) for card_id in queue]
    states: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for play_count in range(len(plays) + 1):
        states.append((tuple(sorted(hand)), tuple(cycle)))
        if play_count == len(plays):
            continue
        card_id = int(plays[play_count])
        if card_id not in hand or len(cycle) != 4:
            raise AssertionError("selected deal rejected an observed play")
        hand.remove(card_id)
        hand.add(cycle.pop(0))
        cycle.append(card_id)
    return _CompatibleDealInvariantTrace(
        state_count_by_play_count=(1,) * len(states), unique_state_by_play_count=tuple(states)
    )


def _select_cycle(
    deck_cards: Sequence[_DeckCard], plays: Sequence[int], *, replay_tag: str, owner: int
) -> _SelectedCycle:
    ids = tuple(item.info.card_id for item in deck_cards)
    forms_by_id = {item.info.card_id: item.form_mask for item in deck_cards}
    omitted = tuple(item.info.card_id for item in deck_cards if item.info.omit_from_starting_hand)
    selected = _random_compatible_cycle(
        ids,
        plays,
        replay_tag=replay_tag,
        owner=owner,
        salt=("native-deal-v4-random-single-omit" if omitted else "native-deal-v4-random-single"),
        omitted_starting_hand_ids=omitted,
    )
    if selected is None:
        raise RoyaleAPIReplayError(f"owner {owner} 的出牌序列与采集牌组不相容；可能是 Duel 页面牌组与具体小局不一致")
    opening, queue = selected
    invariant_trace = _selected_deal_trace(opening, queue, plays)

    opening_slots, queue_slots = NATIVE_DEAL_LAYOUTS[owner]
    configured: list[int | None] = [None] * 8
    for hand_index, deck_slot in enumerate(opening_slots):
        configured[deck_slot] = opening[hand_index]
    for card_id, deck_slot in zip(queue, queue_slots):
        configured[deck_slot] = card_id
    if any(item is None for item in configured):
        raise AssertionError("native deal mapping did not fill all deck slots")
    deck = tuple(int(item) for item in configured if item is not None)
    forms = tuple(forms_by_id[card_id] for card_id in deck)

    hand_slots = _cycle_hand_slots(opening, queue, plays)
    if hand_slots is None:
        raise AssertionError("selected compatible cycle diverged")

    return _SelectedCycle(
        source_deck=ids,
        observed_plays=tuple(int(card_id) for card_id in plays),
        omitted_starting_hand_ids=omitted,
        deck=deck,
        forms=forms,
        hand_slots=tuple(hand_slots),
        opening=opening,
        queue=queue,
        candidate_count=1,
        invariant_trace=invariant_trace,
    )


def _policy_deal_invariance_boundaries(
    *,
    replay_tag: str,
    selected_by_owner: Sequence[_SelectedCycle],
    grouped_events: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[tuple[PolicyDealInvariantBoundaryV1, PolicyDealInvariantBoundaryV1], str]:
    """Bind each owner's first exact private-deal state to an output boundary."""

    if len(selected_by_owner) != 2:
        raise AssertionError("deal invariance requires exactly two owners")
    ordered_groups = tuple(sorted(grouped_events.items()))
    evidence: list[PolicyDealInvariantBoundaryV1] = []
    for owner, selected in enumerate(selected_by_owner):
        trace = selected.invariant_trace
        first_play_count = next(
            (play_count for play_count, state in enumerate(trace.unique_state_by_play_count) if state is not None), None
        )
        first_hand: tuple[int, ...] = ()
        first_cycle: tuple[int, ...] = ()
        if first_play_count is not None:
            first_state = trace.unique_state_by_play_count[first_play_count]
            if first_state is None:
                raise AssertionError("deal invariance convergence has no state")
            first_hand, first_cycle = first_state
            if any(state is None for state in trace.unique_state_by_play_count[first_play_count:]):
                raise AssertionError("deal invariance did not persist")

        prior_play_count = 0
        output_group_index: int | None = None
        output_command_tick: int | None = None
        output_prior_play_count: int | None = None
        output_hand: tuple[int, ...] = ()
        output_cycle: tuple[int, ...] = ()
        for group_index, (_native_tick, events) in enumerate(ordered_groups):
            if first_play_count is not None and prior_play_count >= first_play_count and output_group_index is None:
                source_ticks = {int(event["source_tick"]) for event in events}
                if len(source_ticks) != 1:
                    raise RoyaleAPIReplayError("one native command group has multiple source boundaries")
                state = trace.unique_state_by_play_count[prior_play_count]
                if state is None:
                    raise AssertionError("deal output boundary does not have an invariant state")
                output_group_index = group_index
                output_command_tick = next(iter(source_ticks))
                output_prior_play_count = prior_play_count
                output_hand, output_cycle = state
            prior_play_count += sum(
                1 for event in events if int(event["owner"]) == owner and event["kind"] == "play_card"
            )
        if prior_play_count != len(selected.observed_plays):
            raise AssertionError("deal invariance play count disagrees with command groups")
        evidence.append(
            PolicyDealInvariantBoundaryV1(
                owner=owner,
                candidate_count=selected.candidate_count,
                source_deck=selected.source_deck,
                observed_plays=selected.observed_plays,
                omitted_starting_hand_ids=(selected.omitted_starting_hand_ids),
                state_count_by_play_count=(trace.state_count_by_play_count),
                first_invariant_play_count=first_play_count,
                first_invariant_hand=first_hand,
                first_invariant_cycle=first_cycle,
                first_output_command_group_index=output_group_index,
                first_output_command_tick=output_command_tick,
                first_output_prior_play_count=output_prior_play_count,
                first_output_hand=output_hand,
                first_output_cycle=output_cycle,
            )
        )
    pair = (evidence[0], evidence[1])
    return pair, _policy_deal_evidence_id(replay_tag, pair)


def _policy_deal_evidence_id(replay_tag: str, evidence: Sequence[PolicyDealInvariantBoundaryV1]) -> str:
    return content_hash(
        {
            "version": POLICY_DEAL_INVARIANCE_VERSION,
            "command_boundary_semantics": (POLICY_DEAL_COMMAND_BOUNDARY_SEMANTICS),
            "replay_tag": replay_tag,
            "owner_evidence_ids": tuple(item.evidence_id for item in evidence),
        }
    )


def _retargeted_selected_deal_evidence(
    prepared: PreparedCollectedReplay,
    openings: Sequence[Sequence[int]],
    queues: Sequence[Sequence[int]],
    plays: Sequence[Sequence[int]],
) -> tuple[tuple[PolicyDealInvariantBoundaryV1, PolicyDealInvariantBoundaryV1], str]:
    updated: list[PolicyDealInvariantBoundaryV1] = []
    for owner, previous in enumerate(prepared.policy_deal_invariance):
        trace = _selected_deal_trace(openings[owner], queues[owner], plays[owner])
        first_hand, first_cycle = trace.unique_state_by_play_count[0] or ((), ())
        output_hand: tuple[int, ...] = ()
        output_cycle: tuple[int, ...] = ()
        prior_play_count = previous.first_output_prior_play_count
        if prior_play_count is not None:
            output_state = trace.unique_state_by_play_count[prior_play_count]
            if output_state is None:
                raise AssertionError("selected deal output state is unavailable")
            output_hand, output_cycle = output_state
        updated.append(
            replace(
                previous,
                candidate_count=1,
                state_count_by_play_count=trace.state_count_by_play_count,
                first_invariant_play_count=0,
                first_invariant_hand=first_hand,
                first_invariant_cycle=first_cycle,
                first_output_hand=output_hand,
                first_output_cycle=output_cycle,
            )
        )
    pair = (updated[0], updated[1])
    return pair, _policy_deal_evidence_id(prepared.replay_tag, pair)


def policy_action_stream_id(replay_tag: str, operations: Sequence[SnapshotOperationV1]) -> str:
    """Hash the immutable public command stream, excluding calibrated slots."""

    def metadata_sequence(metadata: Mapping[str, Any], key: str) -> object:
        value = metadata.get(key, ())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return tuple(value)
        # Preserve malformed scalar/None values in the digest.  Validation can
        # then fail closed on an ID mismatch rather than raising an incidental
        # ``tuple(None)`` TypeError before native mutation is guarded.
        return value

    actions: list[Mapping[str, Any]] = []
    for operation_index, operation in enumerate(operations):
        for action_index, action in enumerate(operation.actions):
            if action.kind not in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_ABILITY}:
                continue
            target_tick = operation.start_native_tick + max(1, action.execute_offset_ticks or 1)
            metadata = action.metadata
            actions.append(
                {
                    "operation_index": operation_index,
                    "action_index": action_index,
                    "action_id": action.action_id,
                    "owner": action.owner,
                    "kind": action.kind.value,
                    "card_id": action.card_id,
                    "source_entity": action.source_entity,
                    "ability_id": action.ability_id,
                    "target_kind": action.target_kind.value,
                    "target_grid": action.target_grid,
                    "target_entity": action.target_entity,
                    "subcell_offset": action.subcell_offset,
                    "execute_offset_ticks": action.execute_offset_ticks,
                    "next_decision_ticks": action.next_decision_ticks,
                    "operation_start_native_tick": (operation.start_native_tick),
                    "operation_end_native_tick": operation.end_native_tick,
                    "target_native_tick": target_tick,
                    "source_index": metadata.get("source_index"),
                    "source_event_index": metadata.get("source_event_index"),
                    "source_command_tick": metadata.get("source_command_tick"),
                    "native_observable_tick": metadata.get("native_observable_tick"),
                    "ability_source_keys": metadata_sequence(metadata, "ability_source_keys"),
                    "ability_source_names": metadata_sequence(metadata, "ability_source_names"),
                    "ability_runtime_hints": metadata_sequence(metadata, "ability_runtime_hints"),
                }
            )
    return content_hash({"version": POLICY_ACTION_STREAM_VERSION, "replay_tag": replay_tag, "actions": actions})


def _source_replay_tag(payload: Mapping[str, Any]) -> str:
    source = payload.get("source", {})
    tag = str(source.get("replay_tag") or "") if isinstance(source, Mapping) else ""
    if not tag:
        raise RoyaleAPIReplayError("采集 payload 缺少 source.replay_tag")
    anonymous_id = source.get("anonymized") is True and re.fullmatch(r"[0-9a-f]{32}", tag)
    if not (_SAFE_REPLAY_TAG.fullmatch(tag) or anonymous_id):
        raise RoyaleAPIReplayError(f"无效 replay tag：{tag!r}")
    return tag


def _winner(result: object) -> int | None:
    normalized = str(result or "").strip().casefold()
    if normalized in {"win", "victory"}:
        return 0
    if normalized in {"defeat", "loss", "lose"}:
        return 1
    return None


def _source_end_tick(payload: Mapping[str, Any], *, last_event_tick: int) -> int:
    replay = payload.get("replay")
    duration = replay.get("duration") if isinstance(replay, Mapping) else None
    candidates: list[int] = []
    if isinstance(duration, Mapping):
        timeline_seconds = duration.get("timeline_seconds")
        if (
            isinstance(timeline_seconds, (int, float))
            and not isinstance(timeline_seconds, bool)
            and float(timeline_seconds) >= 0
        ):
            candidates.append(int(round(float(timeline_seconds) * 20.0)))
        display_seconds = duration.get("display_seconds")
        if (
            isinstance(display_seconds, (int, float))
            and not isinstance(display_seconds, bool)
            and float(display_seconds) >= 0
        ):
            candidates.append(int(round(float(display_seconds) * 20.0)))
    return max([last_event_tick, *candidates])


def _created_at_ns(payload: Mapping[str, Any]) -> int:
    source = payload.get("source", {})
    value = source.get("fetched_at") if isinstance(source, Mapping) else None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(parsed.timestamp() * 1_000_000_000)
        except ValueError:
            pass
    return 1


def _display_name(player: Mapping[str, Any], fallback: str) -> str:
    value = str(player.get("name") or player.get("tag") or fallback).strip()
    return value[:48] or fallback


def _tower_troop(player: Mapping[str, Any], label: str) -> tuple[str, int, bool]:
    tower = player.get("tower_card")
    if not isinstance(tower, Mapping):
        return ("tower-princess", ROYALAPI_TOWER_TROOP_IDS["tower-princess"], True)
    key = str(tower.get("card_key") or "").strip().casefold()
    key = {"princess-tower": "tower-princess", "chef-tower": "royal-chef"}.get(key, key)
    if not key:
        return ("tower-princess", ROYALAPI_TOWER_TROOP_IDS["tower-princess"], True)
    try:
        return key, ROYALAPI_TOWER_TROOP_IDS[key], False
    except KeyError as error:
        raise RoyaleAPIReplayError(f"{label} 使用了尚未支持的塔兵 {key!r}；拒绝静默替换成公主塔") from error


def prepare_collected_replay(
    payload: Mapping[str, Any], *, bottom_player_tag: str | None = None
) -> PreparedCollectedReplay:
    """Convert one normalized collector payload into an in-memory replay."""

    replay_tag = _source_replay_tag(payload)
    battle = payload.get("battle")
    if not isinstance(battle, Mapping):
        raise RoyaleAPIReplayError("采集 payload 缺少 battle")
    team_side = battle.get("team")
    opponent_side = battle.get("opponent")
    if not isinstance(team_side, Mapping) or not isinstance(opponent_side, Mapping):
        raise RoyaleAPIReplayError("采集 battle 缺少双方数据")
    team_player = _player(team_side, "team")
    opponent_player = _player(opponent_side, "opponent")
    raw_events = payload.get("events", ())
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise RoyaleAPIReplayError("采集 payload 的 events 不是数组")
    data_i = _replay_data_i(raw_events)

    # Collector coordinates are normalized into a team-as-blue, own-side-at-
    # bottom view. Restore the source true side before assigning native owners:
    # data_i=0 means team was already True Blue (bottom), while data_i=1 means
    # team was True Red (top) and the collector rotated its X coordinate.
    if data_i == 0:
        bottom_side = "team"
        flip_horizontal = False
        flip_vertical = True
    else:
        bottom_side = "opponent"
        flip_horizontal = True
        flip_vertical = False
    top_side = "opponent" if bottom_side == "team" else "team"
    owner_for_side = {top_side: 0, bottom_side: 1}
    side_for_owner = {owner: side for side, owner in owner_for_side.items()}
    normalized_bottom_tag = _normalized_search_query(
        str((team_player if bottom_side == "team" else opponent_player).get("tag") or "")
    )
    requested_bottom_tag = _normalized_search_query(bottom_player_tag or "")
    if requested_bottom_tag:
        requested_bottom_side = _bottom_side_for_player(team_player, opponent_player, requested_bottom_tag)
        if requested_bottom_side != bottom_side:
            raise RoyaleAPIReplayError(
                f"bottom player #{requested_bottom_tag} conflicts with replay True Red / True Blue data_i={data_i}"
            )
    team_tower_key, team_tower_id, team_tower_missing = _tower_troop(team_player, "team")
    opponent_tower_key, opponent_tower_id, opponent_tower_missing = _tower_troop(opponent_player, "opponent")
    team_deck = _load_deck(team_player)
    opponent_deck = _load_deck(opponent_player)
    decks = {"team": team_deck, "opponent": opponent_deck}
    players = {"team": team_player, "opponent": opponent_player}
    sides = {"team": team_side, "opponent": opponent_side}
    tower_keys = {"team": team_tower_key, "opponent": opponent_tower_key}
    tower_ids = {"team": team_tower_id, "opponent": opponent_tower_id}

    parsed_events: list[dict[str, Any]] = []
    play_ids: dict[str, list[int]] = {"team": [], "opponent": []}
    deck_ids = {side: {item.info.card_id for item in deck} for side, deck in decks.items()}
    ability_count = 0
    uniquely_bound_ability_count = 0
    ambiguous_ability_count = 0
    for event_index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise RoyaleAPIReplayError(f"event {event_index} 不是对象")
        side = str(raw_event.get("side") or "")
        if side not in owner_for_side:
            raise RoyaleAPIReplayError(f"event {event_index} side 无效")
        kind = str(raw_event.get("kind") or "")
        source_tick = int(raw_event.get("replay_tick_20hz", -1))
        if source_tick < 1:
            raise RoyaleAPIReplayError(f"event {event_index} tick 无效")
        native_tick = source_tick + ROYALAPI_COMMAND_TO_NATIVE_OBSERVABLE_TICKS
        parsed = {
            "event_index": event_index,
            "source_index": int(raw_event.get("source_index", event_index)),
            "side": side,
            "owner": owner_for_side[side],
            "kind": kind,
            "source_tick": source_tick,
            "native_tick": native_tick,
            "raw": raw_event,
        }
        if kind == "play_card":
            key = str(raw_event.get("card_key") or "")
            info = resolve_native_card(key, None)
            if info.card_id not in deck_ids[side]:
                raise RoyaleAPIReplayError(
                    f"event {event_index} 的 {key} 不在 {side} 牌组中；该记录无法可靠 native 重演"
                )
            parsed["card_id"] = info.card_id
            play_ids[side].append(info.card_id)
            coordinates = raw_event.get("coordinates")
            if not isinstance(coordinates, Mapping) or not bool(coordinates.get("inside_18x32_arena")):
                raise RoyaleAPIReplayError(f"event {event_index} 没有有效 native 坐标")
        elif kind == "activate_ability":
            ability_count += 1
            source_cards = _ability_source_cards(raw_event, decks[side])
            parsed["ability_source_keys"] = tuple(card.key for card in source_cards)
            parsed["ability_source_names"] = tuple(card.name or card.info.name for card in source_cards)
            parsed["ability_runtime_hints"] = _ability_runtime_hints(source_cards)
            if len(source_cards) == 1:
                uniquely_bound_ability_count += 1
            else:
                ambiguous_ability_count += 1
        else:
            raise RoyaleAPIReplayError(f"event {event_index} kind {kind!r} 不受支持")
        parsed_events.append(parsed)

    selected_by_side: dict[str, _SelectedCycle] = {}
    slots_by_event: dict[int, int] = {}
    for side in ("team", "opponent"):
        owner = owner_for_side[side]
        selected = _select_cycle(decks[side], play_ids[side], replay_tag=replay_tag, owner=owner)
        selected_by_side[side] = selected
        play_event_indices = [
            int(event["event_index"])
            for event in parsed_events
            if event["side"] == side and event["kind"] == "play_card"
        ]
        slots_by_event.update(zip(play_event_indices, selected.hand_slots))

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in parsed_events:
        grouped[int(event["native_tick"])].append(event)
    operations: list[SnapshotOperationV1] = []
    previous_tick = 0
    for tick in sorted(grouped):
        interval = tick - previous_tick
        if interval < 1:
            raise RoyaleAPIReplayError("回放事件 tick 未严格递增分组")
        actions: list[ActionV1] = []
        for event in sorted(grouped[tick], key=lambda item: (int(item["source_index"]), int(item["event_index"]))):
            owner = int(event["owner"])
            source_action_id = (
                "royaleapi-"
                + hashlib.sha256(
                    json.dumps(
                        (replay_tag, int(event["source_index"]), int(event["event_index"]), owner, str(event["kind"])),
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()[:24]
            )
            if event["kind"] == "play_card":
                coordinates = event["raw"]["coordinates"]
                grid = coordinates.get("grid_cell_floor")
                offset = coordinates.get("subcell_offset_from_floor_center")
                if not isinstance(grid, Mapping):
                    raise RoyaleAPIReplayError("play_card 缺少 grid_cell_floor")
                target_grid, subcell_offset = _oriented_grid_target(
                    grid,
                    offset if isinstance(offset, Mapping) else None,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                )
                actions.append(
                    ActionV1(
                        owner=owner,
                        kind=ActionKind.PLAY_CARD,
                        hand_slot=slots_by_event[int(event["event_index"])],
                        card_id=int(event["card_id"]),
                        target_kind=TargetKind.GRID,
                        target_grid=target_grid,
                        subcell_offset=subcell_offset,
                        execute_offset_ticks=interval,
                        next_decision_ticks=interval,
                        action_id=source_action_id,
                        metadata={
                            "source": "royaleapi",
                            "replay_tag": replay_tag,
                            "source_index": int(event["source_index"]),
                            "source_event_index": int(event["event_index"]),
                            "source_command_tick": int(event["source_tick"]),
                            "native_observable_tick": tick,
                        },
                    )
                )
            else:
                # The collector knows the side and tick but not a stable native
                # entity key.  The direct renderer resolves the unique ready
                # ability source for that owner at the target tick.
                actions.append(
                    ActionV1(
                        owner=owner,
                        kind=ActionKind.ACTIVATE_ABILITY,
                        source_entity=0,
                        execute_offset_ticks=interval,
                        next_decision_ticks=interval,
                        action_id=source_action_id,
                        metadata={
                            "source": "royaleapi",
                            "replay_tag": replay_tag,
                            "source_index": int(event["source_index"]),
                            "source_event_index": int(event["event_index"]),
                            "source_command_tick": int(event["source_tick"]),
                            "native_observable_tick": tick,
                            "source_entity_inferred_at_render": True,
                            "ability_source_keys": list(event.get("ability_source_keys", ())),
                            "ability_source_names": list(event.get("ability_source_names", ())),
                            "ability_runtime_hints": list(event.get("ability_runtime_hints", ())),
                        },
                    )
                )
        present = {action.owner for action in actions}
        for owner in (0, 1):
            if owner not in present:
                actions.append(ActionV1.wait(owner, ticks=interval, metadata={"source": "royaleapi-gap"}))
        operations.append(
            SnapshotOperationV1(
                actions=tuple(actions),
                advance_ticks=interval,
                requested_advance_ticks=interval,
                start_native_tick=previous_tick,
                end_native_tick=tick,
            )
        )
        previous_tick = tick

    last_source_event_tick = max((int(event["source_tick"]) for event in parsed_events), default=0)
    last_native_event_tick = previous_tick
    source_end_tick = _source_end_tick(payload, last_event_tick=last_source_event_tick)
    native_end_tick = max(source_end_tick, last_native_event_tick)
    if native_end_tick > previous_tick:
        interval = native_end_tick - previous_tick
        operations.append(
            SnapshotOperationV1(
                actions=(
                    ActionV1.wait(0, ticks=interval, metadata={"source": "royaleapi-terminal-gap"}),
                    ActionV1.wait(1, ticks=interval, metadata={"source": "royaleapi-terminal-gap"}),
                ),
                advance_ticks=interval,
                requested_advance_ticks=interval,
                start_native_tick=previous_tick,
                end_native_tick=native_end_tick,
            )
        )
        previous_tick = native_end_tick

    selected_by_owner = tuple(selected_by_side[side_for_owner[owner]] for owner in (0, 1))
    policy_deal_invariance, policy_deal_invariance_id = _policy_deal_invariance_boundaries(
        replay_tag=replay_tag, selected_by_owner=selected_by_owner, grouped_events=grouped
    )
    policy_stream_id = policy_action_stream_id(replay_tag, operations)
    deck_cards_by_owner = tuple(decks[side_for_owner[owner]] for owner in (0, 1))
    omitted_starting_hand_by_owner = tuple(
        tuple(card.info.card_id for card in deck_cards_by_owner[owner] if card.info.omit_from_starting_hand)
        for owner in (0, 1)
    )
    player_by_owner = tuple(players[side_for_owner[owner]] for owner in (0, 1))
    tower_key_by_owner = tuple(tower_keys[side_for_owner[owner]] for owner in (0, 1))
    tower_id_by_owner = tuple(tower_ids[side_for_owner[owner]] for owner in (0, 1))
    levels = [
        value for deck in (team_deck, opponent_deck) for item in deck for value in (item.level,) if value is not None
    ]
    for player in (team_player, opponent_player):
        tower = player.get("tower_card")
        if isinstance(tower, Mapping) and isinstance(tower.get("level"), (int, float)):
            levels.append(int(tower["level"]))
    level_counts = Counter(levels)
    selected_level = sorted(level_counts.items(), key=lambda item: (-item[1], -item[0]))[0][0] if level_counts else 11
    warnings: list[str] = [
        "双方初始手牌未被采集；已按 replay tag 随机选取一组能合法复现公开出牌序列的初始手牌与循环顺序。",
        "这是用 RoyaleAPI 公开动作轨迹新建的 native 动作重建，不是"
        "原官方客户端的完整 replay 状态；若引擎内容或未公开状态不同，"
        "塔血与终局可能分叉。",
    ]
    for owner, omitted_ids in enumerate(omitted_starting_hand_by_owner):
        if not omitted_ids:
            continue
        names_by_id = {card.info.card_id: card.info.name for card in deck_cards_by_owner[owner]}
        names = "、".join(names_by_id[card_id] for card_id in omitted_ids)
        side_label = "上方" if owner == 0 else "下方"
        warnings.append(
            f"{side_label}牌组中的 {names} 不会出现在初始手牌；随机牌序与 native 校准已启用禁止初手牌规则。"
        )
    warnings.append(
        f"Preserved RoyaleAPI true sides from data_i={data_i}: "
        f"{top_side}=True Red/owner 0, {bottom_side}=True Blue/owner 1; "
        f"coordinate flips x={flip_horizontal}, y={flip_vertical}."
    )
    if requested_bottom_tag:
        warnings.append(
            f"Verified bottom player assertion #{requested_bottom_tag} against the replay true-side marker."
        )
    if len(level_counts) > 1:
        warnings.append(f"原局存在不同卡等；当前 native match 只支持统一等级，本次采用众数等级 {selected_level}。")
    if selected_level > 16:
        warnings.append("native v15 最高等级为 16，已把显示等级截到 16。")
    selected_level = min(max(int(selected_level), 1), 16)
    if team_tower_missing or opponent_tower_missing:
        warnings.append("部分玩家缺少塔兵字段；仅缺失的一方回退为公主塔。")
    configured_towers = f"上方 {tower_key_by_owner[0]} / 下方 {tower_key_by_owner[1]}"
    if team_tower_key != "tower-princess" or opponent_tower_key != "tower-princess":
        warnings.append(f"已按原局配置双方塔兵：{configured_towers}。")
    if ability_count and ambiguous_ability_count == 0:
        warnings.append(
            f"{ability_count} 个英雄技能事件已按牌组唯一绑定；"
            "RoyaleAPI 原页面不提供战场实例 ID，播放时会在目标 tick "
            "验证对应的唯一 Ready 实例。"
        )
    elif ability_count:
        warnings.append(
            f"{ability_count} 个英雄技能事件中，"
            f"{uniquely_bound_ability_count} 个已按牌组唯一绑定，"
            f"{ambiguous_ability_count} 个仍需在播放时从 Ready 实例解析。"
        )

    source_game_mode = str(battle.get("game_mode") or "")
    (native_game_mode, native_arena, native_location, native_arena_profile) = _native_battle_presentation(
        source_game_mode
    )
    if native_arena_profile == "classic-legendary-arena.v1":
        warnings.append(f"已按 {source_game_mode} 配置经典挑战场地 Arena_Legendary / PvP_champion。")

    episode = EpisodeConfigV1(
        ruleset_id=NATIVE_RENDER_RULESET_ID,
        deck0=selected_by_owner[0].deck,
        deck1=selected_by_owner[1].deck,
        seed=NATIVE_DEAL_SEED,
        game_mode=native_game_mode,
        arena=native_arena,
        decision_hz=20.0,
        event_driven_decisions=False,
        tags={
            "deck0_form_availability": selected_by_owner[0].forms,
            "deck1_form_availability": selected_by_owner[1].forms,
            "deck0_omit_from_starting_hand_ids": omitted_starting_hand_by_owner[0],
            "deck1_omit_from_starting_hand_ids": omitted_starting_hand_by_owner[1],
            "tower_troop0_id": tower_id_by_owner[0],
            "tower_troop1_id": tower_id_by_owner[1],
            "tower_troop0_key": tower_key_by_owner[0],
            "tower_troop1_key": tower_key_by_owner[1],
            "level_cap": selected_level,
            "minimum_card_level": selected_level,
            "king_tower_level": selected_level,
            "owner0_name": _display_name(player_by_owner[0], "RoyaleAPI-Top"),
            "owner1_name": _display_name(player_by_owner[1], "RoyaleAPI-Bottom"),
            "end_tick": native_end_tick,
            "royaleapi_replay_tag": replay_tag,
            "bottom_player_tag": normalized_bottom_tag,
            "royaleapi_data_i": data_i,
            "true_side_mapping_version": TRUE_SIDE_MAPPING_VERSION,
            "source_team_true_side": "blue" if data_i == 0 else "red",
            "source_true_blue_side": bottom_side,
            "source_true_red_side": top_side,
            "source_team_owner": owner_for_side["team"],
            "arena_horizontal_flip": flip_horizontal,
            "arena_vertical_flip": flip_vertical,
            "source_game_mode": source_game_mode,
            "native_arena_profile": native_arena_profile,
            "location": native_location,
            "initial_order_policy": "random-single-compatible.v3",
            "reconstruction_mode": "royaleapi-public-actions.v3",
            "source_tick_semantics": POLICY_DEAL_COMMAND_BOUNDARY_SEMANTICS,
            "native_observable_tick_offset": (ROYALAPI_COMMAND_TO_NATIVE_OBSERVABLE_TICKS),
            "policy_deal_invariance_version": (POLICY_DEAL_INVARIANCE_VERSION),
            "policy_deal_invariance_id": policy_deal_invariance_id,
            "policy_deal_invariance": tuple(item.binding() for item in policy_deal_invariance),
            "policy_action_stream_version": POLICY_ACTION_STREAM_VERSION,
            "policy_action_stream_id": policy_stream_id,
            "source_end_tick": source_end_tick,
            "last_source_event_tick": last_source_event_tick,
            "last_native_event_tick": last_native_event_tick,
        },
    )
    episode, policy_episode_id = _bind_policy_episode_config(replay_tag, episode)
    source_winner = _winner(battle.get("result"))
    winner = owner_for_side["team" if source_winner == 0 else "opponent"] if source_winner is not None else None
    replay = TrainingReplayV1(
        completed_match_index=1,
        ruleset_id=NATIVE_RENDER_RULESET_ID,
        episode_config=episode,
        episode_id=f"royaleapi-{replay_tag}",
        operations=tuple(operations),
        terminal={
            "winner": winner,
            "source_result": str(battle.get("result") or ""),
            "native_tick": native_end_tick,
            "source_end_tick": source_end_tick,
            "last_source_event_tick": last_source_event_tick,
            "last_native_event_tick": last_native_event_tick,
            "players": tuple(
                {"owner": owner, "crowns": int(sides[side_for_owner[owner]].get("crowns", 0))} for owner in (0, 1)
            ),
        },
        source={
            "site": "RoyaleAPI",
            "replay_tag": replay_tag,
            "requested_player_tag": str(battle.get("requested_player_tag") or ""),
            "played_at_utc": battle.get("played_at_utc"),
            "adapter": NATIVE_RENDER_RULESET_ID,
            "warnings": warnings,
        },
        created_at_ns=_created_at_ns(payload),
    )
    return PreparedCollectedReplay(
        replay=replay,
        replay_tag=replay_tag,
        event_count=len(parsed_events),
        warnings=tuple(warnings),
        opening_cards=(selected_by_owner[0].opening, selected_by_owner[1].opening),
        queue_cards=(selected_by_owner[0].queue, selected_by_owner[1].queue),
        policy_deal_invariance=policy_deal_invariance,
        policy_deal_invariance_id=policy_deal_invariance_id,
        policy_action_stream_id=policy_stream_id,
        policy_episode_config_id=policy_episode_id,
    )
