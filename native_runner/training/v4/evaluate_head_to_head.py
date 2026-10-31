"""Play IL/PPO checkpoints against each other on the resident engine cluster.

Every model pair plays the SAME schedule rows (common random numbers): the
first ``--games-per-pair / 2`` configurations of a ``make_head_to_head_schedule``
file, each twice with the models swapping seats while decks stay with seats.
Within a pair (A, B) configuration ``k`` makes A the collector's "current"
role when ``k`` is even and B when ``k`` is odd, so both models play both role
labels on both seats.  The frozen side runs as a ``current-history`` league
opponent whose checkpoint id is the model NAME; the worker treats current and
frozen owners identically for action selection (roles only change PPO
rewards/storage and the legacy role-based opening force, which is unused here).

Waves are packed per current model and run through
``collect_async_cluster_wave_v4`` with ``segment_decision_steps`` (default 160)
and a segment callback that keeps the policy id and accumulates the per-segment
``collection.reports``.  Rollout storage cannot be disabled cleanly in the
collector, so the current side's observations of one segment stay resident:
roughly ``lanes x segment_decision_steps`` packed rows per wave.

Usage:
    python -m native_runner.training.v4.evaluate_head_to_head \
        --topology topology.json --deck-pool v4-ppo-deck-pool-use5.json \
        --schedule h2h/schedule.jsonl \
        --model IL=/path/IL.pt:gate_logit_offset=0 --model C1=/path/c1.pt:gate_logit_offset=-2.0794 \
        [--model C1E=/path/c1.pt:delay_policy=earliest] [--model HOG=/path/hog.pt:deck=<deck_id>] \
        (--pairs IL:C1 [--pairs ...] | --round-robin) --games-per-pair 400 \
        --mode sample|argmax [--keep-checkpoint-temperatures] \
        --forced-opening none|deck-rule [--force-tick 300] --warmup-ticks 90 \
        --lanes-per-engine 1 --output-dir run/ --device cuda:0 --torch-seed 20261023 \
        [--engine-indices 0,1,...] [--max-waves N] [--resume]

Per-model options: ``gate_logit_offset=X``, ``mode=sample|argmax`` and
``delay_policy=sample|earliest`` (default ``sample``: the delay head picks the
bin; ``earliest``: the first legal delay bin, see
``UniversalCardPolicyV4.set_delay_policy``).  Register one checkpoint under two
names to play it against itself with different options.  A non-default
delay policy is added to the model's identity/behavior key; default runs keep
their previous identity, so ``--resume`` of older runs still matches.

``deck=<deck_id>`` pins a specialist to one deck of the pool: whenever that
model sits in a seat, the seat plays this deck instead of the schedule's deck
for it, while the opponent keeps the schedule's deck of its own seat (so over
a configuration's two seat-swapped rows the opponent plays both scheduled
decks).  Game rows then carry the decks actually played in ``deck0_id`` /
``deck1_id`` plus the scheduled ones in ``schedule_deck0_id`` /
``schedule_deck1_id``; the pinned deck enters the model's identity and
behavior key.  Without the option nothing changes.

Exit status is non-zero when any game was truncated / did not end (hard
failure) or when fewer games than planned were completed without --max-waves.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .aggregate_head_to_head import TIEBREAK_TICK, pair_summaries_v4
from .async_cluster_self_play import AsyncResidentEngineSpecV4, collect_async_cluster_wave_v4
from .checkpoint import load_actor_critic_checkpoint
from .cluster_self_play import ClusterCollectionV4, ClusterMatchAssignmentV4
from .config import ModelConfigV4
from .decoding import DELAY_POLICIES, DELAY_POLICY_SAMPLE
from .deck_pool import DeckIdentityV4, DeckPoolManifestV4
from .factory import production_semantic_bundle
from .league import LeagueEntryV4
from .make_head_to_head_schedule import load_head_to_head_schedule_v4
from .matchmaking import POLICY_HISTORY, ScheduledMatchupV4
from .model import UniversalCardPolicyV4
from .ppo_runtime import PPO_DECISION_TICKS, ppo_environment_config_v4
from .tensors import GATE_ACT, GATE_WAIT


GAME_ROW_SCHEMA_V4 = "v4-head-to-head-game.v1"
DECISION_MODES = ("sample", "argmax")
FORCED_OPENING_MODES = ("none", "deck-rule")
_GATE_OFFSET_ATTRIBUTE = "_head_to_head_gate_logit_offset_v4"
_MODEL_OPTION = re.compile(r"^(?P<rest>.+):(?P<key>gate_logit_offset|mode|delay_policy|deck)=(?P<value>[^:]+)$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
GAMMA_PER_DECISION = 0.9997
SHAPING_BETA = 0.05


@dataclass(frozen=True, slots=True)
class ModelSpecV4:
    name: str
    path: Path
    gate_logit_offset: float = 0.0
    mode: str | None = None
    delay_policy: str = DELAY_POLICY_SAMPLE
    fixed_deck_id: str | None = None

    def __post_init__(self) -> None:
        if not _MODEL_NAME.match(self.name):
            raise ValueError(f"model name {self.name!r} must match {_MODEL_NAME.pattern}")
        if not math.isfinite(self.gate_logit_offset):
            raise ValueError("gate logit offset must be finite")
        if self.mode is not None and self.mode not in DECISION_MODES:
            raise ValueError(f"model mode must be one of {DECISION_MODES}")
        if self.delay_policy not in DELAY_POLICIES:
            raise ValueError(f"model delay_policy must be one of {DELAY_POLICIES}")
        if self.fixed_deck_id is not None and not self.fixed_deck_id.strip():
            raise ValueError("model deck must be a non-empty deck id")


def parse_model_spec_v4(text: str) -> ModelSpecV4:
    """Parse ``NAME=PATH[:gate_logit_offset=X][:mode=sample|argmax][:delay_policy=sample|earliest][:deck=ID]``.

    Options may come in any order.
    """

    if "=" not in text:
        raise ValueError(f"model spec {text!r} needs NAME=PATH")
    name, rest = text.split("=", 1)
    options: dict[str, str] = {}
    while True:
        match = _MODEL_OPTION.match(rest)
        if match is None:
            break
        if match.group("key") in options:
            raise ValueError(f"model spec {text!r} repeats {match.group('key')}")
        options[match.group("key")] = match.group("value")
        rest = match.group("rest")
    if not rest:
        raise ValueError(f"model spec {text!r} has an empty path")
    try:
        offset = float(options.get("gate_logit_offset", "0"))
    except ValueError as error:
        raise ValueError(f"model spec {text!r} has a non-numeric gate_logit_offset") from error
    return ModelSpecV4(
        name=name.strip(),
        path=Path(rest),
        gate_logit_offset=offset,
        mode=options.get("mode"),
        delay_policy=options.get("delay_policy", DELAY_POLICY_SAMPLE),
        fixed_deck_id=options.get("deck"),
    )


def parse_pairs_v4(values: Sequence[str], names: Sequence[str], *, round_robin: bool) -> tuple[tuple[str, str], ...]:
    known = tuple(names)
    if len(known) != len(set(known)):
        raise ValueError("model names must be unique")
    if round_robin:
        if values:
            raise ValueError("--pairs and --round-robin are mutually exclusive")
        if len(known) < 2:
            raise ValueError("round robin needs at least two models")
        return tuple((known[i], known[j]) for i in range(len(known)) for j in range(i + 1, len(known)))
    pairs: list[tuple[str, str]] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(f"pair {item!r} must be NAME:NAME")
            first, second = parts[0].strip(), parts[1].strip()
            if first not in known or second not in known:
                raise ValueError(f"pair {item!r} names an unknown model")
            if first == second:
                raise ValueError(f"pair {item!r} repeats one model; register the checkpoint twice for A-vs-A")
            pairs.append((first, second))
    if not pairs:
        raise ValueError("no model pairs were requested")
    unordered = [frozenset(pair) for pair in pairs]
    if len(unordered) != len(set(unordered)):
        raise ValueError("a model pair was requested twice")
    return tuple(pairs)


def model_config_from_contract_v4(mapping: Mapping[str, Any]) -> ModelConfigV4:
    """Rebuild the checkpoint's ModelConfigV4 so architecture variants load exactly."""

    known = {item.name: item for item in fields(ModelConfigV4)}
    unknown = sorted(set(mapping).difference(known))
    if unknown:
        raise ValueError(f"checkpoint model_config has fields this code does not know: {unknown}")
    defaults = ModelConfigV4()
    values = {}
    for key, value in mapping.items():
        if isinstance(getattr(defaults, key), tuple) and isinstance(value, list):
            value = tuple(value)
        values[key] = value
    return ModelConfigV4(**values)


def build_model_from_checkpoint_v4(path: Path, device: torch.device) -> tuple[UniversalCardPolicyV4, dict[str, Any]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    contract = raw.get("contract") if isinstance(raw, dict) else None
    if not isinstance(contract, Mapping) or not isinstance(contract.get("model_config"), Mapping):
        raise ValueError(f"{path} is not a V4 actor-critic checkpoint with a model_config contract")
    config = model_config_from_contract_v4(contract["model_config"])
    del raw
    bundle = production_semantic_bundle()
    model = UniversalCardPolicyV4(
        bundle.card_catalog,
        ability_catalog=bundle.ability_catalog,
        entity_archetype_catalog=bundle.entity_archetype_catalog,
        effect_catalog=bundle.effect_catalog,
        mechanic_profile_catalog=bundle.mechanic_profile_catalog,
        config=config,
    ).to(device)
    payload = load_actor_critic_checkpoint(path, model, map_location=device)
    model.eval()
    return model, payload


def apply_gate_logit_offset_v4(model: UniversalCardPolicyV4, offset: float) -> None:
    """Add ``offset`` to the ACT logit of the gate head exactly once.

    ``-log(w)`` inverts a training-time ACT class weight ``w`` exactly: the
    weighted cross-entropy optimum has ACT odds inflated by ``w``.
    """

    if not math.isfinite(offset):
        raise ValueError("gate logit offset must be finite")
    if model.__dict__.get(_GATE_OFFSET_ATTRIBUTE) is not None:
        raise RuntimeError("gate logit offset was already applied to this model")
    gate = model.gate_head[-1]
    if not isinstance(gate, nn.Linear) or gate.out_features != 2 or gate.bias is None:
        raise TypeError("gate head must end in a two-way Linear layer with bias")
    if GATE_ACT == GATE_WAIT:
        raise RuntimeError("gate ACT/WAIT indices collide")
    with torch.no_grad():
        gate.bias[GATE_ACT] += float(offset)
    model.__dict__[_GATE_OFFSET_ATTRIBUTE] = float(offset)


def configure_decision_mode_v4(model: UniversalCardPolicyV4, mode: str, *, keep_checkpoint_temperatures: bool) -> None:
    if mode == "sample":
        model.set_ppo_rollout_sampling(True)
        if not keep_checkpoint_temperatures:
            model.set_ppo_gate_temperature(1.0)
            model.set_ppo_action_temperature(1.0)
            model.set_ppo_continue_temperature(1.0)
    elif mode == "argmax":
        model.set_ppo_rollout_sampling(False)
    else:
        raise ValueError(f"decision mode must be one of {DECISION_MODES}")


def behavior_key_v4(
    checkpoint_id: str, offset: float, mode: str, model: UniversalCardPolicyV4, *, fixed_deck_id: str | None = None
) -> str:
    temperatures = (
        "argmax"
        if mode == "argmax"
        else f"T={model.ppo_gate_temperature:g},{model.ppo_action_temperature:g},{model.ppo_continue_temperature:g}"
    )
    key = f"{checkpoint_id}|gate_logit_offset={offset!r}|mode={mode}|{temperatures}"
    # Appended only when non-default so existing behavior keys are unchanged.
    delay_policy = getattr(model, "delay_policy", DELAY_POLICY_SAMPLE)
    if delay_policy != DELAY_POLICY_SAMPLE:
        key = f"{key}|delay_policy={delay_policy}"
    return key if fixed_deck_id is None else f"{key}|deck={fixed_deck_id}"


@dataclass(frozen=True, slots=True)
class GameJobV4:
    game_index: int
    pair_index: int
    model_a: str
    model_b: str
    schedule_row: Mapping[str, Any]
    seat_models: tuple[str, str]
    current_model: str

    @property
    def pair(self) -> str:
        return f"{self.model_a}:{self.model_b}"

    @property
    def current_seat(self) -> int:
        return self.seat_models.index(self.current_model)

    @property
    def frozen_model(self) -> str:
        return self.seat_models[1 - self.current_seat]


def plan_head_to_head_games_v4(
    schedule_rows: Sequence[Mapping[str, Any]], pairs: Sequence[tuple[str, str]], *, games_per_pair: int
) -> tuple[GameJobV4, ...]:
    if games_per_pair <= 0 or games_per_pair % 2:
        raise ValueError("games per pair must be a positive even number")
    if games_per_pair > len(schedule_rows):
        raise ValueError(f"schedule has {len(schedule_rows)} rows; {games_per_pair} games per pair requested")
    jobs: list[GameJobV4] = []
    for pair_index, (model_a, model_b) in enumerate(pairs):
        for row in schedule_rows[:games_per_pair]:
            slots = {"A": model_a, "B": model_b}
            seat_models = (slots[str(row["owner0_slot"])], slots[str(row["owner1_slot"])])
            current = model_a if int(row["config_id"]) % 2 == 0 else model_b
            jobs.append(
                GameJobV4(
                    game_index=len(jobs),
                    pair_index=pair_index,
                    model_a=model_a,
                    model_b=model_b,
                    schedule_row=row,
                    seat_models=seat_models,
                    current_model=current,
                )
            )
    return tuple(jobs)


def plan_waves_v4(
    jobs: Sequence[GameJobV4], *, capacity: int, model_order: Sequence[str]
) -> tuple[tuple[str, tuple[GameJobV4, ...]], ...]:
    """Group games by current model (one current model per collector wave) and chunk by lane capacity."""

    if capacity <= 0:
        raise ValueError("wave capacity must be positive")
    grouped: dict[str, list[GameJobV4]] = defaultdict(list)
    for job in jobs:
        grouped[job.current_model].append(job)
    waves: list[tuple[str, tuple[GameJobV4, ...]]] = []
    for name in model_order:
        group = grouped.pop(name, [])
        for start in range(0, len(group), capacity):
            waves.append((name, tuple(group[start : start + capacity])))
    if grouped:
        raise ValueError(f"games reference unknown current models: {sorted(grouped)}")
    return tuple(waves)


def seat_deck_ids_v4(job: GameJobV4, fixed_decks: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Deck ids actually played per seat: a model's pinned deck replaces its seat's scheduled deck."""

    row = job.schedule_row
    pinned = fixed_decks or {}
    return (
        str(pinned.get(job.seat_models[0], row["deck0_id"])),
        str(pinned.get(job.seat_models[1], row["deck1_id"])),
    )


def assignment_for_job_v4(
    job: GameJobV4,
    decks: Mapping[str, DeckIdentityV4],
    model_paths: Mapping[str, str],
    fixed_decks: Mapping[str, str] | None = None,
) -> ClusterMatchAssignmentV4:
    row = job.schedule_row
    deck0_id, deck1_id = seat_deck_ids_v4(job, fixed_decks)
    try:
        deck0, deck1 = decks[deck0_id], decks[deck1_id]
    except KeyError as error:
        raise ValueError(f"schedule deck {error} is not in the deck pool") from error
    matchup = ScheduledMatchupV4(
        sequence=job.game_index + 1,
        category=str(row["category"]),  # type: ignore[arg-type]
        policy_matchup=POLICY_HISTORY,
        deck0=deck0,
        deck1=deck1,
        current_owners=(job.current_seat,),
        battle_level=int(row["battle_level"]),
        seed=int(row["seed"]),
    )
    layout0, layout1 = matchup.runtime_layouts()
    # A pinned deck replaces the scheduled one, so only scheduled seats can be checked against the row.
    if (deck0_id == str(row["deck0_id"]) and list(layout0.card_ids) != list(row["deck0"])) or (
        deck1_id == str(row["deck1_id"]) and list(layout1.card_ids) != list(row["deck1"])
    ):
        raise ValueError(f"schedule row {row['schedule_row_id']} runtime layout differs from its seed")
    opponent = LeagueEntryV4(
        checkpoint_id=job.frozen_model, checkpoint_path=model_paths[job.frozen_model], update_step=0
    )
    return ClusterMatchAssignmentV4(matchup, opponent=opponent)


def engine_specs_for_wave_v4(
    topology: Mapping[str, Any],
    engine_indices: Sequence[int],
    *,
    games: int,
    lanes_per_engine: int,
    force_tick: int | None,
    warmup_ticks: int,
) -> tuple[AsyncResidentEngineSpecV4, ...]:
    if not 1 <= lanes_per_engine <= 8:
        raise ValueError("lanes per engine must be in [1, 8]")
    engines_needed = math.ceil(games / lanes_per_engine)
    if games <= 0 or engines_needed > len(engine_indices):
        raise ValueError("wave does not fit the selected engines")
    specs = []
    remaining = games
    for index in engine_indices[:engines_needed]:
        lanes = min(lanes_per_engine, remaining)
        remaining -= lanes
        specs.append(
            AsyncResidentEngineSpecV4(
                engine_index=int(index),
                cpu=int(topology["worker_cpus"][index]),
                host=str(topology["engine_hosts"][index]),
                base_port=int(topology["engine_ports"][index]) - int(index),
                lanes=lanes,
                timeout=600.0,
                capture_mode="compact",
                worker_host=str(topology["engine_worker_hosts"][index]),
                worker_port=int(topology["engine_worker_ports"][index]),
                force_deck_rule_opening_tick=force_tick,
                environment_warmup_ticks=warmup_ticks,
                skip_shaping_telemetry=True,
            )
        )
    if remaining:
        raise RuntimeError("wave lane packing left games unassigned")
    return tuple(specs)


class SegmentReportAccumulatorV4:
    """Segment callback: keep one policy id and accumulate reports cleared per segment."""

    def __init__(self, policy_state_id: str) -> None:
        if not policy_state_id:
            raise ValueError("policy state id must be non-empty")
        self.policy_state_id = policy_state_id
        self.reports: list[dict[str, object]] = []
        self.timings: list[dict[str, float]] = []

    def __call__(self, collection: ClusterCollectionV4) -> str:
        self.reports.extend(dict(report) for report in collection.reports)
        self.timings.append(dict(collection.timing))
        return self.policy_state_id

    @property
    def segments(self) -> int:
        return len(self.timings)

    def timing_summary(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for timing in self.timings:
            for key, value in timing.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    totals[key] += float(value)
        return dict(sorted(totals.items()))


def _pair_tuple(value: object, name: str) -> tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"report field {name} must hold two owners")
    return (value[0], value[1])


def game_row_from_report_v4(
    job: GameJobV4,
    report: Mapping[str, Any],
    *,
    wave_index: int,
    behaviors: Mapping[str, str],
    fixed_decks: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    row = job.schedule_row
    deck_ids = seat_deck_ids_v4(job, fixed_decks)
    expected = {"deck0_id": deck_ids[0], "deck1_id": deck_ids[1], "battle_level": row["battle_level"]}
    for key, value in expected.items():
        if str(report[key]) != str(value):
            raise RuntimeError(f"game {job.game_index} report {key} differs from its schedule row")
    if tuple(report["current_owners"]) != (job.current_seat,):
        raise RuntimeError(f"game {job.game_index} report current owner differs from the plan")
    result = _pair_tuple(report["result"], "result")
    result_seat0 = float(result[0])
    forced_owner = report.get("opening_forced_owner")
    truncated = bool(report["truncated"])
    ended = bool(report["terminal_ended"])
    seat0_score = 0.5 * (result_seat0 + 1.0)
    result: dict[str, Any] = {
        "schema": GAME_ROW_SCHEMA_V4,
        "game_id": job.game_index,
        "wave": wave_index,
        "pair": job.pair,
        "pair_index": job.pair_index,
        "model_a": job.model_a,
        "model_b": job.model_b,
        "schedule_row_id": int(row["schedule_row_id"]),
        "config_id": int(row["config_id"]),
        "seed": int(row["seed"]),
        "battle_level": int(row["battle_level"]),
        "category": str(row["category"]),
        "deck0_id": deck_ids[0],
        "deck1_id": deck_ids[1],
        "seat0_model": job.seat_models[0],
        "seat1_model": job.seat_models[1],
        "seat0_behavior": behaviors[job.seat_models[0]],
        "seat1_behavior": behaviors[job.seat_models[1]],
        "current_model": job.current_model,
        "current_seat": job.current_seat,
        "frozen_model": job.frozen_model,
        "result_seat0": result_seat0,
        "outcome_seat0": "win" if result_seat0 > 0 else ("loss" if result_seat0 < 0 else "draw"),
        "score_a": seat0_score if job.seat_models[0] == job.model_a else 1.0 - seat0_score,
        "crowns": list(_pair_tuple(report["crowns"], "crowns")),
        "terminal_ended": ended,
        "terminal_reason": report.get("terminal_reason"),
        "terminal_winner": report.get("terminal_winner"),
        "terminal_tick": int(report["terminal_tick"]),
        "final_tower_hitpoints": list(report.get("final_tower_hitpoints") or ()),
        "active_actions": list(_pair_tuple(report["active_actions"], "active_actions")),
        "rejected_actions": list(_pair_tuple(report["rejected_actions"], "rejected_actions")),
        # Additive: decision steps with ACT / with a second micro-action (older reports lack them).
        "act_steps": list(_pair_tuple(report.get("act_steps", (0, 0)), "act_steps")),
        "second_action_steps": list(_pair_tuple(report.get("second_action_steps", (0, 0)), "second_action_steps")),
        "rejection_details": list(report.get("rejection_details") or ()),
        "first_nonwait_action_tick": list(_pair_tuple(report["first_nonwait_action_tick"], "first_nonwait")),
        "forced_owner": forced_owner,
        "forced_model": None if forced_owner is None else job.seat_models[int(forced_owner)],
        "forced_tick": report.get("opening_forced_tick"),
        "deck_average_elixir": list(_pair_tuple(report["deck_average_elixir"], "deck_average_elixir")),
        "decision_steps": int(report["decision_steps"]),
        "engine_index": int(report["engine_index"]),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "truncated": truncated,
        "hard_failure": truncated or not ended,
    }
    if deck_ids != (str(row["deck0_id"]), str(row["deck1_id"])):
        # Additive: only games with a pinned deck carry the scheduled decks.
        result["schedule_deck0_id"] = str(row["deck0_id"])
        result["schedule_deck1_id"] = str(row["deck1_id"])
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_completed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_wave_v4(
    collect: Callable[..., ClusterCollectionV4],
    *,
    specs: Sequence[AsyncResidentEngineSpecV4],
    jobs: Sequence[GameJobV4],
    assignments: Sequence[ClusterMatchAssignmentV4],
    models: Mapping[str, UniversalCardPolicyV4],
    current_model: str,
    policy_state_id: str,
    segment_decision_steps: int,
    max_decision_steps: int,
) -> tuple[list[dict[str, object]], SegmentReportAccumulatorV4, float]:
    """Run one wave; return reports ordered like ``jobs`` plus segment timing."""

    accumulator = SegmentReportAccumulatorV4(policy_state_id)
    frozen = sorted({job.frozen_model for job in jobs})
    started = time.perf_counter()
    collect(
        tuple(specs),
        tuple(assignments),
        models[current_model],
        models[current_model],
        {name: models[name] for name in frozen},
        gamma_per_decision=GAMMA_PER_DECISION,
        shaping_beta=SHAPING_BETA,
        mutual_elixir_overflow_penalty=0.0,
        mutual_elixir_overflow_grace=4.0,
        unilateral_elixir_overflow_penalty=0.0,
        elixir_overflow_step_penalty_cap=0.1,
        policy_state_id=policy_state_id,
        max_decision_steps=max_decision_steps,
        validate_tensors=False,
        segment_decision_steps=segment_decision_steps,
        segment_callback=accumulator,
    )
    elapsed = time.perf_counter() - started
    by_sequence: dict[int, dict[str, object]] = {}
    for report in accumulator.reports:
        sequence = int(report["sequence"])  # type: ignore[arg-type]
        if sequence in by_sequence:
            raise RuntimeError(f"wave {policy_state_id} reported game {sequence} twice")
        by_sequence[sequence] = report
    expected = [job.game_index + 1 for job in jobs]
    missing = sorted(set(expected).difference(by_sequence))
    extra = sorted(set(by_sequence).difference(expected))
    if missing or extra:
        raise RuntimeError(
            f"wave {policy_state_id} reports do not match its games: missing={missing[:10]} extra={extra[:10]} "
            f"(segments={accumulator.segments})"
        )
    return [by_sequence[sequence] for sequence in expected], accumulator, elapsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--deck-pool", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=PATH[:gate_logit_offset=X][:mode=M][:delay_policy=sample|earliest][:deck=DECK_ID]",
    )
    parser.add_argument("--pairs", action="append", default=[], help="NAME:NAME[,NAME:NAME...]; A is the first")
    parser.add_argument("--round-robin", action="store_true")
    parser.add_argument("--games-per-pair", type=int, required=True)
    parser.add_argument("--mode", choices=DECISION_MODES, required=True)
    parser.add_argument("--keep-checkpoint-temperatures", action="store_true")
    parser.add_argument("--forced-opening", choices=FORCED_OPENING_MODES, default="none")
    parser.add_argument("--force-tick", type=int, default=300)
    parser.add_argument("--warmup-ticks", type=int, default=90)
    parser.add_argument("--lanes-per-engine", type=int, default=1)
    parser.add_argument("--segment-decision-steps", type=int, default=160)
    parser.add_argument("--engine-indices", help="comma-separated topology engine indices (default: all)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-seed", type=int, default=20261023)
    parser.add_argument("--max-waves", type=int, help="stop after N waves (pilot); summary marks the run partial")
    parser.add_argument("--resume", action="store_true", help="skip games already in OUTPUT/games.jsonl")
    args = parser.parse_args(argv)

    if args.force_tick <= 0 or args.warmup_ticks <= 0 or args.segment_decision_steps <= 0:
        raise ValueError("force tick, warmup ticks and segment decision steps must be positive")
    if args.max_waves is not None and args.max_waves <= 0:
        raise ValueError("max waves must be positive")
    specs_by_name = [parse_model_spec_v4(value) for value in args.model]
    names = [spec.name for spec in specs_by_name]
    pairs = parse_pairs_v4(args.pairs, names, round_robin=args.round_robin)
    used = {name for pair in pairs for name in pair}
    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    engine_count = int(topology["engine_count"])
    for key in ("engine_hosts", "engine_ports", "engine_worker_hosts", "engine_worker_ports", "worker_cpus"):
        if len(topology[key]) != engine_count:
            raise ValueError(f"topology field {key} does not cover {engine_count} engines")
    engine_indices = (
        tuple(int(value) for value in args.engine_indices.split(","))
        if args.engine_indices
        else tuple(range(engine_count))
    )
    if not engine_indices or len(set(engine_indices)) != len(engine_indices):
        raise ValueError("engine indices must be unique and non-empty")
    if any(not 0 <= index < engine_count for index in engine_indices):
        raise ValueError("engine index is outside the topology")
    if not 1 <= args.lanes_per_engine <= 8:
        raise ValueError("lanes per engine must be in [1, 8]")
    capacity = len(engine_indices) * args.lanes_per_engine
    schedule_rows = load_head_to_head_schedule_v4(args.schedule)
    manifest = DeckPoolManifestV4.load(args.deck_pool)
    decks = {deck.deck_id: deck for deck in manifest.decks}
    jobs = plan_head_to_head_games_v4(schedule_rows, pairs, games_per_pair=args.games_per_pair)
    fixed_decks = {spec.name: spec.fixed_deck_id for spec in specs_by_name if spec.fixed_deck_id is not None}
    unknown_decks = sorted(set(fixed_decks.values()).difference(decks))
    if unknown_decks:
        raise ValueError(f"pinned model decks are not in the deck pool: {unknown_decks}")
    if any(first in fixed_decks and second in fixed_decks for first, second in pairs):
        raise ValueError("a pair cannot pin the decks of both models")
    force_tick = args.force_tick if args.forced_opening == "deck-rule" else None

    device = torch.device(args.device)
    torch.manual_seed(args.torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.torch_seed)
        torch.cuda.set_device(device)
    torch.set_num_threads(8)
    models: dict[str, UniversalCardPolicyV4] = {}
    identities: dict[str, dict[str, Any]] = {}
    behaviors: dict[str, str] = {}
    for spec in specs_by_name:
        if spec.name not in used:
            continue
        model, payload = build_model_from_checkpoint_v4(spec.path, device)
        apply_gate_logit_offset_v4(model, spec.gate_logit_offset)
        # Before any collection: the collector's CUDA Graphs bake it in.
        model.set_delay_policy(spec.delay_policy)
        mode = spec.mode or args.mode
        configure_decision_mode_v4(model, mode, keep_checkpoint_temperatures=args.keep_checkpoint_temperatures)
        models[spec.name] = model
        behaviors[spec.name] = behavior_key_v4(
            str(payload["checkpoint_id"]), spec.gate_logit_offset, mode, model, fixed_deck_id=spec.fixed_deck_id
        )
        identities[spec.name] = {
            "path": str(spec.path),
            "sha256": _sha256(spec.path),
            "checkpoint_id": str(payload["checkpoint_id"]),
            "update_step": int(payload["update_step"]),
            "training_stage": str(payload.get("training_stage")),
            # Recorded, not enforced: IL checkpoints store the ACT class weight in extra.
            "checkpoint_gate_act_weight": dict(payload.get("extra") or {}).get("gate_act_weight"),
            "gate_logit_offset": spec.gate_logit_offset,
            "mode": mode,
            "rollout_sampling": model.ppo_rollout_sampling,
            "temperatures": {
                "gate": model.ppo_gate_temperature,
                "action": model.ppo_action_temperature,
                "continue": model.ppo_continue_temperature,
            },
            "behavior": behaviors[spec.name],
        }
        if spec.delay_policy != DELAY_POLICY_SAMPLE:
            # Only non-default policies enter the run key (old runs still resume).
            identities[spec.name]["delay_policy"] = spec.delay_policy
        if spec.fixed_deck_id is not None:
            identities[spec.name]["fixed_deck_id"] = spec.fixed_deck_id
    model_paths = {spec.name: str(spec.path) for spec in specs_by_name}
    run_key = {
        "models": identities,
        "pairs": [list(pair) for pair in pairs],
        "games_per_pair": args.games_per_pair,
        "schedule_sha256": _sha256(args.schedule),
        "deck_pool_sha256": _sha256(args.deck_pool),
        "forced_opening": args.forced_opening,
        "force_tick": force_tick,
        "warmup_ticks": args.warmup_ticks,
        "role_assignment": "config_id even -> model A current; odd -> model B current",
    }
    identity = {
        "schema": "v4-head-to-head-identity.v1",
        "run_key": run_key,
        "mode": args.mode,
        "keep_checkpoint_temperatures": args.keep_checkpoint_temperatures,
        "lanes_per_engine": args.lanes_per_engine,
        "engine_indices": list(engine_indices),
        "segment_decision_steps": args.segment_decision_steps,
        "torch_seed": args.torch_seed,
        "skip_shaping_telemetry": True,
        "topology": str(args.topology),
        "schedule": str(args.schedule),
        "deck_pool": str(args.deck_pool),
        "planned_games": len(jobs),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    games_path = args.output_dir / "games.jsonl"
    identity_path = args.output_dir / "identity.json"
    waves_path = args.output_dir / "waves.jsonl"
    completed_rows = _load_completed(games_path)
    if completed_rows and not args.resume:
        raise ValueError(f"{games_path} already holds games; pass --resume or use a new output directory")
    if args.resume and identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous.get("run_key") != json.loads(json.dumps(run_key)):
            raise ValueError("resume identity differs from the existing run (models, pairs, schedule or rules)")
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "head_to_head_identity", **identity}, sort_keys=True), flush=True)

    done = {(str(row["pair"]), int(row["schedule_row_id"])) for row in completed_rows}
    pending = [job for job in jobs if (job.pair, int(job.schedule_row["schedule_row_id"])) not in done]
    waves = plan_waves_v4(pending, capacity=capacity, model_order=names)
    environment = ppo_environment_config_v4(gamma_per_decision=GAMMA_PER_DECISION, shaping_beta=SHAPING_BETA)
    max_decision_steps = math.ceil(environment.max_battle_ticks / PPO_DECISION_TICKS) + 4
    wave_offset = 1 + max((int(row["wave"]) for row in completed_rows), default=-1)
    run_started = time.perf_counter()
    run_games = 0
    wave_lines: list[dict[str, Any]] = []
    for wave_number, (current_name, wave_jobs) in enumerate(waves):
        if args.max_waves is not None and wave_number >= args.max_waves:
            break
        wave_index = wave_offset + wave_number
        specs = engine_specs_for_wave_v4(
            topology,
            engine_indices,
            games=len(wave_jobs),
            lanes_per_engine=args.lanes_per_engine,
            force_tick=force_tick,
            warmup_ticks=args.warmup_ticks,
        )
        assignments = [assignment_for_job_v4(job, decks, model_paths, fixed_decks) for job in wave_jobs]
        reports, accumulator, elapsed = run_wave_v4(
            collect_async_cluster_wave_v4,
            specs=specs,
            jobs=wave_jobs,
            assignments=assignments,
            models=models,
            current_model=current_name,
            policy_state_id=f"h2h{wave_index:05d}",
            segment_decision_steps=args.segment_decision_steps,
            max_decision_steps=max_decision_steps,
        )
        rows = [
            game_row_from_report_v4(job, report, wave_index=wave_index, behaviors=behaviors, fixed_decks=fixed_decks)
            for job, report in zip(wave_jobs, reports, strict=True)
        ]
        with games_path.open("a", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        completed_rows.extend(rows)
        run_games += len(rows)
        line = {
            "event": "head_to_head_wave",
            "wave": wave_index,
            "current_model": current_name,
            "games": len(rows),
            "engines": len(specs),
            "segments": accumulator.segments,
            "elapsed_seconds": elapsed,
            "games_per_minute": 60.0 * len(rows) / max(elapsed, 1e-9),
            "hard_failures": sum(int(row["hard_failure"]) for row in rows),
            "collector_timing": accumulator.timing_summary(),
        }
        wave_lines.append(line)
        with waves_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
        print(json.dumps(line, sort_keys=True), flush=True)

    run_elapsed = time.perf_counter() - run_started
    hard_failures = sum(int(bool(row["hard_failure"])) for row in completed_rows)
    partial = args.max_waves is not None and len(waves) > args.max_waves
    complete = len(completed_rows) == len(jobs)
    summary = {
        "schema": "v4-head-to-head-summary.v1",
        "ok": hard_failures == 0 and (complete or partial),
        "complete": complete,
        "partial_by_max_waves": partial,
        "planned_games": len(jobs),
        "completed_games": len(completed_rows),
        "hard_failures": hard_failures,
        "collector_timing": {
            "this_invocation_games": run_games,
            "this_invocation_elapsed_seconds": run_elapsed,
            "games_per_minute": 60.0 * run_games / max(run_elapsed, 1e-9),
            "waves": wave_lines,
        },
        "pairs": pair_summaries_v4(completed_rows, tiebreak_tick=TIEBREAK_TICK) if completed_rows else {},
        "identity": identity,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "event": "head_to_head_summary",
                "ok": summary["ok"],
                "completed_games": len(completed_rows),
                "planned_games": len(jobs),
                "hard_failures": hard_failures,
                "games_per_minute": summary["collector_timing"]["games_per_minute"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
