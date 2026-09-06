"""Expert timing/window contracts for online RoyaleAPI replay training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ...contracts import ActionKind, ActionV1, TargetKind
from ...perspective import PerspectiveTransformV1
from ..replay_archive import TrainingReplayV1
from .config import ModelConfigV4
from .decoding import ShadowCandidateLegality
from .tensors import (
    ActionSequenceV4,
    CANDIDATE_ABILITY,
    CANDIDATE_DEPLOY,
    GATE_ACT,
    GATE_WAIT,
    TARGET_GRID,
    UniversalSemanticBatchV4,
)


FIRST_POLICY_DECISION_TICK = 90
POLICY_DECISION_TICKS = 5
MAX_EXPERT_MICRO_ACTIONS = 2
NATIVE_TICK_MS = 50


class ExpertActionAlignmentError(ValueError):
    """An expert command cannot be represented by the current legal candidates."""


@dataclass(frozen=True, slots=True)
class TimedExpertActionV4:
    source_tick: int
    action: ActionV1
    source_index: int = 0

    @property
    def owner(self) -> int:
        return self.action.owner


@dataclass(frozen=True, slots=True)
class ExpertTimelineReportV4:
    action_count: int
    active_window_count: int
    overflow_window_count: int
    early_action_count: int
    maximum_actions_in_window: int

    @property
    def accepted(self) -> bool:
        return self.overflow_window_count == 0 and self.early_action_count == 0


def replay_expert_actions(replay: TrainingReplayV1) -> tuple[TimedExpertActionV4, ...]:
    """Extract authoritative source-command ticks from a prepared replay."""

    result: list[TimedExpertActionV4] = []
    for operation in replay.operations:
        for action in operation.actions:
            if action.kind not in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_ABILITY}:
                continue
            raw_tick = action.metadata.get("source_command_tick")
            if type(raw_tick) is not int:
                raise ExpertActionAlignmentError("expert action lacks an integer source_command_tick")
            raw_index = action.metadata.get("source_index", len(result))
            if type(raw_index) is not int:
                raise ExpertActionAlignmentError("expert source_index is not an integer")
            result.append(TimedExpertActionV4(source_tick=raw_tick, action=action, source_index=raw_index))
    return tuple(sorted(result, key=lambda item: (item.source_tick, item.source_index, item.owner)))


def decision_tick_for_action(
    source_tick: int,
    *,
    first_decision_tick: int = FIRST_POLICY_DECISION_TICK,
    decision_ticks: int = POLICY_DECISION_TICKS,
) -> int:
    if decision_ticks <= 0:
        raise ValueError("decision_ticks must be positive")
    if source_tick < first_decision_tick:
        raise ExpertActionAlignmentError(
            f"expert action tick {source_tick} precedes first decision tick {first_decision_tick}"
        )
    return first_decision_tick + ((source_tick - first_decision_tick) // decision_ticks) * decision_ticks


def group_expert_action_windows(
    actions: Sequence[TimedExpertActionV4],
    *,
    first_decision_tick: int = FIRST_POLICY_DECISION_TICK,
    decision_ticks: int = POLICY_DECISION_TICKS,
) -> dict[tuple[int, int], tuple[TimedExpertActionV4, ...]]:
    grouped: dict[tuple[int, int], list[TimedExpertActionV4]] = {}
    for item in actions:
        decision_tick = decision_tick_for_action(
            item.source_tick, first_decision_tick=first_decision_tick, decision_ticks=decision_ticks
        )
        grouped.setdefault((decision_tick, item.owner), []).append(item)
    return {
        key: tuple(sorted(value, key=lambda item: (item.source_tick, item.source_index)))
        for key, value in grouped.items()
    }


def screen_expert_timeline(
    actions: Sequence[TimedExpertActionV4],
    *,
    first_decision_tick: int = FIRST_POLICY_DECISION_TICK,
    decision_ticks: int = POLICY_DECISION_TICKS,
    max_micro_actions: int = MAX_EXPERT_MICRO_ACTIONS,
) -> ExpertTimelineReportV4:
    """Count overflow/early failures before any engine slot is consumed."""

    early = sum(item.source_tick < first_decision_tick for item in actions)
    eligible = tuple(item for item in actions if item.source_tick >= first_decision_tick)
    grouped = group_expert_action_windows(
        eligible, first_decision_tick=first_decision_tick, decision_ticks=decision_ticks
    )
    counts = tuple(len(value) for value in grouped.values())
    return ExpertTimelineReportV4(
        action_count=len(actions),
        active_window_count=len(grouped),
        overflow_window_count=sum(value > max_micro_actions for value in counts),
        early_action_count=early,
        maximum_actions_in_window=max(counts, default=0),
    )


def require_accepted_timeline(report: ExpertTimelineReportV4, *, replay_id: str = "") -> None:
    if report.overflow_window_count:
        label = f" {replay_id}" if replay_id else ""
        raise ExpertActionAlignmentError(
            f"replay{label} has {report.overflow_window_count} decision windows with more than two expert actions"
        )
    if report.early_action_count:
        label = f" {replay_id}" if replay_id else ""
        raise ExpertActionAlignmentError(
            f"replay{label} has {report.early_action_count} actions before the first policy decision"
        )


def _unique_candidate(matches: torch.Tensor, *, label: str, context: str = "") -> int:
    rows = matches.nonzero(as_tuple=False).flatten()
    if rows.numel() != 1:
        suffix = f"; {context}" if context else ""
        raise ExpertActionAlignmentError(f"{label} matched {int(rows.numel())} legal candidates; expected one{suffix}")
    return int(rows[0].item())


def build_expert_action_batch(
    batch: UniversalSemanticBatchV4,
    windows: Sequence[Sequence[TimedExpertActionV4]],
    *,
    decision_ticks_by_row: Sequence[int],
    perspectives: Sequence[PerspectiveTransformV1],
    config: ModelConfigV4 | None = None,
    validate: bool = True,
) -> ActionSequenceV4:
    """Align one five-tick expert window per batch row to candidate UIDs."""

    actual = config or ModelConfigV4()
    batch_size = batch.batch_size
    if not (len(windows) == len(decision_ticks_by_row) == len(perspectives) == batch_size):
        raise ValueError("expert windows, ticks, perspectives, and batch rows differ")
    max_micro = actual.max_micro_actions
    gate = torch.full((batch_size,), GATE_WAIT, dtype=torch.long)
    count = torch.zeros(batch_size, dtype=torch.long)
    candidate_index = torch.full((batch_size, max_micro), -1, dtype=torch.long)
    candidate_uid = torch.full_like(candidate_index, -1)
    target_cell = torch.full_like(candidate_index, -1)
    delay_offset_bin = torch.full_like(candidate_index, -1)

    candidates = batch.candidates
    for row, raw_window in enumerate(windows):
        window = tuple(sorted(raw_window, key=lambda item: (item.source_tick, item.source_index)))
        if len(window) > max_micro:
            raise ExpertActionAlignmentError(f"row {row} has {len(window)} expert actions in one decision window")
        if not window:
            continue
        decision_tick = int(decision_ticks_by_row[row])
        perspective = perspectives[row]
        gate[row] = GATE_ACT
        count[row] = len(window)
        for step, timed in enumerate(window):
            action = timed.action
            if action.owner != perspective.actor_owner:
                raise ExpertActionAlignmentError("expert owner and perspective differ")
            delay_ticks = timed.source_tick - decision_tick
            if not 0 <= delay_ticks < POLICY_DECISION_TICKS:
                raise ExpertActionAlignmentError("expert action is outside its five-tick decision window")
            try:
                delay_bin = actual.delay_offset_ms.index(delay_ticks * NATIVE_TICK_MS)
            except ValueError as error:
                raise ExpertActionAlignmentError("expert delay is absent from the V4 delay vocabulary") from error

            legal = candidates.mask[row]
            if action.kind is ActionKind.PLAY_CARD:
                if action.hand_slot is None or action.card_id is None:
                    raise ExpertActionAlignmentError("expert card play lacks native hand/card identity")
                matches = (
                    legal
                    & (candidates.variant[row] == CANDIDATE_DEPLOY)
                    & (candidates.native_hand_slot[row] == action.hand_slot)
                    & (candidates.native_visible_card_id[row] == action.card_id)
                )
                deploy_rows = (legal & (candidates.variant[row] == CANDIDATE_DEPLOY)).nonzero(as_tuple=False).flatten()
                available = [
                    (int(candidates.native_hand_slot[row, index]), int(candidates.native_visible_card_id[row, index]))
                    for index in deploy_rows.tolist()
                ]
                selected = _unique_candidate(
                    matches,
                    label="expert card play",
                    context=(
                        f"decision_tick={decision_tick} source_tick={timed.source_tick} "
                        f"owner={action.owner} wanted=(slot={action.hand_slot},"
                        f"card={action.card_id}) legal_deploy={available}"
                    ),
                )
                model_action = perspective.action_to_model(action)
                if model_action.target_kind is not TargetKind.GRID or model_action.target_grid is None:
                    raise ExpertActionAlignmentError("expert card play has no GRID target")
                x, y = map(int, model_action.target_grid)
                if bool(candidates.is_building[row, selected].item()):
                    anchor_x = float(candidates.building_offset_x[row, selected].item())
                    anchor_y = float(candidates.building_offset_y[row, selected].item())
                    source_offset = model_action.subcell_offset or (anchor_x, anchor_y)
                    canonical_x = x + float(source_offset[0]) - anchor_x
                    canonical_y = y + float(source_offset[1]) - anchor_y
                    rounded_x = round(canonical_x)
                    rounded_y = round(canonical_y)
                    if abs(canonical_x - rounded_x) > 1e-6 or abs(canonical_y - rounded_y) > 1e-6:
                        raise ExpertActionAlignmentError(
                            "expert building target cannot be represented by its native placement anchor"
                        )
                    x, y = rounded_x, rounded_y
                if not 0 <= x < actual.board_width or not 0 <= y < actual.board_height:
                    raise ExpertActionAlignmentError("expert GRID target is out of bounds")
                selected_target = y * actual.board_width + x
            elif action.kind is ActionKind.ACTIVATE_ABILITY:
                matches = legal & (candidates.variant[row] == CANDIDATE_ABILITY)
                inferred = action.metadata.get("source_entity_inferred_at_render") is True
                if action.source_entity is not None and action.source_entity > 0 and not inferred:
                    matches &= candidates.native_source_entity[row] == action.source_entity
                ability_rows = (
                    (legal & (candidates.variant[row] == CANDIDATE_ABILITY)).nonzero(as_tuple=False).flatten()
                )
                available = [
                    (
                        int(candidates.native_source_entity[row, index]),
                        int(candidates.native_visible_card_id[row, index]),
                        int(candidates.ability_vocab_id[row, index]),
                    )
                    for index in ability_rows.tolist()
                ]
                context = (
                    f"decision_tick={decision_tick} source_tick={timed.source_tick} "
                    f"owner={action.owner} wanted_source={action.source_entity} "
                    f"source_inferred={inferred} legal_abilities={available}"
                )
                if inferred:
                    rows = matches.nonzero(as_tuple=False).flatten()
                    if rows.numel() == 0:
                        raise ExpertActionAlignmentError(f"expert ability matched no legal candidates; {context}")
                    source_ids = candidates.native_source_entity[row, rows]
                    newest_source = int(source_ids.max().item())
                    selected = _unique_candidate(
                        matches & (candidates.native_source_entity[row] == newest_source),
                        label="newest legal expert ability",
                        context=context,
                    )
                else:
                    selected = _unique_candidate(matches, label="expert ability", context=context)
                selected_target = -1
            else:
                raise ExpertActionAlignmentError(f"unsupported expert action kind {action.kind.value}")
            candidate_index[row, step] = selected
            candidate_uid[row, step] = candidates.uid[row, selected].cpu()
            target_cell[row, step] = selected_target
            delay_offset_bin[row, step] = delay_bin

    sequence = ActionSequenceV4(
        gate=gate,
        micro_action_count=count,
        candidate_index=candidate_index,
        candidate_uid=candidate_uid,
        target_cell=target_cell,
        delay_offset_bin=delay_offset_bin,
    )
    if validate:
        sequence.validate(actual, candidate_count=int(candidates.mask.shape[1]))

    if not torch.any(count):
        return sequence

    # Validate compound-action shadow legality before the sequence reaches the
    # loss.  This catches duplicate cards, exclusion groups, elixir, and delay
    # ordering at the producer boundary.
    shadow = ShadowCandidateLegality(candidates, actual)
    for step in range(max_micro):
        active = count > step
        selected = candidate_index[:, step].clamp_min(0).to(candidates.mask.device)
        legal = shadow.candidate_mask()
        if torch.any(active.to(legal.device) & ~legal.gather(1, selected[:, None])[:, 0]):
            raise ExpertActionAlignmentError("expert compound action is illegal under V4 shadow state")
        selected_mode = candidates.target_mode.to(legal.device).gather(1, selected[:, None])[:, 0]
        grid = active.to(legal.device) & (selected_mode == TARGET_GRID)
        selected_target = target_cell[:, step].to(legal.device)
        safe_target = selected_target.clamp_min(0)
        placement = shadow.selected_placement(selected)
        target_legal = placement.flatten(1).gather(1, safe_target[:, None])[:, 0]
        if torch.any(grid & ~target_legal):
            raise ExpertActionAlignmentError("expert GRID target is illegal under V4 shadow placement")
        shadow.apply(
            active.to(legal.device),
            selected,
            selected_target,
            delay_offset_bin[:, step].clamp_min(0).to(legal.device),
            step=step,
        )
    return sequence
