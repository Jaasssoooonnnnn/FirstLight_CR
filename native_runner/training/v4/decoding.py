"""V4 shadow legality, timing, and fail-closed native action decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor

from ...contracts import ActionKind, ActionV1, ObservationV1, TargetKind
from ...perspective import PerspectiveTransformV1
from .native_actions import (
    DecodedActionSequenceV4,
    MIRROR_CARD_ID,
    NATIVE_TICK_MS,
    ability_runtime_contract,
    mirror_play_runtime_contract,
)
from .catalog import CardCatalogV1
from .config import MATCH_SCALAR_NAMES, ModelConfigV4
from .tensors import (
    ActionCandidatesV4,
    ActionSequenceV4,
    CANDIDATE_ABILITY,
    CANDIDATE_DEPLOY,
    GATE_ACT,
    GATE_WAIT,
    TARGET_GRID,
    TARGET_NONE,
    target_xy_from_cell,
)


def candidate_uid_v4(
    *,
    variant: int,
    visible_card_vocab_id: int,
    effective_card_vocab_id: int,
    native_hand_slot: int = -1,
    native_source_entity: int = -1,
    ability_vocab_id: int = 0,
) -> int:
    """Return a stable, non-sentinel signed-int64 candidate identity.

    The UID is only an address used to replay a sampled candidate after rows
    have been reordered.  It is deliberately not a learnable model feature.
    """

    fields = {
        "ability_vocab_id": int(ability_vocab_id),
        "effective_card_vocab_id": int(effective_card_vocab_id),
        "native_hand_slot": int(native_hand_slot),
        "native_source_entity": int(native_source_entity),
        "variant": int(variant),
        "visible_card_vocab_id": int(visible_card_vocab_id),
    }
    payload = json.dumps(fields, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    # Keep the value positive so -1 remains an unambiguous padding sentinel.
    return int.from_bytes(digest, "big", signed=False) & ((1 << 63) - 1)


# Delay-aware legality (C1 fix): a candidate is legal when its cost is
# affordable at some allowed delay bin of the decision window, i.e. at the
# touch tick d + k (k = offset_ms / 50 native ticks) rather than at the
# decision tick d:
#
#     e(k) = min(10, e(0) + regen_per_tick * k) - spent
#
# where e(0) is ``candidates.elixir`` (BattleEnvV1's ``effective_elixir``:
# raw elixir minus elixir reserved by in-flight own commands, clamped at 0)
# and ``spent`` is the cost of the earlier micro-actions of the same decision
# (touched commands are reserved, not deducted, until the engine executes them
# 20 ticks later, so the raw bar keeps capping at 10).  BattleEnvV1's action
# mask (``affordable_from_bin``) and ``_validate_compound_actions`` apply the
# same rule, so a sampled compound action always passes the environment.
#
# The switch is ``ModelConfigV4.delay_aware_legality`` (default True); this
# module constant only mirrors that default.  ``ShadowCandidateLegality`` reads
# the flag from its ``config`` unless ``delay_aware=`` overrides it.  Pre-fix
# checkpoints load with the decision-tick rule (``checkpoint.LEGACY_FIELD_DEFAULTS``).
DELAY_AWARE_LEGALITY_DEFAULT: bool = ModelConfigV4().delay_aware_legality
ELIXIR_CAP = 10.0
ELIXIR_MULTIPLIER_SCALAR_INDEX = MATCH_SCALAR_NAMES.index("elixir_multiplier_over_3")
# Full 10-elixir bar duration per match multiplier.  Mirrors
# ``battle_env.ELIXIR_SEGMENTS`` (1x 28 s, 2x 14 s, 3x 9.3 s): one elixir per
# 2.8 s = 56 ticks at 1x.  Kept local so this module never imports the engine.
ELIXIR_FULL_BAR_MS_BY_MULTIPLIER: tuple[tuple[int, int], ...] = ((1, 28_000), (2, 14_000), (3, 9_300))
_AFFORDABLE_TOLERANCE = 1e-6

# Inference delay-bin policy (``UniversalCardPolicyV4.set_delay_policy``).
# ``sample`` uses the delay head (sample or argmax, as before); ``earliest``
# ignores the head and plays the first bin of the legal delay mask, i.e. the
# first bin at which the chosen candidate is affordable under the shadow's
# sequential, order-preserving legality (and the inference cap).  Legacy
# checkpoints (``delay_aware_legality=False``) therefore always get bin 0.
DELAY_POLICY_SAMPLE = "sample"
DELAY_POLICY_EARLIEST = "earliest"
DELAY_POLICIES: tuple[str, ...] = (DELAY_POLICY_SAMPLE, DELAY_POLICY_EARLIEST)


def validate_delay_policy(policy: object) -> str:
    if not isinstance(policy, str) or policy not in DELAY_POLICIES:
        raise ValueError(f"delay policy must be one of {DELAY_POLICIES}, got {policy!r}")
    return policy


def earliest_delay_offset_bin(delay_offset_mask: Tensor) -> Tensor:
    """First legal bin ``[B]`` of a legal delay mask ``[B,K]``.

    An empty row maps to bin 0, the never-observed fallback the delay
    categorical uses for such rows.  Shape-static (CUDA Graph safe).
    """

    if delay_offset_mask.dtype != torch.bool or delay_offset_mask.ndim != 2 or delay_offset_mask.shape[1] <= 0:
        raise ValueError("delay offset mask must be bool [B,K]")
    # Number of leading illegal bins == index of the first legal bin; a row
    # with no legal bin counts K, which ``where`` folds back to bin 0.
    leading = delay_offset_mask.long().cumsum(dim=-1).eq(0).sum(dim=-1)
    return torch.where(leading >= delay_offset_mask.shape[1], torch.zeros_like(leading), leading)


def elixir_multiplier_from_match_scalars(match_scalars: Tensor) -> Tensor:
    """Return the per-row match elixir multiplier (1, 2 or 3) as float32 [B]."""

    if match_scalars.ndim != 2 or match_scalars.shape[1] <= ELIXIR_MULTIPLIER_SCALAR_INDEX:
        raise ValueError("match_scalars must have shape [B,F] with the elixir multiplier column")
    return match_scalars[:, ELIXIR_MULTIPLIER_SCALAR_INDEX].to(torch.float32) * 3.0


def elixir_regen_per_tick(multiplier: Tensor) -> Tensor:
    """Native elixir regeneration per 50 ms tick for each row's multiplier.

    Stored match scalars are fp16 (``2/3`` becomes ``0.66650``), so known
    multipliers are snapped to the native segment table; any other positive
    value falls back to the linear ``multiplier / 56`` model.
    """

    value = multiplier.to(torch.float32)
    rate = value.clamp_min(0.0) * float(10.0 * NATIVE_TICK_MS / ELIXIR_FULL_BAR_MS_BY_MULTIPLIER[0][1])
    for level, full_bar_ms in ELIXIR_FULL_BAR_MS_BY_MULTIPLIER:
        rate = torch.where((value - float(level)).abs() < 0.25, float(10.0 * NATIVE_TICK_MS / full_bar_ms), rate)
    return rate


class ShadowCandidateLegality:
    """Open-loop legality state for at most two candidate micro-actions.

    With ``delay_aware`` (default ``config.delay_aware_legality``) a candidate
    is legal when ``cost <= elixir_at(k)`` for some delay bin ``k`` that is
    allowed (``allowed_delay_bins``, e.g. the model's inference delay cap) and
    not earlier than the previous micro-action's bin, where::

        elixir_at(k) = max(e0, min(10, e0 + regen_per_tick * ticks(k))) - spent

    ``e0`` is ``candidates.elixir`` (the environment's effective elixir at the
    decision tick) and ``spent`` the cost of the earlier micro-actions; touched
    commands are reserved rather than deducted, so the raw bar keeps capping at
    10 while they are in flight.  This is exactly BattleEnvV1's sequential
    compound check.  Without regeneration (``delay_aware=False``) it reduces
    bit-exactly to the historical ``cost <= elixir_at_d - spent`` rule.

    ``elixir_multiplier`` is the per-row match multiplier (1/2/3); it defaults
    to 1x (conservative).  ``remaining_elixir`` keeps its historical meaning
    (elixir at ``d`` minus spent cost) because it is a model input via
    ``shadow_features``.
    """

    def __init__(
        self,
        candidates: ActionCandidatesV4,
        config: ModelConfigV4,
        *,
        elixir_multiplier: Tensor | float | None = None,
        delay_aware: bool | None = None,
        allowed_delay_bins: Tensor | None = None,
    ) -> None:
        self.base = candidates
        self.config = config
        batch, count = candidates.mask.shape
        bins = len(config.delay_offset_ms)
        self.delay_aware = bool(config.delay_aware_legality) if delay_aware is None else bool(delay_aware)
        self.remaining_elixir = candidates.elixir.clone()
        if allowed_delay_bins is not None:
            shape = tuple(allowed_delay_bins.shape)
            if allowed_delay_bins.dtype != torch.bool or shape not in ((bins,), (batch, bins)):
                raise ValueError("allowed_delay_bins must be bool [K] or [B,K]")
            allowed = allowed_delay_bins.to(candidates.mask.device)
            self.allowed_delay_bins: Tensor | None = allowed[None].expand(batch, -1) if allowed.ndim == 1 else allowed
        else:
            self.allowed_delay_bins = None
        if self.delay_aware:
            if elixir_multiplier is None:
                multiplier = candidates.elixir.new_ones(batch, dtype=torch.float32)
            elif isinstance(elixir_multiplier, Tensor):
                multiplier = elixir_multiplier.to(device=candidates.elixir.device, dtype=torch.float32)
                if tuple(multiplier.shape) != (batch,):
                    raise ValueError("elixir_multiplier must have shape [B]")
            else:
                multiplier = candidates.elixir.new_full((batch,), float(elixir_multiplier), dtype=torch.float32)
            self.regen_per_tick: Tensor | None = elixir_regen_per_tick(multiplier).to(candidates.elixir.dtype)
            # ``ModelConfigV4`` freezes the offsets at 0..200 ms in 50 ms steps,
            # so bin k is exactly k native ticks.  ``arange`` keeps this free of
            # host-to-device copies (safe inside the live CUDA Graph capture).
            if any(int(value) != index * NATIVE_TICK_MS for index, value in enumerate(config.delay_offset_ms)):
                raise ValueError("delay-aware legality requires delay offsets of k * 50 ms")
            self.delay_ticks: Tensor | None = torch.arange(
                bins, dtype=candidates.elixir.dtype, device=candidates.elixir.device
            )
            # Cost of the earlier micro-actions of this decision (reserved).
            self.spent_elixir: Tensor | None = torch.zeros_like(candidates.elixir)
        else:
            self.regen_per_tick = None
            self.delay_ticks = None
            self.spent_elixir = None
        self.used_candidate = torch.zeros_like(candidates.mask)
        self.used_ability = torch.zeros(batch, dtype=torch.bool, device=candidates.mask.device)
        self.used_hand_row = torch.zeros(batch, config.max_own_cards, dtype=torch.bool, device=candidates.mask.device)
        self.used_exclusion = torch.full(
            (batch, config.max_micro_actions),
            -1,
            dtype=candidates.exclusion_group_id.dtype,
            device=candidates.mask.device,
        )
        self.planned_x = candidates.cost.new_zeros(batch, config.max_micro_actions)
        self.planned_y = candidates.cost.new_zeros(batch, config.max_micro_actions)
        self.planned_half_width = candidates.cost.new_zeros(batch, config.max_micro_actions)
        self.planned_half_height = candidates.cost.new_zeros(batch, config.max_micro_actions)
        self.planned_count = torch.zeros(batch, dtype=torch.long, device=candidates.mask.device)
        self.selected_count = torch.zeros(batch, dtype=torch.long, device=candidates.mask.device)
        self.previous_delay_offset_ms = candidates.cost.new_zeros(batch)
        self.previous_delay_offset_bin = torch.zeros(batch, dtype=torch.long, device=candidates.mask.device)
        self._candidate_rows = torch.arange(count, device=candidates.mask.device)
        self._hand_rows = torch.arange(config.max_own_cards, device=candidates.mask.device)

    def placement(self) -> Tensor:
        result = self.base.placement.clone()
        config = self.config
        x = torch.arange(config.board_width, device=result.device, dtype=self.planned_x.dtype).view(1, 1, 1, -1)
        y = torch.arange(config.board_height, device=result.device, dtype=self.planned_y.dtype).view(1, 1, -1, 1)
        candidate_x = x + 0.5 + self.base.building_offset_x[:, :, None, None]
        candidate_y = y + 0.5 + self.base.building_offset_y[:, :, None, None]
        for planned_index in range(config.max_micro_actions):
            active = self.planned_count > planned_index
            x_overlap = (candidate_x - self.planned_x[:, planned_index, None, None, None]).abs() < (
                self.base.building_half_width[:, :, None, None]
                + self.planned_half_width[:, planned_index, None, None, None]
            )
            y_overlap = (candidate_y - self.planned_y[:, planned_index, None, None, None]).abs() < (
                self.base.building_half_height[:, :, None, None]
                + self.planned_half_height[:, planned_index, None, None, None]
            )
            blocked = x_overlap & y_overlap & active[:, None, None, None] & self.base.is_building[:, :, None, None]
            result &= ~blocked
        return result

    def elixir_at_bins(self) -> Tensor:
        """Elixir available at each delay bin ``[B,K]`` for the next micro-action.

        ``max(e0, min(10, e0 + regen * k)) - spent``: regeneration of the raw
        bar stops at the cap (never lowering an already-higher value), and the
        earlier micro-actions' costs stay reserved.  The ordering constraint in
        ``delay_offset_mask`` excludes bins before the previous micro-action.
        """

        bins = len(self.config.delay_offset_ms)
        if self.regen_per_tick is None or self.delay_ticks is None or self.spent_elixir is None:
            return self.remaining_elixir[:, None].expand(-1, bins)
        start = self.base.elixir[:, None]
        accrued = start + self.regen_per_tick[:, None] * self.delay_ticks[None, :]
        capped = torch.maximum(accrued.clamp_max(ELIXIR_CAP), start.expand(-1, bins))
        return capped - self.spent_elixir[:, None]

    def affordable_at_bins(self) -> Tensor:
        """Whether each candidate's cost is affordable at each delay bin ``[B,C,K]``."""

        return self.base.cost[:, :, None] <= self.elixir_at_bins()[:, None, :] + _AFFORDABLE_TOLERANCE

    def _ordering_mask(self, *, step: int) -> Tensor:
        bins = torch.arange(len(self.config.delay_offset_ms), device=self.previous_delay_offset_bin.device)
        if step == 0:
            return torch.ones(
                self.previous_delay_offset_bin.shape[0], bins.shape[0], dtype=torch.bool, device=bins.device
            )
        return bins[None] >= self.previous_delay_offset_bin[:, None]

    def _reachable_bins(self) -> Tensor:
        """Bins ``[B,K]`` a next micro-action may use: not before the previous
        micro-action (``previous_delay_offset_bin`` is 0 before the first) and
        allowed by ``allowed_delay_bins``."""

        bins = torch.arange(len(self.config.delay_offset_ms), device=self.previous_delay_offset_bin.device)
        reachable = bins[None] >= self.previous_delay_offset_bin[:, None]
        if self.allowed_delay_bins is not None:
            reachable = reachable & self.allowed_delay_bins
        return reachable

    def candidate_mask(self) -> Tensor:
        if self.delay_aware:
            # Legal when affordable at some reachable bin.  A candidate whose
            # only affordable bins are excluded by the inference delay cap is
            # illegal here rather than relying on the delay head's fallback.
            affordable = (self.affordable_at_bins() & self._reachable_bins()[:, None, :]).any(dim=-1)
        else:
            affordable = self.base.cost <= self.remaining_elixir[:, None] + _AFFORDABLE_TOLERANCE
        result = self.base.mask & ~self.used_candidate & affordable
        result &= ~((self.selected_count[:, None] > 0) & (self.base.native_visible_card_id == MIRROR_CARD_ID))
        result &= ~(self.used_ability[:, None] & (self.base.variant == CANDIDATE_ABILITY))
        deploy = self.base.variant == CANDIDATE_DEPLOY
        has_hand = self.base.own_card_row >= 0
        safe_hand = self.base.own_card_row.clamp(0, self.config.max_own_cards - 1)
        hand_used = self.used_hand_row.gather(1, safe_hand)
        result &= ~(deploy & has_hand & hand_used)
        for index in range(self.config.max_micro_actions):
            used = self.used_exclusion[:, index]
            result &= ~((used[:, None] >= 0) & (self.base.exclusion_group_id == used[:, None]))
        has_grid_target = self.placement().flatten(2).any(dim=-1)
        result &= (self.base.target_mode != TARGET_GRID) | has_grid_target
        return result

    def selected_placement(self, candidate_index: Tensor) -> Tensor:
        placement = self.placement()
        return placement.gather(
            1, candidate_index[:, None, None, None].expand(-1, 1, self.config.board_height, self.config.board_width)
        )[:, 0]

    def delay_offset_mask(self, *, step: int, candidate_index: Tensor | None = None) -> Tensor:
        """Legal delay bins ``[B,K]``: order-preserving and, when delay-aware
        and ``candidate_index`` is given, affordable for that candidate and
        allowed by ``allowed_delay_bins``.

        A row whose selected candidate is affordable at no such bin (an
        inactive row, or an illegal candidate rejected by ``candidate_mask``)
        falls back to the ordering mask so the distribution stays defined.
        The legacy rule (``delay_aware=False``) returns the ordering mask
        unchanged; callers apply their own inference cap on top.
        """

        ordering = self._ordering_mask(step=step)
        if not self.delay_aware or candidate_index is None:
            return ordering
        selected = candidate_index.clamp(0, self.base.mask.shape[1] - 1)
        bins = ordering.shape[1]
        affordable = self.affordable_at_bins().gather(1, selected[:, None, None].expand(-1, 1, bins))[:, 0]
        allowed = ordering & affordable
        if self.allowed_delay_bins is not None:
            allowed = allowed & self.allowed_delay_bins
        return torch.where(allowed.any(dim=-1, keepdim=True), allowed, ordering)

    def shadow_features(self, *, step: int) -> Tensor:
        output = self.remaining_elixir.new_zeros(self.remaining_elixir.shape[0], self.config.shadow_feature_dim)
        legal = self.candidate_mask()
        output[:, 0] = self.remaining_elixir / 10.0
        output[:, 1] = float(step) / float(self.config.max_micro_actions)
        output[:, 2] = self.used_candidate.float().mean(dim=-1)
        output[:, 3] = legal.float().mean(dim=-1)
        output[:, 4] = self.planned_count.to(output.dtype) / float(self.config.max_micro_actions)
        output[:, 5] = self.previous_delay_offset_ms / 200.0
        output[:, 6] = self.used_hand_row.float().mean(dim=-1)
        output[:, 7] = legal.any(dim=-1).to(output.dtype)
        return output

    def apply(
        self, rows: Tensor, candidate_index: Tensor, target_cell: Tensor, delay_offset_bin: Tensor, *, step: int
    ) -> None:
        if not (rows.is_cuda and torch.cuda.is_current_stream_capturing()):
            # Preserve the established sparse eager/training path. The fixed
            # masked implementation below is selected only while recording the
            # dedicated batch-1 live CUDA Graph.
            indices = rows.nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                return
            chosen = candidate_index[indices]
            if self.spent_elixir is not None:
                self.spent_elixir[indices] += self.base.cost[indices, chosen]
            self.remaining_elixir[indices] -= self.base.cost[indices, chosen]
            self.used_candidate[indices, chosen] = True
            self.selected_count[indices] += 1
            variant = self.base.variant[indices, chosen]
            self.used_ability[indices] |= variant == CANDIDATE_ABILITY
            own_row = self.base.own_card_row[indices, chosen]
            use_hand = (variant == CANDIDATE_DEPLOY) & (own_row >= 0)
            self.used_hand_row[indices[use_hand], own_row[use_hand]] = True
            exclusion = self.base.exclusion_group_id[indices, chosen]
            self.used_exclusion[indices, step] = exclusion
            offset = torch.tensor(
                self.config.delay_offset_ms,
                dtype=self.previous_delay_offset_ms.dtype,
                device=self.previous_delay_offset_ms.device,
            )[delay_offset_bin[indices]]
            self.previous_delay_offset_ms[indices] = offset
            self.previous_delay_offset_bin[indices] = delay_offset_bin[indices]

            building = self.base.is_building[indices, chosen] & (target_cell[indices] >= 0)
            building_rows = indices[building]
            if building_rows.numel() == 0:
                return
            building_candidates = chosen[building]
            x, y = target_xy_from_cell(
                target_cell[building_rows], width=self.config.board_width, height=self.config.board_height
            )
            planned_index = self.planned_count[building_rows]
            self.planned_x[building_rows, planned_index] = (
                x.to(self.planned_x.dtype) + 0.5 + self.base.building_offset_x[building_rows, building_candidates]
            )
            self.planned_y[building_rows, planned_index] = (
                y.to(self.planned_y.dtype) + 0.5 + self.base.building_offset_y[building_rows, building_candidates]
            )
            self.planned_half_width[building_rows, planned_index] = self.base.building_half_width[
                building_rows, building_candidates
            ].to(self.planned_half_width.dtype)
            self.planned_half_height[building_rows, planned_index] = self.base.building_half_height[
                building_rows, building_candidates
            ].to(self.planned_half_height.dtype)
            self.planned_count[building_rows] += 1
            return

        # All live tensors have fixed shapes. Keep inactive rows as masked
        # no-ops instead of materializing data-dependent ``nonzero`` indices;
        # this preserves the shadow state exactly and permits CUDA Graph
        # capture for batch-1 deterministic inference.
        chosen = candidate_index.clamp(0, self.base.mask.shape[1] - 1)
        chosen_column = chosen[:, None]
        selected_cost = self.base.cost.gather(1, chosen_column)[:, 0]
        if self.spent_elixir is not None:
            self.spent_elixir = self.spent_elixir + selected_cost * rows.to(selected_cost.dtype)
        self.remaining_elixir -= selected_cost * rows.to(selected_cost.dtype)
        self.used_candidate |= rows[:, None] & (self._candidate_rows[None] == chosen_column)
        self.selected_count += rows.to(self.selected_count.dtype)
        variant = self.base.variant.gather(1, chosen_column)[:, 0]
        self.used_ability |= rows & (variant == CANDIDATE_ABILITY)
        own_row = self.base.own_card_row.gather(1, chosen_column)[:, 0]
        use_hand = rows & (variant == CANDIDATE_DEPLOY) & (own_row >= 0)
        self.used_hand_row |= use_hand[:, None] & (self._hand_rows[None] == own_row[:, None])
        exclusion = self.base.exclusion_group_id.gather(1, chosen_column)[:, 0]
        self.used_exclusion[:, step] = torch.where(rows, exclusion, self.used_exclusion[:, step])
        offset = torch.zeros_like(self.previous_delay_offset_ms)
        for offset_bin, milliseconds in enumerate(self.config.delay_offset_ms):
            offset = torch.where(delay_offset_bin == offset_bin, float(milliseconds), offset)
        self.previous_delay_offset_ms = torch.where(rows, offset, self.previous_delay_offset_ms)
        self.previous_delay_offset_bin = torch.where(rows, delay_offset_bin, self.previous_delay_offset_bin)

        selected_building = self.base.is_building.gather(1, chosen_column)[:, 0]
        building = rows & selected_building & (target_cell >= 0)
        safe_target = target_cell.clamp(0, self.config.board_width * self.config.board_height - 1)
        x = safe_target.remainder(self.config.board_width)
        y = torch.div(safe_target, self.config.board_width, rounding_mode="floor")
        planned_index = self.planned_count.clamp_max(self.config.max_micro_actions - 1)
        planned_x = x.to(self.planned_x.dtype) + 0.5 + self.base.building_offset_x.gather(1, chosen_column)[:, 0]
        planned_y = y.to(self.planned_y.dtype) + 0.5 + self.base.building_offset_y.gather(1, chosen_column)[:, 0]
        half_width = self.base.building_half_width.gather(1, chosen_column)[:, 0]
        half_height = self.base.building_half_height.gather(1, chosen_column)[:, 0]
        for planned_slot in range(self.config.max_micro_actions):
            write = building & (planned_index == planned_slot)
            self.planned_x[:, planned_slot] = torch.where(write, planned_x, self.planned_x[:, planned_slot])
            self.planned_y[:, planned_slot] = torch.where(write, planned_y, self.planned_y[:, planned_slot])
            self.planned_half_width[:, planned_slot] = torch.where(
                write, half_width, self.planned_half_width[:, planned_slot]
            )
            self.planned_half_height[:, planned_slot] = torch.where(
                write, half_height, self.planned_half_height[:, planned_slot]
            )
        self.planned_count += building.to(self.planned_count.dtype)


def delay_offset_ms(sequence: ActionSequenceV4, config: ModelConfigV4) -> Tensor:
    """Return each active micro-action's offset from the decision start."""

    bins = sequence.delay_offset_bin.clamp_min(0)
    vocabulary = torch.tensor(config.delay_offset_ms, device=bins.device, dtype=torch.long)
    offsets = vocabulary[bins]
    active = torch.arange(config.max_micro_actions, device=bins.device)[None] < sequence.micro_action_count[:, None]
    return offsets.masked_fill(~active, -1)


def wait_ticks(sequence: ActionSequenceV4, config: ModelConfigV4) -> Tensor:
    """V4 WAIT is always exactly one fixed policy turn."""

    return torch.where(
        sequence.gate == GATE_WAIT,
        torch.full_like(sequence.gate, config.decision_ticks),
        torch.zeros_like(sequence.gate),
    )


def _candidate_row_for_uid(candidates: ActionCandidatesV4, *, batch_row: int, candidate_uid: int) -> int:
    matches = (
        (candidates.mask[batch_row] & (candidates.uid[batch_row] == candidate_uid)).nonzero(as_tuple=False).flatten()
    )
    if matches.numel() != 1:
        raise ValueError("sampled candidate UID is missing or duplicated in the live set")
    return int(matches[0].item())


def _placement_entry(observation: ObservationV1, *, native_hand_slot: int) -> Mapping[str, Any]:
    raw = observation.action_mask.placement_masks
    entry = raw.get(str(native_hand_slot), raw.get(native_hand_slot))
    if not isinstance(entry, Mapping):
        raise ValueError("selected deployment has no native placement contract")
    return entry


def _model_subcell_offset(placement_entry: Mapping[str, Any], *, horizontal_mirror: bool) -> tuple[float, float] | None:
    raw = placement_entry.get("model_subcell_offset")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 2:
        raise ValueError("selected deployment has an invalid anchor offset")
    result = (float(raw[0]), float(raw[1]))
    if any(not math.isfinite(value) or not -0.5 <= value <= 0.5 for value in result):
        raise ValueError("selected deployment anchor offset is outside [-0.5, 0.5]")
    return (-result[0] if horizontal_mirror else result[0], result[1])


def _native_placement_allows(
    placement_entry: Mapping[str, Any], *, native_grid: tuple[int, int], config: ModelConfigV4
) -> None:
    rows = placement_entry.get("row_major")
    if rows is None:
        return
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or len(rows) != config.board_height
        or any(
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes, bytearray))
            or len(item) != config.board_width
            for item in rows
        )
    ):
        raise ValueError("native placement contract is not row-major [32][18]")
    x, y = native_grid
    if not bool(rows[y][x]):
        raise ValueError("model target is masked by the native placement contract")


def _timing_metadata(
    *, step: int, delay_offset_bin: int, delay_offset_ms: int, base_latency_ticks: int, base_latency_ms: float
) -> dict[str, int | float | str]:
    return {
        "policy_action_schema": "universal-candidate-action.v4",
        "policy_micro_action_step": int(step),
        "policy_delay_offset_bin": int(delay_offset_bin),
        "policy_delay_offset_ms": int(delay_offset_ms),
        "base_latency_ms": float(base_latency_ms),
        "base_latency_ticks": int(base_latency_ticks),
        "execute_offset_ticks": int(base_latency_ticks + delay_offset_ms // NATIVE_TICK_MS),
    }


def decode_action_sequence_v4(
    sequence: ActionSequenceV4,
    candidates: ActionCandidatesV4,
    *,
    row: int,
    observation: ObservationV1,
    catalog: CardCatalogV1,
    deck: Sequence[int],
    card_costs: Mapping[int, float],
    ability_id_by_vocab_id: Mapping[int, str] | None = None,
    horizontal_mirror: bool = False,
    hand_slot_permutation: Sequence[int] = (0, 1, 2, 3),
    config: ModelConfigV4 | None = None,
    base_latency_ticks: int = 1,
    base_latency_ms: float = 0.0,
    validate: bool = True,
) -> DecodedActionSequenceV4:
    """Decode one V4 row against the same live FAIR legality contract.

    Candidate UIDs, rather than stored candidate row indices, are authoritative
    during replay.  This keeps native decoding correct when a candidate builder
    emits the same set in a different order.
    """

    actual = config or ModelConfigV4()
    if not 0 <= row < candidates.mask.shape[0]:
        raise IndexError("action batch row is outside the candidate batch")
    if candidates.mask.shape[0] != sequence.gate.shape[0]:
        raise ValueError("action and candidate batch sizes differ")
    if validate:
        sequence.validate(actual, candidate_count=int(candidates.mask.shape[1]))
    if observation.owner not in (0, 1):
        raise ValueError("V4 native decoding requires a FAIR actor observation")
    owner = int(observation.owner)
    if len(deck) != actual.max_own_cards:
        raise ValueError("V4 native decoding requires eight deck identities")
    if base_latency_ticks < 1:
        raise ValueError("base_latency_ticks must be positive")
    if not math.isfinite(base_latency_ms) or base_latency_ms < 0.0:
        raise ValueError("base_latency_ms must be finite and non-negative")
    if not isinstance(horizontal_mirror, bool):
        raise TypeError("horizontal_mirror must be boolean")

    gate = int(sequence.gate[row].item())
    if gate == GATE_WAIT:
        return DecodedActionSequenceV4(
            owner=owner,
            actions=(
                ActionV1.wait(
                    owner,
                    ticks=actual.decision_ticks,
                    metadata={
                        "policy_action_schema": "universal-candidate-action.v4",
                        "policy_wait_mode": "fixed-policy-turn",
                        "policy_wait_ticks": actual.decision_ticks,
                        "policy_wait_ms": actual.decision_ticks * NATIVE_TICK_MS,
                    },
                ),
            ),
        )
    if gate != GATE_ACT:
        raise ValueError("V4 action gate must be WAIT=0 or ACT=1")

    effective_elixir = observation.action_mask.reasons.get("effective_elixir")
    if effective_elixir is None or not math.isfinite(float(effective_elixir)):
        raise ValueError("native action mask has no exact effective elixir")
    if not math.isclose(float(candidates.elixir[row].item()), float(effective_elixir), abs_tol=1e-6):
        raise ValueError("candidate set and native action mask disagree on elixir")
    player = next((item for item in observation.players if item.owner == owner), None)
    if player is None or player.elixir_exact is None:
        raise ValueError("V4 native decoding requires exact own elixir")

    perspective = PerspectiveTransformV1(
        actor_owner=owner,
        horizontal_mirror=horizontal_mirror,
        hand_slot_permutation=tuple(int(value) for value in hand_slot_permutation),
    )
    # Same legality rule as sampling and as ``BattleEnvV1._validate_compound_actions``
    # (``config.delay_aware_legality``): each micro-action must be affordable
    # at its own touch bin, with the match multiplier the tensorizer exported.
    raw_multiplier = observation.time.elixir_multiplier
    multiplier = (
        float(raw_multiplier) if raw_multiplier is not None and math.isfinite(float(raw_multiplier)) else 1.0
    )
    shadow = ShadowCandidateLegality(candidates, actual, elixir_multiplier=multiplier)
    active_row = torch.zeros(candidates.mask.shape[0], dtype=torch.bool, device=candidates.mask.device)
    active_row[row] = True
    decoded: list[ActionV1] = []
    action_count = int(sequence.micro_action_count[row].item())

    for step in range(action_count):
        uid = int(sequence.candidate_uid[row, step].item())
        candidate_row = _candidate_row_for_uid(candidates, batch_row=row, candidate_uid=uid)
        legal = shadow.candidate_mask()
        if not bool(legal[row, candidate_row].item()):
            raise ValueError("sampled candidate is illegal under V4 shadow state")

        target_cell = int(sequence.target_cell[row, step].item())
        target_mode = int(candidates.target_mode[row, candidate_row].item())
        variant = int(candidates.variant[row, candidate_row].item())
        if target_mode == TARGET_GRID:
            if target_cell < 0:
                raise ValueError("GRID candidate has no target cell")
            selected_placement = shadow.selected_placement(
                torch.full((candidates.mask.shape[0],), candidate_row, dtype=torch.long, device=candidates.mask.device)
            )
            x_tensor, y_tensor = target_xy_from_cell(
                torch.tensor([target_cell], dtype=torch.long, device=candidates.mask.device),
                width=actual.board_width,
                height=actual.board_height,
            )
            model_grid = (int(x_tensor[0].item()), int(y_tensor[0].item()))
            if not bool(selected_placement[row, model_grid[1], model_grid[0]].item()):
                raise ValueError("sampled target is illegal under V4 shadow state")
        elif target_mode == TARGET_NONE:
            if target_cell != -1:
                raise ValueError("target-free candidate carries a target cell")
            model_grid = None
        else:
            raise ValueError("native V4 decoder does not support entity targets yet")

        delay_bin = int(sequence.delay_offset_bin[row, step].item())
        if not 0 <= delay_bin < len(actual.delay_offset_ms):
            raise ValueError("sampled delay bin is outside the V4 delay vocabulary")
        if shadow.delay_aware:
            selected_candidates = torch.full(
                (candidates.mask.shape[0],), candidate_row, dtype=torch.long, device=candidates.mask.device
            )
            if not bool(shadow.affordable_at_bins()[row, candidate_row, delay_bin].item()) or not bool(
                shadow.delay_offset_mask(step=step, candidate_index=selected_candidates)[row, delay_bin].item()
            ):
                raise ValueError("sampled delay bin is unaffordable under V4 shadow state")
        offset_ms = int(actual.delay_offset_ms[delay_bin])
        timing = _timing_metadata(
            step=step,
            delay_offset_bin=delay_bin,
            delay_offset_ms=offset_ms,
            base_latency_ticks=base_latency_ticks,
            base_latency_ms=base_latency_ms,
        )
        execute_offset_ticks = int(timing["execute_offset_ticks"])
        cost = float(candidates.cost[row, candidate_row].item())

        if variant == CANDIDATE_DEPLOY:
            if target_mode != TARGET_GRID or model_grid is None:
                raise ValueError("DEPLOY candidate must use a GRID target")
            native_hand_slot = int(candidates.native_hand_slot[row, candidate_row].item())
            if not 0 <= native_hand_slot < len(observation.action_mask.hand_slots):
                raise ValueError("DEPLOY candidate has an invalid native hand slot")
            if not observation.action_mask.hand_slots[native_hand_slot]:
                raise ValueError("DEPLOY candidate is masked by the native hand contract")
            placement_entry = _placement_entry(observation, native_hand_slot=native_hand_slot)
            native_visible_card_id = int(candidates.native_visible_card_id[row, candidate_row].item())
            visible_vocab_id = int(candidates.visible_card_vocab_id[row, candidate_row].item())
            effective_vocab_id = int(candidates.effective_card_vocab_id[row, candidate_row].item())
            catalog_visible_card_id = catalog.raw_card_id(visible_vocab_id)
            effective_card_id = catalog.raw_card_id(effective_vocab_id)
            if catalog_visible_card_id != native_visible_card_id:
                raise ValueError("candidate catalog and native visible card IDs disagree")
            if effective_card_id is None:
                raise ValueError("DEPLOY candidate has no exact effective card identity")
            effective_form = int(candidates.effective_form[row, candidate_row].item())
            if native_visible_card_id == MIRROR_CARD_ID:
                mirror_card_id, mirror_cost, mirror_form = mirror_play_runtime_contract(
                    player=player,
                    hand_slot=native_hand_slot,
                    deck=deck,
                    card_costs=card_costs,
                    placement_entry=placement_entry,
                )
                if (
                    mirror_card_id != effective_card_id
                    or not math.isclose(mirror_cost, cost, abs_tol=1e-6)
                    or mirror_form != effective_form
                ):
                    raise ValueError("Mirror candidate disagrees with its live runtime contract")
            else:
                if effective_card_id != native_visible_card_id:
                    raise ValueError("non-Mirror candidate changed its effective card identity")
                runtime_cost = placement_entry.get("effective_cost")
                if (
                    isinstance(runtime_cost, bool)
                    or not isinstance(runtime_cost, (int, float))
                    or not math.isclose(float(runtime_cost), cost, abs_tol=1e-6)
                ):
                    raise ValueError("DEPLOY candidate cost disagrees with its live runtime contract")

            try:
                model_hand_slot = perspective.hand_slot_permutation.index(native_hand_slot)
            except ValueError as error:
                raise ValueError("native hand slot is absent from the perspective") from error
            model_subcell_offset = _model_subcell_offset(placement_entry, horizontal_mirror=horizontal_mirror)
            if bool(candidates.is_building[row, candidate_row].item()):
                candidate_subcell_offset = (
                    float(candidates.building_offset_x[row, candidate_row].item()),
                    float(candidates.building_offset_y[row, candidate_row].item()),
                )
                if model_subcell_offset is None or any(
                    not math.isclose(actual_value, expected_value, abs_tol=1e-6)
                    for actual_value, expected_value in zip(candidate_subcell_offset, model_subcell_offset, strict=True)
                ):
                    raise ValueError("building candidate and native placement anchor disagree")
            model_action = ActionV1(
                owner=owner,
                kind=ActionKind.PLAY_CARD,
                hand_slot=model_hand_slot,
                card_id=native_visible_card_id,
                target_kind=TargetKind.GRID,
                target_grid=model_grid,
                subcell_offset=model_subcell_offset,
                execute_offset_ticks=execute_offset_ticks,
                next_decision_ticks=actual.decision_ticks,
                metadata={
                    **timing,
                    "policy_candidate_uid": uid,
                    "policy_visible_card_id": native_visible_card_id,
                    "policy_effective_card_id": effective_card_id,
                    "policy_effective_cost": cost,
                    "policy_effective_form_code": effective_form,
                },
            )
            native_action = perspective.action_to_native(model_action)
            assert native_action.target_grid is not None
            _native_placement_allows(placement_entry, native_grid=native_action.target_grid, config=actual)
            decoded.append(native_action)
        elif variant == CANDIDATE_ABILITY:
            if target_mode != TARGET_NONE:
                raise ValueError("current native abilities must be target-free")
            source_entity = int(candidates.native_source_entity[row, candidate_row].item())
            if source_entity not in observation.action_mask.ability_sources:
                raise ValueError("ability candidate source is absent from the native mask")
            runtime_contract = ability_runtime_contract(observation, source_entity=source_entity, deck=deck)
            ability_id = runtime_contract.ability_id
            runtime_cost = runtime_contract.cost
            ability_vocab_id = int(candidates.ability_vocab_id[row, candidate_row].item())
            if ability_id_by_vocab_id is not None:
                expected_ability_id = ability_id_by_vocab_id.get(ability_vocab_id)
                if expected_ability_id != ability_id:
                    raise ValueError("ability vocabulary and live ability identity disagree")
            if not math.isclose(runtime_cost, cost, abs_tol=1e-6):
                raise ValueError("ability candidate cost disagrees with its live source")
            visible_card_id = catalog.raw_card_id(int(candidates.visible_card_vocab_id[row, candidate_row].item()))
            effective_card_id = catalog.raw_card_id(int(candidates.effective_card_vocab_id[row, candidate_row].item()))
            if (
                visible_card_id != runtime_contract.source_card_id
                or effective_card_id != runtime_contract.source_card_id
            ):
                raise ValueError("ability candidate card semantics disagree with its live source")
            effective_form = int(candidates.effective_form[row, candidate_row].item())
            decoded.append(
                ActionV1(
                    owner=owner,
                    kind=ActionKind.ACTIVATE_ABILITY,
                    source_entity=source_entity,
                    ability_id=ability_id,
                    execute_offset_ticks=execute_offset_ticks,
                    next_decision_ticks=actual.decision_ticks,
                    metadata={
                        **timing,
                        "policy_candidate_uid": uid,
                        "policy_ability_vocab_id": ability_vocab_id,
                        "policy_visible_card_id": visible_card_id,
                        "policy_effective_card_id": effective_card_id,
                        "policy_effective_form_code": effective_form,
                    },
                )
            )
        else:
            raise ValueError("V4 candidate has an unsupported variant")

        selected_rows = torch.zeros(candidates.mask.shape[0], dtype=torch.long, device=candidates.mask.device)
        selected_rows[row] = candidate_row
        selected_targets = torch.full((candidates.mask.shape[0],), -1, dtype=torch.long, device=candidates.mask.device)
        selected_targets[row] = target_cell
        selected_delays = torch.zeros(candidates.mask.shape[0], dtype=torch.long, device=candidates.mask.device)
        selected_delays[row] = delay_bin
        shadow.apply(active_row, selected_rows, selected_targets, selected_delays, step=step)

    return DecodedActionSequenceV4(owner=owner, actions=tuple(decoded))
