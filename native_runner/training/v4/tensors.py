"""Live tensor contracts for ``universal-semantic-observation.v4``."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Self

import torch
from torch import Tensor

from .config import ModelConfigV4
from .mechanics import ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES


GATE_WAIT = 0
GATE_ACT = 1

CANDIDATE_DEPLOY = 0
CANDIDATE_ABILITY = 1

TARGET_NONE = 0
TARGET_GRID = 1
TARGET_ENTITY = 2

EFFECT_PARENT_CHILD = 0
EFFECT_PARENT_TOWER = 1


def _map_tensors(value: object, transform: object) -> object:
    if isinstance(value, Tensor):
        return transform(value)  # type: ignore[operator]
    if is_dataclass(value):
        return type(value)(**{item.name: _map_tensors(getattr(value, item.name), transform) for item in fields(value)})
    return value


def _cat_records(values: tuple[object, ...]) -> object:
    first = values[0]
    if isinstance(first, Tensor):
        if not all(isinstance(item, Tensor) for item in values):
            raise TypeError("cannot concatenate mixed tensor/non-tensor fields")
        return torch.cat(values, dim=0)  # type: ignore[arg-type]
    if is_dataclass(first):
        if not all(type(item) is type(first) for item in values):
            raise TypeError("cannot concatenate different V4 record types")
        return type(first)(
            **{item.name: _cat_records(tuple(getattr(value, item.name) for value in values)) for item in fields(first)}
        )
    if not all(item == first for item in values[1:]):
        raise ValueError("non-tensor V4 record fields differ")
    return first


def _cat_padded_records(values: tuple[object, ...], padding_value: int = 0) -> object:
    first = values[0]
    if isinstance(first, Tensor):
        if not all(isinstance(item, Tensor) for item in values):
            raise TypeError("cannot concatenate mixed tensor/non-tensor fields")
        ranks = {item.ndim for item in values}
        if len(ranks) != 1:
            raise ValueError("cannot pad tensors with different ranks")
        target = tuple(max(int(item.shape[dimension]) for item in values) for dimension in range(1, first.ndim))
        padded = []
        for item in values:
            if tuple(item.shape[1:]) == target:
                padded.append(item)
                continue
            output = item.new_full((int(item.shape[0]), *target), padding_value)
            slices = (slice(None),) + tuple(slice(0, int(size)) for size in item.shape[1:])
            output[slices] = item
            padded.append(output)
        return torch.cat(tuple(padded), dim=0)
    if is_dataclass(first):
        if not all(type(item) is type(first) for item in values):
            raise TypeError("cannot concatenate different V4 record types")
        return type(first)(
            **{
                item.name: _cat_padded_records(
                    tuple(getattr(value, item.name) for value in values), tensor_padding_value(first, item.name)
                )
                for item in fields(first)
            }
        )
    if not all(item == first for item in values[1:]):
        raise ValueError("non-tensor V4 record fields differ")
    return first


def concatenate_tensor_records(values: tuple[TensorRecordV4, ...]) -> TensorRecordV4:
    """Concatenate matching nested V4 records along their batch dimension."""

    if not values:
        raise ValueError("at least one V4 record is required")
    return _cat_records(values)  # type: ignore[return-value]


def concatenate_padded_tensor_records(values: tuple[TensorRecordV4, ...]) -> TensorRecordV4:
    """Concatenate records using canonical padding for new masked dimensions."""

    if not values:
        raise ValueError("at least one V4 record is required")
    return _cat_padded_records(values)  # type: ignore[return-value]


class TensorRecordV4:
    """Recursive device/detach helpers shared by nested V4 records."""

    def to(self, device: torch.device | str) -> Self:
        return _map_tensors(self, lambda item: item.to(device))  # type: ignore[return-value]

    def detach(self) -> Self:
        return _map_tensors(self, Tensor.detach)  # type: ignore[return-value]

    def to_storage(
        self, device: torch.device | str = "cpu", *, float_dtype: torch.dtype | None = torch.float16
    ) -> Self:
        """Move a record to rollout storage, optionally packing float fields."""

        if float_dtype not in {None, torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("V4 storage float dtype must be fp16, bf16, fp32, or None")

        def move(item: Tensor) -> Tensor:
            dtype = float_dtype if item.is_floating_point() else None
            with torch.inference_mode(False):
                stored = item.detach().to(device=device, dtype=dtype)
                # ``Tensor.to`` may return the original tensor when dtype and
                # device already match.  Rollouts collected under
                # inference_mode must become ordinary tensors before PPO.
                return stored.clone() if torch.is_inference(stored) else stored

        return _map_tensors(self, move)  # type: ignore[return-value]

    def to_model_input(self, device: torch.device | str, *, float_dtype: torch.dtype = torch.float32) -> Self:
        """Restore packed floating inputs to the model's compute dtype."""

        if not float_dtype.is_floating_point:
            raise ValueError("V4 model input dtype must be floating point")

        def move(item: Tensor) -> Tensor:
            dtype = float_dtype if item.is_floating_point() else None
            return item.to(device=device, dtype=dtype)

        return _map_tensors(self, move)  # type: ignore[return-value]

    def index_select(self, indices: Tensor) -> Self:
        return _map_tensors(self, lambda item: item.index_select(0, indices))  # type: ignore[return-value]

    def narrow_batch(self, start: int, length: int) -> Self:
        """Return a zero-copy contiguous view along the batch dimension."""

        if start < 0 or length <= 0:
            raise ValueError("V4 batch narrowing needs a non-negative start and length")

        def narrow(item: Tensor) -> Tensor:
            if start + length > int(item.shape[0]):
                raise ValueError("V4 batch narrowing is outside a tensor")
            return item.narrow(0, start, length)

        return _map_tensors(self, narrow)  # type: ignore[return-value]


def target_cell_from_xy(x: Tensor, y: Tensor, *, width: int = 18, height: int = 32) -> Tensor:
    """Flatten model coordinates using canonical row-major ``y * W + x``."""

    if x.shape != y.shape:
        raise ValueError("x and y tensors must have identical shapes")
    if torch.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
        raise ValueError("target coordinate is outside the model grid")
    return y.to(torch.long) * width + x.to(torch.long)


def target_xy_from_cell(target_cell: Tensor, *, width: int = 18, height: int = 32) -> tuple[Tensor, Tensor]:
    """Decode canonical row-major target cells into ``(x, y)``."""

    _require_range(target_cell, 0, width * height, "target cell is outside the model grid")
    cell = target_cell.to(torch.long)
    return cell.remainder(width), torch.div(cell, width, rounding_mode="floor")


def _require_bool(tensor: Tensor, label: str) -> None:
    if tensor.dtype != torch.bool:
        raise TypeError(f"{label} must be bool")


def _require_integer(tensor: Tensor, label: str) -> None:
    if tensor.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError(f"{label} must use an integer dtype")


def _require_floating_finite(tensor: Tensor, label: str) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{label} must use a floating dtype")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} must contain only finite values")


def _require_shape(tensor: Tensor, shape: tuple[int, ...], label: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{label} has shape {tuple(tensor.shape)}, expected {shape}")


def _validate_vocab_ids(ids: Tensor, mask: Tensor, *, vocab_size: int, label: str) -> None:
    _require_integer(ids, label)
    _require_range(ids, 0, vocab_size, f"{label} contains an out-of-range vocabulary ID")


def _require_padding_value(tensor: Tensor, mask: Tensor, value: int | float | bool, label: str) -> None:
    if torch.any(tensor[~mask] != value):
        raise ValueError(f"{label} has non-canonical padding")


def _require_range(
    values: Tensor, minimum: int | float, maximum: int | float, message: str, *, inclusive_max: bool = False
) -> None:
    above = values > maximum if inclusive_max else values >= maximum
    if torch.any((values < minimum) | above):
        raise ValueError(message)


def _validate_fields(
    record: TensorRecordV4,
    label: str,
    shape: tuple[int, ...],
    *,
    integers: str = "",
    booleans: dict[str, tuple[int, ...]] | None = None,
    floats: dict[str, tuple[int, ...]] | None = None,
) -> None:
    """Validate each record field once from its scalar type and trailing shape."""
    specifications = (
        (_require_integer, {name: () for name in integers.split()}),
        (_require_bool, booleans or {}),
        (_require_floating_finite, floats or {}),
    )
    for validate_type, dimensions in specifications:
        for name, tail in dimensions.items():
            value = getattr(record, name)
            field_label = f"{label}.{name}"
            _require_shape(value, (*shape, *tail), field_label)
            validate_type(value, field_label)


def _validate_padding(record: TensorRecordV4, label: str, mask: Tensor, names: str, **overrides: int | float) -> None:
    for name in names.split():
        _require_padding_value(getattr(record, name), mask, overrides.get(name, 0), f"{label}.{name}")


@dataclass(slots=True)
class CardSetV4(TensorRecordV4):
    card_vocab_id: Tensor
    runtime_form: Tensor
    role_bits: Tensor
    runtime_features: Tensor
    mask: Tensor

    def validate(
        self,
        config: ModelConfigV4,
        *,
        batch_size: int,
        vocab_size: int,
        expected_count: int | None,
        maximum_count: int,
        label: str,
    ) -> None:
        _require_bool(self.mask, f"{label}.mask")
        if self.mask.ndim != 2 or self.mask.shape[0] != batch_size:
            raise ValueError(f"{label}.mask must have shape [B,N]")
        count = int(self.mask.shape[1])
        if expected_count is not None and count != expected_count:
            raise ValueError(f"{label} must contain exactly {expected_count} rows")
        if count > maximum_count:
            raise ValueError(f"{label} exceeds its configured capacity")
        shape = (batch_size, count)
        _validate_fields(
            self,
            label,
            shape,
            integers="card_vocab_id runtime_form",
            booleans={"role_bits": (2,)},
            floats={"runtime_features": (config.card_runtime_feature_dim,)},
        )
        _validate_vocab_ids(self.card_vocab_id, self.mask, vocab_size=vocab_size, label=f"{label}.card_vocab_id")
        forms = self.runtime_form[self.mask]
        _require_range(forms, 0, config.form_type_count, f"{label}.runtime_form is outside the form vocabulary")
        if torch.any(self.role_bits[..., 0] & self.role_bits[..., 1]):
            raise ValueError("elite and evolution deck roles are mutually exclusive")
        active_roles = self.role_bits & self.mask.unsqueeze(-1)
        if torch.any(active_roles[..., 0].sum(dim=1) > 2):
            raise ValueError("a deck cannot have more than two elite roles")
        if torch.any(active_roles[..., 1].sum(dim=1) > 2):
            raise ValueError("a deck cannot have more than two evolution roles")
        if torch.any(active_roles.sum(dim=(1, 2)) > 3):
            raise ValueError("elite plus evolution roles cannot exceed three")
        _validate_padding(self, label, self.mask, "card_vocab_id runtime_form role_bits runtime_features")


@dataclass(slots=True)
class TowerSetV4(TensorRecordV4):
    tower_type: Tensor
    tower_troop_type: Tensor
    owner_type: Tensor
    features: Tensor
    position: Tensor
    extent: Tensor
    mask: Tensor

    def validate(self, config: ModelConfigV4, *, batch_size: int) -> None:
        _require_bool(self.mask, "towers.mask")
        _require_shape(self.mask, (batch_size, config.max_towers), "towers.mask")
        shape = (batch_size, config.max_towers)
        _validate_fields(
            self,
            "towers",
            shape,
            integers="tower_type tower_troop_type owner_type",
            floats={"features": (config.tower_feature_dim,), "position": (2,), "extent": (4,)},
        )
        _require_range(self.tower_type[self.mask], 0, config.tower_type_count, "tower type is outside its vocabulary")
        _require_range(
            self.tower_troop_type[self.mask],
            0,
            config.tower_troop_type_count,
            "tower troop type is outside its vocabulary",
        )
        _require_range(self.owner_type[self.mask], 0, config.owner_type_count, "tower owner is outside its vocabulary")
        _validate_padding(
            self, "towers", self.mask, "tower_type tower_troop_type owner_type features position extent", owner_type=2
        )


@dataclass(slots=True)
class ActiveEffectSetV4(TensorRecordV4):
    """Variable-length live effects attached locally to Child or Tower rows."""

    effect_vocab_id: Tensor
    parent_type: Tensor
    parent_index: Tensor
    source_owner_type: Tensor
    runtime_features: Tensor
    mask: Tensor

    def validate(
        self, config: ModelConfigV4, *, batch_size: int, effect_vocab_size: int, child_count: int, tower_count: int
    ) -> None:
        _require_bool(self.mask, "active_effects.mask")
        if self.mask.ndim != 2 or self.mask.shape[0] != batch_size:
            raise ValueError("active_effects.mask must have shape [B,M]")
        shape = tuple(self.mask.shape)
        _validate_fields(
            self,
            "active_effects",
            shape,
            integers="effect_vocab_id parent_type parent_index source_owner_type",
            floats={"runtime_features": (len(ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES),)},
        )
        _validate_vocab_ids(
            self.effect_vocab_id, self.mask, vocab_size=effect_vocab_size, label="active_effects.effect_vocab_id"
        )
        if torch.any(self.mask & (self.effect_vocab_id == 0)):
            raise ValueError("active effects cannot use the PAD effect identity")
        parent_types = self.parent_type[self.mask]
        _require_range(
            parent_types,
            EFFECT_PARENT_CHILD,
            EFFECT_PARENT_TOWER,
            "active effect parent type is invalid",
            inclusive_max=True,
        )
        source_owners = self.source_owner_type[self.mask]
        _require_range(source_owners, 0, config.owner_type_count, "active effect source owner is invalid")
        child = self.mask & (self.parent_type == EFFECT_PARENT_CHILD)
        tower = self.mask & (self.parent_type == EFFECT_PARENT_TOWER)
        if torch.any(child & ((self.parent_index < 0) | (self.parent_index >= child_count))):
            raise ValueError("active effect child address is invalid")
        if torch.any(tower & ((self.parent_index < 0) | (self.parent_index >= tower_count))):
            raise ValueError("active effect tower address is invalid")
        _validate_padding(
            self,
            "active_effects",
            self.mask,
            "effect_vocab_id parent_type parent_index source_owner_type runtime_features",
            parent_index=-1,
        )


@dataclass(slots=True)
class BattleGroupSetV4(TensorRecordV4):
    source_card_vocab_id: Tensor
    group_type: Tensor
    owner_type: Tensor
    runtime_form: Tensor
    features: Tensor
    position: Tensor
    extent: Tensor
    mask: Tensor

    child_archetype_id: Tensor
    child_group_index: Tensor
    child_type: Tensor
    child_features: Tensor
    child_position: Tensor
    child_extent: Tensor
    child_radius: Tensor
    child_mask: Tensor

    def validate(
        self, config: ModelConfigV4, *, batch_size: int, card_vocab_size: int, entity_archetype_vocab_size: int
    ) -> None:
        _require_bool(self.mask, "groups.mask")
        _require_bool(self.child_mask, "groups.child_mask")
        if self.mask.ndim != 2 or self.mask.shape[0] != batch_size:
            raise ValueError("groups.mask must have shape [B,G]")
        if self.child_mask.ndim != 2 or self.child_mask.shape[0] != batch_size:
            raise ValueError("groups.child_mask must have shape [B,N_child]")
        groups = int(self.mask.shape[1])
        children = int(self.child_mask.shape[1])
        if groups > config.max_battle_groups:
            raise ValueError("battle-group capacity exceeded")
        if children > config.max_children_total:
            raise ValueError("child capacity exceeded")
        group_shape = (batch_size, groups)
        child_shape = (batch_size, children)
        _validate_fields(
            self,
            "groups",
            group_shape,
            integers="source_card_vocab_id group_type owner_type runtime_form",
            floats={"features": (config.group_feature_dim,), "position": (2,), "extent": (4,)},
        )
        _validate_fields(
            self,
            "groups",
            child_shape,
            integers="child_archetype_id child_group_index child_type",
            floats={
                "child_features": (config.child_feature_dim,),
                "child_position": (2,),
                "child_extent": (4,),
                "child_radius": (),
            },
        )
        _validate_vocab_ids(
            self.source_card_vocab_id, self.mask, vocab_size=card_vocab_size, label="groups.source_card_vocab_id"
        )
        _require_range(self.group_type[self.mask], 0, config.group_type_count, "group type is outside its vocabulary")
        _require_range(self.owner_type[self.mask], 0, config.owner_type_count, "group owner is outside its vocabulary")
        _require_range(self.runtime_form[self.mask], 0, config.form_type_count, "group form is outside its vocabulary")
        child_group = self.child_group_index[self.child_mask]
        _require_range(child_group, 0, groups, "child references an out-of-range group row")
        batch_rows = torch.arange(batch_size, device=self.mask.device).unsqueeze(1)
        safe_groups = self.child_group_index.clamp(0, max(groups - 1, 0))
        referenced_group_mask = self.mask[batch_rows, safe_groups]
        if torch.any(self.child_mask & ~referenced_group_mask):
            raise ValueError("a retained child references a masked group")
        _require_range(
            self.child_archetype_id[self.child_mask],
            0,
            entity_archetype_vocab_size,
            "child archetype is outside its vocabulary",
        )
        _require_range(
            self.child_type[self.child_mask], 0, config.child_type_count, "child type is outside its vocabulary"
        )
        active_child_extents = self.child_extent[self.child_mask]
        if torch.any(
            (active_child_extents[:, 2] < active_child_extents[:, 0])
            | (active_child_extents[:, 3] < active_child_extents[:, 1])
        ):
            raise ValueError("child extent bounds are reversed")
        if torch.any(self.child_radius[self.child_mask] < 0):
            raise ValueError("child radius must be non-negative")
        _validate_padding(
            self,
            "groups",
            self.mask,
            "source_card_vocab_id group_type owner_type runtime_form features position extent",
            owner_type=2,
        )
        _validate_padding(
            self,
            "groups",
            self.child_mask,
            "child_archetype_id child_group_index child_type child_features child_position child_extent child_radius",
        )


@dataclass(slots=True)
class EventSetV4(TensorRecordV4):
    event_type: Tensor
    owner_type: Tensor
    source_card_vocab_id: Tensor
    source_form: Tensor
    source_group_index: Tensor
    target_token_index: Tensor
    features: Tensor
    mask: Tensor

    def validate(
        self, config: ModelConfigV4, *, batch_size: int, vocab_size: int, group_count: int, token_count: int
    ) -> None:
        _require_bool(self.mask, "events.mask")
        if self.mask.ndim != 2 or self.mask.shape[0] != batch_size:
            raise ValueError("events.mask must have shape [B,E]")
        count = int(self.mask.shape[1])
        if count > config.max_recent_events:
            raise ValueError("recent-event capacity exceeded")
        shape = (batch_size, count)
        _validate_fields(
            self,
            "events",
            shape,
            integers="event_type owner_type source_card_vocab_id source_form source_group_index target_token_index",
            floats={"features": (config.event_feature_dim,)},
        )
        _validate_vocab_ids(
            self.source_card_vocab_id, self.mask, vocab_size=vocab_size, label="events.source_card_vocab_id"
        )
        _require_range(self.event_type[self.mask], 0, config.event_type_count, "event type is outside its vocabulary")
        _require_range(self.owner_type[self.mask], 0, config.owner_type_count, "event owner is outside its vocabulary")
        forms = self.source_form[self.mask]
        _require_range(forms, 0, config.form_type_count, "event source form is outside its vocabulary")
        groups = self.source_group_index[self.mask]
        _require_range(groups, -1, group_count, "event source-group address is invalid")
        targets = self.target_token_index[self.mask]
        _require_range(targets, -1, token_count, "event target-token address is invalid")
        _validate_padding(
            self,
            "events",
            self.mask,
            "event_type owner_type source_card_vocab_id source_form source_group_index target_token_index features",
            owner_type=2,
            source_group_index=-1,
            target_token_index=-1,
        )


@dataclass(slots=True)
class RelationEdgesV4(TensorRecordV4):
    source: Tensor
    target: Tensor
    relation_type: Tensor
    mask: Tensor

    def validate(self, config: ModelConfigV4, *, batch_size: int, token_count: int) -> None:
        _require_bool(self.mask, "relation_edges.mask")
        if self.mask.ndim != 2 or self.mask.shape[0] != batch_size:
            raise ValueError("relation_edges.mask must have shape [B,M]")
        shape = tuple(self.mask.shape)
        _validate_fields(self, "relation_edges", shape, integers="source target relation_type")
        for name in ("source", "target"):
            values = getattr(self, name)[self.mask]
            _require_range(values, 0, token_count, f"relation edge {name} address is invalid")
        types = self.relation_type[self.mask]
        _require_range(types, 0, config.relation_type_count, "relation type is outside its vocabulary")
        _validate_padding(self, "relation_edges", self.mask, "source target relation_type")


@dataclass(slots=True)
class PreviousActionV4(TensorRecordV4):
    gate: Tensor
    micro_action_count: Tensor
    variant: Tensor
    visible_card_vocab_id: Tensor
    effective_card_vocab_id: Tensor
    effective_form: Tensor
    ability_vocab_id: Tensor
    target_cell: Tensor
    delay_offset_bin: Tensor
    source_position: Tensor
    source_mask: Tensor
    action_mask: Tensor

    def validate(
        self, config: ModelConfigV4, *, batch_size: int, card_vocab_size: int, ability_vocab_size: int
    ) -> None:
        shape = (batch_size, config.max_micro_actions)
        _validate_fields(self, "previous_action", (batch_size,), integers="gate micro_action_count")
        _validate_fields(
            self,
            "previous_action",
            shape,
            integers="variant visible_card_vocab_id effective_card_vocab_id effective_form "
            "ability_vocab_id target_cell delay_offset_bin",
            booleans={"action_mask": (), "source_mask": ()},
            floats={"source_position": (2,)},
        )
        _require_range(self.gate, GATE_WAIT, GATE_ACT, "previous action gate must be WAIT or ACT", inclusive_max=True)
        _require_range(
            self.micro_action_count,
            0,
            config.max_micro_actions,
            "previous micro-action count is invalid",
            inclusive_max=True,
        )
        expected_mask = (
            torch.arange(config.max_micro_actions, device=self.gate.device)[None] < self.micro_action_count[:, None]
        )
        if not torch.equal(expected_mask, self.action_mask):
            raise ValueError("previous action mask does not match its count")
        if torch.any((self.gate == GATE_WAIT) & (self.micro_action_count != 0)) or torch.any(
            (self.gate == GATE_ACT) & (self.micro_action_count == 0)
        ):
            raise ValueError("previous gate and micro-action count disagree")
        for name in ("visible_card_vocab_id", "effective_card_vocab_id"):
            _validate_vocab_ids(
                getattr(self, name), self.action_mask, vocab_size=card_vocab_size, label=f"previous_action.{name}"
            )
        variants = self.variant[self.action_mask]
        _require_range(
            variants, CANDIDATE_DEPLOY, CANDIDATE_ABILITY, "previous action variant is invalid", inclusive_max=True
        )
        forms = self.effective_form[self.action_mask]
        _require_range(forms, 0, config.form_type_count, "previous action form is outside its vocabulary")
        ability_ids = self.ability_vocab_id[self.action_mask]
        _require_range(ability_ids, 0, ability_vocab_size, "previous action ability ID is outside its vocabulary")
        expected_source = self.action_mask & (self.variant == CANDIDATE_ABILITY)
        if not torch.equal(self.source_mask, expected_source):
            raise ValueError("previous action source mask must identify Ability sources")
        targets = self.target_cell[self.action_mask]
        _require_range(targets, -1, config.board_width * config.board_height, "previous action target cell is invalid")
        offsets = self.delay_offset_bin[self.action_mask]
        _require_range(offsets, 0, len(config.delay_offset_ms), "previous action delay offset is invalid")
        if torch.any(self.action_mask[:, 1:] & (self.delay_offset_bin[:, 1:] < self.delay_offset_bin[:, :-1])):
            raise ValueError("previous action delay offsets must be nondecreasing")
        _validate_padding(
            self,
            "previous_action",
            self.action_mask,
            "variant visible_card_vocab_id effective_card_vocab_id effective_form "
            "ability_vocab_id target_cell delay_offset_bin",
            target_cell=-1,
            delay_offset_bin=-1,
        )
        _require_padding_value(self.source_position, self.source_mask, 0.0, "previous_action.source_position")


@dataclass(slots=True)
class ActionCandidatesV4(TensorRecordV4):
    mask: Tensor
    elixir: Tensor
    uid: Tensor
    variant: Tensor
    visible_card_vocab_id: Tensor
    effective_card_vocab_id: Tensor
    effective_form: Tensor
    own_card_row: Tensor
    source_group_index: Tensor
    source_child_index: Tensor
    ability_vocab_id: Tensor
    cost: Tensor
    runtime_features: Tensor
    target_mode: Tensor
    placement: Tensor
    is_building: Tensor
    building_half_width: Tensor
    building_half_height: Tensor
    building_offset_x: Tensor
    building_offset_y: Tensor
    exclusion_group_id: Tensor
    native_hand_slot: Tensor
    native_source_entity: Tensor
    native_visible_card_id: Tensor

    def validate(
        self,
        config: ModelConfigV4,
        *,
        batch_size: int,
        card_vocab_size: int,
        ability_vocab_size: int,
        own_card_count: int,
        group_count: int,
        child_count: int,
    ) -> None:
        _require_bool(self.mask, "candidates.mask")
        if self.mask.ndim != 2 or self.mask.shape[0] != batch_size:
            raise ValueError("candidates.mask must have shape [B,C]")
        count = int(self.mask.shape[1])
        if count <= 0:
            raise ValueError("action-candidate tensor needs at least one row")
        if count > config.max_action_candidates:
            raise ValueError("action-candidate capacity exceeded")
        shape = (batch_size, count)
        _require_shape(self.elixir, (batch_size,), "candidates.elixir")
        _require_floating_finite(self.elixir, "candidates.elixir")
        _require_range(self.elixir, 0.0, 10.0001, "candidate elixir must be in [0, 10]", inclusive_max=True)
        _validate_fields(
            self,
            "candidates",
            shape,
            integers="uid variant visible_card_vocab_id effective_card_vocab_id effective_form "
            "own_card_row source_group_index source_child_index ability_vocab_id "
            "target_mode exclusion_group_id native_hand_slot native_source_entity "
            "native_visible_card_id",
            floats={
                "cost": (),
                "building_half_width": (),
                "building_half_height": (),
                "building_offset_x": (),
                "building_offset_y": (),
                "runtime_features": (config.candidate_runtime_feature_dim,),
            },
            booleans={"placement": (config.board_height, config.board_width), "is_building": ()},
        )
        for name in ("visible_card_vocab_id", "effective_card_vocab_id"):
            _validate_vocab_ids(getattr(self, name), self.mask, vocab_size=card_vocab_size, label=f"candidates.{name}")
        variants = self.variant[self.mask]
        _require_range(
            variants, CANDIDATE_DEPLOY, CANDIDATE_ABILITY, "candidate variant is invalid", inclusive_max=True
        )
        if torch.any(self.uid[self.mask] == -1):
            raise ValueError("legal candidate UID cannot use the padding sentinel -1")
        deploy = self.mask & (self.variant == CANDIDATE_DEPLOY)
        ability = self.mask & (self.variant == CANDIDATE_ABILITY)
        if torch.any(deploy & ((self.native_hand_slot < 0) | (self.native_hand_slot >= 4))):
            raise ValueError("DEPLOY candidate needs a native hand slot in 0..3")
        if torch.any(deploy & (self.native_visible_card_id <= 0)):
            raise ValueError("DEPLOY candidate needs a positive native card ID")
        if torch.any(ability & (self.native_source_entity < 0)):
            raise ValueError("ABILITY candidate needs an exact native source entity")
        if torch.any(ability & (self.native_hand_slot != -1)):
            raise ValueError("ABILITY candidate must not carry a native hand slot")
        forms = self.effective_form[self.mask]
        _require_range(forms, 0, config.form_type_count, "candidate form is outside its vocabulary")
        card_rows = self.own_card_row[self.mask]
        _require_range(card_rows, -1, own_card_count, "candidate own-card address is invalid")
        group_rows = self.source_group_index[self.mask]
        _require_range(group_rows, -1, group_count, "candidate source-group address is invalid")
        child_rows = self.source_child_index[self.mask]
        _require_range(child_rows, -1, child_count, "candidate source-child address is invalid")
        ability_ids = self.ability_vocab_id[self.mask]
        _require_range(ability_ids, 0, ability_vocab_size, "candidate ability ID is outside its vocabulary")
        target_modes = self.target_mode[self.mask]
        if torch.any((target_modes != TARGET_NONE) & (target_modes != TARGET_GRID)):
            raise ValueError("current V4 candidates support only NONE and GRID targets")
        if torch.any(self.cost[self.mask] < 0):
            raise ValueError("candidate cost must be non-negative")
        if torch.any(self.building_half_width[self.mask] < 0) or torch.any(self.building_half_height[self.mask] < 0):
            raise ValueError("candidate building footprint must be non-negative")
        grid = self.mask & (self.target_mode == TARGET_GRID)
        if torch.any(grid & ~self.placement.flatten(2).any(dim=-1)):
            raise ValueError("a legal GRID candidate needs at least one placement")
        nongrid = self.mask & (self.target_mode != TARGET_GRID)
        if torch.any(nongrid & self.placement.flatten(2).any(dim=-1)):
            raise ValueError("non-GRID candidates must not carry placement cells")
        for row in range(batch_size):
            uids = self.uid[row, self.mask[row]]
            if uids.numel() != torch.unique(uids).numel():
                raise ValueError("candidate UIDs must be unique within each row")
        _validate_padding(
            self,
            "candidates",
            self.mask,
            "uid variant visible_card_vocab_id effective_card_vocab_id effective_form "
            "own_card_row source_group_index source_child_index ability_vocab_id cost "
            "runtime_features target_mode placement is_building building_half_width "
            "building_half_height building_offset_x building_offset_y exclusion_group_id "
            "native_hand_slot native_source_entity native_visible_card_id",
            own_card_row=-1,
            source_group_index=-1,
            source_child_index=-1,
            exclusion_group_id=-1,
            native_hand_slot=-1,
            native_source_entity=-1,
            native_visible_card_id=-1,
        )


@dataclass(slots=True)
class UniversalSemanticBatchV4(TensorRecordV4):
    match_scalars: Tensor
    own_cards: CardSetV4
    opponent_cards: CardSetV4
    towers: TowerSetV4
    groups: BattleGroupSetV4
    active_effects: ActiveEffectSetV4
    events: EventSetV4
    relation_edges: RelationEdgesV4
    explicit_spatial_planes: Tensor
    previous_action: PreviousActionV4
    candidates: ActionCandidatesV4

    @property
    def batch_size(self) -> int:
        return int(self.match_scalars.shape[0])

    @property
    def global_token_count(self) -> int:
        return int(
            self.towers.mask.shape[1]
            + self.own_cards.mask.shape[1]
            + self.opponent_cards.mask.shape[1]
            + self.groups.mask.shape[1]
        )

    def to_storage(
        self, device: torch.device | str = "cpu", *, float_dtype: torch.dtype | None = torch.float16
    ) -> Self:
        """Pack model features without rounding action-legality budgets."""

        stored = TensorRecordV4.to_storage(self, device, float_dtype=float_dtype)
        if float_dtype not in {torch.float16, torch.bfloat16}:
            return stored

        def exact_float32(item: Tensor) -> Tensor:
            with torch.inference_mode(False):
                exact = item.detach().to(device=device, dtype=torch.float32)
                return exact.clone() if torch.is_inference(exact) else exact

        # A native effective elixir such as 4.999 rounds to 5.0 in fp16.  If
        # these fields are packed, rollout sampling can admit a two-card cost
        # of exactly five that the authoritative native decoder then rejects.
        stored.candidates.elixir = exact_float32(self.candidates.elixir)
        stored.candidates.cost = exact_float32(self.candidates.cost)
        return stored

    def validate(
        self,
        config: ModelConfigV4,
        *,
        card_vocab_size: int,
        ability_vocab_size: int,
        entity_archetype_vocab_size: int,
        effect_vocab_size: int,
    ) -> None:
        if self.match_scalars.ndim != 2:
            raise ValueError("match_scalars must have shape [B,F]")
        batch_size = self.batch_size
        _require_shape(self.match_scalars, (batch_size, config.generic_scalar_dim), "match_scalars")
        _require_floating_finite(self.match_scalars, "match_scalars")
        self.own_cards.validate(
            config,
            batch_size=batch_size,
            vocab_size=card_vocab_size,
            expected_count=config.max_own_cards,
            maximum_count=config.max_own_cards,
            label="own_cards",
        )
        self.opponent_cards.validate(
            config,
            batch_size=batch_size,
            vocab_size=card_vocab_size,
            expected_count=None,
            maximum_count=config.max_opponent_cards,
            label="opponent_cards",
        )
        self.towers.validate(config, batch_size=batch_size)
        self.groups.validate(
            config,
            batch_size=batch_size,
            card_vocab_size=card_vocab_size,
            entity_archetype_vocab_size=entity_archetype_vocab_size,
        )
        self.active_effects.validate(
            config,
            batch_size=batch_size,
            effect_vocab_size=effect_vocab_size,
            child_count=self.groups.child_mask.shape[1],
            tower_count=self.towers.mask.shape[1],
        )
        token_count = self.global_token_count
        self.events.validate(
            config,
            batch_size=batch_size,
            vocab_size=card_vocab_size,
            group_count=self.groups.mask.shape[1],
            token_count=token_count,
        )
        self.relation_edges.validate(config, batch_size=batch_size, token_count=token_count)
        _require_shape(
            self.explicit_spatial_planes,
            (batch_size, config.explicit_spatial_channels, config.board_height, config.board_width),
            "explicit_spatial_planes",
        )
        _require_floating_finite(self.explicit_spatial_planes, "explicit_spatial_planes")
        self.previous_action.validate(
            config, batch_size=batch_size, card_vocab_size=card_vocab_size, ability_vocab_size=ability_vocab_size
        )
        self.candidates.validate(
            config,
            batch_size=batch_size,
            card_vocab_size=card_vocab_size,
            ability_vocab_size=ability_vocab_size,
            own_card_count=self.own_cards.mask.shape[1],
            group_count=self.groups.mask.shape[1],
            child_count=self.groups.child_mask.shape[1],
        )


@dataclass(slots=True)
class RecurrentPolicyStateV4(TensorRecordV4):
    hidden: Tensor
    cell: Tensor


@dataclass(slots=True)
class ActionSequenceV4(TensorRecordV4):
    gate: Tensor
    micro_action_count: Tensor
    candidate_index: Tensor
    candidate_uid: Tensor
    target_cell: Tensor
    delay_offset_bin: Tensor

    def validate(self, config: ModelConfigV4, *, candidate_count: int) -> None:
        batch_size = int(self.gate.shape[0])
        shape = (batch_size, config.max_micro_actions)
        _validate_fields(self, "actions", (batch_size,), integers="gate micro_action_count")
        _validate_fields(self, "actions", shape, integers="candidate_index candidate_uid target_cell delay_offset_bin")
        _require_range(self.gate, GATE_WAIT, GATE_ACT, "action gate must be WAIT or ACT", inclusive_max=True)
        _require_range(
            self.micro_action_count, 0, config.max_micro_actions, "micro-action count is invalid", inclusive_max=True
        )
        if torch.any((self.gate == GATE_WAIT) & (self.micro_action_count != 0)) or torch.any(
            (self.gate == GATE_ACT) & (self.micro_action_count == 0)
        ):
            raise ValueError("gate and micro-action count disagree")
        active = (
            torch.arange(config.max_micro_actions, device=self.gate.device)[None] < self.micro_action_count[:, None]
        )
        indices = self.candidate_index[active]
        _require_range(indices, 0, candidate_count, "active action has an invalid candidate index")
        if torch.any(self.candidate_index[~active] != -1):
            raise ValueError("inactive action must use candidate index -1")
        if torch.any(self.candidate_uid[~active] != -1):
            raise ValueError("inactive action must use candidate UID -1")
        targets = self.target_cell[active]
        _require_range(targets, -1, config.board_width * config.board_height, "action target cell is invalid")
        offsets = self.delay_offset_bin[active]
        _require_range(offsets, 0, len(config.delay_offset_ms), "action delay offset is invalid")
        if torch.any(active[:, 1:] & (self.delay_offset_bin[:, 1:] < self.delay_offset_bin[:, :-1])):
            raise ValueError("action delay offsets must be nondecreasing")
        if torch.any(self.target_cell[~active] != -1) or torch.any(self.delay_offset_bin[~active] != -1):
            raise ValueError("inactive action fields must use -1 padding")


@dataclass(slots=True)
class ActionEvaluationComponentsV4(TensorRecordV4):
    gate_log_prob: Tensor
    gate_entropy: Tensor
    candidate_log_prob: Tensor
    candidate_entropy: Tensor
    target_log_prob: Tensor
    target_entropy: Tensor
    delay_offset_log_prob: Tensor
    delay_offset_log_probs: Tensor
    delay_offset_legal_mask: Tensor
    delay_offset_entropy: Tensor
    continue_log_prob: Tensor
    continue_entropy: Tensor


@dataclass(slots=True)
class EncodedObservationV4(TensorRecordV4):
    scene_state: Tensor
    token_memory: Tensor
    token_mask: Tensor
    own_card_memory: Tensor
    group_memory: Tensor
    child_memory: Tensor
    spatial_memory: Tensor
    spatial_summary: Tensor
    scalar_summary: Tensor
    event_summary: Tensor
    previous_action_summary: Tensor
    candidate_memory: Tensor
    candidate_summary: Tensor


@dataclass(slots=True)
class PolicyContextV4(TensorRecordV4):
    policy_context: Tensor
    value_context: Tensor
    encoded: EncodedObservationV4
    next_state: RecurrentPolicyStateV4
    value: Tensor


@dataclass(slots=True)
class PolicyOutputV4(TensorRecordV4):
    actions: ActionSequenceV4
    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    next_state: RecurrentPolicyStateV4
    action_components: ActionEvaluationComponentsV4 | None = None


_PADDING_VALUES = {
    TowerSetV4: {"owner_type": 2},
    BattleGroupSetV4: {"owner_type": 2},
    EventSetV4: {"owner_type": 2, "source_group_index": -1, "target_token_index": -1},
    ActiveEffectSetV4: {"parent_index": -1},
    ActionCandidatesV4: dict.fromkeys(
        (
            "own_card_row",
            "source_group_index",
            "source_child_index",
            "exclusion_group_id",
            "native_hand_slot",
            "native_source_entity",
            "native_visible_card_id",
        ),
        -1,
    ),
    PreviousActionV4: {"target_cell": -1, "delay_offset_bin": -1},
    ActionSequenceV4: dict.fromkeys(("candidate_index", "candidate_uid", "target_cell", "delay_offset_bin"), -1),
}


def tensor_padding_value(record: object, field_name: str) -> int:
    """Return a field's canonical fill value; existing tensor values remain authoritative."""
    record_type = record if isinstance(record, type) else type(record)
    return _PADDING_VALUES.get(record_type, {}).get(field_name, 0)
