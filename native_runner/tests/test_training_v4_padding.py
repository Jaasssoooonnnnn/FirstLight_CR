from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.mechanics import ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES
from native_runner.training.v4.tensors import (
    ActionCandidatesV4,
    ActionSequenceV4,
    ActiveEffectSetV4,
    BattleGroupSetV4,
    EventSetV4,
    PreviousActionV4,
    RelationEdgesV4,
    TowerSetV4,
    concatenate_padded_tensor_records,
    tensor_padding_value,
)


def _effects(count: int) -> ActiveEffectSetV4:
    return ActiveEffectSetV4(
        effect_vocab_id=torch.ones(1, count, dtype=torch.long),
        parent_type=torch.zeros(1, count, dtype=torch.long),
        parent_index=torch.zeros(1, count, dtype=torch.long),
        source_owner_type=torch.ones(1, count, dtype=torch.long),
        runtime_features=torch.full((1, count, len(ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES)), 0.25),
        mask=torch.ones(1, count, dtype=torch.bool),
    )


def _validate_effects(record: ActiveEffectSetV4) -> None:
    record.validate(ModelConfigV4(), batch_size=record.mask.shape[0], effect_vocab_size=2, child_count=1, tower_count=1)


def _assert_identical(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif is_dataclass(left):
        assert type(left) is type(right)
        for field in fields(left):
            _assert_identical(getattr(left, field.name), getattr(right, field.name))
    else:
        assert left == right


@pytest.mark.parametrize(
    "record_type,field_name,expected",
    [
        (ActiveEffectSetV4, "parent_index", -1),
        (ActiveEffectSetV4, "source_owner_type", 0),
        (TowerSetV4, "owner_type", 2),
        (BattleGroupSetV4, "owner_type", 2),
        (BattleGroupSetV4, "child_group_index", 0),
        (EventSetV4, "owner_type", 2),
        (EventSetV4, "source_group_index", -1),
        (EventSetV4, "target_token_index", -1),
        (ActionCandidatesV4, "native_source_entity", -1),
        (ActionCandidatesV4, "uid", 0),
        (PreviousActionV4, "target_cell", -1),
        (PreviousActionV4, "delay_offset_bin", -1),
        (ActionSequenceV4, "candidate_index", -1),
        (ActionSequenceV4, "candidate_uid", -1),
        (RelationEdgesV4, "source", 0),
        (dict, "parent_index", 0),
    ],
)
def test_padding_defaults_follow_the_record_contract(record_type, field_name, expected) -> None:
    assert tensor_padding_value(record_type, field_name) == expected


def test_empty_effect_collection_gets_canonical_capacity_without_changing_active_rows() -> None:
    empty, active = _effects(0), _effects(1)
    _validate_effects(empty)
    _validate_effects(active)
    combined = concatenate_padded_tensor_records((empty, active))
    _validate_effects(combined)
    assert combined.mask.tolist() == [[False], [True]]
    assert combined.parent_index.tolist() == [[-1], [0]]
    _assert_identical(combined.narrow_batch(1, 1), active)
    assert empty.parent_index.shape == (1, 0)
    assert active.parent_index.tolist() == [[0]]


def test_padding_never_repairs_existing_invalid_source_values() -> None:
    bad = replace(
        _effects(1),
        effect_vocab_id=torch.zeros(1, 1, dtype=torch.long),
        source_owner_type=torch.zeros(1, 1, dtype=torch.long),
        runtime_features=torch.zeros(1, 1, len(ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES)),
        mask=torch.zeros(1, 1, dtype=torch.bool),
    )
    combined = concatenate_padded_tensor_records((bad, _effects(2)))
    assert combined.parent_index[0].tolist() == [0, -1]
    with pytest.raises(ValueError, match="active_effects.parent_index has non-canonical padding"):
        _validate_effects(combined)


def test_nested_variable_collections_preserve_canonical_addresses_and_unknown_owners() -> None:
    short, long = _batch(1), _batch(1)
    short.events = type(short.events)(**{f.name: getattr(short.events, f.name)[:, :1] for f in fields(short.events)})
    short.candidates = type(short.candidates)(
        **{
            f.name: (value if f.name == "elixir" else value[:, :4])
            for f in fields(short.candidates)
            for value in (getattr(short.candidates, f.name),)
        }
    )
    long.active_effects = _effects(1)
    combined = concatenate_padded_tensor_records((short, long))
    model = _model()
    combined.validate(
        model.config,
        card_vocab_size=model.catalog.vocab_size,
        ability_vocab_size=model.ability_catalog.vocab_size,
        entity_archetype_vocab_size=model.entity_archetype_catalog.vocab_size,
        effect_vocab_size=model.effect_catalog.vocab_size,
    )
    assert combined.active_effects.parent_index[0].tolist() == [-1]
    assert combined.events.owner_type[0, 1].item() == 2
    assert combined.events.target_token_index[0, 1].item() == -1
    assert combined.events.source_group_index[0, 1].item() == -1
    assert combined.candidates.own_card_row[0, 4].item() == -1
    assert combined.candidates.native_source_entity[0, 4].item() == -1
    _assert_identical(combined.narrow_batch(1, 1), long)


def test_canonical_padding_preserves_forward_values_and_enables_validation() -> None:
    torch.manual_seed(605)
    model = _model().eval()
    empty, active = _batch(1), _batch(1)
    active.active_effects = _effects(1)
    canonical = concatenate_padded_tensor_records((empty, active))
    old_zero_padded = replace(
        canonical,
        active_effects=replace(
            canonical.active_effects,
            parent_index=canonical.active_effects.parent_index.masked_fill(~canonical.active_effects.mask, 0),
        ),
    )
    with torch.no_grad():
        state = model.initial_state(2)
        before = model.forward(old_zero_padded, state, validate=False)
        after = model.forward(canonical, state, validate=True)
    _assert_identical(before, after)
    with pytest.raises(ValueError, match="active_effects.parent_index has non-canonical padding"):
        model.forward(old_zero_padded, state, validate=True)
