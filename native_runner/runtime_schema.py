"""Shared wire validation for the three authoritative native movement rings.

Card-specific event proofs remain in their adapters. This module owns only
the common entity identity, capability, and epoch/retention protocol.
"""

from dataclasses import dataclass, fields
from functools import cache
from typing import Any, Mapping

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


@cache
def wire_fields(record_type: type) -> frozenset[str]:
    """Dataclass names are the snake_case spelling of these wire fields."""
    return frozenset(
        field.name.split("_")[0] + "".join(part.title() for part in field.name.split("_")[1:])
        for field in fields(record_type)
    )


def same_entity_identity(before: object, after: object) -> bool:
    return (
        before.native_object_id == after.native_object_id
        and before.entity_key == after.entity_key
        and before.owner == after.owner
        and before.object_index == after.object_index
        and before.secondary_index == after.secondary_index
        and before.card_id == after.card_id
        and before.object_kind == after.object_kind
    )


@dataclass(frozen=True, slots=True)
class RuntimeSchema:
    error: type[ValueError]
    label: str
    version: str
    capacity: int
    capability_source: str
    capability_validation: str
    capability_type: type
    entity_type: type
    envelope_type: type
    owner_bounds: tuple[int, int] = (0, 1)
    character_cards: frozenset[int] | None = None
    object_kind_min: int = 0

    def mapping(self, value: object, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise self.error(f"{label} must be a mapping")
        return value

    def integer(self, value: object, label: str, *, minimum=None, maximum=None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.error(f"{label} must be an integer")
        if minimum is not None and value < minimum:
            raise self.error(f"{label} must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise self.error(f"{label} must be <= {maximum}")
        return value

    def boolean(self, value: object, label: str) -> bool:
        if not isinstance(value, bool):
            raise self.error(f"{label} must be a boolean")
        return value

    def point(self, value: object, label: str) -> tuple[int, int]:
        if not isinstance(value, list) or len(value) != 2:
            raise self.error(f"{label} must contain two integers")
        return (
            self.integer(value[0], f"{label}[0]", minimum=INT32_MIN, maximum=INT32_MAX),
            self.integer(value[1], f"{label}[1]", minimum=INT32_MIN, maximum=INT32_MAX),
        )

    def entity_key(self, value: object, label: str) -> tuple[int, int, int]:
        if not isinstance(value, list) or len(value) != 3:
            raise self.error(f"{label} must contain three integers")
        low, high = (0, 1) if self.character_cards is not None else (INT32_MIN, INT32_MAX)
        return (
            self.integer(value[0], f"{label}[0]", minimum=low, maximum=high),
            self.integer(value[1], f"{label}[1]", minimum=INT32_MIN, maximum=INT32_MAX),
            self.integer(value[2], f"{label}[2]", minimum=1, maximum=UINT32_MAX),
        )

    def record(self, value: object, label: str, fields, schema_name: str):
        raw = self.mapping(value, label)
        if set(raw) != set(fields):
            raise self.error(f"{label} fields do not match the {schema_name} schema")
        return WireRecord(self, raw, label)

    def entity_fact(self, value: object, label: str):
        record = self.record(
            value,
            label,
            (
                "validated",
                "present",
                "nativeObjectId",
                "entityKey",
                "owner",
                "objectIndex",
                "secondaryIndex",
                "cardId",
                "objectKind",
                "position",
                "visibilityValidated",
                "invisibleCount",
            ),
            "combat entity",
        )
        for marker in ("validated", "present"):
            if record.boolean(marker) is not True:
                raise self.error(f"{label} identity is not {marker}")
        native_object_id = record.integer("nativeObjectId", 1, UINT32_MAX)
        # Character state validates owner before its key; the other rings
        # validate the key first. Keep error ordering as well as accepted data.
        if self.character_cards is not None:
            owner = record.integer("owner", *self.owner_bounds)
        entity_key = self.entity_key(record.raw.get("entityKey"), f"{label}.entityKey")
        if self.character_cards is None:
            owner = record.integer("owner", *self.owner_bounds)
        identity_matches = entity_key == (owner, -2, native_object_id)
        identity_error = f"{label}.entityKey does not match authoritative identity fields"
        if self.character_cards is not None and not identity_matches:
            raise self.error(identity_error)
        object_index = record.integer("objectIndex", INT32_MIN, INT32_MAX)
        secondary_index = record.integer("secondaryIndex", INT32_MIN, INT32_MAX)
        if not identity_matches:
            raise self.error(identity_error)
        card_id = record.integer("cardId", INT32_MIN, INT32_MAX)
        if self.character_cards is not None and card_id not in self.character_cards:
            raise self.error(f"{label}.cardId is outside the exact allowlist")
        object_kind = record.integer("objectKind", self.object_kind_min, 31)
        if self.character_cards is not None and object_kind != 5:
            raise self.error(f"{label}.objectKind is not a native LogicCharacter")
        position = self.point(record.raw.get("position"), f"{label}.position")
        visibility_validated = record.boolean("visibilityValidated")
        invisible_count = record.optional_integer("invisibleCount", 0, 64)
        if visibility_validated != (invisible_count is not None):
            raise self.error(f"{label} visibility validation/value relationship is invalid")
        return self.entity_type(
            native_object_id=native_object_id,
            entity_key=entity_key,
            owner=owner,
            object_index=object_index,
            secondary_index=secondary_index,
            card_id=card_id,
            object_kind=object_kind,
            position=position,
            visibility_validated=visibility_validated,
            invisible_count=invisible_count,
        )

    def capability(self, value: object):
        label = f"{self.label}.capability"
        record = self.record(value, label, ("status", "source", "validation", "confidence", "failClosed"), "capability")
        status = record.raw.get("status")
        if status not in {"derived", "unavailable"}:
            raise self.error(f"{label}.status is invalid")
        for name, expected in (
            ("source", self.capability_source),
            ("validation", self.capability_validation),
            ("confidence", "high"),
        ):
            if record.raw.get(name) != expected:
                raise self.error(f"{label}.{name} is invalid")
        if record.boolean("failClosed") is not True:
            raise self.error(f"{label} is not fail-closed")
        return self.capability_type(
            status=str(status),
            source=self.capability_source,
            validation=self.capability_validation,
            confidence="high",
            fail_closed=True,
        )

    def envelope(self, value, generation, state_epoch, observation_tick, event_parser):
        label = self.label
        record = self.record(
            value,
            label,
            (
                "ok",
                "schema",
                "generation",
                "stateEpoch",
                "observationTick",
                "capacity",
                "hookSetAttested",
                "hookSetInstalled",
                "capability",
                "epochFirstSequence",
                "oldestRetainedSequence",
                "nextSequence",
                "overflowCount",
                "rejectedCount",
                "sequenceGapBeforeOldest",
                "complete",
                "events",
            ),
            "v1",
        )
        raw = record.raw
        if raw.get("ok") is not True or raw.get("schema") != self.version:
            raise self.error(f"{label} schema/ok marker is invalid")
        wire_generation = record.integer("generation", 0, UINT64_MAX)
        wire_state_epoch = record.integer("stateEpoch", 0, UINT64_MAX)
        wire_tick = record.integer("observationTick", -1, INT32_MAX)
        if (wire_generation, wire_state_epoch, wire_tick) != (generation, state_epoch, observation_tick):
            raise self.error(f"{label} observation identity is inconsistent")
        capacity = record.integer("capacity", 1)
        if capacity != self.capacity:
            raise self.error(f"{label} capacity is unexpected")
        attested = record.boolean("hookSetAttested")
        installed = record.boolean("hookSetInstalled")
        if installed and not attested:
            verb = "hook is" if self.character_cards is not None else "hooks are"
            raise self.error(f"{label} {verb} installed without attestation")
        capability = self.capability(raw.get("capability"))
        if capability.status == "derived" and (not attested or not installed):
            noun = "an installed attested hook" if self.character_cards is not None else "installed attested hooks"
            raise self.error(f"{label} derived capability lacks {noun}")
        first = record.integer("epochFirstSequence", 1, UINT64_MAX)
        oldest = record.integer("oldestRetainedSequence", first, UINT64_MAX)
        next_sequence = record.integer("nextSequence", oldest, UINT64_MAX)
        overflow = record.integer("overflowCount", 0, UINT64_MAX)
        if overflow != max(0, next_sequence - first - capacity):
            raise self.error(f"{label} overflow accounting is inconsistent")
        if oldest != max(first, next_sequence - capacity):
            raise self.error(f"{label} retained window is inconsistent")
        gap = record.boolean("sequenceGapBeforeOldest")
        if gap != (oldest > first):
            raise self.error(f"{label} retention-gap marker is inconsistent")
        rejected = record.integer("rejectedCount", 0, UINT64_MAX)
        complete = record.boolean("complete")
        if complete != (capability.status == "derived" and attested and installed and rejected == 0):
            raise self.error(f"{label} complete marker contradicts capture state")
        raw_events = raw.get("events")
        if not isinstance(raw_events, list) or len(raw_events) > capacity:
            raise self.error(f"{label}.events is invalid")
        if len(raw_events) != next_sequence - oldest:
            raise self.error(f"{label} retained event range is incomplete")
        if capability.status == "unavailable" and raw_events:
            raise self.error(f"unavailable {label} exposed event records")
        events = []
        previous_tick = None
        for index, value in enumerate(raw_events):
            event = event_parser(
                value,
                index=index,
                expected_sequence=oldest + index,
                observation_tick=observation_tick,
                previous_tick=previous_tick,
            )
            events.append(event)
            previous_tick = event.tick
        return self.envelope_type(
            generation=wire_generation,
            state_epoch=wire_state_epoch,
            observation_tick=wire_tick,
            capacity=capacity,
            hook_set_attested=attested,
            hook_set_installed=installed,
            capability=capability,
            epoch_first_sequence=first,
            oldest_retained_sequence=oldest,
            next_sequence=next_sequence,
            overflow_count=overflow,
            rejected_count=rejected,
            sequence_gap_before_oldest=gap,
            complete=complete,
            events=tuple(events),
        )


@dataclass(frozen=True, slots=True)
class WireRecord:
    """Bind field names to one error domain and path without repeating them."""

    schema: RuntimeSchema
    raw: Mapping[str, Any]
    label: str

    def integer(self, name: str, minimum=None, maximum=None) -> int:
        return self.schema.integer(self.raw.get(name), f"{self.label}.{name}", minimum=minimum, maximum=maximum)

    def optional_integer(self, name: str, minimum=None, maximum=None) -> int | None:
        return None if self.raw.get(name) is None else self.integer(name, minimum, maximum)

    def boolean(self, name: str) -> bool:
        return self.schema.boolean(self.raw.get(name), f"{self.label}.{name}")

    def event_header(self, expected_sequence, observation_tick, previous_tick):
        sequence = self.integer("sequence", 1, UINT64_MAX)
        if sequence != expected_sequence:
            raise self.schema.error(f"{self.schema.label} event sequences are not contiguous")
        tick = self.integer("tick", 0, INT32_MAX)
        if observation_tick < 0 or tick > observation_tick:
            raise self.schema.error(f"{self.label} occurs after the observation")
        if previous_tick is not None and tick < previous_tick:
            raise self.schema.error(f"{self.schema.label} event ticks are not monotonic")
        return sequence, tick
