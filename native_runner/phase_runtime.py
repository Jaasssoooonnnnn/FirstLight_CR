"""Exact-build non-visual phase/timing semantics for Null's Royale 15.535.13.

The native layer deliberately emits raw facts.  This module performs the
small amount of stateful joining needed for attack phases, Inferno damage
ramp, classic movement charge, deployment gating, and effect-scaled movement
or attack timing.  Every resolver fails closed when its required native hook
or static AttackSequence evidence is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ATTACK_STATE_FIELDS, AttackPhase, AttackStateV1, SemanticEvidenceLevel, SemanticProvenanceV1


PHASE_RUNTIME_SCHEMA = "native-phase-runtime.v1"
EXACT_LIBG_SHA256 = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783"
EXACT_LIBG_BUILD_ID = "90e6f351f0dec4a28b494c3812d2629fd45d5a83"
SIMULATION_TICK_MS = 50
CLASSIC_CHARGE_READY = 10_000
ATTACK_SEQUENCE_PROGRESS_LIMIT = 50
ATTACK_SEQUENCE_DECAY_DURATION_MS = 7_000
NATIVE_OBJECT_ID_ENTITY_KEY_TAG = -2


class PhaseRuntimeError(ValueError):
    """Raised when raw phase telemetry contradicts the exact-build contract."""


def _nonnegative_fields(record, names: Sequence[str], context: str, *, optional: bool = True) -> None:
    for name in names:
        value = getattr(record, name)
        if optional and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PhaseRuntimeError(f"{context} {name} is invalid")


def _mapping_fields(record, camel_case: bool = False):
    # Version is always the first wire field, independent of dataclass order.
    for item in sorted(fields(record), key=lambda item: item.name != "version"):
        parts = item.name.split("_")
        wire = parts[0] + "".join(part.title() for part in parts[1:])
        yield item.name, wire if camel_case else item.name


def _runtime_mapping(record, *, camel_case: bool = False) -> Mapping[str, Any]:
    values = {}
    for name, wire in _mapping_fields(record, camel_case):
        value = getattr(record, name)
        values[wire] = value.value if name in {"phase", "classic_charge_phase"} else value
    return MappingProxyType(values)


class DamageRampIdentity(str, Enum):
    INFERNO_DRAGON = "inferno_dragon"
    INFERNO_TOWER = "inferno_tower"
    MIGHTY_MINER = "mighty_miner"
    OTHER = "other"


class ClassicChargePhase(str, Enum):
    UNAVAILABLE = "unavailable"
    ACCUMULATING = "accumulating"
    READY = "ready"


class MovementPhase(str, Enum):
    UNAVAILABLE = "unavailable"
    STATIONARY = "stationary"
    MOVING = "moving"
    UNKNOWN = "unknown"


class DeploymentPhase(str, Enum):
    DEPLOYING = "deploying"
    ACTIVE = "active"
    UNKNOWN = "unknown"


class PhaseHookKind(str, Enum):
    ATTACK_START = "attack_start"
    ATTACK_RELEASE = "attack_release"
    EFFECT_APPLY = "effect_apply"
    ATTACK_SCALE = "attack_scale"
    MOVEMENT_SCALE = "movement_scale"
    EFFECTIVE_MOVEMENT_SPEED = "effective_movement_speed"
    MOVEMENT_DELTA = "movement_delta"
    TARGET_RESET = "target_reset"
    CLASSIC_CHARGE_READY = "classic_charge_ready"
    DEPLOY_SCALE = "deploy_scale"


@dataclass(frozen=True, slots=True)
class PhaseRuntimeWindowV1:
    """Cursor-aware retained window for the fixed native phase-event ring."""

    capacity: int
    epoch_first_sequence: int
    oldest_retained_sequence: int
    next_sequence: int
    overflow_count: int
    rejected_count: int
    complete: bool
    version: str = field(default=PHASE_RUNTIME_SCHEMA, init=False)

    def __post_init__(self) -> None:
        _nonnegative_fields(
            self,
            (
                "capacity",
                "epoch_first_sequence",
                "oldest_retained_sequence",
                "next_sequence",
                "overflow_count",
                "rejected_count",
            ),
            "phase ring",
            optional=False,
        )
        if self.capacity <= 0:
            raise PhaseRuntimeError("phase ring capacity must be positive")
        if not isinstance(self.complete, bool):
            raise PhaseRuntimeError("phase ring complete must be boolean")
        if not (self.epoch_first_sequence <= self.oldest_retained_sequence <= self.next_sequence):
            raise PhaseRuntimeError("phase ring sequence window is inconsistent")
        if self.next_sequence - self.oldest_retained_sequence > self.capacity:
            raise PhaseRuntimeError("phase ring retained window exceeds capacity")
        expected_overflow = max(self.next_sequence - self.epoch_first_sequence - self.capacity, 0)
        if self.overflow_count != expected_overflow:
            raise PhaseRuntimeError("phase ring overflow accounting is inconsistent")
        if self.complete and self.rejected_count != 0:
            raise PhaseRuntimeError("phase ring cannot be complete after rejecting native facts")

    def has_unconsumed_gap(self, next_expected_sequence: int) -> bool:
        if (
            isinstance(next_expected_sequence, bool)
            or not isinstance(next_expected_sequence, int)
            or next_expected_sequence < self.epoch_first_sequence
            or next_expected_sequence > self.next_sequence
        ):
            raise PhaseRuntimeError("phase ring host cursor is outside the epoch")
        return next_expected_sequence < self.oldest_retained_sequence

    def usable_from(self, next_expected_sequence: int) -> bool:
        return self.complete and self.rejected_count == 0 and not self.has_unconsumed_gap(next_expected_sequence)

    def to_mapping(self) -> Mapping[str, Any]:
        return _runtime_mapping(self, camel_case=True)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhaseRuntimeWindowV1":
        names = dict(_mapping_fields(cls, camel_case=True))
        if set(value) != set(names.values()) or value.get("version") != PHASE_RUNTIME_SCHEMA:
            raise PhaseRuntimeError("phase ring mapping does not match v1 schema")
        return cls(**{name: value[wire] for name, wire in names.items() if name != "version"})


@dataclass(frozen=True, slots=True)
class EffectScale:
    """The two independent extrema used by native Buff multiplier consumers."""

    positive_percent: int = 100
    negative_magnitude: int = 0

    def __post_init__(self) -> None:
        if self.positive_percent < 0 or self.negative_magnitude < 0:
            raise PhaseRuntimeError("effect scale extrema must be non-negative")

    @property
    def remaining_percent(self) -> int:
        return max(min(100 - self.negative_magnitude, 100), 0)


def combine_effect_multipliers(values: Iterable[int]) -> EffectScale:
    """Match libg+f5b3c8/f5b4ac extrema selection exactly.

    Positive Buff fields do not stack: the largest value wins over the native
    baseline 100.  Negative fields are magnitudes and likewise do not stack:
    the most negative value wins.  Positive and negative extrema then compose.
    """

    positive = 100
    negative = 0
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise PhaseRuntimeError("effect multipliers must be integers")
        if raw >= 1:
            positive = max(positive, raw)
        elif raw < 0:
            negative = max(negative, -raw)
    return EffectScale(positive_percent=positive, negative_magnitude=negative)


def scale_effect_value(value: int, multipliers: Iterable[int] | EffectScale) -> int:
    """Apply the native two-stage integer division without float drift."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhaseRuntimeError("scaled native value must be a non-negative integer")
    scale = multipliers if isinstance(multipliers, EffectScale) else combine_effect_multipliers(multipliers)
    first = value * scale.positive_percent // 100
    return first * scale.remaining_percent // 100


def wall_time_for_native_progress(
    native_remaining: int, native_step_per_tick: int, *, tick_ms: int = SIMULATION_TICK_MS
) -> int | None:
    """Convert a native timeline budget to wall/simulation time.

    A zero effective step is an exact paused/frozen timeline, so the remaining
    wall time is intentionally unknown instead of being reported as infinity
    or zero.
    """

    for name, value in (
        ("native_remaining", native_remaining),
        ("native_step_per_tick", native_step_per_tick),
        ("tick_ms", tick_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PhaseRuntimeError(f"{name} must be an integer")
    if native_remaining < 0 or native_step_per_tick < 0 or tick_ms <= 0:
        raise PhaseRuntimeError("native timing values are outside their valid range")
    if native_remaining == 0:
        return 0
    if native_step_per_tick == 0:
        return None
    ticks = (native_remaining + native_step_per_tick - 1) // native_step_per_tick
    return ticks * tick_ms


def classic_charge_phase(progress: int | None) -> ClassicChargePhase:
    if progress is None or progress == -1:
        return ClassicChargePhase.UNAVAILABLE
    if isinstance(progress, bool) or not isinstance(progress, int) or progress < -1:
        raise PhaseRuntimeError("classic charge progress must be -1 or non-negative")
    return ClassicChargePhase.READY if progress >= CLASSIC_CHARGE_READY else ClassicChargePhase.ACCUMULATING


def classic_charge_increment(accepted_movement_delta: int, charge_range: int) -> int:
    if (
        isinstance(accepted_movement_delta, bool)
        or not isinstance(accepted_movement_delta, int)
        or isinstance(charge_range, bool)
        or not isinstance(charge_range, int)
        or charge_range <= 0
    ):
        raise PhaseRuntimeError("classic charge delta/range is invalid")
    return max(accepted_movement_delta, 0) * 1_000 // charge_range


@dataclass(frozen=True, slots=True)
class DamageRampSpec:
    identity: DamageRampIdentity
    stage_durations_ms: tuple[int, ...]
    stage_damage: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", DamageRampIdentity(self.identity))
        object.__setattr__(self, "stage_durations_ms", tuple(self.stage_durations_ms))
        object.__setattr__(self, "stage_damage", tuple(self.stage_damage))
        if (
            not self.stage_durations_ms
            or len(self.stage_durations_ms) != len(self.stage_damage)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.stage_durations_ms
            )
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.stage_damage)
        ):
            raise PhaseRuntimeError("damage-ramp stages must be aligned positive data")

    @property
    def is_inferno(self) -> bool:
        return self.identity in {DamageRampIdentity.INFERNO_DRAGON, DamageRampIdentity.INFERNO_TOWER}

    @property
    def is_continuous_lock_ramp(self) -> bool:
        return self.is_inferno or self.identity == DamageRampIdentity.MIGHTY_MINER


@dataclass(frozen=True, slots=True)
class DamageRampRuntime:
    identity: DamageRampIdentity
    stage: int
    stage_elapsed_native_ms: int
    next_stage_remaining_native_ms: int | None
    next_stage_remaining_wall_ms: int | None
    damage: int
    damage_multiplier: float | None
    locked: bool


def stage_from_durations(timeline_ms: int, durations_ms: Sequence[int]) -> int:
    """Match libg+d99404, including promotion exactly on a duration boundary."""

    if isinstance(timeline_ms, bool) or not isinstance(timeline_ms, int) or timeline_ms < 0 or not durations_ms:
        raise PhaseRuntimeError("AttackSequence timeline/durations are invalid")
    remaining = timeline_ms
    for index, duration in enumerate(durations_ms):
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            raise PhaseRuntimeError("AttackSequence duration must be positive")
        remaining -= duration
        if remaining < 0:
            return index
    return len(durations_ms) - 1


def resolve_damage_ramp(
    *,
    raw_stage: int | None,
    timeline_ms: int | None,
    target_locked: bool,
    attack_step_native_ms: int | None,
    spec: DamageRampSpec | None,
) -> DamageRampRuntime | None:
    """Promote +0x20 only for a named continuous-lock damage ramp."""

    if spec is None or not spec.is_continuous_lock_ramp:
        return None
    if raw_stage is None or timeline_ms is None:
        return None
    if (
        isinstance(raw_stage, bool)
        or not isinstance(raw_stage, int)
        or raw_stage < 0
        or raw_stage >= len(spec.stage_durations_ms)
        or isinstance(timeline_ms, bool)
        or not isinstance(timeline_ms, int)
        or timeline_ms < 0
    ):
        raise PhaseRuntimeError("native damage-ramp stage/timeline is invalid")
    calculated = stage_from_durations(timeline_ms, spec.stage_durations_ms)
    if calculated != raw_stage:
        raise PhaseRuntimeError("native AttackSequence stage contradicts the joined stage durations")
    stage_start = sum(spec.stage_durations_ms[:raw_stage])
    elapsed = timeline_ms - stage_start
    next_native: int | None
    next_wall: int | None = None
    if raw_stage + 1 < len(spec.stage_durations_ms):
        next_native = spec.stage_durations_ms[raw_stage] - elapsed
        if next_native <= 0:
            raise PhaseRuntimeError("damage-ramp stage should already have advanced")
        if attack_step_native_ms is not None:
            next_wall = wall_time_for_native_progress(next_native, attack_step_native_ms)
    else:
        next_native = None
    base_damage = spec.stage_damage[0]
    damage = spec.stage_damage[raw_stage]
    multiplier = None if base_damage == 0 else damage / base_damage
    return DamageRampRuntime(
        identity=spec.identity,
        stage=raw_stage,
        stage_elapsed_native_ms=elapsed,
        next_stage_remaining_native_ms=next_native,
        next_stage_remaining_wall_ms=next_wall,
        damage=damage,
        damage_multiplier=multiplier,
        locked=bool(target_locked),
    )


@dataclass(frozen=True, slots=True)
class PhaseHookEvent:
    sequence: int
    tick: int
    kind: PhaseHookKind
    entity_key: tuple[int, int, int]
    target_key: tuple[int, int, int] | None = None
    hook_offset: int | None = None
    caller_offset: int | None = None
    success: bool | None = None
    buff_global_id: int | None = None
    buff_remaining_ms: int | None = None
    speed_multiplier: int | None = None
    hit_speed_multiplier: int | None = None
    input_step: int | None = None
    output_step: int | None = None
    timeline_before: int | None = None
    timeline_after: int | None = None
    classic_charge_before: int | None = None
    classic_charge_after: int | None = None
    target_present_before: bool | None = None
    target_present_after: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PhaseHookKind(self.kind))
        object.__setattr__(self, "entity_key", _entity_key(self.entity_key))
        if self.target_key is not None:
            object.__setattr__(self, "target_key", _entity_key(self.target_key))
        if self.sequence <= 0 or self.tick < 0:
            raise PhaseRuntimeError("phase hook sequence/tick is invalid")
        _nonnegative_fields(
            self,
            ("hook_offset", "caller_offset", "buff_global_id", "output_step", "timeline_before", "timeline_after"),
            "phase hook",
            optional=True,
        )
        if self.input_step is not None and (isinstance(self.input_step, bool) or not isinstance(self.input_step, int)):
            raise PhaseRuntimeError("phase hook input_step is invalid")
        if self.buff_remaining_ms is not None and self.buff_remaining_ms < -1:
            raise PhaseRuntimeError("phase hook Buff remaining time is invalid")
        for name in ("classic_charge_before", "classic_charge_after"):
            value = getattr(self, name)
            if value is not None:
                classic_charge_phase(value)

    @property
    def successful_release(self) -> bool:
        return self.kind == PhaseHookKind.ATTACK_RELEASE and self.success is True

    @property
    def full_attack_stop_effect(self) -> bool:
        return (
            self.kind == PhaseHookKind.EFFECT_APPLY
            and self.hit_speed_multiplier is not None
            and self.hit_speed_multiplier <= -100
        )

    @property
    def attack_timeline_blocked(self) -> bool:
        return (
            self.kind == PhaseHookKind.ATTACK_SCALE
            and self.input_step is not None
            and self.input_step > 0
            and self.output_step == 0
        )

    @property
    def destructive_attack_reset(self) -> bool:
        return (
            self.kind == PhaseHookKind.TARGET_RESET
            and self.timeline_before is not None
            and self.timeline_before > 0
            and self.timeline_after == 0
        )


@dataclass(frozen=True, slots=True)
class ConfirmedAttackInterrupt:
    entity_key: tuple[int, int, int]
    tick: int
    effect_sequence: int
    effect_global_id: int
    timeline_pause_confirmed: bool
    destructive_reset_confirmed: bool


def correlate_attack_interrupts(events: Iterable[PhaseHookEvent]) -> tuple[ConfirmedAttackInterrupt, ...]:
    """Join start + full-stop Buff + exact consumer/reset facts.

    Ice Wizard's -30 HitSpeedMultiplier does not satisfy this predicate.  An
    Electro Wizard/Zap-style full stop is reported as an interrupt occurrence
    only if f5b4ac actually returns zero in the same tick.  A separate flag
    records whether f5c894 also proved destructive attack-state reset.
    """

    ordered = sorted(events, key=lambda item: item.sequence)
    pending_start: dict[tuple[int, int, int], PhaseHookEvent] = {}
    results: list[ConfirmedAttackInterrupt] = []
    for event in ordered:
        key = event.entity_key
        if event.kind == PhaseHookKind.ATTACK_START:
            pending_start[key] = event
            continue
        if event.successful_release:
            pending_start.pop(key, None)
            continue
        if not event.full_attack_stop_effect or key not in pending_start:
            continue
        same_tick = [candidate for candidate in ordered if candidate.entity_key == key and candidate.tick == event.tick]
        paused = any(candidate.attack_timeline_blocked for candidate in same_tick)
        if not paused or event.buff_global_id is None:
            continue
        reset = any(candidate.destructive_attack_reset for candidate in same_tick)
        results.append(
            ConfirmedAttackInterrupt(
                entity_key=key,
                tick=event.tick,
                effect_sequence=event.sequence,
                effect_global_id=event.buff_global_id,
                timeline_pause_confirmed=True,
                destructive_reset_confirmed=reset,
            )
        )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class AttackRuntimeRaw:
    entity_key: tuple[int, int, int]
    tick: int
    target_validated: bool
    target_entity: int | None
    attack_sequence_stage: int | None
    attack_timeline_ms: int | None
    load_remaining_ms: int | None
    hit_speed_ms: int | None
    attack_dash_time_ms: int | None
    attack_step_native_ms: int | None
    hook_set_attested: bool
    attack_sequence_progress: int | None = None
    attack_sequence_progress_limit: int | None = None
    attack_sequence_decay_remaining_ms: int | None = None
    attack_sequence_decay_duration_ms: int | None = None
    events: tuple[PhaseHookEvent, ...] = ()
    damage_ramp_spec: DamageRampSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_key", _entity_key(self.entity_key))
        object.__setattr__(self, "events", tuple(self.events))
        if self.tick < 0:
            raise PhaseRuntimeError("attack snapshot tick must be non-negative")
        _nonnegative_fields(
            self,
            (
                "target_entity",
                "attack_sequence_stage",
                "attack_sequence_progress",
                "attack_sequence_progress_limit",
                "attack_sequence_decay_remaining_ms",
                "attack_sequence_decay_duration_ms",
                "attack_timeline_ms",
                "load_remaining_ms",
                "hit_speed_ms",
                "attack_dash_time_ms",
                "attack_step_native_ms",
            ),
            "attack snapshot",
            optional=True,
        )
        for amount, limit, label, bound in (
            (self.attack_sequence_progress, self.attack_sequence_progress_limit, "progress", "limit"),
            (self.attack_sequence_decay_remaining_ms, self.attack_sequence_decay_duration_ms, "decay", "duration"),
        ):
            if (amount is None) != (limit is None):
                raise PhaseRuntimeError(f"attack sequence {label} pair is incomplete")
            if limit is not None and (limit <= 0 or amount > limit):
                raise PhaseRuntimeError(f"attack sequence {label} is outside its {bound}")


@dataclass(frozen=True, slots=True)
class AttackTimingRuntime:
    load_remaining_native_ms: int | None
    load_remaining_wall_ms: int | None
    release_remaining_native_ms: int | None
    release_remaining_wall_ms: int | None
    attack_step_native_ms: int | None


@dataclass(frozen=True, slots=True)
class AttackRuntimeProjection:
    state: AttackStateV1
    timing: AttackTimingRuntime
    damage_ramp: DamageRampRuntime | None
    interrupt: ConfirmedAttackInterrupt | None


def resolve_attack_runtime(raw: AttackRuntimeRaw) -> AttackRuntimeProjection:
    events = tuple(event for event in raw.events if event.entity_key == raw.entity_key)
    ordered = sorted(events, key=lambda item: item.sequence)
    interrupts = correlate_attack_interrupts(ordered)
    latest_interrupt = interrupts[-1] if interrupts else None

    latest_start = latest_release = latest_reset = None
    for event in ordered:
        if event.kind == PhaseHookKind.ATTACK_START:
            latest_start = event
        if event.successful_release:
            latest_release = event
        if event.destructive_attack_reset:
            latest_reset = event
    start_pending = latest_start is not None and all(
        item is None or item.sequence < latest_start.sequence for item in (latest_release, latest_reset)
    )

    phase = AttackPhase.UNKNOWN
    phase_started_tick: int | None = None
    interrupted: bool | None = None
    phase_notes: list[str] = []
    if not raw.hook_set_attested:
        phase_notes.append("phase hooks are not exact-build attested")
    elif (
        latest_interrupt is not None
        and latest_interrupt.tick == raw.tick
        and latest_start is not None
        and latest_interrupt.effect_sequence > latest_start.sequence
    ):
        phase = AttackPhase.INTERRUPTED
        phase_started_tick = latest_interrupt.tick
        interrupted = True
        phase_notes.append("full-stop Buff and f5b4ac zero-step consumer joined in one tick")
        if latest_interrupt.destructive_reset_confirmed:
            phase_notes.append("f5c894 also confirmed destructive timeline reset")
        else:
            phase_notes.append("timeline pause is exact; destructive cancellation was not observed")
    elif latest_release is not None and latest_release.tick == raw.tick and latest_release.successful_release:
        phase = AttackPhase.RELEASE
        phase_started_tick = raw.tick
        interrupted = False
    elif start_pending:
        phase = AttackPhase.WINDUP
        phase_started_tick = latest_start.tick
        interrupted = False
    elif raw.load_remaining_ms is not None and raw.load_remaining_ms > 0:
        phase = AttackPhase.COOLDOWN
        interrupted = False
        phase_notes.append("phase is the exact +0x28 LoadTime gate")
    elif raw.target_validated and raw.target_entity is None and raw.attack_timeline_ms == 0:
        phase = AttackPhase.IDLE
        interrupted = False
    elif raw.hook_set_attested:
        phase_notes.append("snapshot cannot distinguish backswing/channel/cooldown without a hook edge")

    load_wall = (
        wall_time_for_native_progress(raw.load_remaining_ms, SIMULATION_TICK_MS)
        if raw.load_remaining_ms is not None
        else None
    )
    release_native: int | None = None
    release_wall: int | None = None
    if (
        phase == AttackPhase.WINDUP
        and raw.attack_timeline_ms is not None
        and raw.hit_speed_ms is not None
        and raw.hit_speed_ms > 0
        and raw.attack_dash_time_ms is not None
    ):
        remainder = (raw.attack_timeline_ms + raw.attack_dash_time_ms) % raw.hit_speed_ms
        release_native = raw.hit_speed_ms - remainder
        if raw.attack_step_native_ms is not None:
            release_wall = wall_time_for_native_progress(release_native, raw.attack_step_native_ms)

    try:
        ramp = resolve_damage_ramp(
            raw_stage=raw.attack_sequence_stage,
            timeline_ms=raw.attack_timeline_ms,
            target_locked=raw.target_validated and raw.target_entity is not None,
            attack_step_native_ms=raw.attack_step_native_ms,
            spec=raw.damage_ramp_spec,
        )
    except PhaseRuntimeError as error:
        if str(error) not in {
            "native AttackSequence stage contradicts the joined stage durations",
            "damage-ramp stage should already have advanced",
        }:
            raise
        # The native sequence index can legitimately survive a target/timeline
        # reset for one snapshot.  Keep that authoritative raw field, but do
        # not invent derived charge timing from a temporarily incoherent pair.
        ramp = None
        phase_notes.append(
            "damage-ramp stage/timeline were transiently incoherent; derived charge fields were withheld"
        )

    evidence = {name: SemanticEvidenceLevel.UNKNOWN for name in ATTACK_STATE_FIELDS}
    sources: dict[str, tuple[str, ...]] = {}
    values: dict[str, Any] = {}

    def publish(name, value, level, *origin, known=None):
        values[name] = value
        if known is None:
            known = value is not None
        if known:
            evidence[name] = level
            sources[name] = origin

    authoritative = SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
    derived = SemanticEvidenceLevel.NATIVE_DERIVED
    publish(
        "target_entity",
        raw.target_entity,
        authoritative,
        "phaseSnapshot.targetEntity",
        known=bool(raw.target_validated),
    )
    publish("sequence_index", raw.attack_sequence_stage, authoritative, "type0+0x20")
    publish(
        "sequence_progress", raw.attack_sequence_progress, authoritative, "LogicGameObject+0xc8 runtime variable map"
    )
    publish(
        "sequence_progress_limit",
        raw.attack_sequence_progress_limit,
        authoritative,
        "typed ActionSetVariable min bound",
    )
    publish(
        "sequence_decay_remaining_ms",
        raw.attack_sequence_decay_remaining_ms,
        authoritative,
        "LogicGameObject+0xc8 runtime variable map",
    )
    publish(
        "sequence_decay_duration_ms",
        raw.attack_sequence_decay_duration_ms,
        authoritative,
        "typed LogicVariableData fallback",
    )
    publish(
        "phase",
        phase,
        derived,
        "phaseRuntime.events[]",
        "type0 snapshot",
        known=bool(raw.hook_set_attested and phase != AttackPhase.UNKNOWN),
    )
    publish("phase_started_tick", phase_started_tick, derived, "phaseRuntime.events[].tick")
    publish(
        "phase_remaining_ms",
        release_wall if phase == AttackPhase.WINDUP else None,
        derived,
        "type0+0x24",
        "LogicCharacterData+0x418/+0x6a0",
        "f5b4ac return",
    )
    publish("cooldown_remaining_ms", load_wall if phase == AttackPhase.COOLDOWN else None, derived, "type0+0x28")
    publish("interrupted", interrupted, derived, "phaseRuntime.events[]")
    for name, attribute, origin in (
        ("charge_stage", "stage", ("type0+0x20", "named static AttackSequence")),
        ("charge_elapsed_ms", "stage_elapsed_native_ms", ("type0+0x24", "AttackSequence entry+0xa0")),
        ("damage_multiplier", "damage_multiplier", ("named static AttackSequence damage",)),
        ("locked", "locked", ("type0+0x10",)),
    ):
        publish(name, getattr(ramp, attribute) if ramp is not None else None, derived, *origin, known=ramp is not None)

    state = AttackStateV1(
        **values,
        provenance=SemanticProvenanceV1(
            field_evidence=evidence, source_fields=sources, observed_tick=raw.tick, notes=tuple(phase_notes)
        ),
    )
    return AttackRuntimeProjection(
        state=state,
        timing=AttackTimingRuntime(
            load_remaining_native_ms=raw.load_remaining_ms,
            load_remaining_wall_ms=load_wall,
            release_remaining_native_ms=release_native,
            release_remaining_wall_ms=release_wall,
            attack_step_native_ms=raw.attack_step_native_ms,
        ),
        damage_ramp=ramp,
        interrupt=latest_interrupt,
    )


@dataclass(frozen=True, slots=True)
class MovementRuntimeRaw:
    entity_key: tuple[int, int, int]
    tick: int
    component_validated: bool
    hook_set_attested: bool
    movement_delta: int | None
    effective_speed: int | None
    speed_input: int | None
    speed_after_effects: int | None
    classic_charge_progress: int | None
    classic_charge_speed_multiplier: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_key", _entity_key(self.entity_key))
        if self.tick < 0:
            raise PhaseRuntimeError("movement snapshot tick must be non-negative")
        _nonnegative_fields(
            self,
            ("effective_speed", "speed_input", "speed_after_effects", "classic_charge_speed_multiplier"),
            "movement snapshot",
            optional=True,
        )
        classic_charge_phase(self.classic_charge_progress)


@dataclass(frozen=True, slots=True)
class MovementRuntimeV1:
    phase: MovementPhase
    effective_speed: int | None
    effect_scaled_speed: int | None
    movement_delta: int | None
    classic_charge_phase: ClassicChargePhase
    classic_charge_progress: int | None
    classic_charge_speed_multiplier: int | None
    observed_tick: int
    complete: bool
    version: str = field(default="movement-runtime.v1", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", MovementPhase(self.phase))
        object.__setattr__(self, "classic_charge_phase", ClassicChargePhase(self.classic_charge_phase))

    def to_mapping(self) -> Mapping[str, Any]:
        return _runtime_mapping(self)


def resolve_movement_runtime(raw: MovementRuntimeRaw) -> MovementRuntimeV1:
    if not raw.component_validated:
        phase = MovementPhase.UNAVAILABLE
    elif raw.hook_set_attested and raw.movement_delta is not None:
        phase = MovementPhase.MOVING if raw.movement_delta > 0 else MovementPhase.STATIONARY
    else:
        phase = MovementPhase.UNKNOWN
    complete = (
        raw.component_validated
        and raw.hook_set_attested
        and raw.movement_delta is not None
        and raw.effective_speed is not None
    )
    return MovementRuntimeV1(
        phase=phase,
        effective_speed=raw.effective_speed,
        effect_scaled_speed=raw.speed_after_effects,
        movement_delta=raw.movement_delta,
        classic_charge_phase=classic_charge_phase(raw.classic_charge_progress),
        classic_charge_progress=raw.classic_charge_progress,
        classic_charge_speed_multiplier=raw.classic_charge_speed_multiplier,
        observed_tick=raw.tick,
        complete=complete,
    )


@dataclass(frozen=True, slots=True)
class DeploymentRuntimeRaw:
    tick: int
    component_validated: bool
    remaining_native_ms: int | None
    previous_remaining_native_ms: int | None
    configured_deploy_time_ms: int | None
    uses_effect_scaled_step: bool | None
    observed_step_native_ms: int | None

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise PhaseRuntimeError("deployment snapshot tick must be non-negative")
        _nonnegative_fields(
            self,
            (
                "remaining_native_ms",
                "previous_remaining_native_ms",
                "configured_deploy_time_ms",
                "observed_step_native_ms",
            ),
            "deployment snapshot",
            optional=True,
        )


@dataclass(frozen=True, slots=True)
class DeploymentRuntimeV1:
    phase: DeploymentPhase
    remaining_native_ms: int | None
    remaining_wall_ms: int | None
    configured_deploy_time_ms: int | None
    observed_step_native_ms: int | None
    observed_tick: int
    complete: bool
    version: str = field(default="deployment-runtime.v1", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", DeploymentPhase(self.phase))

    def to_mapping(self) -> Mapping[str, Any]:
        return _runtime_mapping(self)


def resolve_deployment_runtime(raw: DeploymentRuntimeRaw) -> DeploymentRuntimeV1:
    remaining = raw.remaining_native_ms
    if not raw.component_validated or remaining is None:
        phase = DeploymentPhase.UNKNOWN
        step = None
        wall = None
        complete = False
    else:
        phase = DeploymentPhase.DEPLOYING if remaining > 0 else DeploymentPhase.ACTIVE
        if raw.uses_effect_scaled_step is False:
            step = SIMULATION_TICK_MS
        elif raw.uses_effect_scaled_step is True:
            step = raw.observed_step_native_ms
        else:
            step = None
        wall = wall_time_for_native_progress(remaining, step) if step is not None else (0 if remaining == 0 else None)
        complete = remaining == 0 or step is not None
    return DeploymentRuntimeV1(
        phase=phase,
        remaining_native_ms=remaining,
        remaining_wall_ms=wall,
        configured_deploy_time_ms=raw.configured_deploy_time_ms,
        observed_step_native_ms=step,
        observed_tick=raw.tick,
        complete=complete,
    )


def _entity_key(value: Sequence[int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise PhaseRuntimeError("entity key must contain owner/index/secondary")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise PhaseRuntimeError("entity key values must be integers")
    ordinary = all(item >= 0 for item in result)
    native_id_tagged = result[0] >= 0 and result[1] == NATIVE_OBJECT_ID_ENTITY_KEY_TAG and result[2] > 0
    if not ordinary and not native_id_tagged:
        raise PhaseRuntimeError("entity key must be non-negative or use the exact native-ID tag")
    return result  # type: ignore[return-value]
