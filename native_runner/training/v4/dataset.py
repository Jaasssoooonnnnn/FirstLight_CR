"""Streaming access and cheap preflight checks for RoyaleAPI IL replays."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS


@dataclass(frozen=True, slots=True)
class ReplayPreflightV4:
    replay_tag: str
    action_count: int
    early_action_count: int
    overflow_window_count: int
    maximum_actions_in_window: int
    data_i: int | None

    @property
    def accepted(self) -> bool:
        return (
            self.action_count > 0
            and self.early_action_count == 0
            and self.overflow_window_count == 0
            and self.data_i in (0, 1)
        )

    @property
    def rejection_reason(self) -> str | None:
        if self.action_count == 0:
            return "no_expert_actions"
        if self.data_i not in (0, 1):
            return "missing_or_mixed_data_i"
        if self.early_action_count:
            return "action_before_first_decision"
        if self.overflow_window_count:
            return "more_than_two_actions_in_window"
        return None


def screen_replay_payload(
    payload: Mapping[str, Any],
    *,
    first_decision_tick: int = FIRST_POLICY_DECISION_TICK,
    decision_ticks: int = POLICY_DECISION_TICKS,
) -> ReplayPreflightV4:
    """Reject unrepresentable timelines before launching a battle engine."""

    source = payload.get("source")
    source_tag = source.get("replay_tag") if isinstance(source, Mapping) else None
    replay_tag = str(payload.get("replay_tag") or payload.get("replayTag") or source_tag or "")
    raw_events = payload.get("events", ())
    if not isinstance(raw_events, list):
        raw_events = ()
    windows: dict[tuple[str, int], int] = {}
    data_i_values: set[int] = set()
    action_count = 0
    early = 0
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("kind") not in {"play_card", "activate_ability"}:
            continue
        tick = raw.get("replay_tick_20hz")
        side = str(raw.get("side") or "")
        if type(tick) is not int or side not in {"team", "opponent"}:
            continue
        action_count += 1
        source_fields = raw.get("source_fields")
        if isinstance(source_fields, Mapping) and type(source_fields.get("data_i")) is int:
            data_i_values.add(int(source_fields["data_i"]))
        if tick < first_decision_tick:
            early += 1
            continue
        decision_tick = first_decision_tick + ((tick - first_decision_tick) // decision_ticks) * decision_ticks
        key = (side, decision_tick)
        windows[key] = windows.get(key, 0) + 1
    counts = tuple(windows.values())
    data_i = next(iter(data_i_values)) if len(data_i_values) == 1 else None
    return ReplayPreflightV4(
        replay_tag=replay_tag,
        action_count=action_count,
        early_action_count=early,
        overflow_window_count=sum(value > 2 for value in counts),
        maximum_actions_in_window=max(counts, default=0),
        data_i=data_i,
    )


class ParquetReplayStreamV4:
    """Yield every source replay and its preflight report for cache partitioning."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)

    def __iter__(self) -> Iterator[tuple[dict[str, Any], ReplayPreflightV4]]:
        try:
            import pyarrow.dataset as arrow_dataset
        except ImportError as error:
            raise RuntimeError("PyArrow is required for Parquet replay streaming") from error
        replay_root = self.dataset_root / "replays"
        if not replay_root.is_dir():
            raise FileNotFoundError(f"missing replay Parquet directory: {replay_root}")
        scanner = arrow_dataset.dataset(replay_root, format="parquet").scanner(
            columns=["replay_tag", "payload_json"], batch_size=256, use_threads=True
        )
        for record_batch in scanner.to_batches():
            payloads = record_batch.column("payload_json").to_pylist()
            for raw_payload in payloads:
                payload = json.loads(str(raw_payload))
                if not isinstance(payload, dict):
                    continue
                report = screen_replay_payload(payload)
                yield payload, report
