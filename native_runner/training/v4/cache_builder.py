"""Build complete, partitionable V4 IL caches before training starts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
from itertools import islice
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from ...cr_native_env import NativeClashEnv
from ...royaleapi_replay import calibrate_collected_replay_deal, prepare_collected_replay
from .cache import (
    assert_il_sequences_exact,
    load_il_cache_index,
    load_il_cache_shard,
    write_il_cache_index,
    write_il_cache_shard,
)
from .config import ModelConfigV4
from .dataset import ParquetReplayStreamV4, ReplayPreflightV4
from .factory import build_episode_tensorizers_v4, build_resident_collected_replay_batch_v4
from .expert import replay_expert_actions, require_accepted_timeline, screen_expert_timeline
from .producer import ReplayProductionResultV4, produce_il_replay_batch


IndexedPayload = tuple[int, Mapping[str, Any]]
IndexedBatch = tuple[int, tuple[IndexedPayload, ...]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[2]
    # Bind the concrete code and competitive facts, including shared schemas.
    # A fixed filename inventory misses changes when implementations are split.
    paths = [
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and "tests" not in path.parts
        and not path.name.startswith("test_")
        and path.suffix.lower() in {".py", ".cpp", ".h", ".inc", ".s"}
    ]
    paths.extend((package_root / "data" / "competitive").glob("*.json"))
    return {path.relative_to(package_root).as_posix(): _sha256_file(path) for path in sorted(paths)}


def _dataset_snapshot(dataset_root: Path) -> dict[str, object]:
    replay_root = dataset_root / "replays"
    files = tuple(sorted(replay_root.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"no replay Parquet files under {replay_root}")
    rows = [{"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in files]
    return {"root": str(dataset_root.resolve()), "replay_files": rows, "snapshot_sha256": _canonical_sha256(rows)}


def _runtime_reports(paths: Sequence[Path]) -> list[dict[str, object]]:
    return [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path)} for path in paths]


def _headless_calibration_probe(native: Any) -> Any:
    """Expose the stock-deal calibration API on the semantic headless engine."""

    class HeadlessCalibrationProbe:
        def create_native_match(self, config: Any, *, wait_timeout: float = 30.0) -> Mapping[str, Any]:
            del wait_timeout
            return native.create_match(config)

        @staticmethod
        def pause() -> Mapping[str, Any]:
            return {"ok": True, "paused": True, "mode": "headless"}

        def __getattr__(self, name: str) -> Any:
            return getattr(native, name)

    return HeadlessCalibrationProbe()


def _failed_replay_result(
    payload: Mapping[str, Any], prepared: Any | None, *, phase: str, error: BaseException
) -> ReplayProductionResultV4:
    if prepared is None:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        source = payload.get("source")
        source_tag = source.get("replay_tag") if isinstance(source, Mapping) else None
        replay_tag = str(
            payload.get("replay_tag")
            or payload.get("replayTag")
            or source_tag
            or f"unprepared-{hashlib.sha256(encoded).hexdigest()[:16]}"
        )
        source_end_tick = 0
        expert_action_count = 0
    else:
        replay_tag = str(prepared.replay_tag)
        source_end_tick = int(prepared.replay.end_native_tick)
        try:
            expert_action_count = len(replay_expert_actions(prepared.replay))
        except Exception:
            expert_action_count = 0
    return ReplayProductionResultV4(
        replay_tag=replay_tag,
        sequences=None,
        completed=False,
        failure_reason=f"{phase}: {type(error).__name__}: {error}",
        first_untrusted_tick=0,
        simulated_end_tick=0,
        source_end_tick=source_end_tick,
        expert_action_count=expert_action_count,
        executed_expert_action_count=0,
        winner_matches=None,
        crowns_match=None,
        decision_frame_count=0,
    )


def _failure_payload(shard_id: str, results: Sequence[ReplayProductionResultV4]) -> dict[str, object]:
    return {
        "schema": "cr-v4-il-cache-failures.v1",
        "shard_id": shard_id,
        "results": [
            {
                "replay_tag": result.replay_tag,
                "failure_reason": result.failure_reason,
                "first_untrusted_tick": result.first_untrusted_tick,
                "simulated_end_tick": result.simulated_end_tick,
                "source_end_tick": result.source_end_tick,
                "expert_action_count": result.expert_action_count,
                "executed_expert_action_count": result.executed_expert_action_count,
                "decision_frame_count": result.decision_frame_count,
            }
            for result in results
        ],
    }


def _is_retryable_result(result: ReplayProductionResultV4) -> bool:
    if result.sequences is not None or not result.failure_reason:
        return False
    lowered = result.failure_reason.lower()
    return any(
        marker in lowered
        for marker in (
            "broken pipe",
            "connection aborted",
            "connection refused",
            "connection reset",
            "eof",
            "resident batch",
            "socket",
            "timed out",
            "timeout",
        )
    )


def _convert_batch(
    *,
    native: NativeClashEnv,
    payload_batch: Sequence[IndexedPayload],
    host: str,
    port: int,
    gamma: float,
    max_frames: int | None,
    timeout: float,
    resident_slots: int,
    capture_mode: str,
) -> tuple[ReplayProductionResultV4, ...]:
    # Deal calibration uses singleton commands. Restore that surface after a
    # previous shard's resident session before calibrating the next shard.
    native.stop_resident_mode()
    calibration_probe = _headless_calibration_probe(native)
    by_source_index: dict[int, ReplayProductionResultV4] = {}
    prepared_rows: list[tuple[int, Any, Any]] = []
    payload_by_source = dict(payload_batch)
    for source_index, payload in payload_batch:
        prepared = None
        phase = "prepare"
        try:
            prepared = prepare_collected_replay(payload)
            phase = "calibrate"
            calibrated = calibrate_collected_replay_deal(prepared, calibration_probe)
            phase = "screen"
            expert_actions = replay_expert_actions(calibrated.replay)
            require_accepted_timeline(screen_expert_timeline(expert_actions), replay_id=calibrated.replay_tag)
            phase = "tensorizers"
            tensorizers = build_episode_tensorizers_v4(calibrated.replay.episode_config)
            prepared_rows.append((source_index, calibrated, tensorizers))
        except Exception as error:
            by_source_index[source_index] = _failed_replay_result(payload, prepared, phase=phase, error=error)

    if prepared_rows:
        initial_count = min(resident_slots, len(prepared_rows))
        initial = prepared_rows[:initial_count]
        replacements = prepared_rows[initial_count:]
        environments: tuple[Any, ...] = ()
        try:
            environments, coordinator = build_resident_collected_replay_batch_v4(
                tuple(item[1] for item in initial),
                env_ids=tuple(range(initial_count)),
                host=host,
                port=port,
                timeout=timeout,
                capture_mode=capture_mode,
            )
            produced = produce_il_replay_batch(
                environments,
                tuple(item[1] for item in initial),
                tuple(item[2] for item in initial),
                gamma_per_decision=gamma,
                batch_coordinator=coordinator,
                replacement_prepared_replays=tuple(item[1] for item in replacements),
                replacement_tensorizers_by_replay=tuple(item[2] for item in replacements),
                max_frames=max_frames,
                validate_tensors=False,
            )
            if len(produced) != len(prepared_rows):
                raise RuntimeError(f"resident producer returned {len(produced)} of {len(prepared_rows)} replay results")
            for (source_index, _prepared, _tensorizers), result in zip(prepared_rows, produced, strict=True):
                by_source_index[source_index] = result
        except Exception as error:
            for source_index, prepared, _tensorizers in prepared_rows:
                by_source_index[source_index] = _failed_replay_result(
                    payload_by_source[source_index], prepared, phase="produce", error=error
                )
        finally:
            for environment in environments:
                try:
                    environment.native.close()
                except Exception:
                    pass

    if set(by_source_index) != {item[0] for item in payload_batch}:
        raise RuntimeError("cache conversion lost one or more source rows")
    return tuple(by_source_index[index] for index, _payload in payload_batch)


def _load_resumable_summary(summary_path: Path, *, expected_replays: int) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"invalid existing cache summary {summary_path}")
    data_file = summary.get("data_file")
    if (
        not isinstance(data_file, str)
        or int(summary.get("replay_results", -1)) != expected_replays
        or not (summary_path.parent / data_file).is_file()
    ):
        raise ValueError(f"incomplete existing cache shard {summary_path}")
    return summary


def _sequence_to_storage(sequence: Any) -> Any:
    return replace(
        sequence,
        observations=tuple(
            observation.to_storage("cpu", float_dtype=torch.float16) for observation in sequence.observations
        ),
    )


def _worker(
    *,
    worker_index: int,
    partition_index: int,
    host: str,
    port: int,
    batches: Sequence[IndexedBatch],
    cache_root: Path,
    contract: Mapping[str, object],
    gamma: float,
    max_frames: int | None,
    compression_level: int,
    verify_roundtrip: bool,
    timeout: float,
    resident_slots: int,
    capture_mode: str,
    infrastructure_retries: int,
    resume: bool,
) -> tuple[dict[str, object], ...]:
    """Convert one engine endpoint's deterministic shard assignment."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    summaries: list[dict[str, object]] = []
    native = NativeClashEnv(host=host, port=port, timeout=timeout)
    native.wait_ready(timeout=timeout)
    worker_started = time.perf_counter()
    try:
        for local_index, (shard_index, payload_batch) in enumerate(batches, start=1):
            shard_id = f"part-{partition_index:02d}-shard-{shard_index:06d}"
            data_path = cache_root / f"{shard_id}.cril.zst"
            summary_path = data_path.with_suffix(".json")
            existing = _load_resumable_summary(summary_path, expected_replays=len(payload_batch)) if resume else None
            if existing is not None:
                summary = dict(existing)
                summary["summary_file"] = summary_path.name
                summary["port"] = port
                summary["host"] = host
                summaries.append(summary)
                continue

            results: tuple[ReplayProductionResultV4, ...] = ()
            for attempt in range(infrastructure_retries + 1):
                results = _convert_batch(
                    native=native,
                    payload_batch=payload_batch,
                    host=host,
                    port=port,
                    gamma=gamma,
                    max_frames=max_frames,
                    timeout=timeout,
                    resident_slots=resident_slots,
                    capture_mode=capture_mode,
                )
                if any(result.sequences is not None for result in results):
                    break
                if attempt >= infrastructure_retries or not any(_is_retryable_result(result) for result in results):
                    break
                time.sleep(min(5.0, float(2**attempt)))

            if not any(result.sequences is not None for result in results):
                failure_path = cache_root / f"{shard_id}.failures.json"
                failure_path.write_text(
                    json.dumps(_failure_payload(shard_id, results), ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "event": "cache-shard-rejected",
                            "partition": partition_index,
                            "worker": worker_index,
                            "port": port,
                            "shard": shard_id,
                            "rejected_replays": len(results),
                            "failure_file": failure_path.name,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            summary = write_il_cache_shard(
                data_path, results, shard_id=shard_id, contract=contract, compression_level=compression_level
            )
            summary["summary_file"] = summary_path.name
            summary["port"] = port
            summary["host"] = host
            if verify_roundtrip:
                loaded = load_il_cache_shard(summary_path)
                sequence_index = 0
                for result in results:
                    if result.sequences is None:
                        continue
                    for sequence in result.sequences:
                        assert_il_sequences_exact(_sequence_to_storage(sequence), loaded.sequence(sequence_index))
                        sequence_index += 1
                if sequence_index != len(loaded.descriptors):
                    raise AssertionError("cache round-trip sequence count differs")
            summaries.append(summary)
            elapsed = time.perf_counter() - worker_started
            print(
                json.dumps(
                    {
                        "event": "cache-shard",
                        "partition": partition_index,
                        "worker": worker_index,
                        "port": port,
                        "shard": shard_id,
                        "worker_shards_done": local_index,
                        "worker_shards_total": len(batches),
                        "worker_elapsed_s": round(elapsed, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        try:
            native.stop_resident_mode()
        except Exception:
            pass
    return tuple(summaries)


def _partition_payloads(
    payloads: Sequence[IndexedPayload], *, ports: Sequence[int], replays_per_shard: int
) -> tuple[tuple[IndexedBatch, ...], ...]:
    chunks = tuple(
        (shard_index, tuple(payloads[start : start + replays_per_shard]))
        for shard_index, start in enumerate(range(0, len(payloads), replays_per_shard))
    )
    assignments: list[list[IndexedBatch]] = [[] for _ in ports]
    for shard_index, chunk in chunks:
        assignments[shard_index % len(ports)].append((shard_index, chunk))
    return tuple(tuple(item) for item in assignments)


def _source_stream(dataset_roots: Sequence[Path]) -> Iterable[tuple[int, Path, Mapping[str, Any], ReplayPreflightV4]]:
    source_index = 0
    for dataset_root in dataset_roots:
        stream = ParquetReplayStreamV4(dataset_root)
        for payload, report in stream:
            yield source_index, dataset_root, payload, report
            source_index += 1


def _select_partition(
    dataset_roots: Sequence[Path], *, source_count: int, partition_index: int, partition_count: int
) -> tuple[tuple[IndexedPayload, ...], tuple[dict[str, object], ...]]:
    stream = _source_stream(dataset_roots)
    selected_source = islice(stream, source_count) if source_count else stream
    payloads: list[IndexedPayload] = []
    labels: list[dict[str, object]] = []
    seen = 0
    seen_tags: set[str] = set()
    for source_index, dataset_root, payload, report in selected_source:
        seen += 1
        if source_index % partition_count != partition_index:
            continue
        if report.replay_tag in seen_tags:
            raise ValueError(f"duplicate replay tag in source selection: {report.replay_tag}")
        seen_tags.add(report.replay_tag)
        payloads.append((source_index, payload))
        labels.append(
            {
                "source_index": source_index,
                "dataset_root": str(dataset_root.resolve()),
                "replay_tag": report.replay_tag,
                "preflight": asdict(report),
                "preflight_accepted": report.accepted,
                "preflight_rejection_reason": report.rejection_reason,
            }
        )
    if source_count and seen != source_count:
        raise ValueError(f"requested {source_count} source rows, found {seen}")
    if not payloads:
        raise ValueError("source selection assigned no replay rows to this partition")
    return tuple(payloads), tuple(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--additional-dataset-root", type=Path, action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", required=True)
    parser.add_argument("--replays-per-shard", type=int, default=4)
    parser.add_argument("--source-count", type=int, default=100)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--resident-slots", type=int, default=1)
    parser.add_argument("--capture-mode", choices=("zlib-json", "json", "compact"), default="zlib-json")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--compression-level", type=int, default=-5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--infrastructure-retries", type=int, default=2)
    parser.add_argument("--runtime-report", type=Path, action="append", default=[])
    parser.add_argument("--verify-roundtrip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    source_count = args.source_count
    ports = tuple(int(value) for value in args.ports.split(",") if value)
    if (
        not ports
        or len(set(ports)) != len(ports)
        or args.replays_per_shard <= 0
        or source_count < 0
        or args.max_frames < 0
        or not 0 <= args.partition_index < args.partition_count
        or not 1 <= args.resident_slots <= 16
        or args.infrastructure_retries < 0
    ):
        raise ValueError("cache builder arguments are invalid")

    cache_root = args.cache_root.resolve()
    index_path = cache_root / "index.json"
    if index_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to replace existing cache index {index_path}")
        print(json.dumps(load_il_cache_index(index_path), sort_keys=True), flush=True)
        return
    cache_root.mkdir(parents=True, exist_ok=True)

    dataset_roots = tuple(Path(item).resolve() for item in (args.dataset_root, *args.additional_dataset_root))
    payloads, labels = _select_partition(
        dataset_roots,
        source_count=source_count,
        partition_index=args.partition_index,
        partition_count=args.partition_count,
    )
    config_payload = json.dumps(asdict(ModelConfigV4()), sort_keys=True, separators=(",", ":")).encode()
    shard_contract: dict[str, object] = {
        "version": "v4-il-cache-contract.v2",
        "datasets": [_dataset_snapshot(path) for path in dataset_roots],
        "runtime_reports": _runtime_reports(args.runtime_report),
        "source_sha256": _source_hashes(),
        "model_config": asdict(ModelConfigV4()),
        "model_config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "decision_ticks": ModelConfigV4().decision_ticks,
        "storage_float_dtype": "float16",
        "gamma_per_decision": args.gamma,
        "max_frames": None if args.max_frames == 0 else args.max_frames,
        "capture_mode": args.capture_mode,
        "headless": True,
        "raster_rendering": False,
        "pipeline_phase": "cache-build-only",
        "concurrent_training": False,
        "source_count": source_count,
        "partition_index": args.partition_index,
        "partition_count": args.partition_count,
        "partition_rule": "zero_based_source_index_modulo_partition_count",
        "partition_replay_count": len(payloads),
        "partition_selection_sha256": _canonical_sha256(labels),
    }
    index_contract = dict(shard_contract)
    index_contract["selected_replays"] = labels
    assignments = _partition_payloads(payloads, ports=ports, replays_per_shard=args.replays_per_shard)
    active = tuple(
        (index, port, batches) for index, (port, batches) in enumerate(zip(ports, assignments, strict=True)) if batches
    )
    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(active), mp_context=context) as executor:
        futures = {
            executor.submit(
                _worker,
                worker_index=index,
                partition_index=args.partition_index,
                host=args.host,
                port=port,
                batches=batches,
                cache_root=cache_root,
                contract=shard_contract,
                gamma=args.gamma,
                max_frames=None if args.max_frames == 0 else args.max_frames,
                compression_level=args.compression_level,
                verify_roundtrip=args.verify_roundtrip,
                timeout=args.timeout,
                resident_slots=args.resident_slots,
                capture_mode=args.capture_mode,
                infrastructure_retries=args.infrastructure_retries,
                resume=args.resume,
            ): port
            for index, port, batches in active
        }
        for future in as_completed(futures):
            summaries.extend(future.result())
    summaries.sort(key=lambda item: str(item["shard_id"]))
    index = write_il_cache_index(index_path, summaries, contract=index_contract)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "ok": True,
                "cache_root": str(cache_root),
                "elapsed_s": round(elapsed, 6),
                "partition_index": args.partition_index,
                "partition_count": args.partition_count,
                "partition_replay_count": len(payloads),
                "producer_match_decision_frames_s": round(int(index["match_decision_frames"]) / elapsed, 3),
                "compression_ratio": round(int(index["container_bytes"]) / max(1, int(index["compressed_bytes"])), 6),
                **{
                    key: index[key]
                    for key in (
                        "shard_count",
                        "replay_results",
                        "sequences",
                        "owner_frames",
                        "match_decision_frames",
                        "compressed_bytes",
                        "container_bytes",
                        "completed_replays",
                        "failed_replays",
                        "zero_frame_failures",
                        "failures_before_40s",
                        "unexpected_unknown_failures",
                        "winner_comparable_replays",
                        "winner_mismatches",
                        "crowns_comparable_replays",
                        "crowns_mismatches",
                        "expert_actions",
                        "executed_expert_actions",
                    )
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
