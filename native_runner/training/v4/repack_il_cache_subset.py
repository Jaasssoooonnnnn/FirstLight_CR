"""Repack a replay subset of V4 IL caches into a dense cache (N selected replays per shard).

Subsets chosen by replay tag from a full cache leave most of every shard
unselected, so a group of ``--shards-per-rank`` shards carries a subset-sized
fraction of the lanes.  This tool writes the selected replays into new shards
of ``--replays-per-shard`` replays each (both owner lanes), in the unchanged
on-disk format (zstd container of a canonical-JSON manifest plus safetensors,
per-shard summary JSON, canonical ``index.json``), so the unmodified trainer
and evaluator take ``--index <output>/node-0/index.json`` with the same tag file.

Pipeline (each source shard is decoded exactly once):

1. Plan with the trainer's own ``build_training_plan`` over the source indexes
   (so duplicate replay copies resolve exactly as in training) and assign the
   sorted tags, shuffled with ``--seed``, to destination shards in chunks.
2. Source pass (``--workers`` processes): decode a source shard, cut every
   selected replay's frames out of every tensor, write each replay to a
   per-replay spill container (same format, zstd) and record a canonical
   SHA-256 of its tensors.
3. Destination pass: per destination shard, stream its replays' spill files
   into preallocated tensors, re-pad the three dynamic collections to the new
   shard's capacities exactly as ``write_il_cache_shard`` does, and stream the
   container through zstd and SHA-256 without an extra in-memory copy.
4. Write the index; ``--verify`` then re-reads every source and destination
   shard and checks per-replay digests, manifests and the trainer plan.

Tensor bytes, descriptors (except ``offset``) and per-replay ``results`` rows
are carried verbatim; shard and index aggregates are recomputed like the
writer does.  ``<output>/repack.json`` records the provenance and assignment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import time
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .cache import (
    _CONTAINER_HEADER,
    _CONTAINER_MAGIC,
    _DYNAMIC_COUNTS,
    _NEGATIVE_PADDING_FIELDS,
    IL_CACHE_SHARD_SCHEMA,
    IL_CACHE_SUMMARY_SCHEMA,
    _byte_view,
    _canonical_json,
    _decode_il_cache_shard,
    _tensor_from_spec,
    load_il_cache_index,
    write_il_cache_index,
)
from .train_imitation_cache import _read_replay_tags, build_training_plan

REPACK_SCHEMA = "v4-il-dense-subset-repack.v1"
REPACK_HOST = "dense-repack"  # index rows carry host/port of the converter; a repack has none
REPACK_PORT = 0
DEFAULT_COMPRESSION_LEVEL = -5  # cache_builder's default; ~90x on the IL caches
# Contract keys that describe which replays a conversion node/run handled, not
# what the tensors mean.  They differ between the per-node partitions of one
# conversion (partition_index, partition_replay_count,
# partition_selection_sha256 in shard manifests, plus selected_replays in the
# index contracts).  A dense shard mixes partitions, so it carries the first
# source's value and repack.json records every source value.  Any other
# differing key (gamma, model_config, storage dtype, decision ticks, ...) is
# fatal unless named with --tolerate-contract-key.
TOLERATED_CONTRACT_KEYS = frozenset(
    {
        "partition_index",
        "partition_count",
        "partition_replay_count",
        "partition_rule",
        "partition_selection_sha256",
        "selected_replays",
        "runtime_reports",
    }
)
_COUNT_TENSORS = {
    "active_effects": "active_effect_counts",
    "relation_edges": "relation_edge_counts",
    "candidates": "candidate_counts",
}
assert set(_COUNT_TENSORS) == set(_DYNAMIC_COUNTS)
_KNOWN_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "shard_id",
        "contract",
        "sequences",
        "results",
        "owner_frames",
        "match_decision_frames",
        "dynamic_capacities",
        "tensor_count",
    }
)
# safetensors' Dtype enum order; ``safetensors.torch.save`` writes tensors by
# descending dtype, then ascending name, and pads the header with spaces to 8.
_SAFETENSORS_DTYPE_ORDER = (
    "BOOL",
    "U8",
    "I8",
    "F8_E5M2",
    "F8_E4M3",
    "I16",
    "U16",
    "F16",
    "BF16",
    "I32",
    "U32",
    "F32",
    "F64",
    "I64",
    "U64",
)
_TORCH_TO_SAFETENSORS = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int32: "I32",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int64: "I64",
}
for _name, _attr in (
    ("F8_E5M2", "float8_e5m2"),
    ("F8_E4M3", "float8_e4m3fn"),
    ("U16", "uint16"),
    ("U32", "uint32"),
    ("U64", "uint64"),
):
    if hasattr(torch, _attr):
        _TORCH_TO_SAFETENSORS[getattr(torch, _attr)] = _name


# ---------------------------------------------------------------------------
# Container writing (streamed: no full-shard bytes copy)
# ---------------------------------------------------------------------------


def safetensors_layout(tensors: Mapping[str, Tensor]) -> tuple[bytes, tuple[str, ...]]:
    """The 8-byte length + JSON header and the data order that ``safetensors.torch.save`` produces."""

    def key(name: str) -> tuple[int, str]:
        return (-_SAFETENSORS_DTYPE_ORDER.index(_TORCH_TO_SAFETENSORS[tensors[name].dtype]), name)

    order = tuple(sorted(tensors, key=key))
    entries = []
    offset = 0
    for name in order:
        tensor = tensors[name]
        size = tensor.numel() * tensor.element_size()
        entries.append(
            json.dumps(name, ensure_ascii=False)
            + ":"
            + json.dumps(
                {
                    "dtype": _TORCH_TO_SAFETENSORS[tensor.dtype],
                    "shape": list(tensor.shape),
                    "data_offsets": [offset, offset + size],
                },
                separators=(",", ":"),
            )
        )
        offset += size
    header = ("{" + ",".join(entries) + "}").encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header, order


class _HashingFile:
    def __init__(self, handle: object) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes | memoryview) -> int:
        self.digest.update(data)
        self.size += len(data)
        return getattr(self._handle, "write")(data)

    def flush(self) -> None:
        getattr(self._handle, "flush")()


def write_container(
    data_path: Path, manifest: Mapping[str, object], tensors: Mapping[str, Tensor], *, compression_level: int
) -> dict[str, object]:
    """Write ``header + canonical manifest + safetensors`` through zstd; returns the size/digest fields."""

    import zstandard

    manifest_payload = _canonical_json(manifest)
    safetensors_header, order = safetensors_layout(tensors)
    contiguous = {name: tensors[name].contiguous() for name in order}
    container_bytes = (
        _CONTAINER_HEADER.size
        + len(manifest_payload)
        + len(safetensors_header)
        + sum(tensor.numel() * tensor.element_size() for tensor in contiguous.values())
    )
    container_digest = hashlib.sha256()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            sink = _HashingFile(handle)
            compressor = zstandard.ZstdCompressor(level=compression_level, threads=1)
            with compressor.stream_writer(sink, size=container_bytes, closefd=False) as writer:
                parts: Iterable[bytes | memoryview] = (
                    _CONTAINER_HEADER.pack(_CONTAINER_MAGIC, len(manifest_payload)),
                    manifest_payload,
                    safetensors_header,
                )
                for part in parts:
                    container_digest.update(part)
                    writer.write(part)
                for tensor in contiguous.values():
                    if tensor.numel() == 0:
                        continue
                    view = _byte_view(tensor.view(-1).view(torch.uint8))
                    container_digest.update(view)
                    writer.write(view)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, data_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "compressed_sha256": sink.digest.hexdigest(),
        "container_sha256": container_digest.hexdigest(),
        "compressed_bytes": sink.size,
        "container_bytes": container_bytes,
        "compression_ratio": round(container_bytes / max(1, sink.size), 6),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Per-replay extraction and canonical digests
# ---------------------------------------------------------------------------


def _dynamic_collection(name: str) -> str | None:
    parts = name.split(".")
    if len(parts) == 3 and parts[0] == "observations" and parts[1] in _COUNT_TENSORS and parts[2] != "elixir":
        return parts[1]
    return None


def _raw_tensors(summary_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, Tensor]]:
    decoded = _decode_il_cache_shard(summary_path, verify=True)
    tensors = {spec.name: _tensor_from_spec(decoded.buffer, spec) for spec in decoded.specs}
    return decoded.summary, decoded.manifest, tensors


def extract_replay(
    tensors: Mapping[str, Tensor], descriptors: Sequence[Mapping[str, object]]
) -> tuple[dict[str, Tensor], dict[str, int]]:
    """One replay's frames of every frame tensor, dynamic collections narrowed to the replay's own capacity.

    The narrowed width is the replay's maximum per-frame count, so the result
    is independent of which shard (and shard capacity) the replay came from:
    the writer pads every frame canonically beyond its count.
    """

    frames = int(tensors["episode_start"].shape[0])
    spans = [(int(item["offset"]), int(item["length"])) for item in descriptors]
    contiguous = all(spans[index + 1][0] == spans[index][0] + spans[index][1] for index in range(len(spans) - 1))

    def cut(tensor: Tensor) -> Tensor:
        if int(tensor.shape[0]) != frames:
            raise ValueError("cache tensor does not have the shard's frame count")
        if contiguous:
            return tensor.narrow(0, spans[0][0], sum(length for _offset, length in spans))
        return torch.cat([tensor.narrow(0, offset, length) for offset, length in spans])

    counts = {collection: cut(tensors[name]) for collection, name in _COUNT_TENSORS.items()}
    widths = {collection: int(value.max()) if value.numel() else 0 for collection, value in counts.items()}
    result: dict[str, Tensor] = {}
    for name, tensor in tensors.items():
        if name == "sequence_offsets":
            continue
        piece = cut(tensor)
        collection = _dynamic_collection(name)
        if collection is not None:
            piece = piece.narrow(1, 0, widths[collection])
        result[name] = piece
    return result, widths


def replay_digest(tensors: Mapping[str, Tensor]) -> str:
    """SHA-256 over names, dtypes, shapes and bytes of ``extract_replay`` output (canonical per replay)."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        if tensor.numel():
            digest.update(_byte_view(tensor.view(-1).view(torch.uint8)))
    return digest.hexdigest()


def _replay_descriptors(manifest: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    by_tag: dict[str, list[dict[str, object]]] = {}
    for raw in manifest["sequences"]:  # type: ignore[union-attr]
        by_tag.setdefault(str(raw["replay_tag"]), []).append(dict(raw))
    for rows in by_tag.values():
        rows.sort(key=lambda row: int(row["offset"]))
    return by_tag


def _results_by_tag(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for raw in manifest.get("results", []):  # type: ignore[union-attr]
        tag = str(raw["replay_tag"])
        if tag in rows:
            raise ValueError(f"cache shard has two result rows for replay {tag}")
        rows[tag] = dict(raw)
    return rows


# ---------------------------------------------------------------------------
# Worker tasks (module level: they run in spawned processes)
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    torch.set_num_threads(1)


def _spill_paths(spill_dir: Path, tag: str) -> tuple[Path, Path]:
    data = spill_dir / f"{tag}.cril.zst"
    return data, data.with_suffix(".json")


def source_task(
    summary_path: str, tags: Sequence[str], spill_dir: str | None, compression_level: int
) -> dict[str, dict[str, object]]:
    """Decode one source shard; spill (optional) and digest each selected replay."""

    summary, manifest, tensors = _raw_tensors(Path(summary_path))
    descriptors = _replay_descriptors(manifest)
    results = _results_by_tag(manifest)
    extra_keys = sorted(set(manifest).difference(_KNOWN_MANIFEST_KEYS))
    out: dict[str, dict[str, object]] = {}
    for tag in tags:
        rows = descriptors.get(tag)
        if not rows or [int(row["owner"]) for row in rows] != [0, 1]:
            raise ValueError(f"replay {tag} does not have owner lanes 0 and 1 in {summary_path}")
        if tag not in results:
            raise ValueError(f"replay {tag} has no results row in {summary_path}")
        piece, widths = extract_replay(tensors, rows)
        info: dict[str, object] = {
            "source_summary": summary_path,
            "source_shard_id": manifest.get("shard_id"),
            "descriptors": rows,
            "result": results[tag],
            "widths": widths,
            "frames": sum(int(row["length"]) for row in rows),
            "digest": replay_digest(piece),
            # Contracts are ~16 KB; only their digest travels back per replay.
            "contract_sha256": hashlib.sha256(_canonical_json(manifest.get("contract"))).hexdigest(),
            "extra_manifest": {key: manifest[key] for key in extra_keys},
            "storage": {
                name: [
                    str(tensor.dtype),
                    [
                        None if _dynamic_collection(name) and axis == 0 else size
                        for axis, size in enumerate(tensor.shape[1:])
                    ],
                ]
                for name, tensor in piece.items()
            },
        }
        if spill_dir is not None:
            data_path, json_path = _spill_paths(Path(spill_dir), tag)
            spill_manifest = {"schema": IL_CACHE_SHARD_SCHEMA, "spill_replay_tag": tag, "widths": widths}
            fields = write_container(data_path, spill_manifest, piece, compression_level=compression_level)
            _write_json(json_path, {"schema": IL_CACHE_SUMMARY_SCHEMA, "data_file": data_path.name, **fields})
        out[tag] = info
    del tensors, summary
    return out


def _pad_value(name: str) -> int:
    return -1 if name.rsplit(".", 1)[-1] in _NEGATIVE_PADDING_FIELDS else 0


def _summary_aggregates(results: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """``write_il_cache_shard``'s per-shard result aggregates, from the carried ``results`` rows."""

    def failed(row: Mapping[str, object]) -> bool:
        return not bool(row["completed"])

    return {
        "completed_replays": sum(bool(row["completed"]) for row in results),
        "failed_replays": sum(failed(row) for row in results),
        "zero_frame_failures": sum(failed(row) and int(row["decision_frame_count"]) == 0 for row in results),
        "failures_before_40s": sum(
            failed(row) and int(row["decision_frame_count"]) > 0 and int(row["simulated_end_tick"]) < 800
            for row in results
        ),
        "unexpected_unknown_failures": sum(
            failed(row) and "unknown" in str(row.get("failure_reason") or "").lower() for row in results
        ),
        "winner_comparable_replays": sum(row.get("winner_matches") is not None for row in results),
        "winner_mismatches": sum(row.get("winner_matches") is False for row in results),
        "crowns_comparable_replays": sum(row.get("crowns_match") is not None for row in results),
        "crowns_mismatches": sum(row.get("crowns_match") is False for row in results),
        "expert_actions": sum(int(row["expert_action_count"]) for row in results),
        "executed_expert_actions": sum(int(row["executed_expert_action_count"]) for row in results),
        "decision_frames": sum(int(row["decision_frame_count"]) for row in results),
    }


def destination_task(
    shard_id: str,
    output_dir: str,
    replays: Sequence[Mapping[str, object]],
    spill_dir: str,
    contract: Mapping[str, object],
    extra_manifest: Mapping[str, object],
    compression_level: int,
) -> dict[str, object]:
    """Assemble one dense shard from its replays' spill files and write it like ``write_il_cache_shard``."""

    frames = sum(int(replay["frames"]) for replay in replays)
    capacities = {
        collection: max(int(replay["widths"][collection]) for replay in replays)  # type: ignore[index]
        for collection in _COUNT_TENSORS
    }
    storage = replays[0]["storage"]
    tensors: dict[str, Tensor] = {}
    for name, (dtype_name, tail) in storage.items():  # type: ignore[union-attr]
        dtype = getattr(torch, str(dtype_name).removeprefix("torch."))
        collection = _dynamic_collection(name)
        shape = [frames, *(capacities[collection] if size is None else size for size in tail)]  # type: ignore[index]
        if collection is not None:
            tensors[name] = torch.full(shape, _pad_value(name), dtype=dtype)
        else:
            tensors[name] = torch.empty(shape, dtype=dtype)
    descriptors: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    position = 0
    for replay in replays:
        tag = str(replay["result"]["replay_tag"])  # type: ignore[index]
        if replay["storage"] != storage:
            raise ValueError(f"replay {tag} has a different tensor layout than its shard")
        _summary, spill_manifest, piece = _raw_tensors(_spill_paths(Path(spill_dir), tag)[1])
        if spill_manifest.get("spill_replay_tag") != tag or set(piece) != set(tensors):
            raise ValueError(f"spill file for {tag} does not match its plan")
        length = int(replay["frames"])
        for name, source in piece.items():
            target = tensors[name].narrow(0, position, length)
            if _dynamic_collection(name) is not None:
                target = target.narrow(1, 0, int(source.shape[1]))
            target.copy_(source)
        del piece
        offset = position
        for row in replay["descriptors"]:  # type: ignore[union-attr]
            descriptors.append({**row, "offset": offset})
            offset += int(row["length"])
        results.append(dict(replay["result"]))  # type: ignore[arg-type]
        position += length
    tensors["sequence_offsets"] = torch.tensor([int(row["offset"]) for row in descriptors] + [frames], dtype=torch.long)
    manifest = {
        **extra_manifest,
        "schema": IL_CACHE_SHARD_SCHEMA,
        "shard_id": shard_id,
        "contract": dict(contract),
        "sequences": descriptors,
        "results": results,
        "owner_frames": frames,
        "match_decision_frames": frames // 2,
        "dynamic_capacities": capacities,
        "tensor_count": len(tensors),
    }
    data_path = Path(output_dir) / f"{shard_id}.cril.zst"
    fields = write_container(data_path, manifest, tensors, compression_level=compression_level)
    summary = {
        "schema": IL_CACHE_SUMMARY_SCHEMA,
        "shard_schema": IL_CACHE_SHARD_SCHEMA,
        "shard_id": shard_id,
        "data_file": data_path.name,
        **fields,
        "replay_results": len(results),
        "sequences": len(descriptors),
        "owner_frames": frames,
        "match_decision_frames": frames // 2,
        **_summary_aggregates(results),
    }
    _write_json(data_path.with_suffix(".json"), summary)
    return summary


def destination_digest_task(summary_path: str) -> dict[str, dict[str, object]]:
    """Re-read one written dense shard: per-replay digests, descriptors and results rows."""

    _summary, manifest, tensors = _raw_tensors(Path(summary_path))
    descriptors = _replay_descriptors(manifest)
    results = _results_by_tag(manifest)
    if set(descriptors) != set(results):
        raise ValueError(f"dense shard {summary_path} has replays without results rows (or the reverse)")
    out = {}
    for tag, rows in descriptors.items():
        piece, _widths = extract_replay(tensors, rows)
        out[tag] = {"digest": replay_digest(piece), "descriptors": rows, "result": results[tag]}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def assign_replays(tags: Iterable[str], *, replays_per_shard: int, seed: int) -> list[list[str]]:
    """Sorted tags, ``random.Random(seed).shuffle``, consecutive chunks (the last may be short)."""

    if replays_per_shard <= 0:
        raise ValueError("replays_per_shard must be positive")
    ordered = sorted(tags)
    random.Random(seed).shuffle(ordered)
    return [ordered[start : start + replays_per_shard] for start in range(0, len(ordered), replays_per_shard)]


class _Runner:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._pool = None
        if workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=torch.multiprocessing.get_context("spawn"), initializer=_worker_init
            )

    def map(self, function, calls: Sequence[tuple], *, label: str, log: bool):
        started = time.perf_counter()
        results = [None] * len(calls)
        if self._pool is None:
            for index, args in enumerate(calls):
                results[index] = function(*args)
        else:
            futures: dict[Future, int] = {self._pool.submit(function, *args): index for index, args in enumerate(calls)}
            done = 0
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if log and (done % max(1, len(calls) // 20) == 0 or done == len(calls)):
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps({"event": label, "done": done, "total": len(calls), "elapsed_s": round(elapsed, 1)}),
                        flush=True,
                    )
        return results, time.perf_counter() - started

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)


def _source_plan(index_paths: Sequence[Path], tags_path: Path, manifest_workers: int):
    plans, metadata = build_training_plan(list(index_paths), tags_path, manifest_workers=manifest_workers)
    return plans, metadata


def _lanes(plans) -> set[tuple[str, int, int]]:
    """(replay_tag, owner, length) of every planned lane."""

    lanes = set()
    for plan in plans:
        for unit in plan.units:
            for owner, length in enumerate(unit.sequence_lengths):
                lanes.add((unit.replay_tag, owner, int(length)))
    return lanes


def _short(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


def _value_key(text: str, limit: int = 256) -> str:
    """A contract value's JSON, or its digest when long (index ``selected_replays`` lists are ~12 MB)."""

    if len(text) <= limit:
        return text
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()} ({len(text)} chars)"


def check_contracts(
    contracts: Mapping[str, Mapping[str, object]], *, tolerated: frozenset[str], what: str
) -> tuple[dict[str, object], dict[str, dict[str, list[str]]]]:
    """First contract plus ``{key: {value: [sources]}}`` for every key that varies.

    ``value`` is the canonical JSON of the value, or ``sha256:<hex> (<n> chars)``
    for values longer than 256 characters.

    Raises ``ValueError`` naming every varying key outside ``tolerated``, with
    example values and sources, so nothing that changes training semantics is
    silently mixed into one dense shard.
    """

    names = list(contracts)
    if not names:
        raise ValueError(f"no {what} to compare")
    first = dict(contracts[names[0]])
    keys = sorted({key for contract in contracts.values() for key in contract})
    variations: dict[str, dict[str, list[str]]] = {}
    samples: dict[str, dict[str, str]] = {}
    for key in keys:
        values: dict[str, list[str]] = {}
        texts: dict[str, str] = {}
        for name in names:
            text = json.dumps(contracts[name].get(key, "<missing>"), sort_keys=True)
            value = _value_key(text)
            texts.setdefault(value, text)
            values.setdefault(value, []).append(name)
        if len(values) > 1:
            variations[key] = values
            samples[key] = texts
    fatal = [key for key in variations if key not in tolerated]
    if fatal:
        lines = []
        for key in fatal:
            examples = [
                f"{_short(samples[key][value])} in {len(sources)} ({sources[0]})"
                for value, sources in list(variations[key].items())[:3]
            ]
            lines.append(f"{key}: " + "; ".join(examples))
        raise ValueError(
            f"{what} disagree on {len(fatal)} contract key(s) that are not tolerated "
            f"(tolerated: {sorted(tolerated)}; add --tolerate-contract-key only if the key does not change "
            "training semantics):\n  " + "\n  ".join(lines)
        )
    return first, variations


def check_source_manifests(
    summary_paths: Sequence[Path], *, tolerated: frozenset[str], manifest_workers: int
) -> dict[str, object]:
    """Fail-fast manifest check of every source shard that will be read (header-only reads)."""

    from concurrent.futures import ThreadPoolExecutor

    from .cache import load_il_cache_manifest

    with ThreadPoolExecutor(max_workers=max(1, min(manifest_workers, len(summary_paths)))) as pool:
        manifests = dict(zip((str(path) for path in summary_paths), pool.map(load_il_cache_manifest, summary_paths)))
    problems = [path for path, manifest in manifests.items() if manifest.get("schema") != IL_CACHE_SHARD_SCHEMA]
    if problems:
        raise ValueError(f"{len(problems)} source shards have an unsupported manifest schema, e.g. {problems[:3]}")
    extras = {
        path: {key: manifest[key] for key in sorted(set(manifest).difference(_KNOWN_MANIFEST_KEYS))}
        for path, manifest in manifests.items()
    }
    extra, extra_variations = check_contracts(
        extras, tolerated=frozenset(), what="source shard manifests (top-level fields)"
    )
    contract, variations = check_contracts(
        {path: manifest.get("contract", {}) for path, manifest in manifests.items()},  # type: ignore[misc]
        tolerated=tolerated,
        what="source shard manifest contracts",
    )
    return {
        "contract": contract,
        "extra": extra,
        "variations": variations,
        "first_source": next(iter(manifests)),
        "contract_sha256": {
            path: hashlib.sha256(_canonical_json(manifest.get("contract"))).hexdigest()
            for path, manifest in manifests.items()
        },
    }


def repack(
    index_paths: Sequence[Path],
    tags_path: Path,
    output: Path,
    *,
    replays_per_shard: int = 16,
    workers: int = 1,
    seed: int = 0,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    manifest_workers: int = 32,
    spill_dir: Path | None = None,
    keep_spill: bool = False,
    log: bool = False,
    tolerate_contract_keys: Iterable[str] = (),
) -> dict[str, object]:
    """Build ``output/node-0/index.json`` (+ shards) and ``output/repack.json``; returns the latter."""

    index_paths = [Path(path).resolve() for path in index_paths]
    tags_path = Path(tags_path).resolve()
    output = Path(output).resolve()
    node_dir = output / "node-0"
    index_path = node_dir / "index.json"
    if index_path.exists():
        raise FileExistsError(f"{index_path} exists; choose a new --output")
    spill = (spill_dir or output / ".spill").resolve()
    tolerated = TOLERATED_CONTRACT_KEYS | frozenset(tolerate_contract_keys)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    plans, source_metadata = _source_plan(index_paths, tags_path, manifest_workers)
    tags = _read_replay_tags(tags_path)
    assignment = assign_replays(tags, replays_per_shard=replays_per_shard, seed=seed)
    # Fail fast: every consistency check runs before the source pass.
    index_contract, index_variations = check_contracts(
        {str(path): load_il_cache_index(path)["contract"] for path in index_paths},  # type: ignore[misc]
        tolerated=tolerated,
        what="source index contracts",
    )
    source_check = check_source_manifests(
        [plan.summary_path for plan in plans], tolerated=tolerated, manifest_workers=manifest_workers
    )
    manifest_contract = source_check["contract"]
    extra_manifest = source_check["extra"]
    timings["plan_s"] = time.perf_counter() - started
    if log:
        print(
            json.dumps(
                {
                    "event": "preflight",
                    "source_shards": len(plans),
                    "tolerated_manifest_contract_variations": sorted(source_check["variations"]),  # type: ignore[arg-type]
                    "tolerated_index_contract_variations": sorted(index_variations),
                    "plan_s": round(timings["plan_s"], 1),
                }
            ),
            flush=True,
        )
    spill.mkdir(parents=True, exist_ok=True)
    node_dir.mkdir(parents=True, exist_ok=True)

    runner = _Runner(workers)
    try:
        calls = [
            (str(plan.summary_path), [unit.replay_tag for unit in plan.units], str(spill), compression_level)
            for plan in plans
        ]
        source_rows, timings["source_pass_s"] = runner.map(source_task, calls, label="source_pass", log=log)
        replays: dict[str, dict[str, object]] = {}
        for rows in source_rows:
            replays.update(rows)  # type: ignore[arg-type]
        if set(replays) != tags:
            raise RuntimeError("source pass did not produce every selected replay")
        contract_digests = source_check["contract_sha256"]
        for tag, row in replays.items():
            if (
                row["contract_sha256"] != contract_digests[row["source_summary"]]
                or row["extra_manifest"] != extra_manifest
            ):  # type: ignore[index]
                raise RuntimeError(f"source shard of replay {tag} changed since the preflight check")

        width = max(6, len(str(len(assignment) - 1)))
        destination_calls = [
            (
                f"dense-shard-{index:0{width}d}",
                str(node_dir),
                [replays[tag] for tag in shard_tags],
                str(spill),
                manifest_contract,
                extra_manifest,
                compression_level,
            )
            for index, shard_tags in enumerate(assignment)
        ]
        summaries, timings["destination_pass_s"] = runner.map(
            destination_task, destination_calls, label="destination_pass", log=log
        )
    finally:
        runner.close()

    rows = [
        {**summary, "summary_file": f"{summary['shard_id']}.cril.json", "host": REPACK_HOST, "port": REPACK_PORT}
        for summary in summaries  # type: ignore[union-attr]
    ]
    index = write_il_cache_index(index_path, rows, contract=index_contract)
    if not keep_spill:
        shutil.rmtree(spill, ignore_errors=True)
    timings["total_s"] = time.perf_counter() - started
    value_by_source = {
        key: {source: value for value, sources in values.items() for source in sources}
        for key, values in source_check["variations"].items()  # type: ignore[union-attr]
    }
    record = {
        "schema": REPACK_SCHEMA,
        "source_indexes": [str(path) for path in index_paths],
        "source_index_sha256": source_metadata["cache_index_sha256"],
        "source_index_contract_differences": sorted(index_variations),
        "tolerated_contract_keys": sorted(tolerated),
        # {key: {value_json: [source index paths]}}; the dense index carries the first index's value.
        "source_index_contract_variations": index_variations,
        # {key: {value_json: [source shard summary paths]}}; dense manifests carry the value of
        # source_manifest_contract_first_source.
        "source_manifest_contract_variations": source_check["variations"],
        "source_manifest_contract_first_source": source_check["first_source"],
        # Per dense shard: the source values of each varying key among its replays.
        "dense_shard_contract_sources": {
            f"dense-shard-{index:0{width}d}": {
                key: sorted({value_by_source[key][replays[tag]["source_summary"]] for tag in shard_tags})  # type: ignore[index]
                for key in value_by_source
            }
            for index, shard_tags in enumerate(assignment)
        },
        "tags_path": str(tags_path),
        "tags_sha256": source_metadata["train_tags_sha256"],
        "replays_per_shard": replays_per_shard,
        "seed": seed,
        "compression_level": compression_level,
        "workers": workers,
        "source_shards_read": len(plans),
        "dense_shards": len(assignment),
        "selected_replays": len(tags),
        "selected_owner_frames": source_metadata["selected_owner_frames"],
        "dense_index": str(index_path),
        "dense_index_sha256": index["index_sha256"],
        "index_row_host_port": [REPACK_HOST, REPACK_PORT],
        "timings_s": {key: round(value, 3) for key, value in timings.items()},
        "assignment": assignment,
        "replays": {
            tag: {
                "source_summary": row["source_summary"],
                "source_shard_id": row["source_shard_id"],
                "digest": row["digest"],
                "frames": row["frames"],
            }
            for tag, row in sorted(replays.items())
        },
    }
    _write_json(output / "repack.json", record)
    return record


def verify(output: Path, *, workers: int = 1, manifest_workers: int = 32, log: bool = False) -> dict[str, object]:
    """Independently re-read sources and the dense cache; raise on any difference."""

    output = Path(output).resolve()
    record = json.loads((output / "repack.json").read_text(encoding="utf-8"))
    index_paths = [Path(path) for path in record["source_indexes"]]
    tags_path = Path(record["tags_path"])
    dense_index = Path(record["dense_index"])
    started = time.perf_counter()
    source_plans, source_metadata = _source_plan(index_paths, tags_path, manifest_workers)
    dense_plans, dense_metadata = _source_plan([dense_index], tags_path, manifest_workers)
    problems: list[str] = []
    if source_metadata["cache_index_sha256"] != record["source_index_sha256"]:
        problems.append("source index files changed since the repack")
    for key in ("selected_replays", "selected_owner_sequences", "selected_owner_frames"):
        if source_metadata[key] != dense_metadata[key]:
            problems.append(f"plan {key}: source {source_metadata[key]} != dense {dense_metadata[key]}")
    if _lanes(source_plans) != _lanes(dense_plans):
        problems.append("planned (replay, owner, length) lanes differ")
    if set(source_metadata["cache_gamma_per_decision"].values()) != set(
        dense_metadata["cache_gamma_per_decision"].values()
    ):
        problems.append("index gamma_per_decision differs")
    runner = _Runner(workers)
    try:
        source_calls = [
            (str(plan.summary_path), [unit.replay_tag for unit in plan.units], None, 0) for plan in source_plans
        ]
        source_rows, source_s = runner.map(source_task, source_calls, label="verify_source", log=log)
        dense_rows, dense_s = runner.map(
            destination_digest_task, [(str(plan.summary_path),) for plan in dense_plans], label="verify_dense", log=log
        )
    finally:
        runner.close()
    source = {tag: row for rows in source_rows for tag, row in rows.items()}
    dense = {tag: row for rows in dense_rows for tag, row in rows.items()}
    if set(source) != set(dense):
        problems.append(f"replay sets differ: {len(set(source) ^ set(dense))} tags")
    mismatched = []
    for tag in sorted(set(source) & set(dense)):
        left, right = source[tag], dense[tag]
        strip = lambda rows: [{key: value for key, value in row.items() if key != "offset"} for row in rows]  # noqa: E731
        if left["digest"] != right["digest"]:
            mismatched.append((tag, "tensors"))
        elif strip(left["descriptors"]) != strip(right["descriptors"]):
            mismatched.append((tag, "descriptors"))
        elif left["result"] != right["result"]:
            mismatched.append((tag, "results row"))
        elif record["replays"].get(tag, {}).get("digest") != left["digest"]:
            mismatched.append((tag, "repack record digest"))
    if mismatched:
        problems.append(f"{len(mismatched)} replays differ, e.g. {mismatched[:5]}")
    report = {
        "event": "verify",
        "ok": not problems,
        "problems": problems,
        "replays_checked": len(set(source) & set(dense)),
        "dense_shards": len(dense_plans),
        "source_shards": len(source_plans),
        "selected_owner_frames": dense_metadata["selected_owner_frames"],
        "source_pass_s": round(source_s, 3),
        "dense_pass_s": round(dense_s, 3),
        "total_s": round(time.perf_counter() - started, 3),
    }
    if problems:
        raise AssertionError(json.dumps(report))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--index", type=Path, action="append", help="source index.json (repeat; later wins on duplicates)"
    )
    parser.add_argument("--tags", type=Path, help="replay tag file (the subset)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays-per-shard", type=int, default=16)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compression-level", type=int, default=DEFAULT_COMPRESSION_LEVEL)
    parser.add_argument("--manifest-workers", type=int, default=32)
    parser.add_argument(
        "--spill-dir",
        type=Path,
        help="per-replay spill files (default <output>/.spill; ~1/90 of the subset's raw bytes)",
    )
    parser.add_argument("--keep-spill", action="store_true")
    parser.add_argument(
        "--tolerate-contract-key",
        action="append",
        default=[],
        help=f"also tolerate this contract key differing between sources (default: {sorted(TOLERATED_CONTRACT_KEYS)})",
    )
    parser.add_argument("--verify", action="store_true", help="after repacking, re-read everything and compare")
    parser.add_argument("--verify-only", action="store_true", help="only verify an existing <output>")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.verify_only:
        if not args.index or args.tags is None:
            parser.error("--index and --tags are required unless --verify-only")
        record = repack(
            args.index,
            args.tags,
            args.output,
            replays_per_shard=args.replays_per_shard,
            workers=args.workers,
            seed=args.seed,
            compression_level=args.compression_level,
            manifest_workers=args.manifest_workers,
            spill_dir=args.spill_dir,
            keep_spill=args.keep_spill,
            log=True,
            tolerate_contract_keys=args.tolerate_contract_key,
        )
        summary = {key: value for key, value in record.items() if key not in ("assignment", "replays")}
        print(json.dumps({"event": "repacked", **summary}, sort_keys=True), flush=True)
    if args.verify or args.verify_only:
        print(
            json.dumps(verify(args.output, workers=args.workers, manifest_workers=args.manifest_workers, log=True)),
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
