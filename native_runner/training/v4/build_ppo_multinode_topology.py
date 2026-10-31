"""Build one strict trainer manifest from resident engine-node manifests.

Two node-manifest flavours are accepted:

* legacy emulator node-ready files: every endpoint row carries a global ``engine_index``
  (contiguous from zero across all nodes);
* standalone (zygote) node-ready files written by ``ppo/ppo_engines.sh``: endpoint rows carry the
  node-local ``engine_id`` (== the CPU the engine pins itself to) and their own worker-service
  ``worker_host``/``worker_port``; global indices are assigned here in node order, then ``engine_id``
  order.

Any number of engines per node, any number of nodes, and any (uniform) resident-slot count per engine
are supported; the PPO single-slot setup uses ``resident_slots_per_engine == 1``.

``worker_cpus`` keeps its trainer-side meaning (CPUs of the *trainer* allocation that
``_bind_rank_cpus`` adds to a rank's affinity; the remote worker service re-pins the real worker).
``--worker-cpus rank-control`` fills it with the owning rank's first control CPU, so a rank is bound to
its control CPUs only -- required when engines share the trainer node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

SCHEMA = "v4-ppo-multinode-topology.v1"
BUILDER_VERSION = 2
STANDALONE_BACKEND = "cr_engine_standalone-zygote"
# The standalone zygote itself runs as "engine23" and keeps 127.0.0.1:(26789+23); a child 23 cannot
# bind its port, so engine id 23 must never appear in a standalone pool.
STANDALONE_RESERVED_ENGINE_ID = 23
RANK_CONTROL = "rank-control"


def parse_cpu_set(value: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if start > end:
                raise ValueError("CPU range is reversed")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    if not cpus or len(cpus) != len(set(cpus)) or min(cpus) < 0:
        raise ValueError("CPU set must contain unique nonnegative CPUs")
    return tuple(cpus)


def _short_host(value: object) -> str:
    return str(value).strip().split(".", 1)[0]


def _uniform(values: set[object], what: str) -> object:
    if len(values) != 1:
        raise ValueError(f"engine nodes do not share one {what}")
    return next(iter(values))


def build_topology(
    node_manifests: Sequence[Mapping[str, object]],
    *,
    trainer_job_id: str,
    trainer_host: str,
    gpu_count: int,
    worker_cpu_pool: Sequence[int] | None = None,
    control_cpus: Sequence[int],
    lanes_per_engine: int | None = None,
    worker_cpu_mode: str = "pool",
    cuda_devices: int | None = None,
) -> dict[str, object]:
    """``gpu_count`` is the trainer WORLD SIZE (number of ranks; field ``gpus``, kept for compatibility).

    ``cuda_devices`` is the number of physical GPUs those ranks share (default: one GPU per rank).  With
    several ranks per GPU, rank r uses local CUDA device ``r // (world_size / cuda_devices)`` (contiguous
    blocks: 4 ranks on 2 GPUs -> ranks 0-1 on cuda:0, 2-3 on cuda:1), published as ``rank_cuda_device``.
    """
    if not trainer_job_id or not trainer_host or gpu_count <= 0:
        raise ValueError("trainer identity and GPU count are required")
    devices = gpu_count if cuda_devices is None else int(cuda_devices)
    if devices <= 0 or devices > gpu_count or gpu_count % devices != 0:
        raise ValueError("world size must be a positive multiple of the CUDA device count")
    ranks_per_device = gpu_count // devices
    if worker_cpu_mode not in {"pool", RANK_CONTROL}:
        raise ValueError("worker CPU mode must be 'pool' or 'rank-control'")
    workers = tuple(int(value) for value in (worker_cpu_pool or ()))
    controls = tuple(int(value) for value in control_cpus)
    if worker_cpu_mode == "pool":
        if (
            not workers
            or not controls
            or len(workers) != len(set(workers))
            or len(controls) != len(set(controls))
            or set(workers).intersection(controls)
        ):
            raise ValueError("worker and control CPU pools must be unique and disjoint")
    else:
        if workers:
            raise ValueError("rank-control worker CPUs take no explicit worker CPU pool")
        if not controls or len(controls) != len(set(controls)):
            raise ValueError("control CPUs must be unique")
    if len(controls) < gpu_count:
        raise ValueError("one trainer control CPU is required per GPU rank")
    if lanes_per_engine is not None and lanes_per_engine <= 0:
        raise ValueError("lanes per engine must be positive")
    if not node_manifests:
        raise ValueError("at least one engine-node manifest is required")

    rows_by_node: list[list[dict[str, object]]] = []
    attestation_digests: set[str] = set()
    decision_ticks: set[int] = set()
    resident_slots: set[int] = set()
    backends: set[str] = set()
    engine_bins: set[str] = set()
    snapshots: set[str] = set()
    node_summaries: list[dict[str, object]] = []
    for node_position, manifest in enumerate(node_manifests):
        if manifest.get("ok") is not True:
            raise ValueError("engine-node manifest is not ready")
        raw_engines = manifest.get("engines")
        if not isinstance(raw_engines, list) or not raw_engines:
            raise ValueError("engine-node manifest has no endpoints")
        normalized = tuple(row for row in raw_engines if isinstance(row, Mapping))
        if len(normalized) != len(raw_engines):
            raise ValueError("engine-node endpoint row is invalid")
        manifest_worker_port = int(manifest.get("remote_worker_port", 29_900))
        if not 1 <= manifest_worker_port <= 65_535:
            raise ValueError("engine-node worker service port is invalid")
        manifest_worker_host = str(manifest.get("remote_worker_host", "")).strip()
        backend = str(manifest.get("engine_backend", "")).strip()
        backends.add(backend)
        node_rows = [
            {
                **row,
                "worker_port": int(row.get("worker_port", manifest_worker_port)),
                "worker_host": str(row.get("worker_host", manifest_worker_host or row.get("host", ""))).strip(),
                "_node_position": node_position,
            }
            for row in normalized
        ]
        if backend == STANDALONE_BACKEND:
            reserved = [row for row in node_rows if int(row.get("engine_id", -1)) == STANDALONE_RESERVED_ENGINE_ID]
            if reserved:
                raise ValueError("standalone engine id 23 is reserved by the zygote")
        rows_by_node.append(node_rows)
        attestation_digests.add(str(manifest.get("attestation_digest", "")))
        decision_ticks.add(int(manifest.get("decision_ticks", -1)))
        resident_slots.add(int(manifest.get("resident_slots_per_engine", -1)))
        if manifest.get("engine_bin_sha256"):
            engine_bins.add(str(manifest["engine_bin_sha256"]))
        if manifest.get("snapshot"):
            snapshots.add(str(manifest["snapshot"]))
        node_summaries.append(
            {
                "host": _short_host(manifest.get("host", node_rows[0].get("host", ""))),
                "slurm_job_id": manifest.get("slurm_job_id"),
                "engine_count": len(node_rows),
                "launch": manifest.get("launch"),
                "worker_services": manifest.get("worker_services"),
            }
        )

    if attestation_digests == {""} or len(attestation_digests) != 1:
        raise ValueError("engine nodes do not share one attestation digest")
    if decision_ticks != {5}:
        raise ValueError("engine nodes do not use five-tick decisions")
    lanes = int(_uniform(resident_slots, "resident slot count"))  # type: ignore[arg-type]
    if lanes <= 0:
        raise ValueError("engine nodes do not declare a positive resident slot count")
    if lanes_per_engine is not None and lanes != lanes_per_engine:
        raise ValueError(f"engine nodes expose {lanes} resident slots, expected {lanes_per_engine}")
    backend = str(_uniform(backends, "engine backend"))
    if len(engine_bins) > 1:
        raise ValueError("engine nodes do not share one engine binary")
    if len(snapshots) > 1:
        raise ValueError("engine nodes do not share one code snapshot")

    all_rows = [row for node_rows in rows_by_node for row in node_rows]
    with_index = sum("engine_index" in row for row in all_rows)
    if with_index == len(all_rows):
        ordered = sorted(all_rows, key=lambda row: int(row["engine_index"]))  # type: ignore[arg-type]
        indices = tuple(int(row["engine_index"]) for row in ordered)  # type: ignore[arg-type]
        if indices != tuple(range(len(ordered))):
            raise ValueError("global engine indices must be contiguous from zero")
    elif with_index == 0:
        if any("engine_id" not in row for row in all_rows):
            raise ValueError("endpoint rows need either a global engine_index or a node-local engine_id")
        ordered = [
            row
            for node_rows in rows_by_node
            for row in sorted(node_rows, key=lambda item: int(item["engine_id"]))  # type: ignore[arg-type]
        ]
        indices = tuple(range(len(ordered)))
    else:
        raise ValueError("endpoint rows mix global engine_index and node-local engine_id")

    engine_count = len(ordered)
    if engine_count < gpu_count:
        raise ValueError("every GPU rank needs at least one engine")
    hosts = tuple(str(row["host"]).strip() for row in ordered)
    ports = tuple(int(row["port"]) for row in ordered)  # type: ignore[arg-type]
    worker_hosts = tuple(str(row["worker_host"]).strip() for row in ordered)
    worker_ports = tuple(int(row["worker_port"]) for row in ordered)  # type: ignore[arg-type]
    if any(not host for host in hosts):
        raise ValueError("engine endpoint host is empty")
    if any(not host for host in worker_hosts):
        raise ValueError("engine worker service host is empty")
    endpoints = tuple(zip(hosts, ports, strict=True))
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("engine endpoints are not unique")
    if any(not 1 <= port <= 65_535 for port in ports):
        raise ValueError("engine endpoint port is invalid")
    if any(not 1 <= port <= 65_535 for port in worker_ports):
        raise ValueError("engine worker service port is invalid")

    engine_ids: list[int] | None = None
    engine_cpus: list[int] | None = None
    if all("engine_id" in row for row in ordered):
        engine_ids = [int(row["engine_id"]) for row in ordered]  # type: ignore[arg-type]
        engine_cpus = [int(row.get("engine_cpu", row["engine_id"])) for row in ordered]  # type: ignore[arg-type]
        placed = list(zip((_short_host(host) for host in hosts), engine_cpus, strict=True))
        if len(placed) != len(set(placed)):
            raise ValueError("two engines are pinned to the same CPU of one node")
        trainer_short = _short_host(trainer_host)
        colocated = {cpu for host, cpu in placed if host == trainer_short}
        if colocated.intersection(controls):
            raise ValueError("trainer control CPUs overlap engine CPUs on the trainer node")
        if worker_cpu_mode == "pool" and colocated.intersection(workers):
            raise ValueError("trainer worker CPU pool overlaps engine CPUs on the trainer node")

    rank_engine_indices = [list(range(rank, engine_count, gpu_count)) for rank in range(gpu_count)]
    if worker_cpu_mode == "pool":
        worker_cpus = [workers[index % len(workers)] for index in indices]
    else:
        # _bind_rank_cpus stripes control CPUs as control[rank::world]; giving every engine the first
        # CPU of its owner's stripe keeps each rank's affinity equal to its control CPUs.
        owner = {index: rank for rank, owned in enumerate(rank_engine_indices) for index in owned}
        worker_cpus = [controls[owner[index]] for index in indices]

    topology: dict[str, object] = {
        "ok": True,
        "schema": SCHEMA,
        "topology_builder_version": BUILDER_VERSION,
        "trainer_slurm_job_id": trainer_job_id,
        "trainer_host": trainer_host,
        "gpus": gpu_count,
        "world_size": gpu_count,
        "cuda_devices": devices,
        "ranks_per_cuda_device": ranks_per_device,
        "rank_cuda_device": [rank // ranks_per_device for rank in range(gpu_count)],
        "engine_count": engine_count,
        "resident_slots_per_engine": lanes,
        "lanes_per_engine": lanes,
        "decision_ticks": 5,
        "engine_backend": backend or "unspecified",
        "engine_hosts": list(hosts),
        "engine_ports": list(ports),
        "engine_worker_hosts": list(worker_hosts),
        "engine_worker_ports": list(worker_ports),
        "worker_cpus": worker_cpus,
        "worker_cpu_mode": worker_cpu_mode,
        "control_cpus": ",".join(str(value) for value in controls),
        "rank_engine_indices": rank_engine_indices,
        "attestation_digest": next(iter(attestation_digests)),
        "nodes": node_summaries,
    }
    if engine_ids is not None and engine_cpus is not None:
        topology["engine_ids"] = engine_ids
        topology["engine_cpus"] = engine_cpus
    if engine_bins:
        topology["engine_bin_sha256"] = next(iter(engine_bins))
    if snapshots:
        topology["snapshot"] = next(iter(snapshots))
    return topology


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node-ready", type=Path, nargs="+", required=True, help="node manifests, in node order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-job-id", required=True)
    parser.add_argument("--trainer-host", required=True)
    parser.add_argument("--gpus", type=int, required=True, help="trainer world size (ranks)")
    parser.add_argument("--cuda-devices", type=int, help="physical GPUs shared by the ranks (default: --gpus)")
    parser.add_argument(
        "--worker-cpus",
        required=True,
        help="trainer-side worker CPU pool (e.g. 31-32) or 'rank-control' (bind each rank to its control CPUs)",
    )
    parser.add_argument("--control-cpus", required=True)
    parser.add_argument("--lanes-per-engine", type=int, help="expected resident slots per engine (1 = single slot)")
    args = parser.parse_args()

    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in args.node_ready]
    rank_control = args.worker_cpus.strip() == RANK_CONTROL
    topology = build_topology(
        manifests,
        trainer_job_id=args.trainer_job_id,
        trainer_host=args.trainer_host,
        gpu_count=args.gpus,
        worker_cpu_pool=None if rank_control else parse_cpu_set(args.worker_cpus),
        control_cpus=parse_cpu_set(args.control_cpus),
        lanes_per_engine=args.lanes_per_engine,
        worker_cpu_mode=RANK_CONTROL if rank_control else "pool",
        cuda_devices=args.cuda_devices,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(topology, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
