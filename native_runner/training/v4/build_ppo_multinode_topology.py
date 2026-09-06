"""Build one strict trainer manifest from resident engine-node manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


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


def build_topology(
    node_manifests: Sequence[Mapping[str, object]],
    *,
    trainer_job_id: str,
    trainer_host: str,
    gpu_count: int,
    worker_cpu_pool: Sequence[int],
    control_cpus: Sequence[int],
) -> dict[str, object]:
    if not trainer_job_id or not trainer_host or gpu_count <= 0:
        raise ValueError("trainer identity and GPU count are required")
    workers = tuple(int(value) for value in worker_cpu_pool)
    controls = tuple(int(value) for value in control_cpus)
    if (
        not workers
        or not controls
        or len(workers) != len(set(workers))
        or len(controls) != len(set(controls))
        or set(workers).intersection(controls)
    ):
        raise ValueError("worker and control CPU pools must be unique and disjoint")
    if len(controls) < gpu_count:
        raise ValueError("one trainer control CPU is required per GPU rank")
    if not node_manifests:
        raise ValueError("at least one engine-node manifest is required")

    engines: list[Mapping[str, object]] = []
    attestation_digests: set[str] = set()
    decision_ticks: set[int] = set()
    resident_slots: set[int] = set()
    for manifest in node_manifests:
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
        engines.extend(
            {
                **row,
                "worker_port": int(row.get("worker_port", manifest_worker_port)),
                "worker_host": str(row.get("worker_host", manifest_worker_host or row.get("host", ""))).strip(),
            }
            for row in normalized
        )
        attestation_digests.add(str(manifest.get("attestation_digest", "")))
        decision_ticks.add(int(manifest.get("decision_ticks", -1)))
        resident_slots.add(int(manifest.get("resident_slots_per_engine", -1)))

    if attestation_digests == {""} or len(attestation_digests) != 1:
        raise ValueError("engine nodes do not share one attestation digest")
    if decision_ticks != {5}:
        raise ValueError("engine nodes do not use five-tick decisions")
    if resident_slots != {8}:
        raise ValueError("engine nodes do not expose eight resident slots")

    ordered = sorted(engines, key=lambda row: int(row["engine_index"]))
    indices = tuple(int(row["engine_index"]) for row in ordered)
    if indices != tuple(range(len(ordered))):
        raise ValueError("global engine indices must be contiguous from zero")
    hosts = tuple(str(row["host"]).strip() for row in ordered)
    ports = tuple(int(row["port"]) for row in ordered)
    worker_hosts = tuple(str(row["worker_host"]).strip() for row in ordered)
    worker_ports = tuple(int(row["worker_port"]) for row in ordered)
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

    worker_cpus = [workers[index % len(workers)] for index in indices]
    return {
        "ok": True,
        "schema": "v4-ppo-multinode-topology.v1",
        "trainer_slurm_job_id": trainer_job_id,
        "trainer_host": trainer_host,
        "gpus": gpu_count,
        "engine_count": len(ordered),
        "resident_slots_per_engine": 8,
        "decision_ticks": 5,
        "engine_hosts": list(hosts),
        "engine_ports": list(ports),
        "engine_worker_hosts": list(worker_hosts),
        "engine_worker_ports": list(worker_ports),
        "worker_cpus": worker_cpus,
        "control_cpus": ",".join(str(value) for value in controls),
        "rank_engine_indices": [list(range(rank, len(ordered), gpu_count)) for rank in range(gpu_count)],
        "attestation_digest": next(iter(attestation_digests)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-ready", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-job-id", required=True)
    parser.add_argument("--trainer-host", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--worker-cpus", required=True)
    parser.add_argument("--control-cpus", required=True)
    args = parser.parse_args()

    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in args.node_ready]
    topology = build_topology(
        manifests,
        trainer_job_id=args.trainer_job_id,
        trainer_host=args.trainer_host,
        gpu_count=args.gpus,
        worker_cpu_pool=parse_cpu_set(args.worker_cpus),
        control_cpus=parse_cpu_set(args.control_cpus),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(topology, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
