from __future__ import annotations

import os

import pytest

from native_runner.training.v4.build_ppo_multinode_topology import (
    build_topology,
    parse_cpu_set,
)
from native_runner.training.v4.train_ppo_self_play_cluster import (
    _bind_rank_cpus,
    _ready_engine_hosts,
    _ready_engine_ports,
    _ready_rank_engine_indices,
    _ready_engine_worker_hosts,
    _ready_engine_worker_ports,
    _validate_topology,
)


def _node(host: str, start: int, count: int) -> dict[str, object]:
    return {
        "ok": True,
        "host": host,
        "slurm_job_id": "cpu-job",
        "decision_ticks": 5,
        "resident_slots_per_engine": 8,
        "attestation_digest": "attestation",
        "engines": [
            {
                "engine_index": index,
                "host": host,
                "port": 28_000 + index,
            }
            for index in range(start, start + count)
        ],
    }


def test_build_multinode_topology_assigns_remote_endpoints_and_local_workers() -> None:
    topology = build_topology(
        (_node("worker-a", 0, 2), _node("worker-b", 2, 2)),
        trainer_job_id="gpu-job",
        trainer_host="trainer-a",
        gpu_count=2,
        worker_cpu_pool=(31, 32),
        control_cpus=(33, 34),
    )

    assert topology["engine_count"] == 4
    assert topology["engine_hosts"] == [
        "worker-a",
        "worker-a",
        "worker-b",
        "worker-b",
    ]
    assert topology["engine_ports"] == [28_000, 28_001, 28_002, 28_003]
    assert topology["engine_worker_hosts"] == [
        "worker-a",
        "worker-a",
        "worker-b",
        "worker-b",
    ]
    assert topology["engine_worker_ports"] == [29_900] * 4
    assert topology["worker_cpus"] == [31, 32, 31, 32]
    assert topology["control_cpus"] == "33,34"


def test_build_multinode_topology_rejects_global_engine_gap() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        build_topology(
            (_node("worker-a", 0, 1), _node("worker-b", 2, 1)),
            trainer_job_id="gpu-job",
            trainer_host="trainer-a",
            gpu_count=1,
            worker_cpu_pool=(31,),
            control_cpus=(32,),
        )


def test_build_multinode_topology_routes_workers_to_an_offload_host() -> None:
    node = _node("worker-a", 0, 2)
    node["remote_worker_host"] = "trainer-a"
    node["remote_worker_port"] = 29_910

    topology = build_topology(
        (node,),
        trainer_job_id="gpu-job",
        trainer_host="trainer-a",
        gpu_count=1,
        worker_cpu_pool=(31,),
        control_cpus=(32,),
    )

    assert topology["engine_hosts"] == ["worker-a", "worker-a"]
    assert topology["engine_worker_hosts"] == ["trainer-a", "trainer-a"]
    assert topology["engine_worker_ports"] == [29_910, 29_910]


def test_cpu_set_parser_supports_slurm_ranges() -> None:
    assert parse_cpu_set("31-33,40") == (31, 32, 33, 40)


def test_bind_rank_cpus_deduplicates_time_shared_worker_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound: list[set[int]] = []
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda _pid, cpus: bound.append(set(cpus)),
        raising=False,
    )

    assigned = _bind_rank_cpus(
        {"control_cpus": "40,41"},
        local_rank=0,
        world=2,
        engine_indices=(0, 2, 4, 6),
        worker_cpus=(0, 1, 2, 3, 0, 1, 2, 3),
    )

    assert assigned == (40, 0, 2)
    assert bound == [{0, 2, 40}]


def test_trainer_resolves_manifest_hosts_and_ports() -> None:
    ready = {
        "engine_hosts": ["worker-a", "worker-b"],
        "engine_ports": [28_000, 28_064],
    }
    assert _ready_engine_hosts(
        ready,
        engine_count=2,
    ) == ("worker-a", "worker-b")
    assert _ready_engine_ports(
        ready,
        engine_count=2,
    ) == (28_000, 28_064)
    assert _ready_engine_worker_ports(
        {"engine_worker_ports": [29_900, 29_901]},
        engine_count=2,
    ) == (29_900, 29_901)
    assert _ready_engine_worker_hosts(
        {"engine_worker_hosts": ["trainer-a", "trainer-a"]},
        engine_count=2,
    ) == ("trainer-a", "trainer-a")


def test_trainer_accepts_explicit_uneven_rank_engine_assignments() -> None:
    assert _ready_rank_engine_indices(
        {"rank_engine_indices": [[0], [1, 2], [3, 4]]},
        engine_count=5,
        world_size=3,
    ) == ((0,), (1, 2), (3, 4))


def test_trainer_rejects_incomplete_rank_engine_assignments() -> None:
    with pytest.raises(ValueError, match="every engine exactly once"):
        _ready_rank_engine_indices(
            {"rank_engine_indices": [[0], [1, 2], [2, 3]]},
            engine_count=5,
            world_size=3,
        )


def test_production_topology_uses_manifest_allocation_identity(monkeypatch) -> None:
    ready = {
        "schema": "v4-ppo-multinode-topology.v1",
        "trainer_slurm_job_id": "gpu-job",
        "trainer_host": "trainer-a",
        "gpus": 8,
        "engine_count": 192,
        "resident_slots_per_engine": 8,
        "decision_ticks": 5,
    }
    monkeypatch.setenv("SLURM_JOB_ID", "gpu-job")
    monkeypatch.setattr(
        "native_runner.training.v4.train_ppo_self_play_cluster.socket.gethostname",
        lambda: "trainer-a",
    )
    monkeypatch.setattr(
        "native_runner.training.v4.train_ppo_self_play_cluster.torch.cuda.device_count",
        lambda: 8,
    )

    assert _validate_topology(ready, world_size=8) == (192, 8)
