"""Single-slot standalone-engine PPO topologies (any engine count, any node count)."""

from __future__ import annotations

import json
import os
import sys

import pytest

from native_runner.training.v4 import build_ppo_multinode_topology as builder
from native_runner.training.v4.build_ppo_multinode_topology import build_topology
from native_runner.training.v4.train_ppo_self_play_cluster import (
    _bind_rank_cpus,
    _ready_engine_hosts,
    _ready_engine_ports,
    _ready_engine_worker_hosts,
    _ready_engine_worker_ports,
    _ready_rank_engine_indices,
)

GPU_NODE_IDS = (*range(0, 23), *range(24, 56))  # 55 engines on gpu-a; id 23 belongs to the zygote


def _standalone_node(
    host: str,
    ids: tuple[int, ...],
    *,
    worker_port_for=lambda engine_id: 29_950,
    slots: int = 1,
    digest: str = "attestation",
    engine_bin: str = "5ec20979",
) -> dict[str, object]:
    return {
        "ok": True,
        "schema": "ppo-standalone-node-ready.v1",
        "engine_backend": builder.STANDALONE_BACKEND,
        "host": host,
        "slurm_job_id": "job",
        "decision_ticks": 5,
        "resident_slots_per_engine": slots,
        "attestation_digest": digest,
        "engine_bin_sha256": engine_bin,
        "snapshot": "/snap/ppo",
        "engines": [
            {
                "engine_id": engine_id,
                "engine_cpu": engine_id,
                "host": host,
                "port": 26_789 + engine_id,
                "worker_host": host,
                "worker_port": worker_port_for(engine_id),
            }
            for engine_id in ids
        ],
    }


def _gpu_node() -> dict[str, object]:
    return _standalone_node("gpu-a", GPU_NODE_IDS, worker_port_for=lambda i: 29_950 if i < 28 else 29_951)


def _cpu_node(host: str, count: int) -> dict[str, object]:
    ids = tuple(cpu for cpu in range(64) if cpu != 23)[:count]
    return _standalone_node(host, ids)


def test_gpu_node_single_slot_layout_routes_each_engine_to_its_worker_service() -> None:
    topology = build_topology(
        (_gpu_node(),),
        trainer_job_id="1001",
        trainer_host="gpu-a",
        gpu_count=4,
        control_cpus=(56, 57, 58, 59),
        lanes_per_engine=1,
        worker_cpu_mode="rank-control",
    )

    assert topology["schema"] == "v4-ppo-multinode-topology.v1"
    assert topology["engine_count"] == 55
    assert topology["resident_slots_per_engine"] == 1
    assert topology["lanes_per_engine"] == 1
    assert topology["engine_backend"] == builder.STANDALONE_BACKEND
    assert topology["engine_ids"] == list(GPU_NODE_IDS)
    assert 23 not in topology["engine_ids"]
    assert topology["engine_cpus"] == list(GPU_NODE_IDS)
    assert topology["engine_ports"] == [26_789 + i for i in GPU_NODE_IDS]
    assert topology["engine_worker_ports"] == [29_950 if i < 28 else 29_951 for i in GPU_NODE_IDS]
    assert set(topology["engine_hosts"]) == {"gpu-a"}
    assert topology["control_cpus"] == "56,57,58,59"
    assert topology["engine_bin_sha256"] == "5ec20979"
    assert topology["snapshot"] == "/snap/ppo"
    ranks = topology["rank_engine_indices"]
    assert [len(row) for row in ranks] == [14, 14, 14, 13]
    for rank, owned in enumerate(ranks):
        assert {topology["worker_cpus"][index] for index in owned} == {56 + rank}


def test_multinode_single_slot_topology_orders_nodes_and_balances_ranks() -> None:
    nodes = (_gpu_node(), _cpu_node("cpu-a", 59), _cpu_node("cpu-b", 7))
    topology = build_topology(
        nodes,
        trainer_job_id="1001",
        trainer_host="gpu-a.cluster.example",
        gpu_count=4,
        control_cpus=(56, 57, 58, 59),
        lanes_per_engine=1,
        worker_cpu_mode="rank-control",
    )
    count = 55 + 59 + 7
    assert topology["engine_count"] == count
    hosts = topology["engine_hosts"]
    assert hosts == ["gpu-a"] * 55 + ["cpu-a"] * 59 + ["cpu-b"] * 7
    assert topology["engine_worker_hosts"] == hosts
    assert topology["engine_ids"][55:58] == [0, 1, 2]
    assert topology["engine_ids"][55 + 22 : 55 + 24] == [22, 24]
    assert [node["host"] for node in topology["nodes"]] == ["gpu-a", "cpu-a", "cpu-b"]
    sizes = [len(row) for row in topology["rank_engine_indices"]]
    assert max(sizes) - min(sizes) <= 1
    # Every rank drives engines on every node (striped), and the trainer readers accept the result.
    for owned in topology["rank_engine_indices"]:
        assert {hosts[index] for index in owned} == {"gpu-a", "cpu-a", "cpu-b"}
    ready = json.loads(json.dumps(topology))
    assert _ready_engine_hosts(ready, engine_count=count) == tuple(hosts)
    assert len(_ready_engine_ports(ready, engine_count=count)) == count
    assert _ready_engine_worker_hosts(ready, engine_count=count) == tuple(hosts)
    assert len(_ready_engine_worker_ports(ready, engine_count=count)) == count
    assert len(_ready_rank_engine_indices(ready, engine_count=count, world_size=4)) == 4


def test_rank_control_binds_each_rank_to_its_control_cpus_only(monkeypatch: pytest.MonkeyPatch) -> None:
    bound: list[set[int]] = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, cpus: bound.append(set(cpus)), raising=False)
    topology = build_topology(
        (_gpu_node(), _cpu_node("cpu-a", 20)),
        trainer_job_id="1001",
        trainer_host="gpu-a",
        gpu_count=4,
        control_cpus=(56, 57, 58, 59, 60, 61, 62, 63),
        lanes_per_engine=1,
        worker_cpu_mode="rank-control",
    )
    for rank, owned in enumerate(topology["rank_engine_indices"]):
        assigned = _bind_rank_cpus(topology, rank, 4, owned, topology["worker_cpus"])
        assert set(assigned) == {56 + rank, 60 + rank}
        assert not set(assigned).intersection(GPU_NODE_IDS)
    assert len(bound) == 4


def test_single_engine_single_gpu_topology() -> None:
    topology = build_topology(
        (_standalone_node("cpu-a", (5,)),),
        trainer_job_id="job",
        trainer_host="gpu-a",
        gpu_count=1,
        control_cpus=(56,),
        worker_cpu_mode="rank-control",
    )
    assert topology["engine_count"] == 1
    assert topology["rank_engine_indices"] == [[0]]
    assert topology["worker_cpus"] == [56]


def test_legacy_eight_slot_pool_mode_still_supported() -> None:
    node = _standalone_node("cpu-a", (0, 1, 2), slots=8)
    topology = build_topology(
        (node,),
        trainer_job_id="job",
        trainer_host="gpu-a",
        gpu_count=1,
        worker_cpu_pool=(31, 32),
        control_cpus=(33,),
    )
    assert topology["resident_slots_per_engine"] == 8
    assert topology["worker_cpus"] == [31, 32, 31]
    assert topology["worker_cpu_mode"] == "pool"


@pytest.mark.parametrize(
    ("nodes", "kwargs", "message"),
    [
        (
            (_standalone_node("gpu-a", (22, 23, 24)),),
            {},
            "id 23",
        ),
        (
            (_gpu_node(),),
            {"control_cpus": (55, 56, 57, 58)},
            "control CPUs overlap",
        ),
        (
            (_gpu_node(),),
            {"worker_cpu_mode": "pool", "worker_cpu_pool": (0, 1), "control_cpus": (56, 57, 58, 59)},
            "pool overlaps",
        ),
        (
            (_gpu_node(), _cpu_node("cpu-a", 4) | {"resident_slots_per_engine": 8}),
            {},
            "resident slot count",
        ),
        (
            (_cpu_node("cpu-a", 4) | {"resident_slots_per_engine": 8},),
            {"lanes_per_engine": 1},
            "expected 1",
        ),
        (
            (_gpu_node(), _cpu_node("cpu-a", 4) | {"attestation_digest": "other"}),
            {},
            "attestation digest",
        ),
        (
            (_gpu_node(), _cpu_node("cpu-a", 4) | {"engine_bin_sha256": "b510b632"}),
            {},
            "engine binary",
        ),
        (
            (_gpu_node(), _cpu_node("cpu-a", 4) | {"snapshot": "/snap/other"}),
            {},
            "code snapshot",
        ),
        (
            (_standalone_node("cpu-a", (0, 1, 2)),),
            {},
            "at least one engine",
        ),
        (
            (_gpu_node(), _gpu_node()),
            {},
            "not unique",
        ),
        (
            (_gpu_node(),),
            {"worker_cpu_pool": (90,)},
            "no explicit worker CPU pool",
        ),
    ],
)
def test_single_slot_topology_rejections(nodes, kwargs, message) -> None:
    arguments = {
        "trainer_job_id": "1001",
        "trainer_host": "gpu-a",
        "gpu_count": 4,
        "control_cpus": (56, 57, 58, 59),
        "worker_cpu_mode": "rank-control",
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        build_topology(nodes, **arguments)


def test_rows_mixing_global_and_local_engine_ids_are_rejected() -> None:
    node = _cpu_node("cpu-a", 3)
    node["engines"][0]["engine_index"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="mix"):
        build_topology(
            (node,),
            trainer_job_id="job",
            trainer_host="gpu-a",
            gpu_count=1,
            control_cpus=(56,),
            worker_cpu_mode="rank-control",
        )


def test_cli_writes_single_slot_topology(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    first = tmp_path / "gpu-a.json"
    second = tmp_path / "cpu-a.json"
    first.write_text(json.dumps(_gpu_node()), encoding="utf-8")
    second.write_text(json.dumps(_cpu_node("cpu-a", 10)), encoding="utf-8")
    output = tmp_path / "out" / "topology.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_ppo_multinode_topology",
            "--node-ready",
            str(first),
            str(second),
            "--output",
            str(output),
            "--trainer-job-id",
            "1001",
            "--trainer-host",
            "gpu-a",
            "--gpus",
            "4",
            "--worker-cpus",
            "rank-control",
            "--control-cpus",
            "56-59",
            "--lanes-per-engine",
            "1",
        ],
    )
    assert builder.main() == 0
    capsys.readouterr()
    topology = json.loads(output.read_text(encoding="utf-8"))
    assert topology["engine_count"] == 65
    assert topology["worker_cpu_mode"] == "rank-control"
    assert topology["lanes_per_engine"] == 1
    assert not output.with_name("topology.json.tmp").exists()


def test_world_size_eight_on_two_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Several ranks share each GPU, one control CPU per rank, lanes split over all ranks."""

    bound: list[set[int]] = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, cpus: bound.append(set(cpus)), raising=False)
    nodes = tuple(_cpu_node(f"cs{600 + i}", 31) for i in range(22))
    topology = build_topology(
        nodes,
        trainer_job_id="1002",
        trainer_host="gb001",
        gpu_count=8,
        control_cpus=tuple(range(8)),
        lanes_per_engine=1,
        worker_cpu_mode="rank-control",
        cuda_devices=2,
    )
    assert topology["gpus"] == topology["world_size"] == 8
    assert topology["cuda_devices"] == 2
    assert topology["ranks_per_cuda_device"] == 4
    assert topology["rank_cuda_device"] == [0, 0, 0, 0, 1, 1, 1, 1]
    sizes = [len(row) for row in topology["rank_engine_indices"]]
    assert sum(sizes) == 22 * 31 and max(sizes) - min(sizes) <= 1
    for rank, owned in enumerate(topology["rank_engine_indices"]):
        assert set(_bind_rank_cpus(topology, rank, 8, owned, topology["worker_cpus"])) == {rank}
    ready = json.loads(json.dumps(topology))
    assert len(_ready_rank_engine_indices(ready, engine_count=22 * 31, world_size=8)) == 8


@pytest.mark.parametrize("devices", [0, 3, 9])
def test_world_size_must_be_a_multiple_of_cuda_devices(devices: int) -> None:
    with pytest.raises(ValueError, match="multiple"):
        build_topology(
            (_cpu_node("cs600", 31),),
            trainer_job_id="j",
            trainer_host="gb001",
            gpu_count=8,
            control_cpus=tuple(range(8)),
            worker_cpu_mode="rank-control",
            cuda_devices=devices,
        )


def test_default_is_one_rank_per_cuda_device() -> None:
    topology = build_topology(
        (_gpu_node(),),
        trainer_job_id="1001",
        trainer_host="gpu-a",
        gpu_count=4,
        control_cpus=(56, 57, 58, 59),
        worker_cpu_mode="rank-control",
    )
    assert topology["cuda_devices"] == 4 and topology["rank_cuda_device"] == [0, 1, 2, 3]
