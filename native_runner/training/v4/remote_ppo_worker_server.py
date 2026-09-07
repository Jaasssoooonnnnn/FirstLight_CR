"""CPU-node service that owns PPO environment/tensorizer worker processes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import multiprocessing as mp
import os
import signal
import socket
import time
from typing import Mapping

from .async_cluster_self_play import AsyncResidentEngineSpecV4, _engine_worker
from .remote_worker_transport import RemoteWorkerConnection


def _worker_cpu_group(
    allowed_cpus: tuple[int, ...], *, local_index: int, engine_count: int, width: int, allow_sharing: bool = False
) -> tuple[int, ...]:
    cpu_span = len(allowed_cpus) if allow_sharing else engine_count
    if (
        engine_count <= 0
        or len(allowed_cpus) < (width if allow_sharing else engine_count)
        or not 0 <= local_index < engine_count
        or width <= 0
        or cpu_span % width != 0
    ):
        raise ValueError("remote worker CPU grouping is invalid")
    stride = cpu_span // width
    group = tuple(allowed_cpus[(local_index + offset * stride) % cpu_span] for offset in range(width))
    if len(group) != len(set(group)):
        raise ValueError("remote worker CPU group contains duplicates")
    return group


def _preload_static_catalogs() -> None:
    from .factory import production_semantic_bundle
    from .ppo_runtime import ppo_environment_config_v4, ppo_ruleset_id_v4

    production_semantic_bundle()
    ppo_ruleset_id_v4(ppo_environment_config_v4())


def _prepare_worker_process(listener_fd: int) -> None:
    """Drop parent-only process state inherited through ``fork``."""

    os.close(listener_fd)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _run_worker(connection: RemoteWorkerConnection, request: Mapping[str, object], listener_fd: int) -> None:
    _prepare_worker_process(listener_fd)
    spec = request.get("spec")
    assignments = request.get("assignments")
    if not isinstance(spec, AsyncResidentEngineSpecV4) or not isinstance(assignments, tuple):
        raise TypeError("remote PPO worker start request is invalid")
    opening_reward = request.get("hog_deploy_opening_reward")
    _engine_worker(
        connection,  # type: ignore[arg-type]
        spec,
        assignments,
        float(request["gamma_per_decision"]),
        float(request["shaping_beta"]),
        float(request["mutual_elixir_overflow_penalty"]),
        float(request["mutual_elixir_overflow_grace"]),
        float(request["unilateral_elixir_overflow_penalty"]),
        float(request["elixir_overflow_step_penalty_cap"]),
        str(request["policy_state_id"]),
        bool(request["validate_tensors"]),
        int(request["max_decision_steps"]),
        float(request.get("hog_deploy_reward", 0.0)),
        float(request.get("hog_deploy_reward_episode_cap", 0.1)),
        None if opening_reward is None else float(opening_reward),
        float(request.get("first_hog_timing_reward_max", 0.0)),
        float(request.get("first_hog_timing_start_seconds", 10.0)),
        float(request.get("first_hog_timing_deadline_seconds", 30.0)),
        float(request.get("first_hog_missed_deadline_penalty", 0.0)),
        float(request.get("first_hog_deferral_penalty", 0.0)),
        float(request.get("first_hog_deferral_penalty_episode_cap", 0.0)),
        **(
            {"fireball_king_activation_penalty": float(request["fireball_king_activation_penalty"])}
            if request.get("fireball_king_activation_penalty", 0.0)
            else {}
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--port", type=int, default=29_900)
    parser.add_argument("--global-start", type=int, required=True)
    parser.add_argument("--engine-count", type=int, required=True)
    parser.add_argument("--engine-base-port", type=int, default=28_000)
    parser.add_argument("--preserve-engine-host", action="store_true")
    parser.add_argument("--worker-affinity-width", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--preload-static-catalogs", action="store_true")
    parser.add_argument("--allow-worker-cpu-sharing", action="store_true")
    args = parser.parse_args()
    if args.global_start < 0 or not 1 <= args.engine_count <= 64:
        raise ValueError("remote PPO worker engine range is invalid")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    allowed_cpus = tuple(sorted(os.sched_getaffinity(0)))
    minimum_cpus = args.worker_affinity_width if args.allow_worker_cpu_sharing else args.engine_count
    if len(allowed_cpus) < minimum_cpus:
        raise RuntimeError("remote PPO worker service received too few CPUs")
    affinity_span = len(allowed_cpus) if args.allow_worker_cpu_sharing else args.engine_count
    if affinity_span % args.worker_affinity_width != 0:
        raise ValueError("worker affinity width must divide the CPU span")
    if args.preload_static_catalogs:
        preload_started = time.perf_counter()
        _preload_static_catalogs()
        print(
            {
                "event": "remote_ppo_worker_static_catalogs_ready",
                "elapsed_seconds": time.perf_counter() - preload_started,
            },
            flush=True,
        )
    context = mp.get_context("fork")
    workers: dict[int, mp.Process] = {}
    stopping = False

    def stop(_signal: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind_host, args.port))
    listener.listen(args.engine_count * 2)
    listener.settimeout(1.0)
    print(
        {
            "event": "remote_ppo_worker_server_ready",
            "host": socket.gethostname().split(".", 1)[0],
            "port": args.port,
            "global_start": args.global_start,
            "engine_count": args.engine_count,
            "cpu_count": len(allowed_cpus),
        },
        flush=True,
    )
    try:
        while not stopping:
            for engine_index, process in tuple(workers.items()):
                if not process.is_alive():
                    process.join(timeout=0.0)
                    del workers[engine_index]
            try:
                sock, _address = listener.accept()
            except TimeoutError:
                continue
            connection = RemoteWorkerConnection(sock)
            try:
                request = connection.recv()
                if not isinstance(request, Mapping) or request.get("kind") != "start":
                    raise TypeError("remote PPO worker expected a start request")
                original_spec = request.get("spec")
                if not isinstance(original_spec, AsyncResidentEngineSpecV4):
                    raise TypeError("remote PPO worker spec is invalid")
                endpoint_port = original_spec.base_port + original_spec.engine_index
                local_index = endpoint_port - (args.engine_base_port + args.global_start)
                if not 0 <= local_index < args.engine_count:
                    raise ValueError("remote PPO worker engine is outside this shard")
                prior = workers.get(endpoint_port)
                if prior is not None:
                    # A new start request is the exact ownership handoff for
                    # this endpoint. Do not serially wait one second for each
                    # abandoned worker after a trainer failure; 64 such waits
                    # were dominating the next wave's startup.
                    if prior.is_alive():
                        prior.terminate()
                    prior.join(timeout=5.0)
                    if prior.is_alive():
                        raise RuntimeError("prior remote PPO worker resisted termination")
                worker_cpus = _worker_cpu_group(
                    allowed_cpus,
                    local_index=local_index,
                    engine_count=args.engine_count,
                    width=args.worker_affinity_width,
                    allow_sharing=args.allow_worker_cpu_sharing,
                )
                local_spec = replace(
                    original_spec,
                    cpu=worker_cpus[0],
                    worker_cpus=worker_cpus,
                    host=(original_spec.host if args.preserve_engine_host else "127.0.0.1"),
                    worker_host=None,
                    worker_port=None,
                )
                local_request = dict(request)
                local_request["spec"] = local_spec
                process = context.Process(
                    target=_run_worker,
                    args=(connection, local_request, listener.fileno()),
                    name=f"remote-ppo-engine-{original_spec.engine_index}",
                )
                process.start()
                workers[endpoint_port] = process
                connection.close_local_copy()
            except BaseException as error:
                try:
                    connection.send({"kind": "error", "error": repr(error)})
                except BaseException:
                    pass
                connection.close()
    finally:
        listener.close()
        for process in workers.values():
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 10.0
        for process in workers.values():
            process.join(timeout=max(0.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
