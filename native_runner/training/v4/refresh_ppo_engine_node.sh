#!/usr/bin/env bash
set -euo pipefail

# Replace the probe and restart engines inside an already-ready emulator shard.
# The owning launcher continues to own QEMU and endpoint proxies.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/../../.." && pwd)
source "${repository_root}/native_runner/local_config.sh"
root=${CR_AI_ROOT:-$repository_root}
export CR_AI_ROOT="$root"
export PYTHONPATH="$repository_root"
artifact=${1:?usage: refresh_ppo_engine_node.sh ARTIFACT GLOBAL_START [ENGINE_COUNT] [BASE_PORT] [EMULATOR_BASE_PORT]}
global_start=${2:?global engine start is required}
engine_count=${3:-${CR_PPO_ENGINE_COUNT:?Set CR_PPO_ENGINE_COUNT in .env}}
base_port=${4:-${CR_PPO_BASE_PORT:-28000}}
emulator_base_port=${5:-${CR_EMULATOR_BASE_PORT:-5554}}
python=${CR_AI_PYTHON:-python}
adb=${CR_TRAINING_ADB:-adb}
probe=${CR_PROBE:-${CR_PPO_PROBE:-$repository_root/native_runner/probe/out/libcrprobe.so}}
probe_installer=$script_dir/install_probe_idle_guests.py
engine_configurator=$script_dir/configure_emulator_engine_guests.py
package=nullsroyale.rel.free

if [[ ! ${SLURM_JOB_ID:-} ]]; then
  echo "engine refresh must run inside a Slurm allocation" >&2
  exit 90
fi
if ((global_start < 0 || engine_count <= 0 || engine_count > 64)); then
  echo "global start must be nonnegative and engine count must be 1..64" >&2
  exit 91
fi

max_guest_engines=${CR_PPO_ENGINES_PER_GUEST:?Set CR_PPO_ENGINES_PER_GUEST in .env}
if ((max_guest_engines < 1 || max_guest_engines > 6)); then
  echo "Engines per guest must be in 1..6" >&2
  exit 91
fi
guest_count=$(((engine_count + max_guest_engines - 1) / max_guest_engines))
if ((
  emulator_base_port < 5554 ||
  emulator_base_port % 2 != 0 ||
  emulator_base_port + (guest_count - 1) * 2 > 5682
)); then
  echo "emulator console ports must be even and within 5554..5682" >&2
  exit 91
fi
for required in \
  "$python" "$adb" "$probe" "$probe_installer" "$engine_configurator"; do
  if [[ ! -e $required ]]; then
    echo "missing engine refresh dependency: $required" >&2
    exit 92
  fi
done

serials=()
host_bases=()
guest_engines=()
remaining=$engine_count
engine_cursor=0
for ((guest=0; guest<guest_count; guest++)); do
  count=$max_guest_engines
  ((remaining < count)) && count=$remaining
  serials+=("emulator-$((emulator_base_port + guest * 2))")
  host_bases+=("$((base_port + global_start + engine_cursor))")
  guest_engines+=("$count")
  remaining=$((remaining - count))
  engine_cursor=$((engine_cursor + count))
done

mkdir -p "$artifact"
stamp=$(date +%Y%m%d-%H%M%S)
logs=$artifact/refresh-logs-$stamp
ready=$artifact/engine-node-ready.json
mkdir -p "$logs"
if [[ -e $ready ]]; then
  echo "refusing to overwrite existing refresh readiness: $ready" >&2
  exit 93
fi

export PYTHONPATH=$repository_root
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "$(date -Is) ENGINE_REFRESH_START job=$SLURM_JOB_ID host=$(hostname -s) start=$global_start engines=$engine_count"
for serial in "${serials[@]}"; do
  if [[ $("$adb" -s "$serial" get-state 2>/dev/null || true) != device ]]; then
    echo "refresh guest is unavailable: $serial" >&2
    exit 94
  fi
  "$adb" -s "$serial" shell am force-stop --user 0 "$package"
done

serial_csv=$(IFS=,; echo "${serials[*]}")
"$python" "$probe_installer" \
  --probe "$probe" \
  --serials "$serial_csv" \
  --output "$artifact/probe-refresh-$stamp.json"

source_reports=()
full_guests=$((engine_count / max_guest_engines))
remainder=$((engine_count % max_guest_engines))
configure_guest_batch=3
for ((batch_start=0; batch_start<full_guests; batch_start+=configure_guest_batch)); do
  batch_count=$configure_guest_batch
  ((batch_start + batch_count <= full_guests)) || batch_count=$((full_guests - batch_start))
  batch_serials=$(IFS=,; echo "${serials[*]:batch_start:batch_count}")
  batch_bases=$(IFS=,; echo "${host_bases[*]:batch_start:batch_count}")
  batch_report=$artifact/engine-refresh-$((batch_count * max_guest_engines))-batch-$batch_start-$stamp.json
  "$python" "$engine_configurator" \
    --serials "$batch_serials" \
    --host-bases "$batch_bases" \
    --engines "$max_guest_engines" \
    --output "$batch_report" \
    >"$logs/configure-batch-$batch_start.log"
  source_reports+=("$batch_report")
done
if ((remainder)); then
  last=$((guest_count - 1))
  remainder_report=$artifact/engine-refresh-$remainder-$stamp.json
  "$python" "$engine_configurator" \
    --serials "${serials[$last]}" \
    --host-bases "${host_bases[$last]}" \
    --engines "$remainder" \
    --output "$remainder_report" \
    >"$logs/configure-remainder.log"
  source_reports+=("$remainder_report")
fi

"$python" - \
  "$ready" "$global_start" "$engine_count" "$base_port" \
  "${source_reports[@]}" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

target = Path(sys.argv[1])
global_start = int(sys.argv[2])
engine_count = int(sys.argv[3])
base_port = int(sys.argv[4])
sources = tuple(Path(value) for value in sys.argv[5:])
reports = [json.loads(source.read_text()) for source in sources]
guests = [guest for report in reports for guest in report["guests"]]
endpoints = [row for guest in guests for row in guest["endpoints"]]
ports = sorted(int(row["host_port"]) for row in endpoints)
pids = {
    (guest["serial"], int(row["status"]["pid"]))
    for guest in guests
    for row in guest["endpoints"]
}
digests = {
    str(row["attestation"].get("attestation_digest", ""))
    for row in endpoints
}
expected_ports = list(
    range(base_port + global_start, base_port + global_start + engine_count)
)
checks = {
    "job_id": bool(os.environ.get("SLURM_JOB_ID")),
    "engine_count": sum(int(report["engine_count"]) for report in reports)
    == engine_count,
    "endpoint_count": len(endpoints) == engine_count,
    "unique_processes": len(pids) == engine_count,
    "contiguous_ports": ports == expected_ports,
    "resident_slots": all(
        int(row["status"].get("slotsPerEngine", -1)) == 8 for row in endpoints
    ),
    "production_ready": all(
        row["attestation"].get("production_ready") is True for row in endpoints
    ),
    "one_attestation": len(digests) == 1 and "" not in digests,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"engine refresh acceptance failed: {failed}")
host = socket.gethostname().split(".", 1)[0]
by_port = {int(row["host_port"]): row for row in endpoints}
summary = {
    "ok": True,
    "schema": "v4-ppo-engine-node.v1",
    "refresh_schema": "v4-ppo-engine-refresh.v1",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "host": host,
    "global_engine_start": global_start,
    "engine_count": engine_count,
    "resident_slots_per_engine": 8,
    "parallel_matches": engine_count * 8,
    "decision_ticks": 5,
    "engines": [
        {
            "engine_index": global_start + offset,
            "host": host,
            "port": port,
            "pid": int(by_port[port]["status"]["pid"]),
        }
        for offset, port in enumerate(expected_ports)
    ],
    "attestation_digest": next(iter(digests)),
    "source_reports": [str(source) for source in sources],
    "manifest_path": str(target),
    "checks": checks,
}
target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

echo "$(date -Is) ENGINE_REFRESH_READY start=$global_start engines=$engine_count"
