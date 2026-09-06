#!/usr/bin/env bash
set -euo pipefail

# Boot one node-local native engine shard for multi-node V4 PPO.  The caller
# supplies globally unique engine indices; TCP ports are base + global index.
# The script owns only its emulator processes and never changes the Slurm job.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/../../.." && pwd)
source "${repository_root}/native_runner/local_config.sh"
root=${CR_AI_ROOT:-$repository_root}
export CR_AI_ROOT="$root"
export PYTHONPATH="$repository_root"
artifact=${1:?usage: launch_ppo_engine_node.sh ARTIFACT GLOBAL_START [ENGINE_COUNT] [BASE_PORT] [EMULATOR_BASE_PORT]}
global_start=${2:?global engine start is required}
engine_count=${3:-${CR_PPO_ENGINE_COUNT:?Set CR_PPO_ENGINE_COUNT in .env}}
base_port=${4:-${CR_PPO_BASE_PORT:-28000}}
emulator_base_port=${5:-${CR_EMULATOR_BASE_PORT:-5554}}
python=$(command -v "${CR_AI_PYTHON:-python}")
adb=$(command -v "${CR_TRAINING_ADB:-adb}")
emulator=${CR_EMULATOR:?Set CR_EMULATOR in .env}
source_avd=${CR_SOURCE_AVD:?Set CR_SOURCE_AVD in .env}
source_ini=${CR_SOURCE_AVD_INI:?Set CR_SOURCE_AVD_INI in .env}
initdata=${CR_EMULATOR_INITDATA:?Set CR_EMULATOR_INITDATA in .env}
apk=${CR_TRAINING_APK:?Set CR_TRAINING_APK in .env}
probe=${CR_PROBE:-${CR_PPO_PROBE:-$repository_root/native_runner/probe/out/libcrprobe.so}}
probe_installer=$script_dir/install_probe_idle_guests.py
engine_configurator=${CR_PPO_ENGINE_CONFIGURATOR:-$script_dir/configure_emulator_engine_guests.py}
pulse_compat=${CR_EMULATOR_COMPAT:?Set CR_EMULATOR_COMPAT in .env}
endpoint_proxy=$script_dir/tcp_endpoint_proxy.py

if [[ ! ${SLURM_JOB_ID:-} ]]; then
  echo "engine node must run inside a Slurm allocation" >&2
  exit 90
fi
if ((global_start < 0 || engine_count <= 0 || engine_count > 64)); then
  echo "global start must be nonnegative and engine count must be 1..64" >&2
  exit 91
fi
if ((base_port + global_start + engine_count > 65536)); then
  echo "engine TCP port range exceeds 65535" >&2
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
lanes_per_engine=8
guest_memory_mb_per_engine=${CR_PPO_GUEST_MEMORY_MB_PER_ENGINE:-4096}
remote_worker_port=${CR_PPO_REMOTE_WORKER_PORT:-29900}
cpu_offset=${CR_PPO_CPU_OFFSET:-0}
if ((
  guest_memory_mb_per_engine < 1024 ||
  remote_worker_port < 1 || remote_worker_port > 65535 ||
  cpu_offset < 0
)); then
  echo "guest memory must be at least 1024 MiB per engine and worker port must be valid" >&2
  exit 91
fi
guest_engines=()
guest_memory_mb=()
ports=()
host_bases=()
serials=()
remaining=$engine_count
engine_cursor=0
for ((guest=0; guest<guest_count; guest++)); do
  count=$max_guest_engines
  ((remaining < count)) && count=$remaining
  port=$((emulator_base_port + guest * 2))
  guest_engines+=("$count")
  guest_memory_mb+=("$((count * guest_memory_mb_per_engine))")
  ports+=("$port")
  host_bases+=("$((base_port + global_start + engine_cursor))")
  serials+=("emulator-$port")
  remaining=$((remaining - count))
  engine_cursor=$((engine_cursor + count))
done
dns_args=()
if [[ -n ${CR_EMULATOR_DNS:-} ]]; then dns_args=(-dns-server "$CR_EMULATOR_DNS"); fi

for required in \
  "$python" "$adb" "$emulator" "$source_avd/config.ini" "$source_ini" \
  "$initdata" "$apk" "$probe" \
  "$probe_installer" "$engine_configurator" "$endpoint_proxy"; do
  if [[ ! -e $required ]]; then
    echo "missing PPO cluster dependency: $required" >&2
    exit 92
  fi
done
if [[ ! -r /dev/kvm ]]; then
  echo "/dev/kvm is not readable" >&2
  exit 93
fi

mapfile -t allocated_cpus < <("$python" - <<'PY'
from pathlib import Path

line = next(
    row for row in Path("/proc/self/status").read_text().splitlines()
    if row.startswith("Cpus_allowed_list:")
)
values = []
for part in line.split(":", 1)[1].strip().split(","):
    if "-" in part:
        start, end = map(int, part.split("-", 1))
        values.extend(range(start, end + 1))
    else:
        values.append(int(part))
for value in values:
    print(value)
PY
)
if ((cpu_offset + engine_count > ${#allocated_cpus[@]})); then
  echo "CPU offset $cpu_offset plus $engine_count engines exceeds ${#allocated_cpus[@]} CPUs" >&2
  exit 94
fi

engine_cpu_pools=()
cpu_cursor=0
for ((guest=0; guest<guest_count; guest++)); do
  count=${guest_engines[$guest]}
  pool=$(IFS=,; echo "${allocated_cpus[*]:cpu_offset+cpu_cursor:count}")
  engine_cpu_pools+=("$pool")
  cpu_cursor=$((cpu_cursor + count))
done
if [[ $cpu_cursor -ne $engine_count ]]; then
  echo "guest engine topology uses $cpu_cursor CPUs, expected $engine_count" >&2
  exit 94
fi
control_cpus=$(
  selected_end=$((cpu_offset + engine_count))
  values=("${allocated_cpus[@]:0:cpu_offset}" "${allocated_cpus[@]:selected_end}")
  IFS=,
  echo "${values[*]}"
)

mkdir -p "$artifact"
stamp=$(date +%Y%m%d-%H%M%S)
host_short=$(hostname -s)
run=${TMPDIR:-/tmp}/firstlight-ppo-${SLURM_JOB_ID}-${host_short}-${global_start}-$stamp
avd_home=$run/avd-home
logs=$artifact/engine-logs-$stamp
ready=$artifact/engine-node-ready.json
stop_file=$artifact/STOP_ENGINES
mkdir -p "$avd_home" "$logs"
if [[ -e $artifact/engine-cluster-failed.json ]]; then
  mv \
    "$artifact/engine-cluster-failed.json" \
    "$artifact/engine-cluster-failed-before-$stamp.json"
fi
if [[ -e $ready ]]; then
  echo "refusing to overwrite existing readiness artifact: $ready" >&2
  exit 95
fi
if pgrep -f "[q]emu-system-.*-avd crppo_${global_start}_" >/dev/null; then
  echo "refusing to overlap an existing crppo emulator cluster" >&2
  exit 96
fi

export PYTHONPATH=$repository_root
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
if [[ -d $pulse_compat ]]; then
  export LD_LIBRARY_PATH=$pulse_compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

launcher_pids=()
proxy_pid=
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for index in $(seq 0 $((guest_count - 1))); do
    "$adb" -s "${serials[$index]}" emu kill >/dev/null 2>&1 || true
  done
  for pid in "${launcher_pids[@]}"; do
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
  [[ ! $proxy_pid ]] || kill -TERM "$proxy_pid" >/dev/null 2>&1 || true
  for pid in "${launcher_pids[@]}"; do
    wait "$pid" >/dev/null 2>&1 || true
  done
  [[ ! $proxy_pid ]] || wait "$proxy_pid" >/dev/null 2>&1 || true
  if [[ $status -ne 0 ]]; then
    printf '{"ok":false,"exit_code":%d,"timestamp":"%s"}\n' \
      "$status" "$(date -Is)" >"$artifact/engine-cluster-failed.json"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "$(date -Is) ENGINE_CLUSTER_START job=$SLURM_JOB_ID host=$(hostname) start=$global_start engines=$engine_count"
echo "engine_cpu_pools=${engine_cpu_pools[*]} control_cpus=$control_cpus"
rm -f "$stop_file"

for index in $(seq 0 $((guest_count - 1))); do
  name=crppo_${global_start}_$index
  avd_dir=$avd_home/$name.avd
  emu_dir=$run/emu$index
  mkdir -p "$avd_dir" "$emu_dir"
  cp "$source_avd/config.ini" "$avd_dir/config.ini"
  sed -i \
    -e "s/^avd.id=.*/avd.id=$name/" \
    -e "s/^avd.name=.*/avd.name=$name/" \
    -e "s/^hw.cpu.ncore=.*/hw.cpu.ncore=${guest_engines[$index]}/" \
    -e "s/^hw.ramSize=.*/hw.ramSize=${guest_memory_mb[$index]}M/" \
    "$avd_dir/config.ini"
  cp --reflink=auto --sparse=always "$source_avd/cache.img" "$avd_dir/cache.img"
  cp "$source_ini" "$avd_home/$name.ini"
  sed -i "s#^path=.*#path=$avd_dir#" "$avd_home/$name.ini"
done

ulimit -n 32768
for index in $(seq 0 $((guest_count - 1))); do
  name=crppo_${global_start}_$index
  port=${ports[$index]}
  pool=${engine_cpu_pools[$index]}
  emu_dir=$run/emu$index
  ANDROID_AVD_HOME=$avd_home \
  ANDROID_HOME="${CR_TRAINING_SDK_ROOT:?Set CR_TRAINING_SDK_ROOT in .env}" \
  ANDROID_SDK_ROOT="${CR_TRAINING_SDK_ROOT:?Set CR_TRAINING_SDK_ROOT in .env}" \
    taskset -c "$pool" "$emulator" \
      -avd "$name" \
      -port "$port" \
      -no-window \
      -no-audio \
      -no-boot-anim \
      -no-snapshot \
      -gpu swiftshader_indirect \
      -accel on \
      -cores "${guest_engines[$index]}" \
      -memory "${guest_memory_mb[$index]}" \
      -data "$emu_dir/userdata-qemu.img" \
      -initdata "$initdata" \
      "${dns_args[@]}" \
      -nojni \
      >"$logs/emulator-$index.log" 2>&1 </dev/null &
  launcher_pids+=("$!")
done

deadline=$((SECONDS + 480))
mapfile -t pending < <(seq 0 $((guest_count - 1)))
while ((${#pending[@]})); do
  next=()
  for index in "${pending[@]}"; do
    serial=${serials[$index]}
    state=$("$adb" -s "$serial" get-state 2>/dev/null || true)
    boot=$("$adb" -s "$serial" shell getprop sys.boot_completed </dev/null 2>/dev/null | tr -d '\r' || true)
    if [[ $state == device && $boot == 1 ]]; then
      guest_nproc=$("$adb" -s "$serial" shell nproc | tr -d '\r')
      if [[ $guest_nproc != "${guest_engines[$index]}" ]]; then
        echo "$serial exposes $guest_nproc CPUs, expected ${guest_engines[$index]}" >&2
        exit 97
      fi
      echo "$(date -Is) BOOT_READY index=$index serial=$serial host_cpus=${engine_cpu_pools[$index]}"
    else
      next+=("$index")
    fi
  done
  pending=("${next[@]}")
  ((${#pending[@]} == 0)) && break
  if ((SECONDS >= deadline)); then
    echo "emulator boot timeout: ${pending[*]}" >&2
    exit 98
  fi
  sleep 2
done

if [[ ${CR_PPO_TRIM_GUEST_APPS:-0} == 1 ]]; then
  trim_packages=(
    com.google.android.googlequicksearchbox
    com.android.chrome
    com.google.android.apps.youtube.music
    com.google.android.apps.messaging
    com.google.android.apps.wellbeing
    com.google.android.inputmethod.latin
    com.google.android.tts
    com.google.android.dialer
  )
  trim_pids=()
  for serial in "${serials[@]}"; do
    (
      for package in "${trim_packages[@]}"; do
        "$adb" -s "$serial" shell pm disable-user --user 0 "$package"
      done
    ) >"$logs/trim-apps-$serial.log" 2>&1 &
    trim_pids+=("$!")
  done
  for pid in "${trim_pids[@]}"; do wait "$pid"; done
fi

cached_apk=/dev/shm/cr-engine-cluster-${SLURM_JOB_ID}-$stamp.apk
cp "$apk" "$cached_apk"
install_pids=()
for index in $(seq 0 $((guest_count - 1))); do
  "$adb" -s "${serials[$index]}" install -r -d -g "$cached_apk" \
    >"$logs/install-$index.log" 2>&1 &
  install_pids+=("$!")
done
for pid in "${install_pids[@]}"; do wait "$pid"; done
rm -f "$cached_apk"

serial_csv=$(IFS=,; echo "${serials[*]}")
"$python" "$probe_installer" \
  --probe "$probe" \
  --serials "$serial_csv" \
  --output "$artifact/probe-install-$stamp.json"

# CR_EMULATOR_INITDATA is a prepared offline guest data image. Each clone
# already contains its matching update cache; never export or push resources.

source_reports=()
full_guests=$((engine_count / max_guest_engines))
remainder=$((engine_count % max_guest_engines))
if ((full_guests)); then
  # Starting every ARM64 process across a full 64-core node at once can race
  # libndk_translation's first-cache build.  Bound only the cold-start fanout;
  # all engines remain resident and fully concurrent after readiness.
  configure_guest_batch=${CR_PPO_CONFIGURE_GUEST_BATCH:-3}
  if ((configure_guest_batch <= 0)); then
    echo "CR_PPO_CONFIGURE_GUEST_BATCH must be positive" >&2
    exit 100
  fi
  for ((batch_start=0; batch_start<full_guests; batch_start+=configure_guest_batch)); do
    batch_count=$configure_guest_batch
    ((batch_start + batch_count <= full_guests)) || batch_count=$((full_guests - batch_start))
    batch_serials=$(IFS=,; echo "${serials[*]:batch_start:batch_count}")
    batch_bases=$(IFS=,; echo "${host_bases[*]:batch_start:batch_count}")
    batch_report=$artifact/engine-startup-$((batch_count * max_guest_engines))-batch-$batch_start-$stamp.json
    "$python" "$engine_configurator" \
      --serials "$batch_serials" \
      --host-bases "$batch_bases" \
      --engines "$max_guest_engines" \
      --output "$batch_report" \
      >"$logs/engine-configure-batch-$batch_start.log"
    source_reports+=("$batch_report")
  done
fi
if ((remainder)); then
  last=$((guest_count - 1))
  remainder_report=$artifact/engine-startup-$remainder-$stamp.json
  "$python" "$engine_configurator" \
    --serials "${serials[$last]}" \
    --host-bases "${host_bases[$last]}" \
    --engines "$remainder" \
    --output "$remainder_report" \
    >"$logs/engine-configure-remainder.log"
  source_reports+=("$remainder_report")
fi

# ADB host forwards bind loopback.  One raw-byte asyncio proxy exposes all
# endpoints on this node's private cluster address without a subprocess per
# connection or ncat's small-message-oriented sh-exec path.
"$python" "$endpoint_proxy" \
  --bind-host "$host_short" \
  --listen-base "$((base_port + global_start))" \
  --target-base "$((base_port + global_start))" \
  --count "$engine_count" \
  >"$logs/endpoint-proxy.log" 2>&1 </dev/null &
proxy_pid=$!
sleep 1
if ! kill -0 "$proxy_pid" 2>/dev/null; then
  echo "the engine endpoint proxy failed to start" >&2
  sed -n '1,80p' "$logs/endpoint-proxy.log" >&2 || true
  exit 99
fi
if ! grep -q '"ready": true' "$logs/endpoint-proxy.log"; then
  echo "the engine endpoint proxy did not report readiness" >&2
  exit 99
fi

"$python" - \
  "$ready" "$global_start" "$engine_count" "$base_port" \
  "$remote_worker_port" "$control_cpus" \
  "${engine_cpu_pools[*]}" "$run" "${source_reports[@]}" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

target = Path(sys.argv[1])
global_start = int(sys.argv[2])
engine_count = int(sys.argv[3])
base_port = int(sys.argv[4])
remote_worker_port = int(sys.argv[5])
control_cpus = sys.argv[6]
engine_cpu_pools = sys.argv[7].split()
runtime_dir = sys.argv[8]
sources = tuple(Path(value) for value in sys.argv[9:])
reports = [json.loads(source.read_text()) for source in sources]
guests = [guest for report in reports for guest in report["guests"]]
endpoints = [row for guest in guests for row in guest["endpoints"]]
ports = sorted(int(row["host_port"]) for row in endpoints)
pids = {(guest["serial"], int(row["status"]["pid"])) for guest in guests for row in guest["endpoints"]}
host = socket.gethostname().split(".", 1)[0]
expected_ports = list(
    range(base_port + global_start, base_port + global_start + engine_count)
)
checks = {
    "job_id": bool(os.environ.get("SLURM_JOB_ID")),
    "guest_count": len(guests) == len(engine_cpu_pools),
    "engine_count": sum(int(report.get("engine_count", 0)) for report in reports) == engine_count,
    "endpoint_count": len(endpoints) == engine_count,
    "unique_processes": len(pids) == engine_count,
    "contiguous_ports": ports == expected_ports,
    "resident_slots": all(int(row["status"].get("slotsPerEngine", -1)) == 8 for row in endpoints),
    "production_ready": all(row["attestation"].get("production_ready") is True for row in endpoints),
}
failed = [name for name, value in checks.items() if not value]
if failed:
    raise RuntimeError(f"engine-node acceptance failed: {failed}")
by_port = {int(row["host_port"]): row for row in endpoints}
engine_rows = [
    {
        "engine_index": global_start + offset,
        "host": host,
        "port": port,
        "pid": int(by_port[port]["status"]["pid"]),
    }
    for offset, port in enumerate(expected_ports)
]
summary = {
    "ok": True,
    "schema": "v4-ppo-engine-node.v1",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "host": host,
    "global_engine_start": global_start,
    "engine_count": engine_count,
    "resident_slots_per_engine": 8,
    "remote_worker_port": remote_worker_port,
    "parallel_matches": engine_count * 8,
    "engine_cpu_count": engine_count,
    "control_cpu_count": len([item for item in control_cpus.split(",") if item]),
    "control_cpus": control_cpus,
    "engine_cpu_pools": engine_cpu_pools,
    "decision_ticks": 5,
    "runtime_dir": runtime_dir,
    "engines": engine_rows,
    "attestation_digest": endpoints[0]["attestation"]["attestation_digest"],
    "source_reports": [str(source) for source in sources],
    "manifest_path": str(target),
    "checks": checks,
}
target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

echo "$(date -Is) ENGINE_CLUSTER_READY start=$global_start engines=$engine_count lanes=8 parallel_matches=$((engine_count * 8))"
while [[ ! -e $stop_file ]]; do
  for pid in "${launcher_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "an emulator launcher exited after readiness" >&2
      exit 99
    fi
  done
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "the engine endpoint proxy exited after readiness" >&2
    exit 99
  fi
  sleep 5
done
echo "$(date -Is) ENGINE_CLUSTER_STOP_REQUESTED"
