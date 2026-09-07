#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/../../.." && pwd)
source "${repository_root}/native_runner/local_config.sh"
root=${CR_AI_ROOT:-$repository_root}
export CR_AI_ROOT="$root"
output_dir=${1:-${CR_PPO_OUTPUT_DIR:?Set CR_PPO_OUTPUT_DIR in .env or pass OUTPUT_DIR}}
topology=${2:-${CR_PPO_TOPOLOGY:?Set CR_PPO_TOPOLOGY in .env or pass TOPOLOGY}}
updates=${3:-${CR_PPO_UPDATES:-2}}
segment_seconds=${4:-${CR_PPO_SEGMENT_SECONDS:-40}}
resume_checkpoint=${5:-${CR_PPO_RESUME_CHECKPOINT:-}}
python=${CR_AI_PYTHON:-python}

resume_args=()
if [[ -n $resume_checkpoint ]]; then
  resume_args=(--resume-checkpoint "$resume_checkpoint")
fi

readarray -t topology_values < <("$python" - "$topology" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text())
for key in ("gpus", "trainer_slurm_job_id", "trainer_host"):
    print(value[key])
PY
)
nproc_per_node=${topology_values[0]}
expected_job=${topology_values[1]}
expected_host=${topology_values[2]}

if [[ ${SLURM_JOB_ID:-} != "$expected_job" ]]; then
  echo "trainer topology expects Slurm job $expected_job" >&2
  exit 90
fi
if [[ $(hostname -s) != "$expected_host" ]]; then
  echo "trainer topology expects host $expected_host" >&2
  exit 91
fi

lock_dir=$root/artifacts/v4-ppo-selfplay-job-${SLURM_JOB_ID}
mkdir -p "$lock_dir"
exec 9>"$lock_dir/ppo-training.lock"
if ! flock -n 9; then
  echo "another PPO trainer already owns the engine cluster" >&2
  exit 93
fi

export PYTHONPATH=$repository_root
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

cd "$repository_root"
exec "$python" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$nproc_per_node" \
  -m native_runner.training.v4.train_ppo_self_play_cluster \
  --deck-pool "${CR_PPO_DECK_POOL:?Set CR_PPO_DECK_POOL in .env}" \
  --checkpoint "${CR_PPO_CHECKPOINT:?Set CR_PPO_CHECKPOINT in .env}" \
  "${resume_args[@]}" \
  --output-dir "$output_dir" \
  --engine-ready "$topology" \
  --timeout "${CR_PPO_TIMEOUT:-600}" \
  --updates "$updates" \
  --rollout-segment-seconds "$segment_seconds" \
  --ppo-epochs "${CR_PPO_EPOCHS:-1}" \
  --lane-minibatch-size "${CR_PPO_LANE_MINIBATCH_SIZE:-384}" \
  --lane-microbatch-size "${CR_PPO_LANE_MICROBATCH_SIZE:-256}" \
  --sequence-chunk-steps "${CR_PPO_SEQUENCE_CHUNK_STEPS:-16}" \
  --seed "${CR_PPO_SEED:-0}"
