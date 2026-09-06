#!/usr/bin/env bash
set -euo pipefail

# Pass the same required --index, --train-tags and --output-dir arguments as
# train_imitation_cache. Dataset/checkpoint locations are caller supplied.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${repo_root}/native_runner/local_config.sh"
python_bin="${CR_AI_PYTHON:-python}"
ranks="${IL_WORLD_SIZE:-1}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${repo_root}"
exec "${python_bin}" -m torch.distributed.run \
  --standalone --nproc-per-node="${ranks}" \
  -m native_runner.training.v4.train_imitation_cache \
  --expected-world-size "${ranks}" \
  --shards-per-rank "${CR_IL_SHARDS_PER_RANK:-1}" \
  --torch-threads "${CR_IL_TORCH_THREADS:-1}" \
  --save-every-groups "${CR_IL_SAVE_EVERY_GROUPS:-25}" \
  --no-validate "$@"
