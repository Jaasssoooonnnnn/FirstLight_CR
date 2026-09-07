#!/usr/bin/env bash
set -euo pipefail

# Pass DATASET_ROOT OUTPUT_DIR and explicit cache_builder options, including
# --ports and --partition-index/--partition-count when distributing conversion.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${repo_root}/native_runner/local_config.sh"
python_bin="${CR_AI_PYTHON:-python}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
cd "${repo_root}"
exec "${python_bin}" -m native_runner.training.v4.cache_builder "$@"
