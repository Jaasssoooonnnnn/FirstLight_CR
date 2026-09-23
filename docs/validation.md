# Checks

[English home](../README.md) · [Training](training.md)

Use these checks after installation, when changing an engine or checkpoint, and before starting a training run.

## Python and models

Run from the repository root after installing the dependencies:

```sh
python -m pip check
python tools/check_install.py
```

`pip check` should report no dependency conflicts. `check_install.py` verifies the data hashes, loads each included checkpoint, and prints `Ready: 5 models` when successful.

## Tests

```sh
python -m pip install pytest
python -m pytest native_runner/tests -q
```

The tests cover environment contracts, card mechanics, observations and actions, native layouts, cache encoding, IL/PPO updates, and model serving. Set `CR_INPUT_APK` to enable original-APK fingerprint checks. CUDA tests need a CUDA device, and native C++ tests need `g++`; tests with unavailable prerequisites are skipped.

## Native engine

Prepare the VM using the [installation guide](getting_started.md), then run:

```powershell
.\launch_interface.cmd --diagnose
.\launch_interface.cmd
```

Check that the configured VM is found, a match starts at level 11, and both manual and model actions execute. When diagnosing a problem, use `runs/interface/firstlight-interface.log` together with the build and installation receipts.

## Replays and training

The [training guide](training.md) provides commands for each step. Check the corresponding outputs before moving to the next step:

| Step | Check |
| --- | --- |
| Replay reconstruction | Actions execute in order and playback reaches the source endpoint |
| IL cache | Completed/failed replay counts, expert/executed action counts, and `--verify-roundtrip` result |
| IL training | Finite loss, advancing training groups, and a checkpoint that loads |
| Linux guests / Linux guest | Readiness checks pass for the configured engines and capacities |
| PPO training | Rollouts complete, optimizer updates finish, and checkpoints are saved |

For a new distributed setup, run the one-update PPO example first and check each simulation node’s readiness report and the trainer output. For model comparisons, inspect the outcome and execution metrics described in [model evaluation](../checkpoints/README.md).

## Repository files

Before publishing repository changes, run:

```sh
python tools/check_release.py
```

This checks the distribution file set, included checkpoint and data hashes, configuration template, and ignore rules. A successful result ends with `0 errors`.
