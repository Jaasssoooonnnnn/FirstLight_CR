# Training and simulation

[English home](../README.md) · [Installation](getting_started.md) · [Training history](training_history.md)

Run commands from the repository root with the Python dependencies installed. Windows supports cache building, inference, evaluation, and the console. The IL/PPO launchers target Linux with CUDA and NCCL.

## 1. Data and workflow examples

[IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) provides 252,238 replays and 17,836,160 actions. Download its files into one directory, keeping the `replays/` and `actions/` subdirectories. Set `CR_REPLAY_DATASET_ROOT` to that directory to browse it in the console.

For a small reproducible example, generate a 45-second synthetic replay, Parquet dataset, tag list, and deck pool:

```sh
python tools/make_demo_data.py
```

Outputs are in `runs/demo`. Use `--output` to choose a new directory on subsequent runs. The console’s collected-replay page can open `runs/demo/dataset`. This synthetic example checks the workflow; use real matches to study agent behavior.

### Replay format

The cache builder reads `replays/part-*.parquet`. Each row contains a string `replay_tag` and a JSON string `payload_json` with schema `royaleapi-battle-replay.v1`: decks, forms, tower troops, 20 Hz action times, sides, coordinates, source-side markers in `source_fields.data_i`, and the replay end time. The console list also uses `requested_player_tag`; time columns are optional. See the generated `runs/demo/payload.json` for a minimal complete example.

## 2. Replay reconstruction → IL cache

Start an idle offline engine first; on Windows use `./native_runner/start_offline.ps1`. Set `--ports` to its actual host control port.

```sh
python -m native_runner.training.v4.cache_builder runs/demo/dataset runs/demo/cache \
  --ports 26789 --source-count 1 --resident-slots 1 --verify-roundtrip
```

Outputs include `index.json`, sequence shards, and shard manifests. Check completed/failed replay counts and expert/executed-action counts before training. For the full dataset, replace the two directories and use `--source-count 0`. Multiple engine ports are comma-separated; `--resident-slots` must fit the configured capacity. Linux partitioned conversion is available through `native_runner/training/v4/launch_cache_partition.sh`.

## 3. Recurrent imitation learning

On the Linux training host, set `CR_AI_PYTHON` to the training interpreter and `IL_WORLD_SIZE` to the GPU count (default 1). `CR_IL_SHARDS_PER_RANK` defaults to 1. Copy the complete cache directory to this host. The training tag file contains one selected replay tag per line, with both owner sequences present in the cache.

```sh
bash native_runner/training/v4/launch_stateful_il.sh \
  --index runs/demo/cache/index.json --train-tags runs/demo/train-tags.txt \
  --output-dir runs/il-smoke --epochs 1 --max-groups 1 --save-every-groups 1
```

For a training run, remove `--max-groups 1` and use your dataset and tag selection. To resume an interrupted IL run, pass its saved training checkpoint with `--resume`. Check loss, completed groups, and saved checkpoints; the resulting model can be loaded in the console or evaluator.

## 4. Native trajectory playback

To create a complete native trajectory, start an idle offline engine and run:

```sh
python tools/make_demo_data.py --output runs/native-demo --native
```

Open `runs/native-demo/training_replays` on the console’s training-replay page. In your own runner, export `TrainingReplayV1` with `capture_training_replay(environment, ...)` from a trace-enabled environment; see [replay_archive.py](../native_runner/training/replay_archive.py). The V4 trainers do not automatically save a visual replay for each match; the viewer expects exported trajectories.

## 5. Windows engine cluster

A dedicated VM supports 1–24 engine processes and 1–8 resident slots per process. Set `CR_BUILD_ENGINE_COUNT` before building the APK, `CR_ENGINE_COUNT` to no more than that value, and `CR_SLOTS_PER_ENGINE` to the desired slot count. Set `CR_VM_MEMORY_GB`; use `CR_CONFIGURE_VM_RESOURCES=false` to retain the VM’s existing resource configuration.

```powershell
./native_runner/start_engine_cluster.ps1
./native_runner/stop_engine_cluster.ps1
```

The start command reports control ports and capacities for the cache builder. Stop the cluster before starting a console task, and vice versa.

## 6. Prepare a Linux AVD

Android Emulator hosts need `/dev/kvm`, root adbd, and an Android system image capable of running ARM64 libraries. In the base AVD, manually install the original APK, open it to finish the matching update, and exit. Copy the locally built offline APK and its adjacent `.build.json` receipt to Linux. Place the compiled probe at `native_runner/probe/out/libcrprobe.so`, or set `CR_PROBE`.

```sh
python tools/prepare_training_guest.py --serial emulator-5554 --apk /path/to/cr-ai-offline.apk
```

ADB uses `CR_TRAINING_ADB`, with `--adb` as an override. Preparation enables root, stops the game, applies dual-stack isolation, and runs the device-local backup/install flow. Configure the base AVD with one vCPU for this single-engine readiness check:

```sh
python -m native_runner.training.v4.configure_emulator_engine_guests \
  --serials emulator-5554 --host-bases 26789 --engines 1 --output runs/guest-ready.json
```

Confirm `production_ready`, then shut down the base AVD. Use the emulator’s `qemu-img` to convert its complete userdata image to standalone raw format. If userdata has a qcow2 overlay, use the topmost overlay as input.

```sh
qemu-img convert -O raw <userdata-source> <prepared-userdata.img>
```

Set `CR_EMULATOR_INITDATA` to this prepared image, `CR_SOURCE_AVD` / `CR_SOURCE_AVD_INI` to the matching base AVD, and `CR_TRAINING_APK` to the same signed APK. The prepared userdata must contain the matching application and content. Node launchers clone it for independent guests.

## 7. Distributed PPO

Configure the Linux resource section of `.env`: SDK, emulator, prepared AVD, APK, and probe. Start with `CR_PPO_ENGINES_PER_GUEST=1`; capacity is 1–6 engines per guest with eight resident slots per engine. The APK must be built for at least this many engines. Allocate at least one CPU per engine plus control CPUs. `CR_EMULATOR_COMPAT` points to the emulator compatibility-library directory; use a valid empty directory if no extra libraries are needed.

Within an existing Slurm allocation, launch a simulation node:

```sh
bash native_runner/training/v4/launch_ppo_engine_node.sh runs/node0 0 1
```

This command holds the engines and writes `runs/node0/engine-node-ready.json`. Continue from another shell. Global engine indices start at zero and must be contiguous without overlap across nodes. Each node uses its own output directory. Create `STOP_ENGINES` in that directory to stop its engines.

Generate the trainer topology using your allocation’s job ID, host, and CPU assignments. Pass all readiness files after one `--node-ready` option for multiple nodes.

```sh
python -m native_runner.training.v4.build_ppo_multinode_topology \
  --node-ready runs/node0/engine-node-ready.json --output runs/topology.json \
  --trainer-job-id <job-id> --trainer-host <host> --gpus 1 \
  --worker-cpus <worker-cpu-ids> --control-cpus <control-cpu-ids>
```

Build a deck pool from the parent of your dataset directories (`datasets/*/replays/*.parquet`), or use `runs/demo/deck-pool.json` for a workflow check:

```sh
python -m native_runner.training.v4.build_ppo_deck_pool \
  --dataset-root ./datasets --output ./runs/deck-pool.json --minimum-uses 5
```

Set `CR_PPO_DECK_POOL` and `CR_PPO_CHECKPOINT` to absolute paths in the training host’s `.env`. The included `checkpoints/General/checkpoint-step-00000460.pt` is one starting point. On the trainer host named in the topology and within the same allocation, run one update with 40-second rollout segments:

```sh
bash native_runner/training/v4/launch_ppo_multinode_training.sh runs/ppo-smoke runs/topology.json 1 40
```

Inspect match completion, rejected actions, optimizer metrics, and output checkpoints. Configure longer runs through the launcher arguments and PPO settings in `.env`; see `train_ppo_self_play_cluster --help` for algorithm options. Metrics default to local offline W&B. Resume a full training run with `CR_PPO_RESUME_CHECKPOINT`.

Run the launchers inside your allocated Slurm job. Check engine readiness and training outputs using the [checks guide](validation.md).
