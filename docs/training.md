# Training and simulation · 训练与模拟

[English home](../README.md) · [中文首页](../README.zh-CN.md) · [Installation / 安装](getting_started.md)

Run commands from the repository root with the Python dependencies installed. Windows supports cache building, inference, evaluation, and the console. The IL/PPO launchers target Linux with CUDA and NCCL.

在仓库根目录、已安装依赖的 Python 环境中执行命令。Windows 支持缓存构建、推理、评测和控制台；IL/PPO 训练入口面向 Linux、CUDA 与 NCCL。

## 1. Data and workflow examples · 数据与流程样例

[IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) provides 252,238 replays and 17,836,160 actions. Download its files into one directory, keeping the `replays/` and `actions/` subdirectories. Set `CR_REPLAY_DATASET_ROOT` to that directory to browse it in the console.

[IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) 提供 252,238 场回放和 17,836,160 条动作。将文件下载到同一目录，保留 `replays/` 和 `actions/` 子目录；将 `CR_REPLAY_DATASET_ROOT` 设为该目录即可在控制台浏览。

For a small reproducible example, generate a 45-second synthetic replay, Parquet dataset, tag list, and deck pool:

先运行一个小规模可复现样例，生成 45 秒合成回放、Parquet 数据集、tag 列表与卡组池：

```sh
python tools/make_demo_data.py
```

Outputs are in `runs/demo`. Use `--output` to choose a new directory on subsequent runs. The console’s collected-replay page can open `runs/demo/dataset`. This synthetic example checks the workflow; use real matches to study agent behavior.

输出位于 `runs/demo`；重复运行时用 `--output` 选择新目录。控制台的采集回放页可打开 `runs/demo/dataset`。合成样例用于检查流程，研究智能体表现时使用真实对局。

### Replay format · 回放格式

The cache builder reads `replays/part-*.parquet`. Each row contains a string `replay_tag` and a JSON string `payload_json` with schema `royaleapi-battle-replay.v1`: decks, forms, tower troops, 20 Hz action times, sides, coordinates, source-side markers in `source_fields.data_i`, and the replay end time. The console list also uses `requested_player_tag`; time columns are optional. See the generated `runs/demo/payload.json` for a minimal complete example.

缓存构建器读取 `replays/part-*.parquet`。每行包含字符串 `replay_tag` 与 JSON 字符串 `payload_json`，采用 `royaleapi-battle-replay.v1` 格式：双方卡组、形态、塔兵、20 Hz 动作时刻、阵营、坐标、`source_fields.data_i` 中的原始侧标记，以及结束时刻。控制台列表还使用 `requested_player_tag`，时间列可选。生成的 `runs/demo/payload.json` 是完整最小示例。

## 2. Replay reconstruction → IL cache · 回放重建 → IL 缓存

Start an idle offline engine first; on Windows use `./native_runner/start_offline.ps1`. Set `--ports` to its actual host control port.

先启动空闲的离线引擎，Windows 使用 `./native_runner/start_offline.ps1`。将 `--ports` 设为实际的主机控制端口。

```sh
python -m native_runner.training.v4.cache_builder runs/demo/dataset runs/demo/cache \
  --ports 26789 --source-count 1 --resident-slots 1 --verify-roundtrip
```

Outputs include `index.json`, sequence shards, and shard manifests. Check completed/failed replay counts and expert/executed-action counts before training. For the full dataset, replace the two directories and use `--source-count 0`. Multiple engine ports are comma-separated; `--resident-slots` must fit the configured capacity. Linux partitioned conversion is available through `native_runner/training/v4/launch_cache_partition.sh`.

输出包括 `index.json`、序列分片和分片清单。训练前检查完成与失败回放数、专家动作与实际执行动作数。处理完整数据集时替换输入输出目录，并使用 `--source-count 0`。多个引擎端口用逗号分隔，`--resident-slots` 不得超过配置容量。Linux 分区转换入口为 `native_runner/training/v4/launch_cache_partition.sh`。

## 3. Recurrent imitation learning · 循环模仿学习

On the Linux training host, set `CR_AI_PYTHON` to the training interpreter and `IL_WORLD_SIZE` to the GPU count (default 1). `CR_IL_SHARDS_PER_RANK` defaults to 1. Copy the complete cache directory to this host. The training tag file contains one selected replay tag per line, with both owner sequences present in the cache.

在 Linux 训练主机配置 `CR_AI_PYTHON` 为训练解释器，`IL_WORLD_SIZE` 为 GPU 数量（默认 1），`CR_IL_SHARDS_PER_RANK` 默认 1。将完整缓存目录复制到训练主机。训练 tag 文件每行一个选定的回放编号，缓存中必须包含该回放的双方序列。

```sh
bash native_runner/training/v4/launch_stateful_il.sh \
  --index runs/demo/cache/index.json --train-tags runs/demo/train-tags.txt \
  --output-dir runs/il-smoke --epochs 1 --max-groups 1 --save-every-groups 1
```

For a training run, remove `--max-groups 1` and use your dataset and tag selection. To resume an interrupted IL run, pass its saved training checkpoint with `--resume`. Check loss, completed groups, and saved checkpoints; the resulting model can be loaded in the console or evaluator.

正式训练时去掉 `--max-groups 1`，使用目标数据集及 tag 列表。恢复中断的 IL 训练时，用 `--resume` 传入该训练保存的 checkpoint。检查 loss、完成组数及输出模型，训练结果可直接用于控制台或评测器。

## 4. Native trajectory playback · 原生轨迹回放

To create a complete native trajectory, start an idle offline engine and run:

生成完整原生轨迹时，先启动空闲离线引擎，再运行：

```sh
python tools/make_demo_data.py --output runs/native-demo --native
```

Open `runs/native-demo/training_replays` on the console’s training-replay page. In your own runner, export `TrainingReplayV1` with `capture_training_replay(environment, ...)` from a trace-enabled environment; see [replay_archive.py](../native_runner/training/replay_archive.py). The V4 trainers do not automatically save a visual replay for each match; the viewer expects exported trajectories.

在控制台训练回放页打开 `runs/native-demo/training_replays`。自定义 runner 可对保留 trace 的环境调用 `capture_training_replay(environment, ...)`，导出 `TrainingReplayV1`，参见 [replay_archive.py](../native_runner/training/replay_archive.py)。V4 训练器不会自动为每场对局保存可视回放，播放器读取导出的轨迹。

## 5. Windows engine cluster · Windows 引擎集群

A dedicated VM supports 1–24 engine processes and 1–8 resident slots per process. Set `CR_BUILD_ENGINE_COUNT` before building the APK, `CR_ENGINE_COUNT` to no more than that value, and `CR_SLOTS_PER_ENGINE` to the desired slot count. Set `CR_VM_MEMORY_GB`; use `CR_CONFIGURE_VM_RESOURCES=false` to retain the VM’s existing resource configuration.

专用 VM 支持 1–24 个引擎进程，每进程 1–8 个 resident 槽位。构建 APK 前设置 `CR_BUILD_ENGINE_COUNT`；运行时 `CR_ENGINE_COUNT` 不得超过该值，`CR_SLOTS_PER_ENGINE` 设置所需槽位数。填写 `CR_VM_MEMORY_GB`；若保留 VM 现有资源配置，设置 `CR_CONFIGURE_VM_RESOURCES=false`。

```powershell
./native_runner/start_engine_cluster.ps1
./native_runner/stop_engine_cluster.ps1
```

The start command reports control ports and capacities for the cache builder. Stop the cluster before starting a console task, and vice versa.

启动命令报告控制端口与容量，可供缓存构建器使用。集群与控制台任务互斥，切换前先停止当前任务。

## 6. Prepare a Linux AVD · 准备 Linux AVD

Android Emulator hosts need `/dev/kvm`, root adbd, and an Android system image capable of running ARM64 libraries. In the base AVD, manually install the original APK, open it to finish the matching update, and exit. Copy the locally built offline APK and its adjacent `.build.json` receipt to Linux. Place the compiled probe at `native_runner/probe/out/libcrprobe.so`, or set `CR_PROBE`.

Android Emulator 主机需要 `/dev/kvm`、root adbd 和可运行 ARM64 库的 Android 系统镜像。在基础 AVD 中手动安装原始 APK，打开并完成匹配更新后退出。将本地构建的离线 APK 及相邻 `.build.json` 回执复制到 Linux；编译好的 probe 放到 `native_runner/probe/out/libcrprobe.so`，或通过 `CR_PROBE` 指定。

```sh
python tools/prepare_training_guest.py --serial emulator-5554 --apk /path/to/cr-ai-offline.apk
```

ADB uses `CR_TRAINING_ADB`, with `--adb` as an override. Preparation enables root, stops the game, applies dual-stack isolation, and runs the device-local backup/install flow. Configure the base AVD with one vCPU for this single-engine readiness check:

ADB 使用 `CR_TRAINING_ADB`，也可通过 `--adb` 覆盖。准备程序启用 root、停止游戏、配置双栈隔离，并在设备内完成备份和安装。以下单引擎就绪检查要求基础 AVD 配置一个 vCPU：

```sh
python -m native_runner.training.v4.configure_emulator_engine_guests \
  --serials emulator-5554 --host-bases 26789 --engines 1 --output runs/guest-ready.json
```

Confirm `production_ready`, then shut down the base AVD. Use the emulator’s `qemu-img` to convert its complete userdata image to standalone raw format. If userdata has a qcow2 overlay, use the topmost overlay as input.

确认 `production_ready` 后关闭基础 AVD。使用模拟器附带的 `qemu-img`，将完整 userdata 转换为独立 raw 镜像；若存在 qcow2 overlay，使用最上层 overlay 作为输入。

```sh
qemu-img convert -O raw <userdata-source> <prepared-userdata.img>
```

Set `CR_EMULATOR_INITDATA` to this prepared image, `CR_SOURCE_AVD` / `CR_SOURCE_AVD_INI` to the matching base AVD, and `CR_TRAINING_APK` to the same signed APK. The prepared userdata must contain the matching application and content. Node launchers clone it for independent guests.

将 `CR_EMULATOR_INITDATA` 设为该已准备镜像，`CR_SOURCE_AVD` / `CR_SOURCE_AVD_INI` 指向匹配的基础 AVD，`CR_TRAINING_APK` 指向同一个签名 APK。准备好的 userdata 必须包含匹配的应用与资源，节点启动器据此克隆独立 guest。

## 7. Distributed PPO · 分布式 PPO

Configure the Linux resource section of `.env`: SDK, emulator, prepared AVD, APK, and probe. Start with `CR_PPO_ENGINES_PER_GUEST=1`; capacity is 1–6 engines per guest with eight resident slots per engine. The APK must be built for at least this many engines. Allocate at least one CPU per engine plus control CPUs. `CR_EMULATOR_COMPAT` points to the emulator compatibility-library directory; use a valid empty directory if no extra libraries are needed.

填写 `.env` 的 Linux 资源区：SDK、模拟器、已准备 AVD、APK 和 probe。最小配置使用 `CR_PPO_ENGINES_PER_GUEST=1`；每个 guest 支持 1–6 个引擎，每引擎八个 resident 槽位。APK 构建容量至少覆盖所需引擎数。每引擎至少分配一个 CPU，并预留控制 CPU。`CR_EMULATOR_COMPAT` 指向模拟器兼容库目录；无需额外兼容库时使用有效空目录。

Within an existing Slurm allocation, launch a simulation node:

在已有 Slurm allocation 内启动模拟节点：

```sh
bash native_runner/training/v4/launch_ppo_engine_node.sh runs/node0 0 1
```

This command holds the engines and writes `runs/node0/engine-node-ready.json`. Continue from another shell. Global engine indices start at zero and must be contiguous without overlap across nodes. Each node uses its own output directory. Create `STOP_ENGINES` in that directory to stop its engines.

该命令持续运行引擎并写入 `runs/node0/engine-node-ready.json`，后续步骤在另一个 shell 执行。全局引擎编号从零开始，跨节点连续且不重叠；各节点使用独立输出目录。在该目录创建 `STOP_ENGINES` 文件可停止对应引擎。

Generate the trainer topology using your allocation’s job ID, host, and CPU assignments. Pass all readiness files after one `--node-ready` option for multiple nodes.

使用本次 allocation 的作业 ID、主机与 CPU 分配生成训练拓扑；多个节点的就绪文件依次列在同一个 `--node-ready` 参数后。

```sh
python -m native_runner.training.v4.build_ppo_multinode_topology \
  --node-ready runs/node0/engine-node-ready.json --output runs/topology.json \
  --trainer-job-id <job-id> --trainer-host <host> --gpus 1 \
  --worker-cpus <worker-cpu-ids> --control-cpus <control-cpu-ids>
```

Build a deck pool from the parent of your dataset directories (`datasets/*/replays/*.parquet`), or use `runs/demo/deck-pool.json` for a workflow check:

从数据集目录的上一级（`datasets/*/replays/*.parquet`）生成卡组池；流程检查可直接使用 `runs/demo/deck-pool.json`：

```sh
python -m native_runner.training.v4.build_ppo_deck_pool \
  --dataset-root ./datasets --output ./runs/deck-pool.json --minimum-uses 5
```

Set `CR_PPO_DECK_POOL` and `CR_PPO_CHECKPOINT` to absolute paths in the training host’s `.env`. The included `checkpoints/General/checkpoint-step-00000460.pt` is one starting point. On the trainer host named in the topology and within the same allocation, run one update with 40-second rollout segments:

在训练主机的 `.env` 中将 `CR_PPO_DECK_POOL` 和 `CR_PPO_CHECKPOINT` 设为绝对路径。随附的 `checkpoints/General/checkpoint-step-00000460.pt` 可作为起点。在拓扑指定的训练主机、同一 allocation 内，执行一次更新，使用 40 秒 rollout segment：

```sh
bash native_runner/training/v4/launch_ppo_multinode_training.sh runs/ppo-smoke runs/topology.json 1 40
```

Inspect match completion, rejected actions, optimizer metrics, and output checkpoints. Configure longer runs through the launcher arguments and PPO settings in `.env`; see `train_ppo_self_play_cluster --help` for algorithm options. Metrics default to local offline W&B. Resume a full training run with `CR_PPO_RESUME_CHECKPOINT`.

检查对局完成情况、动作拒绝、优化器指标和输出 checkpoint。长时间训练通过启动参数与 `.env` 的 PPO 配置调整；算法选项见 `train_ppo_self_play_cluster --help`。指标默认写入本地离线 W&B，使用 `CR_PPO_RESUME_CHECKPOINT` 恢复完整训练。

Run the launchers inside your allocated Slurm job. Check engine readiness and training outputs using the [checks guide](validation.md).

在已分配的 Slurm 作业内运行启动器。引擎就绪状态和训练产物的检查方法见[检查指南](validation.md)。
