# Architecture · 系统架构

[English home](../README.md) · [中文首页](../README.zh-CN.md)

The native engine executes battles, the Python environment builds observations and submits actions, and the policy and trainer handle inference and learning. The engine runs locally from the APK supplied by the user.

原生引擎负责执行对局，Python 环境负责生成观测和提交动作，模型与训练器负责推理和学习。引擎来自用户提供的 APK，在本地运行。

## Source map · 源码导览

| Layer / 层 | Entry / 入口 | Responsibility / 职责 |
| --- | --- | --- |
| Native probe / 原生探针 | [probe/](../native_runner/probe/) | Hooks, addresses, layouts, telemetry, and action execution / hook、地址、布局、遥测与动作执行 |
| Build and deployment / 构建部署 | [offline_build.py](../native_runner/offline_build.py), [offline_install.py](../native_runner/offline_install.py) | Compile, patch, sign, isolate, install, and restore / 编译、注入、签名、隔离、安装与恢复 |
| Engine transport / 引擎通信 | [cr_native_env.py](../native_runner/cr_native_env.py), [resident_batch_channel.py](../native_runner/resident_batch_channel.py) | Native control and resident batch channels / 原生控制与常驻实例批量通道 |
| Battle environment / 对局环境 | [battle_env.py](../native_runner/battle_env.py), [contracts.py](../native_runner/contracts.py) | Episode configuration, reset/step, observations, actions, and terminal state / 对局配置、重置与推进、观测、动作和终局 |
| Policy / 策略 | [model.py](../native_runner/training/v4/model.py), [tensorizer.py](../native_runner/training/v4/tensorizer.py), [decoding.py](../native_runner/training/v4/decoding.py) | Semantic tensors, recurrent actor-critic, and action decoding / 语义张量、循环 actor-critic 与动作解码 |
| Imitation learning / 模仿学习 | [cache_builder.py](../native_runner/training/v4/cache_builder.py), [train_imitation_cache.py](../native_runner/training/v4/train_imitation_cache.py) | Replay reconstruction, sequence caching, and supervised updates / 回放重建、序列缓存与监督更新 |
| Reinforcement learning / 强化学习 | [train_ppo_self_play_cluster.py](../native_runner/training/v4/train_ppo_self_play_cluster.py), [matchmaking.py](../native_runner/training/v4/matchmaking.py), [league.py](../native_runner/training/v4/league.py) | Self-play collection, PPO, and opponent selection / 自博弈采集、PPO 与对手选择 |
| Evaluation and serving / 评测与服务 | [evaluate.py](../native_runner/training/v4/evaluate.py), [serve_policy.py](../native_runner/training/v4/serve_policy.py) | Complete matches and stateful local inference / 完整对局与有状态本地推理 |
| Console and replays / 控制台与回放 | [user_interface.py](../native_runner/user_interface.py), [replay_viewer.py](../native_runner/training/replay_viewer.py), [royaleapi_replay.py](../native_runner/royaleapi_replay.py) | Interactive matches and trajectory reconstruction / 交互对局与轨迹重建 |

## One decision · 一次决策

1. `BattleEnvV1` converts native state into the acting side’s `ObservationV1` under the FAIR visibility contract.
2. The tensorizer combines observations with tracked public events and card/mechanic catalogs.
3. The V4 model encodes entity relations and spatial features, updates recurrent memory, and produces action distributions and a value estimate.
4. Decoding resolves the policy output into `ActionV1` requests. The environment submits them to the native engine and records execution outcomes.

1. `BattleEnvV1` 按 FAIR 可见性契约，将原生状态转换为行动方的 `ObservationV1`。
2. 张量化器结合观测、已追踪的公开事件与卡牌机制目录构造模型输入。
3. V4 模型编码实体关系与空间特征，更新循环记忆，输出动作分布和价值估计。
4. 解码器将策略输出转换为 `ActionV1`，环境提交原生引擎执行并记录结果。

Interactive play uses a continuously rendered battle and applies native deployment timing. Headless/resident execution supports replay reconstruction and training collection. The model decision interval is five native ticks; episode warmup and recurrent state belong to the policy session.

交互对局持续渲染并保留原生出牌等待时间；headless/resident 模式用于回放重建与训练采集。模型每五个原生 tick 决策一次，对局预热与循环状态由策略会话管理。

## Research artifacts · 研究产物

| Artifact / 产物 | Used by / 用途 |
| --- | --- |
| Parquet replay dataset / 回放数据集 | Reconstruction and expert-action extraction / 重建对局并提取专家动作 |
| IL cache and index / IL 缓存与索引 | Recurrent observation/action training sequences / 循环观测与动作训练序列 |
| Checkpoint / 模型文件 | Inference, evaluation, and further training / 推理、评测与继续训练 |
| `TrainingReplayV1` | Native trajectory inspection in the viewer / 在播放器中检查原生对局轨迹 |
| Engine readiness and topology JSON / 引擎就绪与拓扑文件 | Connect allocated simulation workers to the trainer / 将已分配的模拟 worker 连接到训练器 |

Catalog ordering, observation/action versions, and card/form identities are part of model compatibility. When extending a game version or model representation, update the relevant contracts and verify both encoding and native execution. See [compatibility data](../native_runner/data/competitive/README.md).

目录顺序、观测与动作版本、卡牌和形态标识都是模型兼容性的一部分。扩展游戏版本或模型表示时，应同步更新相关契约，并验证编码与原生执行。详见[兼容性数据](../native_runner/data/competitive/README.md)。
