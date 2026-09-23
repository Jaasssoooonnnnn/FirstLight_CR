# Architecture

[English home](../README.md)

The native engine executes battles, the Python environment builds observations and submits actions, and the policy and trainer handle inference and learning. The engine runs locally from the APK supplied by the user.

## Source map

| Layer | Entry | Responsibility |
| --- | --- | --- |
| Native probe | [probe/](../native_runner/probe/) | Hooks, addresses, layouts, telemetry, and action execution |
| Build and deployment | [offline_build.py](../native_runner/offline_build.py), [offline_install.py](../native_runner/offline_install.py) | Compile, patch, sign, isolate, install, and restore |
| Engine transport | [cr_native_env.py](../native_runner/cr_native_env.py), [resident_batch_channel.py](../native_runner/resident_batch_channel.py) | Native control and resident batch channels |
| Battle environment | [battle_env.py](../native_runner/battle_env.py), [contracts.py](../native_runner/contracts.py) | Episode configuration, reset/step, observations, actions, and terminal state |
| Policy | [model.py](../native_runner/training/v4/model.py), [tensorizer.py](../native_runner/training/v4/tensorizer.py), [decoding.py](../native_runner/training/v4/decoding.py) | Semantic tensors, recurrent actor-critic, and action decoding |
| Imitation learning | [cache_builder.py](../native_runner/training/v4/cache_builder.py), [train_imitation_cache.py](../native_runner/training/v4/train_imitation_cache.py) | Replay reconstruction, sequence caching, and supervised updates |
| Reinforcement learning | [train_ppo_self_play_cluster.py](../native_runner/training/v4/train_ppo_self_play_cluster.py), [matchmaking.py](../native_runner/training/v4/matchmaking.py), [league.py](../native_runner/training/v4/league.py) | Self-play collection, PPO, and opponent selection |
| Evaluation and serving | [evaluate.py](../native_runner/training/v4/evaluate.py), [serve_policy.py](../native_runner/training/v4/serve_policy.py) | Complete matches and stateful local inference |
| Console and replays | [user_interface.py](../native_runner/user_interface.py), [replay_viewer.py](../native_runner/training/replay_viewer.py), [royaleapi_replay.py](../native_runner/royaleapi_replay.py) | Interactive matches and trajectory reconstruction |

## One decision

1. `BattleEnvV1` converts native state into the acting side’s `ObservationV1` under the FAIR visibility contract.
2. The tensorizer combines observations with tracked public events and card/mechanic catalogs.
3. The V4 model encodes entity relations and spatial features, updates recurrent memory, and produces action distributions and a value estimate.
4. Decoding resolves the policy output into `ActionV1` requests. The environment submits them to the native engine and records execution outcomes.

Interactive play uses a continuously rendered battle and applies native deployment timing. Headless/resident execution supports replay reconstruction and training collection. The model decision interval is five native ticks; episode warmup and recurrent state belong to the policy session.

## Research artifacts

| Artifact | Used by |
| --- | --- |
| Parquet replay dataset | Reconstruction and expert-action extraction |
| IL cache and index | Recurrent observation/action training sequences |
| Checkpoint | Inference, evaluation, and further training |
| `TrainingReplayV1` | Native trajectory inspection in the viewer |
| Engine readiness and topology JSON | Connect allocated simulation workers to the trainer |

Catalog ordering, observation/action versions, and card/form identities are part of model compatibility. When extending a game version or model representation, update the relevant contracts and verify both encoding and native execution. See [compatibility data](../native_runner/data/competitive/README.md).
