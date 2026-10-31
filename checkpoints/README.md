# Models · 模型

[English home](../README.md) · [中文首页](../README.zh-CN.md) · [Training / 训练](../docs/training.md) · [Training history / 训练历程](../docs/training_history.md)

This branch includes one model, **Everlight**. The five earlier models (General, IL, Active IL and two Hog 2.6 specialists) are on the [`main`](https://github.com/Jaasssoooonnnnn/FirstLight_CR/tree/main) branch.

本分支随附一个模型 **Everlight**。之前的五个模型（General、IL、Active IL 和两个 2.6 速猪专精模型）在 [`main`](https://github.com/Jaasssoooonnnnn/FirstLight_CR/tree/main) 分支。

## Everlight

| File / 文件 | Training / 训练 | Size / 大小 |
| --- | --- | --- |
| [Everlight/v15-general-il100-kl5e-2-v80.pt](Everlight/v15-general-il100-kl5e-2-v80.pt) | IL, then PPO self-play; update 80 / 先 IL，再 PPO 自博弈；第 80 次更新 | 67.3 MB |

Compared with the models on `main`:

- **Timing.** A card or ability takes effect 20 ticks (1 second) after it is played. Training, replay reconstruction and the match interface all follow this, for the player and the model alike.
- **Inputs.** Each card token has 21 runtime features instead of 16. The five new ones describe the model's own card that is on the way: whether it is pending, the ticks left, its target x/y, and whether it is an ability. The opponent's pending cards stay hidden until they land.
- **Network.** Spatial CNN with 128 channels and 5 residual blocks (was 64 and 3); LSTM with 768 units (was 512). About 16.8M trainable parameters.
- **Inference settings.** The checkpoint records `delay_policy: earliest` (each card is played at the earliest legal delay bin) and delay bins up to 200 ms. `load_policy_v4` reads both, so the console, evaluator and policy service use them without extra flags.

和 `main` 上的模型相比：

- **时序。** 下牌和放技能后过 20 tick（1 秒）才生效。训练、回放重建和对战界面都按这个规则执行，玩家和模型一样。
- **输入。** 每张卡牌的运行时特征从 16 维增加到 21 维。新增的 5 维描述模型自己已下出、尚未到场的牌：是否在途、剩余 tick、目标坐标、是否为技能。对手在途的牌在落地前仍然不可见。
- **网络。** 空间 CNN 为 128 通道、5 个残差块（原为 64 通道、3 个）；LSTM 为 768 维（原为 512）。可训练参数约 1680 万。
- **推理设置。** checkpoint 里记录了 `delay_policy: earliest`（每张牌按最早的合法延迟档打出）和最大 200 ms 的延迟档。`load_policy_v4` 会读取这两项，对战界面、评测器和策略服务无需额外参数。

The `main` checkpoints were trained with 16 card runtime features and without the delay, so they do not load here. `native_runner/training/v4/migrate_checkpoint_card_runtime.py` can pad them to 21 features for comparison, but they still play as if cards took effect at once.

`main` 上的模型按 16 维特征、无延迟训练，在本分支无法直接加载。`native_runner/training/v4/migrate_checkpoint_card_runtime.py` 可以把它们补零到 21 维用于对比，但它们的打法仍然默认牌会立即生效。

## Local inference · 本地推理

```sh
python tools/check_install.py
python -m native_runner.training.v4.serve_policy   --checkpoint ./checkpoints/Everlight/v15-general-il100-kl5e-2-v80.pt
```

The service waits for JSON-line requests. See the [policy protocol](../docs/policy_service.md) for episode initialization and action requests. For complete matches, follow the evaluator example in the [main README](../README.md).

策略服务启动后等待 JSON 行请求；对局初始化与动作请求格式见[策略协议](../docs/policy_service.md)。完整对局评测命令见[中文首页](../README.zh-CN.md)。

## Evaluation · 模型评测

Stop any active interface task and start the offline engine. Run the evaluator with two checkpoints (for example Everlight and your own run), using `--games` for the number of matches, `--match-config` for a JSON match configuration, and `--seed` for the starting seed. Configuration fields follow [MatchConfig](../native_runner/match_factory.py). Add `--sample` for sampled actions; the default uses deterministic decoding.

先停止界面中的任务并启动离线引擎，再用评测器比较两个模型（例如 Everlight 和你自己训练的模型）。`--games` 设置对局数，`--match-config` 指定 JSON 对局配置，`--seed` 设置起始种子。配置字段见 [MatchConfig](../native_runner/match_factory.py)。默认采用确定性解码，添加 `--sample` 可改为采样出牌。

The evaluator alternates sides and moves each model’s assigned deck with it. Keep decks, forms, tower troops, levels, starting seeds, and decoding mode consistent across comparisons. For a specialist, explicitly provide its deck rather than using the default match configuration.

评测器逐局交换上下方，卡组随对应模型一起交换。比较不同模型时，保持卡组、形态、塔兵、等级、起始种子和解码方式一致。评测速猪专精模型时，显式配置速猪卡组。

| Output / 输出 | Contents / 内容 |
| --- | --- |
| `identity.json` | Checkpoint hashes, match configuration, and sampling mode / 模型哈希、对局配置与采样方式 |
| `matches.jsonl` | Per-match winner, seed, side assignment, terminal state, non-WAIT actions, and rejections / 逐局胜负、种子、位置、终局、非 WAIT 动作与拒绝数 |
| `summary.json` | Requested/completed matches, A/B wins, draws, and truncations / 请求与完成局数、双方胜场、平局与截断数 |

Use wins, losses, and draws from normally terminated matches to compare performance. Report the game count and opponent with any win rate. Check `truncated` and `rejected_actions` separately to distinguish execution problems from lost matches. `actions_by_owner` counts card plays and abilities and is indexed by arena side; `a_owner` identifies model A’s side.

使用正常结束对局的胜负和平局比较表现，报告胜率时同时给出局数和对手。另行检查 `truncated` 与 `rejected_actions`，将执行问题和正常输局分开。`actions_by_owner` 统计下牌与技能次数，按场地方向排列；`a_owner` 标明模型 A 所在的一方。

## Training · 训练

To train an IL model from replay data, follow the cache-building and IL steps in the [training guide](../docs/training.md). Caches built before the 20-tick change cannot be reused. To continue from Everlight with self-play, set `CR_PPO_CHECKPOINT` to its path and follow the PPO steps; the guide also lists the reward options, including the tower potential. Record the dataset and selected tags, configuration, seed, and starting checkpoint with each run so that you can repeat the experiment.

从回放数据训练 IL 模型，按[训练指南](../docs/training.md)先构建缓存，再运行 IL。20 tick 改动之前生成的缓存不能再用。从 Everlight 继续自博弈训练时，将 `CR_PPO_CHECKPOINT` 设为其路径，再执行 PPO 步骤；训练指南也列出了奖励选项，包括塔势能。每次训练保存数据集与 tag 选择、配置、种子及起始模型，便于重复实验。

The included file is identified by SHA-256 in [manifest.json](manifest.json). Card and feature compatibility is described in [environment data](../native_runner/data/competitive/README.md).

随附模型的 SHA-256 见 [manifest.json](manifest.json)，卡牌与特征兼容性见[环境数据说明](../native_runner/data/competitive/README.md)。
