# Models · 模型

[English home](../README.md) · [中文首页](../README.zh-CN.md) · [Training / 训练](../docs/training.md) · [Training history / 训练历程](../docs/training_history.md)

FirstLight CR includes a **strong general-purpose agent** and a **Hall of Fame–level 2.6 Hog Cycle specialist**, alongside IL checkpoints. Select a model in the match interface, or pass its path to the evaluator or policy service.

FirstLight CR 提供**较强的通用模型**、**名人堂级 2.6 速猪专精模型**和 IL checkpoint。可在对战界面选择模型，也可将文件路径传给评测器或策略服务。

## Checkpoint selection · 模型选择

| Model / 模型 | Use / 用途 | File / 文件 |
| --- | --- | --- |
| General / 通用模型 | Stronger with cheap decks; more ordinary with expensive decks / 小费较好，大费一般 | [General/checkpoint-step-00000460.pt](General/checkpoint-step-00000460.pt) |
| IL / 模仿学习模型 | Human-like play across decks, modest overall strength / 各类卡组打法像真人，整体实力一般 | [IL/checkpoint-step-00029396.pt](IL/checkpoint-step-00029396.pt) |
| Active IL / Active IL 模型 | PPO-adjusted IL: more active, better with expensive decks; cheap-deck play regressed / PPO 调整的积极版 IL：大费较好，小费有所退化 | [active IL/checkpoint-step-00000030.pt](<active IL/checkpoint-step-00000030.pt>) |
| Hog 2.6 specialist 1 / 2.6 速猪专精 1 | The more proactive specialist / 更积极一些的速猪模型 | [2_6hog_expert/hog26-specialist1.pt](2_6hog_expert/hog26-specialist1.pt) |
| Hog 2.6 specialist 2 / 2.6 速猪专精 2 | More passive, stronger measured match results / 更被动，模型对战成绩更强 | [2_6hog_expert/hog26-specialist2.pt](2_6hog_expert/hog26-specialist2.pt) |

These playstyle descriptions summarize the author’s hands-on observations. The author reached Hall of Fame on an account using the Hog specialist; the [training history](../docs/training_history.md) separates that account result from model-versus-model evaluations and explains each model’s origin.

上述打法特点来自作者的实战观察。作者使用速猪模型将一个账号打上了名人堂；[训练历程](../docs/training_history.md)分别记录这项账号成绩和模型对战评测，并说明各模型的来源。

Choose General for a first match. For the trained Hog specialist deck, select Hog Rider, Hero Musketeer, Evolved Cannon, Fireball, The Log, Evolved Skeletons, Ice Golem, and Ice Spirit; set those forms explicitly in the interface.

第一次对战可选 General。体验训练时的速猪卡组，请选择野猪骑士、精英火枪手、觉醒加农炮、火球、滚木、觉醒骷髅兵、冰人和冰精灵，并在界面中明确设置这些形态。

## Local inference · 本地推理

```sh
python tools/check_install.py
python -m native_runner.training.v4.serve_policy \
  --checkpoint ./checkpoints/General/checkpoint-step-00000460.pt
```

The service waits for JSON-line requests. See the [policy protocol](../docs/policy_service.md) for episode initialization and action requests. For complete matches, follow the evaluator example in the [main README](../README.md).

策略服务启动后等待 JSON 行请求；对局初始化与动作请求格式见[策略协议](../docs/policy_service.md)。完整对局评测命令见[中文首页](../README.zh-CN.md)。

## Evaluation · 模型评测

Stop any active interface task and start the offline engine. Run the evaluator with two checkpoints, using `--games` for the number of matches, `--match-config` for a JSON match configuration, and `--seed` for the starting seed. Configuration fields follow [MatchConfig](../native_runner/match_factory.py). Add `--sample` for sampled actions; the default uses deterministic decoding.

先停止界面中的任务并启动离线引擎，再用评测器比较两个模型。`--games` 设置对局数，`--match-config` 指定 JSON 对局配置，`--seed` 设置起始种子。配置字段见 [MatchConfig](../native_runner/match_factory.py)。默认采用确定性解码，添加 `--sample` 可改为采样出牌。

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

To train an IL model from replay data, follow the cache-building and IL steps in the [training guide](../docs/training.md). To improve an included model through self-play, set `CR_PPO_CHECKPOINT` to its path and follow the PPO steps. Record the dataset and selected tags, configuration, seed, and starting checkpoint with each run so that you can repeat the experiment.

从回放数据训练 IL 模型，按[训练指南](../docs/training.md)先构建缓存，再运行 IL。使用随附模型继续自博弈训练时，将 `CR_PPO_CHECKPOINT` 设为其路径，再执行 PPO 步骤。每次训练保存数据集与 tag 选择、配置、种子及起始模型，便于重复实验。

The included files are identified by SHA-256 in [manifest.json](manifest.json). Card and feature compatibility is described in [environment data](../native_runner/data/competitive/README.md).

随附模型的 SHA-256 见 [manifest.json](manifest.json)，卡牌与特征兼容性见[环境数据说明](../native_runner/data/competitive/README.md)。
