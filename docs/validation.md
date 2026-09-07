# Checks · 检查指南

[English home](../README.md) · [中文首页](../README.zh-CN.md) · [Training / 训练](training.md)

Use these checks after installation, when changing an engine or checkpoint, and before starting a training run.

安装完成、更换引擎或模型、开始训练前，可以按以下步骤检查。

## Python and models · Python 与模型

Run from the repository root after installing the dependencies:

安装依赖后，在仓库根目录执行：

```sh
python -m pip check
python tools/check_install.py
```

`pip check` should report no dependency conflicts. `check_install.py` verifies the data hashes, loads each included checkpoint, and prints `Ready: 5 models` when successful.

`pip check` 应报告无依赖冲突。`check_install.py` 校验数据哈希并逐个加载随附模型，成功后打印 `Ready: 5 models`。

## Tests · 自动测试

```sh
python -m pip install pytest
python -m pytest native_runner/tests -q
```

The tests cover environment contracts, card mechanics, observations and actions, native layouts, cache encoding, IL/PPO updates, and model serving. Set `CR_INPUT_APK` to enable original-APK fingerprint checks. CUDA tests need a CUDA device, and native C++ tests need `g++`; tests with unavailable prerequisites are skipped.

测试覆盖环境契约、卡牌机制、观测与动作、原生布局、缓存编码、IL/PPO 更新和模型服务。设置 `CR_INPUT_APK` 可启用原始 APK 指纹检查。CUDA 测试需要 CUDA 设备，原生 C++ 测试需要 `g++`；缺少对应条件时会跳过这些测试。

## Native engine · 原生引擎

Prepare the VM using the [installation guide](getting_started.md), then run:

按[安装指南](getting_started.md)准备 VM 后执行：

```powershell
.\launch_interface.cmd --diagnose
.\launch_interface.cmd
```

Check that the configured VM is found, a match starts at level 11, and both manual and model actions execute. When diagnosing a problem, use `runs/interface/firstlight-interface.log` together with the build and installation receipts.

确认程序找到配置的 VM，对局以等级 11 启动，人工和模型都能正常出牌。排查问题时，查看 `runs/interface/firstlight-interface.log` 以及构建、安装回执。

## Replays and training · 回放与训练

The [training guide](training.md) provides commands for each step. Check the corresponding outputs before moving to the next step:

[训练指南](training.md)提供各步骤的命令。完成一步后，检查对应输出，再继续下一步：

| Step / 步骤 | Check / 检查内容 |
| --- | --- |
| Replay reconstruction / 回放重建 | Actions execute in order and playback reaches the source endpoint / 动作按顺序执行，回放到达源终点 |
| IL cache / IL 缓存 | Completed/failed replay counts, expert/executed action counts, and `--verify-roundtrip` result / 完成与失败回放数、专家与执行动作数、缓存读写一致性结果 |
| IL training / IL 训练 | Finite loss, advancing training groups, and a checkpoint that loads / loss 有限、训练组数递增、保存的模型可加载 |
| Linux guests / Linux guest | Readiness checks pass for the configured engines and capacities / 配置的引擎及容量通过就绪检查 |
| PPO training / PPO 训练 | Rollouts complete, optimizer updates finish, and checkpoints are saved / rollout 完成、优化器更新完成、模型正常保存 |

For a new distributed setup, run the one-update PPO example first and check each simulation node’s readiness report and the trainer output. For model comparisons, inspect the outcome and execution metrics described in [model evaluation](../checkpoints/README.md#evaluation--模型评测).

首次配置分布式环境时，先执行一次更新的 PPO 样例，检查各模拟节点的就绪报告与训练器输出。比较模型时，按[模型评测](../checkpoints/README.md#evaluation--模型评测)检查胜负及执行指标。

## Repository files · 仓库文件

Before publishing repository changes, run:

发布仓库改动前执行：

```sh
python tools/check_release.py
```

This checks the distribution file set, included checkpoint and data hashes, configuration template, and ignore rules. A successful result ends with `0 errors`.

该命令检查分发文件、随附模型与数据哈希、配置模板和忽略规则，成功结果以 `0 errors` 结束。
