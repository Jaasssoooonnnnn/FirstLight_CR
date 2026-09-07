# Getting started · 安装与运行

[English home](../README.md) · [中文首页](../README.zh-CN.md) · [Training / 训练](training.md)

## 1. Requirements · 环境要求

| Component / 组件 | Requirement / 要求 |
| --- | --- |
| Desktop / 桌面 | Windows; Python 3.12 with Tkinter / 含 Tkinter 的 Python 3.12 |
| VM / 虚拟机 | Dedicated rooted MuMu instance with ARM64 application support / 支持 ARM64 应用、开启 root 的 MuMu 专用实例 |
| Build / 构建 | JDK 17, Android NDK, Android SDK build-tools; validated with NDK 27.3 and build-tools 37.0 / 已验证 NDK 27.3 与 build-tools 37.0 |
| Engine / 引擎 | Null’s Royale APK 15.535.13, arm64-v8a; content update 15.535.86 / 更新资源 15.535.86 |
| Inference / 推理 | CPU or compatible CUDA device / CPU 或兼容的 CUDA 设备 |
| IL / PPO launchers / 训练入口 | Linux, CUDA, NCCL; see [training](training.md) / 参见训练指南 |

Exact engine fingerprints are in [supported_engine.json](../native_runner/supported_engine.json).

引擎的精确指纹见 [supported_engine.json](../native_runner/supported_engine.json)。

## 2. Python environment · Python 环境

Run all commands from the repository root. On a GPU training host, install the appropriate CUDA build of PyTorch before installing the requirements.

所有命令从仓库根目录执行。GPU 训练主机先安装适配的 CUDA 版 PyTorch，再安装其余依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe tools/check_install.py
```

`check_install.py` checks the included models and their catalogs in the active Python environment. Success means you can load the policy stack; native matches additionally require the VM setup below.

`check_install.py` 检查当前 Python 环境中的随附模型及其目录。通过后即可加载策略系统；原生对局还需完成以下 VM 配置。

## 3. Configure .env · 配置 .env

Use UTF-8 `KEY=value`. Values are literal, with optional matching quotes; variable expansion is not supported. Existing process environment variables take precedence. Blank values use defaults. `.env` is local and Git-ignored; keep the shared `.env.example` blank.

文件使用 UTF-8 `KEY=value`，值按字面读取，可加成对引号，不展开变量。已有进程环境变量优先，空值使用默认值。`.env` 保留在本机并由 Git 忽略；共享的 `.env.example` 保持空值。

| Setting / 参数 | Value / 填写内容 |
| --- | --- |
| `CR_PYTHON`, `CR_PYTHONW` | Absolute paths to `.venv/Scripts/python.exe` and `pythonw.exe` / 两个解释器的绝对路径 |
| `CR_MUMU_MANAGER` | Absolute path to `MuMuManager.exe` / 管理器绝对路径 |
| `CR_ADB` | Absolute path to Android SDK `adb.exe` / ADB 绝对路径 |
| `CR_VM_INDEX`, `CR_VM_NAME` | Index and exact name of the dedicated MuMu instance / 专用实例编号和精确名称 |
| `CR_ADB_SERIAL` | That instance’s ADB address, e.g. `127.0.0.1:PORT` / 该实例的 ADB 地址 |
| `CR_JAVA`, `CR_KEYTOOL` | JDK `java` and `keytool` executable paths / JDK 可执行文件路径 |
| `CR_BUILD_TOOLS`, `CR_NDK_ROOT` | SDK build-tools version directory and NDK root / build-tools 版本目录与 NDK 根目录 |
| `CR_REPLAY_DATASET_ROOT` | Optional local dataset directory containing `replays/` and `actions/` / 可选，本地数据集目录 |

`CR_CONTROL_PORT` defaults to `26789`; choose a free host port if occupied. `CR_GUEST_CONTROL_PORT` defaults to `26789`. Leave `CR_WORKSPACE_ROOT` and `CR_COMPETITIVE_DATA_ROOT` blank to use the repository defaults. Advanced simulation and Linux settings are covered in [training](training.md).

`CR_CONTROL_PORT` 默认 `26789`，被占用时换成空闲主机端口。`CR_GUEST_CONTROL_PORT` 默认 `26789`。`CR_WORKSPACE_ROOT` 与 `CR_COMPETITIVE_DATA_ROOT` 留空即可使用仓库默认值。批量模拟与 Linux 配置见[训练指南](training.md)。

## 4. Prepare the game and build · 准备游戏与构建

1. Obtain the supported original APK yourself. Install it in the configured VM under Android user 0, open it, and wait for the game to finish downloading the matching content update. Exit the game when ready.
2. Keep the original APK at `user-provided/nulls-royale.apk`, or specify its location with `-InputApk` / `CR_INPUT_APK`.
3. Build the probe and offline APK, then install it in the prepared VM.

1. 自行取得指定原始 APK，安装到配置好的 VM 的 Android 用户 0 中，手动打开并等待游戏完成匹配版本的资源更新，然后退出。
2. 将原始 APK 保留为 `user-provided/nulls-royale.apk`，或通过 `-InputApk` / `CR_INPUT_APK` 指定其位置。
3. 构建 probe 与离线 APK，再安装到准备好的 VM 中。

```powershell
./build_offline_engine.ps1 `
  -InputApk ./user-provided/nulls-royale.apk `
  -OutputApk ./build/cr-ai-offline.apk
./install_offline_engine.ps1 -Apk ./build/cr-ai-offline.apk
```

The build verifies the input, compiles the C++ probe, patches manifest/smali, and aligns and signs the APK. It may download a hash-pinned Apktool release. The probe output is `native_runner/probe/out/libcrprobe.so`; `CR_PROBE` overrides this path for both build and runtime.

构建会校验输入、编译 C++ probe、修改 manifest/smali，再对齐和签名 APK；所需的 Apktool 使用固定哈希版本，可能在构建时下载。probe 输出到 `native_runner/probe/out/libcrprobe.so`；`CR_PROBE` 可同时覆盖构建和运行路径。

The installer uses root ADB, isolates the VM’s IPv4/IPv6 external network, verifies the installed content, and backs up the existing APK and application data before replacement. Keep the game installed only under Android user 0. Updated resources remain inside the VM. Opening the original game and downloading its update is a manual prerequisite, not a program step.

安装器使用 root ADB，隔离 VM 的 IPv4/IPv6 外网，检查资源版本，在替换前备份已有 APK 与应用数据。该游戏仅应安装在 Android 用户 0 下。更新资源始终留在 VM 内；打开原版游戏并完成下载是用户的手动准备步骤。

Build and installation receipts are saved as `.build.json` and `.install.json`. Installation failures attempt rollback. To restore a saved backup, use the `backup` value from the installation receipt:

构建与安装回执分别保存为 `.build.json` 和 `.install.json`。安装失败时会尝试回滚。手动恢复时，使用安装回执中的 `backup` 值：

```powershell
./install_offline_engine.ps1 -RestoreBackup <backup>
```

## 5. Open the console · 打开控制台

```powershell
.\launch_interface.cmd --diagnose
.\launch_interface.cmd
```

Select a model and decks on the first page. The model controls the top side and the overlay controls the bottom side, at level 11 and normal speed. Cards can be selected before enough elixir is available; click the arena to deploy once affordable. The four pages share one VM and run one task at a time.

第一页选择模型与卡组，模型控制上方，覆盖层控制下方，固定等级 11、正常速度。圣水不足时也可预选卡牌，足够后点击竞技场出牌。四个页面共用一个 VM，每次运行一项任务。

Models are discovered under `checkpoints/`. Training playback accepts exported native trajectories; collected playback accepts a Parquet dataset such as [IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay). Sample inputs and export instructions are in [training](training.md). Console logs are written to `runs/interface/firstlight-interface.log`.

模型从 `checkpoints/` 自动发现。训练回放读取导出的原生轨迹；采集回放读取 Parquet 数据集，例如 [IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay)。样例生成与轨迹导出方法见[训练指南](training.md)。控制台日志位于 `runs/interface/firstlight-interface.log`。

## Troubleshooting · 排查

| Symptom / 现象 | Check / 检查 |
| --- | --- |
| VM identity mismatch / 实例身份不匹配 | Match index, exact name, and ADB serial to the same instance / 核对编号、精确名称与 ADB 地址是否属于同一实例 |
| Content fingerprint mismatch / 资源指纹不匹配 | Check the supported APK and update version before rebuilding / 核对指定 APK 与更新资源版本后重新构建 |
| Busy engine / 引擎忙 | Stop the active console, evaluation, or cluster task / 停止占用该引擎的控制台、评测或集群任务 |
| Model compatibility error / 模型兼容错误 | Run `tools/check_install.py` and use a compatible V4 checkpoint / 运行兼容性检查并使用匹配的 V4 模型 |
