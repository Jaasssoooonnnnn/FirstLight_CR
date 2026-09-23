<p align="center">
  <img src="docs/assets/firstlight-cr.png" width="240" alt="FirstLight CR logo">
</p>
<h1 align="center">FirstLight CR</h1>
<p align="center"><strong>From replays to agents. From self-play to the arena.</strong></p>
<p align="center">Train Clash Royale agents and play offline, with imitation learning and PPO self-play.</p>
<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Learning-IL_%2B_PPO-EE4C2C.svg" alt="IL and PPO">
  <a href="https://huggingface.co/datasets/VanguardX101/IL_Replay"><img src="https://img.shields.io/badge/🤗_Dataset-252K_replays-FFD21E.svg" alt="252K replays on Hugging Face"></a>
</p>
<p align="center">
  <a href="docs/getting_started.md">Installation</a> ·
  <a href="docs/training.md">Training</a> ·
  <a href="docs/training_history.md">Training history</a> ·
  <a href="checkpoints/README.md">Models</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/policy_service.md">Policy API</a>
</p>

<p align="center">
  <img src="docs/assets/battle-montage.gif" width="960" alt="18 battle clips in a six-column, three-row montage at 12 times speed">
</p>

> **Video: the model's Hall of Fame push and a walkthrough of the AI architecture**
>
> [Watch in English (YouTube)](https://www.youtube.com/watch?v=TpfhzVlXWqw&lc=UgyyW1z5iYdLX8BZ0xl4AaABAg) · [观看中文版（哔哩哔哩）](https://www.bilibili.com/video/BV17Zb46JE9h)

## Pretrained models

**General is a strong model across decks. The author used the Hog 2.6 specialist to reach Hall of Fame.**

The [training history](docs/training_history.md) covers how the models were trained and evaluated.

Choose **General** for different decks or a **Hog 2.6 specialist** for Hog Cycle. Both are included in the repository. See the [model guide](checkpoints/README.md) for checkpoint details and evaluation.

## Features

- **Native battle environment.** Headless and rendered execution, structured observations, card deployment and abilities, with Python `reset` / `step` interfaces.
- **Imitation learning.** Reconstruct Parquet replays in the engine and build observation/action sequences for IL training.
- **Self-play reinforcement learning.** PPO with matchmaking, opponent pools, resident simulation, and distributed collection.
- **Models and inputs.** The V4 actor-critic uses card features, entity relations, spatial features, and recurrent memory. Training, evaluation, and local inference use the same model and input format.
- **Match and replay interface.** Play against a model, control both sides manually, or watch training and collected replays in one offline VM.
- **Source code.** The probe, APK build tools, battle environment, and training code are included.

**122 supported cards · 5 included checkpoints · 252,238 replays · 17,836,160 actions**

See [compatibility data](native_runner/data/competitive/README.md) for the cards and forms supported by each version. Download the replay dataset from [Hugging Face](https://huggingface.co/datasets/VanguardX101/IL_Replay).

## Training flow

```mermaid
flowchart LR
    R[Replay dataset] --> C[Native reconstruction]
    C --> D[Observation / action sequences]
    D --> IL[Imitation learning]
    IL --> P[Recurrent policy]
    P --> S[Self-play + matchmaking]
    S --> PPO[PPO]
    PPO --> P
    P --> E[Match evaluation]
    P --> H[Play against the agent]
    P --> API[Local policy service]
```

See the [architecture guide](docs/architecture.md) for source entry points and data formats.

## Quick start

For a fresh machine or coding-agent setup, follow the [installation guide](docs/getting_started.md). The steps below are a summary.

**Desktop:** Windows, Python 3.12 with Tkinter, and an Android 12 instance of regular MuMu Player with root and ARM64 application support. CPU inference is supported. Building the offline APK from source requires JDK 17, Android NDK, and SDK build-tools; running an already built APK does not.

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill the Python, VM, and build-tool settings in `.env` using the [installation guide](docs/getting_started.md). The supported engine is **Null’s Royale 15.535.13 / arm64-v8a with content update 15.535.86**.

**First, install and open the unmodified original APK in the selected MuMu instance. Let it update naturally until you can enter the normal game lobby, then exit.** Keep the original APK as `user-provided/nulls-royale.apk`. Check the original APK and downloaded resources before building or injecting the probe:

```powershell
./check_original_game.ps1 -InputApk ./user-provided/nulls-royale.apk
./build_offline_engine.ps1 `
  -InputApk ./user-provided/nulls-royale.apk `
  -OutputApk ./build/cr-ai-offline.apk
./install_offline_engine.ps1 -Apk ./build/cr-ai-offline.apk
```

To open the interface after installation, run `Firstlight_CR\launch_interface.cmd` from its parent folder, or run `./launch_interface.cmd` from the repository root.

## Train and evaluate

Start with [IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) for imitation learning, or generate a small workflow example with `python tools/make_demo_data.py`. The [training guide](docs/training.md) covers cache generation, IL, engine clusters, Linux AVD preparation, and distributed PPO.

To compare two included checkpoints, stop any console task and run:

```powershell
./native_runner/start_offline.ps1
.\.venv\Scripts\python.exe -m native_runner.training.v4.evaluate `
  --checkpoint-a ./checkpoints/General/checkpoint-step-00000460.pt `
  --checkpoint-b ./checkpoints/IL/checkpoint-step-00029396.pt `
  --games 2 --output ./evaluations/comparison
```

Use `--games` for the match count, `--match-config` for decks and levels, and `--seed` for the starting seed. See [model evaluation](checkpoints/README.md#evaluation--模型评测) for results and metrics. To integrate your own actor loop, use the [JSON-lines policy service](docs/policy_service.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Getting started](docs/getting_started.md) | Requirements, `.env`, original game update, and offline installation |
| [Training history](docs/training_history.md) | Five main models, experiments, results, and playstyle tradeoffs |
| [Training](docs/training.md) | Replay cache, IL, batch simulation, Linux guests, and PPO |
| [Architecture](docs/architecture.md) | Environment, policy, training, and replay source map |
| [Models](checkpoints/README.md) | Model selection, loading, and evaluation metrics |
| [Policy service](docs/policy_service.md) | Recurrent episode lifecycle and request protocol |
| [Compatibility data](native_runner/data/competitive/README.md) | Card support, model catalogs, and version maintenance |
| [Checks](docs/validation.md) | Environment checks, tests, and runtime result checks |

## License

FirstLight CR is an independent offline project under [Apache-2.0](LICENSE), unaffiliated with Supercell. The repository does not distribute the game APK or its resources; users supply the supported APK locally. See [NOTICE](NOTICE).
