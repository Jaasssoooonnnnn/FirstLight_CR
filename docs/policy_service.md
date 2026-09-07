# Local policy service · 本地策略服务

[English home](../README.md) · [中文首页](../README.zh-CN.md)

`python -m native_runner.training.v4.serve_policy --checkpoint model.pt`
loads a V4 checkpoint once. It reads one JSON object per input line and writes
one response per output line. It does not connect to any game or network service.
EOF closes the active episode. `--device` defaults to `cpu`; `--sample` enables
sampling instead of deterministic decoding.

该命令加载一个 V4 checkpoint，以标准输入输出传递 JSON 行：每行一个请求，每行一个响应。服务不连接游戏或网络。输入结束时关闭当前对局；`--device` 默认 `cpu`，`--sample` 将确定性解码切换为采样。

Each process owns one recurrent episode. Use another process for another actor
or simultaneous episode. Send requests in observation order, exactly once for
each decision. Calling `act` advances the LSTM and previous-action state.

每个进程维护一个对局的循环状态。并行 actor 或对局使用独立进程，请按观测顺序发送请求，每次决策只调用一次；`act` 会推进 LSTM 和上一动作状态。

## Requests · 请求

| `op` | Required fields | Result |
| --- | --- | --- |
| `start` | `episode`: EpisodeConfigV1; `observation`: initial FAIR ObservationV1; `initial_elixir`: mapping with keys `"0"`, `"1"` | `ok`, checkpoint ID and SHA-256; resets any previous episode |
| `observe` | `observation`: FAIR ObservationV1 | Updates the public tracker during initial warmup without choosing an action |
| `act` | `observation`: FAIR ObservationV1 | `ok`, tick, ActionV1 `actions` list, inference time |
| `end` | none | Ends the episode and releases recurrent state |

| `op` | 必填字段 | 返回与行为 |
| --- | --- | --- |
| `start` | `episode`：EpisodeConfigV1；`observation`：初始 FAIR ObservationV1；`initial_elixir`：键为 `"0"`、`"1"` 的映射 | `ok`、模型 ID 与 SHA-256；重置此前对局 |
| `observe` | `observation`：FAIR ObservationV1 | 预热阶段更新公开信息追踪器，不选动作 |
| `act` | `observation`：FAIR ObservationV1 | `ok`、tick、ActionV1 `actions` 列表、推理耗时 |
| `end` | 无 | 结束对局，释放循环状态 |

Episode and observation objects use the existing `to_dict()` contract encoding
from `native_runner/contracts.py`. The actor is `observation.owner`. Supply the
matching episode configuration from `BattleEnvV1.episode_config`; it includes
both deck/form configurations used by the offline public-event tracker.

对局配置与观测使用 [contracts.py](../native_runner/contracts.py) 的 `to_dict()` 编码。行动方为 `observation.owner`；从 `BattleEnvV1.episode_config` 传入匹配的对局配置，其中包含公开事件追踪器所需的双方卡组与形态。

## Client example · 客户端示例

Example, after resetting an existing `environment` and obtaining `observations`:

以下示例接在已有 `environment` 完成 reset、得到 `observations` 之后；将 `model.pt` 替换为实际 checkpoint 路径。

```python
import json
import subprocess
import sys

process = subprocess.Popen(
    [sys.executable, "-m", "native_runner.training.v4.serve_policy",
     "--checkpoint", "model.pt"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
)

def request(payload):
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())

initial_elixir = {
    str(owner): next(p.elixir_exact for p in obs.players if p.owner == owner)
    for owner, obs in observations.items()
}
identity = request({
    "op": "start", "episode": environment.episode_config.to_dict(),
    "observation": observations[0].to_dict(), "initial_elixir": initial_elixir,
})
```

Send `observe` before `FIRST_POLICY_DECISION_TICK`, then `act` every
`POLICY_DECISION_TICKS` (five native ticks). Both constants are exported by
`training/v4/expert.py`. Deserialize returned actions with `ActionV1.from_mapping`
and submit them to the offline environment; an empty list means WAIT. Rendered
interactive callers also apply the native deployment delay, as `offline_agent`
does. After terminal/truncation, send `end`, or start a new episode.

在 `FIRST_POLICY_DECISION_TICK` 前发送 `observe`，之后每隔 `POLICY_DECISION_TICKS`（五个原生 tick）发送一次 `act`。两个常量见 [expert.py](../native_runner/training/v4/expert.py)。用 `ActionV1.from_mapping` 解析返回动作并提交给离线环境，空列表表示 WAIT。交互渲染调用方还需像 `offline_agent` 一样应用原生出牌等待时间。对局结束或截断后发送 `end`，或开始新对局。

Invalid contract/request input returns `{"ok": false, "error": "..."}` and
discards the episode state. Start again before another inference. Unexpected
engine/model errors terminate the process and are written to stderr.

无效契约或请求返回 `{"ok": false, "error": "..."}` 并清除当前对局状态，后续推理前需重新 `start`。未预期的引擎或模型错误会终止进程，并写入标准错误输出。
