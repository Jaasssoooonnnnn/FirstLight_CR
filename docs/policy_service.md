# Local policy service

[English home](../README.md)

`python -m native_runner.training.v4.serve_policy --checkpoint model.pt`
loads a V4 checkpoint once. It reads one JSON object per input line and writes
one response per output line. It does not connect to any game or network service.
EOF closes the active episode. `--device` defaults to `cpu`; `--sample` enables
sampling instead of deterministic decoding.

Each process owns one recurrent episode. Use another process for another actor
or simultaneous episode. Send requests in observation order, exactly once for
each decision. Calling `act` advances the LSTM and previous-action state.

## Requests

| `op` | Required fields | Result |
| --- | --- | --- |
| `start` | `episode`: EpisodeConfigV1; `observation`: initial FAIR ObservationV1; `initial_elixir`: mapping with keys `"0"`, `"1"` | `ok`, checkpoint ID and SHA-256; resets any previous episode |
| `observe` | `observation`: FAIR ObservationV1 | Updates the public tracker during initial warmup without choosing an action |
| `act` | `observation`: FAIR ObservationV1 | `ok`, tick, ActionV1 `actions` list, inference time |
| `end` | none | Ends the episode and releases recurrent state |

Episode and observation objects use the existing `to_dict()` contract encoding
from `native_runner/contracts.py`. The actor is `observation.owner`. Supply the
matching episode configuration from `BattleEnvV1.episode_config`; it includes
both deck/form configurations used by the offline public-event tracker.

## Client example

Example, after resetting an existing `environment` and obtaining `observations`. Replace `model.pt` with the actual checkpoint path:

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

Invalid contract/request input returns `{"ok": false, "error": "..."}` and
discards the episode state. Start again before another inference. Unexpected
engine/model errors terminate the process and are written to stderr.
