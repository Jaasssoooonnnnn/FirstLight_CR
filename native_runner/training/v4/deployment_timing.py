"""Log decision-to-native-spawn timing without changing policy observations."""

from typing import Any, Callable


class DeploymentTimingRecorder:
    def __init__(self, log: Callable[[dict[str, Any]], None]) -> None:
        self.log = log
        self.seen: set[str] = set()
        self.pending: list[dict[str, Any]] = []
        self.deployments: dict[str, dict[str, Any]] = {}
        self.finished: set[str] = set()

    def submit(self, decision: int, tick: int, actions: list[dict[str, Any]]) -> None:
        for index, action in enumerate(actions):
            if action["kind"] == "play_card":
                item = {"decision": decision, "actionIndex": index, "decisionTick": tick, "cardId": action["cardId"]}
                self.pending.append(item)
                self.log({"event": "deployment_timing_requested", **item})

    def observe(self, events: list[dict[str, Any]], observation_tick: int) -> None:
        # Native rings can overlap observations. Retain exact deployment joins
        # across observations, including a card play whose spawn arrives later.
        fresh = [e for e in events if e["nativeEventId"] not in self.seen]
        self.seen.update(e["nativeEventId"] for e in fresh)
        for event in sorted(fresh, key=lambda e: (e["tick"], e["kind"] != "card_play")):
            self.log({"event": "native_deployment_timing", **event, "observationTick": observation_tick})
            deployment_id = event["deploymentId"]
            if event["kind"] == "card_play":
                candidates = [
                    p for p in self.pending if p["cardId"] == event["cardId"] and p["decisionTick"] <= event["tick"]
                ]
                if len(candidates) != 1:
                    self.log(
                        {
                            "event": "deployment_timing_unmatched",
                            "deploymentId": deployment_id,
                            "candidateCount": len(candidates),
                        }
                    )
                    continue
                selected = candidates[0]
                self.pending.remove(selected)
                self.deployments[deployment_id] = {**selected, "cardPlayTick": event["tick"]}
            elif deployment_id in self.deployments and deployment_id not in self.finished:
                # A projectile is reported separately from a unit/building.
                if event["kind"] == "spawn" and event["nativeObjectId"] is None:
                    continue
                selected = self.deployments[deployment_id]
                delta = event["tick"] - selected["decisionTick"]
                if delta < 0:
                    continue
                self.finished.add(deployment_id)
                self.log(
                    {
                        "event": "deployment_spawn_latency",
                        **selected,
                        "deploymentId": deployment_id,
                        "spawnTick": event["tick"],
                        "spawnKind": event["kind"],
                        "objectKind": event["objectKind"],
                        "nativeObjectId": event["nativeObjectId"],
                        "latencyTicks": delta,
                        "latencyMs": delta * 50,
                        "association": "unique_pending_card_then_native_deployment_id",
                    }
                )

    def finish(self) -> None:
        for pending in self.pending:
            self.log({"event": "deployment_timing_unresolved", **pending, "reason": "no_unique_card_play"})
        for key, selected in self.deployments.items():
            if key not in self.finished:
                self.log(
                    {
                        "event": "deployment_timing_unresolved",
                        **selected,
                        "deploymentId": key,
                        "reason": "no_native_spawn",
                    }
                )
