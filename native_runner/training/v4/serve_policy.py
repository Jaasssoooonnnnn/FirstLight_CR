"""Serve one recurrent policy over newline-delimited JSON on stdin/stdout."""

import argparse
import json
import sys

import torch

from ...contracts import EpisodeConfigV1, ObservationV1
from .expert import FIRST_POLICY_DECISION_TICK
from .factory import build_episode_tensorizer_v4
from .policy_session import PolicySessionV4, load_policy_v4


class PolicyService:
    """A process owns one episode at a time; no game transport or account access."""

    def __init__(self, loaded, *, device="cpu", sample=False):
        self.loaded, self.device, self.sample = loaded, device, sample
        self.session = None

    def end(self):
        if self.session is not None:
            self.session.end_episode()
            self.session = None

    def handle(self, request):
        op = request["op"]
        if op == "start":
            self.end()
            episode = EpisodeConfigV1.from_mapping(request["episode"])
            observation = ObservationV1.from_mapping(request["observation"])
            tensorizer = build_episode_tensorizer_v4(episode, actor_owner=observation.owner)
            self.session = PolicySessionV4(self.loaded.model, tensorizer, device=self.device, sample=self.sample)
            self.session.start_episode(
                observation, initial_elixir={int(k): float(v) for k, v in request["initial_elixir"].items()}
            )
            return {
                "ok": True,
                "checkpoint_id": self.loaded.checkpoint_id,
                "checkpoint_sha256": self.loaded.checkpoint_sha256,
            }
        if op == "end":
            self.end()
            return {"ok": True}
        if op not in {"observe", "act"}:
            raise ValueError("op must be start, observe, act or end")
        if self.session is None:
            raise ValueError("start an episode before inference")
        observation = ObservationV1.from_mapping(request["observation"])
        if observation.owner != self.session.actor_owner:
            raise ValueError("observation owner differs from the active episode")
        if op == "observe":
            self.session.tensorizer.tensorize(observation, validate=False)
            return {"ok": True}
        if observation.tick < FIRST_POLICY_DECISION_TICK:
            raise ValueError("use observe during the initial warmup ticks")
        decision = self.session.decide(observation)
        return {
            "ok": True,
            "tick": observation.tick,
            "actions": [a.to_dict() for a in decision.decoded.actions],
            "inference_ms": decision.inference_ms,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    service = PolicyService(load_policy_v4(args.checkpoint, device=args.device), device=args.device, sample=args.sample)
    try:
        for line in sys.stdin:
            try:
                result = service.handle(json.loads(line))
            except (ValueError, TypeError, KeyError) as error:
                service.end()
                result = {"ok": False, "error": str(error)}
            print(json.dumps(result), flush=True)
    finally:
        service.end()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
