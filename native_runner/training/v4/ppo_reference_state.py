"""Frozen reference recurrence across resident PPO segments."""

import torch

from .model import UniversalCardPolicyV4
from .ppo import StoredRolloutV4
from .tensors import RecurrentPolicyStateV4


class ReferenceStateCacheV4:
    def __init__(self) -> None:
        self.states: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def clear(self) -> None:
        self.states.clear()

    def prepare(self, model: UniversalCardPolicyV4, rollout: StoredRolloutV4) -> RecurrentPolicyStateV4:
        rollout.validate()
        if rollout.lane_keys is None:
            raise ValueError("segmented reference recurrence requires stable rollout lane keys")
        state = model.initial_state(rollout.batch_size, device=torch.device("cpu"))
        valid = rollout.valid_mask.any(dim=0).cpu()
        starts = rollout.episode_start[0].cpu()
        for lane, key in enumerate(rollout.lane_keys):
            if not valid[lane] or starts[lane]:
                continue
            if key not in self.states:
                raise ValueError(f"continuing reference lane {key} has no prior reference state")
            hidden, cell = self.states[key]
            state.hidden[lane].copy_(hidden)
            state.cell[lane].copy_(cell)
        return state

    def commit(self, rollout: StoredRolloutV4, final_state: RecurrentPolicyStateV4) -> None:
        if rollout.lane_keys is None or final_state.hidden.shape[0] != rollout.batch_size:
            raise ValueError("reference final state is not aligned with rollout lanes")
        hidden = final_state.hidden.detach().float().cpu()
        cell = final_state.cell.detach().float().cpu()
        # Resident collectors mark a 40-second segment boundary as truncated
        # for value bootstrapping, while the same game continues next segment.
        # Keep its reference memory. Real resets are identified by episode_start
        # in prepare(); unused truncated lanes are bounded by resident actor keys.
        ended = (rollout.terminated & rollout.valid_mask).any(dim=0).cpu()
        valid = rollout.valid_mask.any(dim=0).cpu()
        for lane, key in enumerate(rollout.lane_keys):
            if ended[lane] or not valid[lane]:
                self.states.pop(key, None)
            else:
                self.states[key] = (hidden[lane].clone(), cell[lane].clone())
