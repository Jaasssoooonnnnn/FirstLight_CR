"""A separate, globally normalized recurrent BC step after each PPO update.

Offline demonstrations are NEVER inserted into the on-policy PPO buffer. Each
expert starts from its real episode beginning. Unsupervised prefix/gap frames
are replayed without gradient to reconstruct hidden state, while supervised
windows use ordinary state-carrying TBPTT. No expert returns train the critic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist

from .cache import load_il_cache_shard
from .imitation import ILSequenceV4, ILLossWeightsV4, imitation_loss, slice_il_sequence
from .learning import evaluate_recurrent_sequence
from .tensors import GATE_ACT

HEADS = ("gate", "candidate", "target", "delay", "continue_action")
COUNT_NAMES = ("gate_count", "candidate_count", "target_count", "delay_count", "continue_count")
HOG_ID = 26000021


def _world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _sum(tensor):
    if _world()[1] > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def selected_window_mask(valid, hog_rows, *, window_steps, rng):
    """Opening plus one later attack/cycle context; never re-enable invalid rows."""
    if window_steps <= 0 or valid.ndim != 2 or valid.shape[1] != 1:
        raise ValueError("expert mask requires [T,1] and a positive window")
    trusted = torch.nonzero(valid[:, 0], as_tuple=False).flatten()
    if not len(trusted):
        raise ValueError("expert has no trusted frames")
    end = int(trusted[-1]) + 1
    selected = torch.zeros_like(valid)
    selected[: min(end, window_steps)] = True
    if end > window_steps:
        later = [r for r in hog_rows if window_steps <= r < end and bool(valid[r, 0])]
        # The window contains normal before/after context, not an isolated Hog label.
        center = rng.choice(later) if later else rng.randrange(window_steps, end)
        start = max(0, min(end - window_steps, center - window_steps // 2))
        selected[start : start + window_steps] = True
    return selected & valid


def hog_action_rows(sequence):
    rows = []
    for t, (obs, act) in enumerate(zip(sequence.observations, sequence.actions, strict=True)):
        if not bool(sequence.valid_mask[t, 0]):
            continue
        for j in range(int(act.micro_action_count[0])):
            index = int(act.candidate_index[0, j])
            if int(obs.candidates.native_visible_card_id[0, index]) == HOG_ID:
                rows.append(t)
    return rows


def cached_sequence_quality(shard, index):
    """Audit actual post-migration labels, not just source replay metadata."""
    descriptor = shard.descriptors[index]
    start, stop = descriptor.offset, descriptor.offset + descriptor.length
    valid = shard.valid_mask[start:stop, 0]
    actions = shard.actions
    counts = actions.micro_action_count[start:stop]
    indices = actions.candidate_index[start:stop].clamp_min(0)
    ids = shard.observations.candidates.native_visible_card_id[start:stop].gather(1, indices)
    micro = torch.arange(indices.shape[1])[None] < counts[:, None]
    hog_mask = micro & (ids == HOG_ID) & valid[:, None]
    hog_frames = torch.nonzero(hog_mask.any(dim=-1), as_tuple=False).flatten().tolist()
    gates = actions.gate[start:stop]
    return dict(
        cached_valid_frames=int(valid.sum()),
        cached_gate_frames=int((shard.gate_loss_mask[start:stop, 0] & valid).sum()),
        cached_action_frames=int(((gates == GATE_ACT) & valid).sum()),
        cached_micro_actions=int((micro & valid[:, None]).sum()),
        cached_hog_count=int(hog_mask.sum()),
        cached_hog_frames=hog_frames,
        cached_episode_start=bool(shard.episode_start[start, 0]),
        cached_natural_time=bool(descriptor.natural_time),
    )


def head_counts(sequence):
    valid = sequence.valid_mask
    gate_mask = valid if sequence.gate_loss_mask is None else sequence.gate_loss_mask
    gates = torch.stack([a.gate for a in sequence.actions])
    counts = torch.stack([a.micro_action_count for a in sequence.actions])
    targets = torch.stack([a.target_cell for a in sequence.actions])
    micro = (torch.arange(targets.shape[-1], device=valid.device)[None, None] < counts[..., None]) & valid[..., None]
    return torch.stack(
        (gate_mask.sum(), micro.sum(), (micro & (targets >= 0)).sum(), micro.sum(), ((gates == GATE_ACT) & valid).sum())
    ).double()


def hog_gate_mask(sequence, *, scope="all", metadata=None, frame_offset=0):
    """Original trusted, gate-supervised ACT frames containing an expert Hog.

    This never promotes WAIT labels or unmasks a failed/committed frame. A
    multi-action decision contributes one gate target, even with several Hogs.
    """
    if scope not in ("all", "proactive-first") or frame_offset < 0:
        raise ValueError("invalid Hog gate scope or chunk offset")
    contains_hog = []
    gates = []
    for obs, action in zip(sequence.observations, sequence.actions, strict=True):
        ids = obs.candidates.native_visible_card_id
        if ids.shape[1] == 0:
            contains_hog.append(torch.zeros_like(action.gate, dtype=torch.bool))
        else:
            chosen = ids.gather(1, action.candidate_index.clamp_min(0))
            micro = torch.arange(chosen.shape[1], device=chosen.device)[None] < action.micro_action_count[:, None]
            contains_hog.append(((chosen == HOG_ID) & micro).any(dim=-1))
        gates.append(action.gate)
    mask = sequence.valid_mask & (torch.stack(gates) == GATE_ACT) & torch.stack(contains_hog)
    if sequence.gate_loss_mask is not None:
        mask &= sequence.gate_loss_mask
    if scope == "proactive-first":
        if metadata is None or "proactive_first_hog_frame" not in metadata:
            raise ValueError("proactive gate scope requires source-verified annotations")
        if mask.shape[1] != 1:
            raise ValueError("proactive expert metadata describes one owner sequence")
        frame = metadata["proactive_first_hog_frame"]
        if frame is None:
            return torch.zeros_like(mask)
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("invalid proactive first-Hog frame")
        positions = torch.arange(mask.shape[0], device=mask.device) + frame_offset
        mask &= (positions == frame)[:, None]
    return mask


def hog_gate_nll_sum(sequence, evaluation, *, scope="all", metadata=None, frame_offset=0):
    """Unnormalized supervised NLL; the caller uses the global Hog population."""
    mask = hog_gate_mask(sequence, scope=scope, metadata=metadata, frame_offset=frame_offset)
    nll = -evaluation.components.gate_log_prob.float()
    if nll.shape != mask.shape:
        raise ValueError("Hog gate evaluation shape mismatch")
    return torch.where(mask, nll, torch.zeros_like(nll)).sum(), mask.sum()


def expert_bc_contract_v4(
    sampler, *, coefficient=0.2, hog_gate_weight=0.0, hog_gate_scope="all", hog_gate_gradient_scope="full"
):
    if not math.isfinite(hog_gate_weight) or hog_gate_weight < 0:
        raise ValueError("expert Hog gate weight must be finite and nonnegative")
    if hog_gate_scope not in ("all", "proactive-first"):
        raise ValueError("unknown Hog gate supervision scope")
    if hog_gate_gradient_scope not in ("full", "gate-head-only"):
        raise ValueError("unknown Hog gate auxiliary gradient scope")
    contract = {**sampler.contract(), "coefficient": coefficient}
    # Preserve strict-resume compatibility for the unchanged, zero-weight path.
    if hog_gate_weight and hog_gate_scope == "all":
        contract.update(
            schema="v4-ppo-expert-bc.v3",
            hog_gate_supervision=dict(
                weight=hog_gate_weight,
                temperature=1.0,
                population="trusted gate-eligible ACT frames containing expert Hog",
                normalization="global eligible Hog decision count",
            ),
        )
    elif hog_gate_weight:
        annotation = getattr(sampler, "proactive_opening_contract", None)
        if not annotation or annotation.get("schema") != "v4-hog-proactive-opening-gate.v1":
            raise ValueError("proactive gate supervision needs an annotated expert manifest")
        contract.update(
            schema="v4-ppo-expert-bc.v4",
            hog_gate_supervision=dict(
                weight=hog_gate_weight,
                temperature=1.0,
                scope=hog_gate_scope,
                population="trusted gate-eligible source-proactive first Hog decisions <=30s",
                normalization="global eligible proactive first-Hog decision count",
                empty_population="skip auxiliary only; normal five-head BC remains active",
                annotation=annotation,
            ),
        )
    if hog_gate_weight and hog_gate_gradient_scope == "gate-head-only":
        contract["schema"] = "v4-ppo-expert-bc.v5"
        contract["hog_gate_supervision"].update(
            gradient_scope="gate-head-only",
            gradient_parameters="gate_head.*",
            normal_bc_gradient_scope="full",
            clipping="unchanged shared global norm; may rescale all normal BC gradients",
        )
    return contract


@torch.no_grad()
def expert_hog_probability_sums(sequence, evaluation, *, rollout_gate_temperature):
    """Read-only first-micro Hog probabilities on teacher-forced expert states.

    These are NOT win-rate or on-policy KL measurements. Separate the gate
    from conditional card choice, and show the actual rollout temperature.
    """
    if rollout_gate_temperature <= 0:
        raise ValueError("rollout gate temperature must be positive")
    first_ids = torch.stack(
        [
            (
                obs.candidates.native_visible_card_id.gather(1, act.candidate_index[:, :1].clamp_min(0))[:, 0]
                if obs.candidates.native_visible_card_id.shape[1]
                else torch.full_like(act.gate, -1)
            )
            for obs, act in zip(sequence.observations, sequence.actions, strict=True)
        ]
    )
    counts = torch.stack([a.micro_action_count for a in sequence.actions])
    mask = sequence.valid_mask & (counts > 0) & (first_ids == HOG_ID)
    gate_logp = evaluation.components.gate_log_prob.detach().double().clamp_max(0)[mask]
    candidate_prob = evaluation.components.candidate_log_prob.detach().double()[..., 0][mask].exp()
    gate_prob = gate_logp.exp()
    # Stable binary-logit conversion, including probabilities near zero/one.
    logit = gate_logp - torch.log(-torch.expm1(gate_logp))
    rollout_prob = torch.sigmoid(logit / rollout_gate_temperature)
    return torch.stack(
        (
            mask.sum().double(),
            gate_prob.sum(),
            rollout_prob.sum(),
            candidate_prob.sum(),
            (rollout_prob * candidate_prob).sum(),
            (gate_prob > 0.5).double().sum(),
        )
    )


@dataclass
class ExpertBatchItemV4:
    sequence: ILSequenceV4
    metadata: dict
    supervised_hogs: int
    burnin_frames: int


class HogExpertSamplerV4:
    """Stateless per-update sampling: strict resume does not need hidden RNG state."""

    def __init__(self, manifest_path, *, seed, records_per_rank=2, window_steps=128):
        self.path = Path(manifest_path).resolve()
        raw = self.path.read_bytes()
        payload = json.loads(raw)
        if payload.get("schema") != "v4-hog-expert-selection.v1" or payload.get("discard_tail_ticks") != 200:
            raise ValueError("unsupported or unsafe expert manifest")
        if payload.get("cache_labels_verified") is not True:
            raise ValueError("BC requires a manifest audited against actual cached labels")
        self.rows = payload["rows"]
        if not self.rows or min(records_per_rank, window_steps) <= 0:
            raise ValueError("BC needs records and positive dimensions")
        keys = [(r["replay_tag"], r["owner"]) for r in self.rows]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate expert owner sequences")
        if any(
            r["cache_status"] not in ("completed", "trusted_prefix")
            or r["outcome"] not in ("win", "close_loss")
            or not math.isfinite(r["weight"])
            or r["weight"] <= 0
            for r in self.rows
        ):
            raise ValueError("invalid expert quality selection")
        if any(
            r.get("cached_hog_count", 0) <= 0 or not r.get("cached_episode_start") or not r.get("cached_natural_time")
            for r in self.rows
        ):
            raise ValueError("expert lacks audited Hog labels or real recurrent start")
        self.proactive_opening_contract = payload.get("proactive_opening_gate")
        if self.proactive_opening_contract is not None:
            annotation = self.proactive_opening_contract
            if (
                annotation.get("schema") != "v4-hog-proactive-opening-gate.v1"
                or annotation.get("primary_cache_start_tick") != 130
                or annotation.get("decision_stride_ticks") != 5
            ):
                raise ValueError("unsupported proactive opening annotation")
            annotated_count = 0
            for row in self.rows:
                if "proactive_first_hog_frame" not in row or "proactive_first_enemy_tick" not in row:
                    raise ValueError("incomplete proactive opening annotation")
                frame, enemy_tick = row["proactive_first_hog_frame"], row["proactive_first_enemy_tick"]
                if frame is None:
                    if enemy_tick is not None:
                        raise ValueError("noneligible proactive row has a partial annotation")
                    continue
                tick = row["source_hog_ticks"][0]
                if (
                    isinstance(frame, bool)
                    or not isinstance(frame, int)
                    or frame < 0
                    or frame != row["cached_hog_frames"][0]
                    or frame >= row["sequence_length"]
                    or not 130 + 5 * frame < tick <= min(600, 135 + 5 * frame)
                    or (
                        enemy_tick is not None
                        and (isinstance(enemy_tick, bool) or not isinstance(enemy_tick, int) or enemy_tick <= tick)
                    )
                ):
                    raise ValueError("proactive first-Hog annotation contradicts cached/source clocks")
                annotated_count += 1
            if annotated_count != annotation.get("eligible_sides") or not annotated_count:
                raise ValueError("proactive opening annotation population mismatch")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.seed, self.records_per_rank, self.window_steps = int(seed), int(records_per_rank), int(window_steps)
        self._shard_path = None
        self._shard = None

    def contract(self):
        return dict(
            schema="v4-ppo-expert-bc.v2",
            manifest=str(self.path),
            manifest_sha256=self.sha256,
            seed=self.seed,
            records_per_rank=self.records_per_rank,
            window_steps=self.window_steps,
            sampling="stateless-seed-update-rank; opening+later-Hog-context",
            state="full-prefix-burnin; detach-at-PPO-chunk-length",
            supervision_temperature=1.0,
            value_loss=False,
            gate_act_weight=8.0,
            delay_neighbor_weight=0.2,
        )

    def rows_for_update(self, update, rank, world_size):
        if update < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid update/rank")
        # All ranks draw the same global list, then take disjoint positions.
        # This does not perturb PPO/collector RNG state and resumes exactly.
        rng = random.Random(f"hog-bc-v1:{self.seed}:{update}")
        selected = rng.choices(
            self.rows, weights=[r["weight"] for r in self.rows], k=world_size * self.records_per_rank
        )
        return selected[rank::world_size]

    def sample(self, update, rank, world_size):
        items = []
        for i, row in enumerate(self.rows_for_update(update, rank, world_size)):
            if self._shard_path != row["summary_path"]:
                self._shard = None
                self._shard = load_il_cache_shard(row["summary_path"], verify=True)
                self._shard_path = row["summary_path"]
            shard = self._shard
            if shard.summary["compressed_sha256"] != row["cache_compressed_sha256"]:
                raise ValueError("expert cache bytes changed after dataset audit")
            index = int(row["sequence_index"])
            descriptor = shard.descriptors[index]
            if (descriptor.replay_tag, descriptor.owner, descriptor.length) != (
                row["replay_tag"],
                row["owner"],
                row["sequence_length"],
            ):
                raise ValueError("expert owner/cache identity changed")
            quality = cached_sequence_quality(shard, index)
            if any(quality.get(k) != row.get(k) for k in quality):
                raise ValueError("cached expert labels changed after dataset audit")
            sequence = shard.sequence(index)
            if not bool(sequence.episode_start[0, 0]) or not sequence.natural_time:
                raise ValueError("expert must start at its real recurrent episode boundary")
            hog_rows = hog_action_rows(sequence)
            if not hog_rows:
                raise ValueError(f"selected expert has no valid cached Hog labels: {row['replay_tag']}")
            rng = random.Random(f"hog-bc-window:{self.seed}:{update}:{rank}:{i}")
            selected = selected_window_mask(sequence.valid_mask, hog_rows, window_steps=self.window_steps, rng=rng)
            last = int(torch.nonzero(selected[:, 0], as_tuple=False)[-1]) + 1
            # Clip only after the last selected window. All earlier context is
            # preserved, including unsupervised waiting/defense frames.
            sequence = slice_il_sequence(sequence, 0, last)
            selected = selected[:last]
            gate = sequence.valid_mask if sequence.gate_loss_mask is None else sequence.gate_loss_mask
            sequence = replace(
                sequence,
                valid_mask=selected,
                gate_loss_mask=gate & selected,
                value_loss_mask=torch.zeros_like(selected),
            )
            sequence.validate()
            items.append(
                ExpertBatchItemV4(
                    sequence, row, sum(bool(selected[t, 0]) for t in hog_rows if t < last), last - int(selected.sum())
                )
            )
        return items


def _reduce_actor_gradients(parameters):
    """Average gradients, preserving None on globally unused critic parameters.

    Assigning zero grads to the critic would still let its Adam momentum move
    it during a nominally policy-only BC step.
    """
    params = [p for p in parameters if p.requires_grad]
    device = params[0].device
    active = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=device)
    if _world()[1] > 1:
        dist.all_reduce(active, op=dist.ReduceOp.MAX)
    selected = [p for p, flag in zip(params, active.tolist(), strict=True) if flag]
    if not selected:
        raise RuntimeError("BC produced no gradients")
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in selected])
    _sum(flat).div_(_world()[1])
    offset = 0
    for p in selected:
        p.grad = flat[offset : offset + p.numel()].view_as(p).clone()
        offset += p.numel()


@torch.no_grad()
def _probe_log_probs(model, rollout, *, device, validate):
    """Small labelled probe, NOT an assertion of full-rollout post-update KL."""
    steps, lanes = min(32, rollout.time_steps), min(4, rollout.batch_size)
    observations = tuple(o.narrow_batch(0, lanes).to_model_input(device) for o in rollout.observations[:steps])
    actions = tuple(a.narrow_batch(0, lanes).to(device) for a in rollout.actions[:steps])
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        ev = evaluate_recurrent_sequence(
            model,
            observations,
            actions,
            rollout.episode_start[:steps, :lanes].to(device),
            initial_state=rollout.initial_state.narrow_batch(0, lanes).to(device),
            gate_temperature=rollout.gate_temperature,
            action_temperature=rollout.action_temperature,
            continue_temperature=rollout.continue_temperature,
            validate=validate,
            preencode_observations=True,
        )
    valid = rollout.valid_mask[:steps, :lanes].to(device).clone()
    if rollout.forced_gate_mask is not None:
        valid &= ~rollout.forced_gate_mask[:steps, :lanes].to(device)
    return ev.log_prob.float(), valid


def distributed_expert_bc_update_v4(
    model,
    optimizer,
    sampler,
    *,
    update_step,
    sequence_chunk_steps,
    coefficient=0.2,
    max_grad_norm=0.5,
    rollout=None,
    validate=False,
    hog_gate_weight=0.0,
    hog_gate_scope="all",
    hog_gate_gradient_scope="full",
):
    if not math.isfinite(coefficient) or coefficient <= 0 or sequence_chunk_steps <= 0:
        raise ValueError("BC coefficient and chunk length must be positive")
    if not math.isfinite(hog_gate_weight) or hog_gate_weight < 0:
        raise ValueError("expert Hog gate weight must be finite and nonnegative")
    if hog_gate_scope not in ("all", "proactive-first"):
        raise ValueError("unknown Hog gate supervision scope")
    if hog_gate_gradient_scope not in ("full", "gate-head-only"):
        raise ValueError("unknown Hog gate auxiliary gradient scope")
    gate_parameters = ()
    if hog_gate_weight and hog_gate_gradient_scope == "gate-head-only":
        gate_head = getattr(model, "gate_head", None)
        gate_parameters = () if gate_head is None else tuple(p for p in gate_head.parameters() if p.requires_grad)
        if not gate_parameters:
            raise ValueError("head-only auxiliary requires trainable gate_head parameters")
    started = time.perf_counter()
    rank, world_size = _world()
    device = next(model.parameters()).device
    model.eval()
    before = None if rollout is None else _probe_log_probs(model, rollout, device=device, validate=validate)
    items = sampler.sample(update_step, rank, world_size)
    totals = _sum(sum((head_counts(x.sequence) for x in items)).to(device))
    if bool(torch.any(totals <= 0)):
        raise RuntimeError("expert batch lacks an action-head training population")
    hog_gate_count = None
    if hog_gate_weight:
        hog_gate_count = _sum(
            sum(hog_gate_mask(x.sequence, scope=hog_gate_scope, metadata=x.metadata).sum() for x in items)
            .to(device)
            .double()
        )
        if hog_gate_scope == "all" and not bool(hog_gate_count > 0):
            raise RuntimeError("expert batch lacks trusted gate-supervised Hog decisions")
    optimizer.zero_grad(set_to_none=True)
    numerators = torch.zeros(5, device=device, dtype=torch.float64)
    hog_prob_sums = torch.zeros(6, device=device, dtype=torch.float64)
    hog_gate_numerator = torch.zeros((), device=device, dtype=torch.float64)
    for item in items:
        sequence = item.sequence
        state = None
        for start in range(0, sequence.time_steps, sequence_chunk_steps):
            chunk = slice_il_sequence(sequence, start, min(start + sequence_chunk_steps, sequence.time_steps)).to(
                device
            )
            supervised = bool(chunk.valid_mask.any())
            with (
                torch.set_grad_enabled(supervised),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
            ):
                evaluation = evaluate_recurrent_sequence(
                    model,
                    chunk.observations,
                    chunk.actions,
                    chunk.episode_start,
                    initial_state=state,
                    gate_temperature=1.0,
                    action_temperature=1.0,
                    continue_temperature=1.0,
                    validate=validate,
                    preencode_observations=True,
                )
                state = evaluation.final_state.detach()
                if supervised:
                    loss = imitation_loss(
                        chunk,
                        evaluation,
                        weights=ILLossWeightsV4(value=0.0),
                        gate_act_weight=8.0,
                        delay_neighbor_weight=0.2,
                    )
                    means = torch.stack([getattr(loss, k) for k in HEADS]).float()
                    counts = torch.stack([getattr(loss, k) for k in COUNT_NAMES]).float()
                    # _reduce_actor_gradients averages ranks, so multiply by world.
                    # Every head is normalized by its GLOBAL valid population.
                    objective = (means * counts / totals.clamp_min(1).float()).sum() * (coefficient * world_size)
                    auxiliary_gradients = None
                    if hog_gate_weight:
                        hog_nll, local_hog_count = hog_gate_nll_sum(
                            chunk, evaluation, scope=hog_gate_scope, metadata=item.metadata, frame_offset=start
                        )
                        auxiliary = (
                            hog_nll / hog_gate_count.clamp_min(1).float() * (coefficient * hog_gate_weight * world_size)
                        )
                        if not bool(torch.isfinite(auxiliary)):
                            raise FloatingPointError("nonfinite Hog gate auxiliary loss")
                        if hog_gate_gradient_scope == "full":
                            objective = objective + auxiliary
                        elif bool(local_hog_count > 0):
                            # Ask only for gate parameter derivatives. Normal five-head
                            # BC below still backpropagates through the whole model.
                            auxiliary_gradients = torch.autograd.grad(
                                auxiliary, gate_parameters, retain_graph=True, allow_unused=True
                            )
                        hog_gate_numerator += hog_nll.detach().double()
                    if not bool(torch.isfinite(objective)):
                        raise FloatingPointError("nonfinite expert BC loss")
                    objective.backward()
                    if auxiliary_gradients is not None:
                        for parameter, extra in zip(gate_parameters, auxiliary_gradients, strict=True):
                            if extra is not None:
                                if parameter.grad is None:
                                    parameter.grad = extra.detach().clone()
                                else:
                                    parameter.grad.add_(extra)
                    del auxiliary_gradients
                    numerators += means.detach().double() * counts.double()
                    hog_prob_sums += expert_hog_probability_sums(
                        chunk, evaluation, rollout_gate_temperature=model.ppo_gate_temperature
                    )
                    del loss, objective, means, counts
                    if hog_gate_weight:
                        del hog_nll, auxiliary
            del chunk, evaluation
    _reduce_actor_gradients(tuple(model.parameters()))
    # This clipping intentionally stays identical in both gradient scopes. The
    # gate-only auxiliary adds no direct backbone gradient, but the shared clip
    # factor can still change the magnitude of other normal-BC updates.
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    losses = _sum(numerators) / totals.clamp_min(1)
    metrics = {f"bc_{name}_loss": float(value) for name, value in zip(HEADS, losses.tolist(), strict=True)}
    metrics.update({f"bc_{name}": float(value) for name, value in zip(COUNT_NAMES, totals.tolist(), strict=True)})
    stats = _sum(
        torch.tensor(
            [
                len(items),
                sum(x.supervised_hogs for x in items),
                sum(int(x.sequence.valid_mask.sum()) for x in items),
                sum(x.burnin_frames for x in items),
                sum(x.metadata["outcome"] == "win" for x in items),
            ],
            device=device,
            dtype=torch.float64,
        )
    )
    metrics.update(
        bc_loss=float(losses.sum()),
        bc_coefficient=coefficient,
        bc_gradient_norm=float(grad_norm),
        bc_optimizer_steps=1,
        bc_expert_records=float(stats[0]),
        bc_hog_labels=float(stats[1]),
        bc_supervised_frames=float(stats[2]),
        bc_burnin_frames=float(stats[3]),
        bc_win_records=float(stats[4]),
    )
    if hog_gate_weight:
        hog_gate_loss = _sum(hog_gate_numerator) / hog_gate_count.clamp_min(1)
        metrics.update(
            bc_hog_gate_weight=hog_gate_weight,
            bc_hog_gate_count=float(hog_gate_count),
            bc_hog_gate_loss=float(hog_gate_loss),
            bc_loss=float(losses.sum() + hog_gate_weight * hog_gate_loss),
        )
        if hog_gate_scope == "proactive-first":
            metrics["bc_proactive_hog_gate_empty_update"] = float(hog_gate_count == 0)
        if hog_gate_gradient_scope == "gate-head-only":
            metrics["bc_hog_gate_head_only"] = 1.0
    hog_probs = _sum(hog_prob_sums)
    metrics["bc_hog_first_micro_labels"] = float(hog_probs[0])
    for name, value in zip(
        (
            "gate_prob_T1",
            "gate_prob_rollout_T",
            "conditional_card_prob",
            "joint_first_micro_prob",
            "gate_argmax_act_fraction",
        ),
        hog_probs[1:],
        strict=True,
    ):
        metrics[f"bc_expert_hog_{name}"] = float(value / hog_probs[0].clamp_min(1))
    if before is not None:
        after, valid = _probe_log_probs(model, rollout, device=device, validate=validate)
        delta = (after - before[0])[valid].double()
        probe = _sum(
            torch.stack(
                (
                    (delta.exp() - 1 - delta).sum(),
                    delta.abs().sum(),
                    torch.tensor(delta.numel(), device=device, dtype=torch.float64),
                )
            )
        )
        metrics.update(
            bc_step_probe_kl=float(probe[0] / probe[2].clamp_min(1)),
            bc_step_probe_abs_logprob_change=float(probe[1] / probe[2].clamp_min(1)),
            bc_step_probe_frames=float(probe[2]),
        )
    elapsed = torch.tensor(time.perf_counter() - started, device=device)
    if world_size > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    metrics["bc_seconds"] = float(elapsed)
    return metrics
