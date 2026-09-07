"""Universal-card relation/spatial recurrent actor-critic, version 4."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .catalog import AbilityCatalogV1, CardCatalogV1, EntityArchetypeCatalogV1
from .components import (
    ActiveEffectEncoder,
    BattleGroupEncoder,
    CandidateEncoder,
    CardTokenEncoder,
    EventSetEncoder,
    EffectSemanticEncoder,
    GenericScalarEncoder,
    GlobalCardEncoder,
    LocalChildSetEncoder,
    MechanicProfileEncoder,
    PreviousActionEncoder,
    RelationTransformerV4,
    SpatialScatterEncoder,
    TowerEncoder,
    assemble_global_tokens,
    gather_memory,
    masked_logits,
    mlp,
)
from .mechanics import EffectSemanticCatalogV1, MechanicProfileCatalogV1
from .config import ModelConfigV4, UNIVERSAL_ACTION_VERSION, UNIVERSAL_OBSERVATION_VERSION, UNIVERSAL_POLICY_VERSION
from .decoding import ShadowCandidateLegality
from .tensors import (
    ActionEvaluationComponentsV4,
    ActionSequenceV4,
    EncodedObservationV4,
    GATE_ACT,
    PolicyContextV4,
    PolicyOutputV4,
    RecurrentPolicyStateV4,
    TARGET_GRID,
    UniversalSemanticBatchV4,
)


POLICY_VERSION_V4 = UNIVERSAL_POLICY_VERSION
OBSERVATION_VERSION_V4 = UNIVERSAL_OBSERVATION_VERSION
ACTION_VERSION_V4 = UNIVERSAL_ACTION_VERSION


def _safe_categorical_mask(mask: Tensor) -> Tensor:
    """Add a never-observed fallback to otherwise empty categorical rows."""

    if mask.shape[-1] <= 0:
        raise ValueError("categorical vocabulary cannot be empty")
    empty = ~mask.any(dim=-1, keepdim=True)
    return torch.cat((mask[..., :1] | empty, mask[..., 1:]), dim=-1)


def _distribution(logits: Tensor, *, mask: Tensor | None, temperature: float, validate: bool) -> Categorical:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("categorical temperature must be finite and positive")
    scaled = logits / float(temperature)
    if mask is not None:
        scaled = masked_logits(scaled, _safe_categorical_mask(mask), validate=validate)
    return Categorical(logits=scaled, validate_args=validate)


def _reduce_log_prob_components_v4(components: ActionEvaluationComponentsV4) -> Tensor:
    return torch.stack(
        (
            components.gate_log_prob,
            components.candidate_log_prob.sum(dim=-1),
            components.target_log_prob.sum(dim=-1),
            components.delay_offset_log_prob.sum(dim=-1),
            components.continue_log_prob,
        ),
        dim=-1,
    )


class UniversalCardPolicyV4(nn.Module):
    """Global-card policy with one LSTM and candidate-pointer actions."""

    def __init__(
        self,
        catalog: CardCatalogV1,
        *,
        ability_catalog: AbilityCatalogV1,
        entity_archetype_catalog: EntityArchetypeCatalogV1,
        effect_catalog: EffectSemanticCatalogV1,
        mechanic_profile_catalog: MechanicProfileCatalogV1,
        config: ModelConfigV4 | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfigV4()
        self.catalog = catalog
        self.ability_catalog = ability_catalog
        self.entity_archetype_catalog = entity_archetype_catalog
        self.effect_catalog = effect_catalog
        self.mechanic_profile_catalog = mechanic_profile_catalog
        config = self.config
        if ability_catalog.card_scope != catalog.raw_card_ids:
            raise ValueError("AbilityCatalog and CardCatalog scopes differ")
        if entity_archetype_catalog.card_scope != catalog.raw_card_ids:
            raise ValueError("EntityArchetypeCatalog and CardCatalog scopes differ")
        if effect_catalog.card_scope != catalog.raw_card_ids:
            raise ValueError("EffectSemanticCatalog and CardCatalog scopes differ")
        if mechanic_profile_catalog.card_scope != catalog.raw_card_ids:
            raise ValueError("MechanicProfileCatalog and CardCatalog scopes differ")
        if len(mechanic_profile_catalog.card_form_profile_ids) != catalog.vocab_size:
            raise ValueError("MechanicProfileCatalog card mapping has the wrong size")
        if len(mechanic_profile_catalog.ability_profile_ids) != ability_catalog.vocab_size:
            raise ValueError("MechanicProfileCatalog Ability mapping has the wrong size")
        if len(mechanic_profile_catalog.archetype_profile_ids) != entity_archetype_catalog.vocab_size:
            raise ValueError("MechanicProfileCatalog archetype mapping has the wrong size")
        if len(mechanic_profile_catalog.tower_troop_profile_ids) != config.tower_troop_type_count:
            raise ValueError("MechanicProfileCatalog Tower Troop mapping is invalid")
        if any(
            operation.effect_vocab_id >= effect_catalog.vocab_size
            or operation.produced_archetype_id >= entity_archetype_catalog.vocab_size
            for operations in mechanic_profile_catalog.profile_operations
            for operation in operations
        ):
            raise ValueError("MechanicProfileCatalog references an out-of-scope identity")
        if ability_catalog.vocab_size > config.ability_vocab_size:
            raise ValueError("AbilityCatalog exceeds ModelConfigV4 capacity")
        if entity_archetype_catalog.vocab_size > config.child_archetype_count:
            raise ValueError("EntityArchetypeCatalog exceeds ModelConfigV4 capacity")

        self.effect_encoder = EffectSemanticEncoder(config, effect_catalog)
        self.mechanic_profile_encoder = MechanicProfileEncoder(config, mechanic_profile_catalog)
        self.active_effect_encoder = ActiveEffectEncoder(config)
        self.card_encoder = GlobalCardEncoder(config, catalog, mechanic_profile_catalog)
        self.child_encoder = LocalChildSetEncoder(config, self.card_encoder, mechanic_profile_catalog)
        self.group_encoder = BattleGroupEncoder(config, self.card_encoder)
        self.tower_encoder = TowerEncoder(config, mechanic_profile_catalog)
        self.card_token_encoder = CardTokenEncoder(config, self.card_encoder)
        self.relation_transformer = RelationTransformerV4(config)
        self.spatial_encoder = SpatialScatterEncoder(config)
        self.scalar_encoder = GenericScalarEncoder(config)
        self.event_encoder = EventSetEncoder(config, self.card_encoder)
        self.previous_action_encoder = PreviousActionEncoder(config, self.card_encoder, mechanic_profile_catalog)
        self.candidate_encoder = CandidateEncoder(
            config,
            self.card_encoder,
            ability_features=ability_catalog.feature_tensor(config),
            mechanic_catalog=mechanic_profile_catalog,
        )

        core_input_width = (
            config.token_dim
            + config.spatial_summary_dim
            + config.scalar_summary_dim
            + config.event_summary_dim
            + config.previous_action_dim
            + config.candidate_summary_dim
        )
        if core_input_width != 896:
            raise ValueError("V4 core concatenation must have width 896")
        self.input_projection = mlp(core_input_width, config.lstm_input_dim)
        self.lstm_core = nn.LSTMCell(config.lstm_input_dim, config.lstm_hidden_dim)
        self.core_output_norm = nn.LayerNorm(config.lstm_hidden_dim)

        self.scene_policy_skip = nn.Linear(config.token_dim, config.lstm_hidden_dim)
        self.spatial_policy_skip = nn.Linear(config.spatial_summary_dim, config.lstm_hidden_dim)
        self.candidate_policy_skip = nn.Linear(config.candidate_summary_dim, config.lstm_hidden_dim)
        self.policy_context_norm = nn.LayerNorm(config.lstm_hidden_dim)

        self.scene_value_skip = nn.Linear(config.token_dim, config.lstm_hidden_dim)
        self.spatial_value_skip = nn.Linear(config.spatial_summary_dim, config.lstm_hidden_dim)
        self.scalar_value_skip = nn.Linear(config.scalar_summary_dim, config.lstm_hidden_dim)
        self.event_value_skip = nn.Linear(config.event_summary_dim, config.lstm_hidden_dim)
        self.value_context_norm = nn.LayerNorm(config.lstm_hidden_dim)
        self.value_head = mlp(config.lstm_hidden_dim, 256, 1)

        self.gate_head = mlp(config.lstm_hidden_dim, 256, 2)
        self.decoder_initial = nn.Linear(config.lstm_hidden_dim, config.decoder_hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.decoder_hidden_dim,
            num_heads=config.attention_heads,
            kdim=config.token_dim,
            vdim=config.token_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_dim)
        self.candidate_query = nn.Linear(config.decoder_hidden_dim, config.candidate_dim)
        self.candidate_key = nn.Linear(config.candidate_dim, config.candidate_dim)

        location_condition_width = config.decoder_hidden_dim + config.candidate_dim
        self.location_condition = mlp(location_condition_width, 256, 128)
        self.location_film = nn.Linear(128, 2 * config.spatial_hidden_channels)
        self.location_key = nn.Conv2d(config.spatial_hidden_channels, 128, kernel_size=1)
        self.location_query = mlp(location_condition_width, 256, 128)
        self.no_target_embedding = nn.Parameter(torch.empty(128))
        nn.init.normal_(self.no_target_embedding, std=0.02)

        self.delay_offset_head = mlp(
            config.decoder_hidden_dim + config.candidate_dim + 128, 256, len(config.delay_offset_ms)
        )
        self.delay_offset_embedding = nn.Embedding(len(config.delay_offset_ms), 32)
        self.action_embedding = mlp(config.candidate_dim + 128 + 32, 256, 128)
        autoregressive_width = 128 + config.shadow_feature_dim + config.candidate_summary_dim
        self.decoder_step = mlp(autoregressive_width, config.decoder_hidden_dim, config.decoder_hidden_dim)
        self.continue_head = mlp(config.decoder_hidden_dim + autoregressive_width, 256, 2)

        self._ppo_gate_temperature = 1.0
        self._ppo_action_temperature = 1.0
        self._ppo_continue_temperature = 1.0
        self._inference_delay_offset_max_ms: int | None = None
        self.register_buffer(
            "_inference_delay_offset_mask", torch.ones(len(config.delay_offset_ms), dtype=torch.bool), persistent=False
        )
        self._ppo_rollout_sampling = True
        self._ppo_location_head_fp32 = False
        self._initialize_action_priors()

    def _initialize_action_priors(self) -> None:
        gate = self.gate_head[-1]
        continuation = self.continue_head[-1]
        if not isinstance(gate, nn.Linear) or not isinstance(continuation, nn.Linear):
            raise TypeError("V4 categorical heads must end in Linear")
        with torch.no_grad():
            gate.weight.zero_()
            gate.bias.copy_(
                torch.log(
                    torch.tensor(
                        (1.0 - self.config.initial_act_probability, self.config.initial_act_probability),
                        dtype=gate.bias.dtype,
                        device=gate.bias.device,
                    )
                )
            )
            continuation.weight.zero_()
            continuation.bias.copy_(
                torch.log(
                    torch.tensor(
                        (1.0 - self.config.initial_second_probability, self.config.initial_second_probability),
                        dtype=continuation.bias.dtype,
                        device=continuation.bias.device,
                    )
                )
            )

    @property
    def ppo_gate_temperature(self) -> float:
        return self._ppo_gate_temperature

    @property
    def ppo_action_temperature(self) -> float:
        return self._ppo_action_temperature

    @property
    def ppo_continue_temperature(self) -> float:
        return self._ppo_continue_temperature

    @property
    def ppo_rollout_sampling(self) -> bool:
        return self._ppo_rollout_sampling

    def set_ppo_gate_temperature(self, temperature: float) -> None:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("PPO gate temperature must be finite and positive")
        self._ppo_gate_temperature = float(temperature)

    def set_ppo_action_temperature(self, temperature: float) -> None:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("PPO action temperature must be finite and positive")
        self._ppo_action_temperature = float(temperature)

    def set_ppo_continue_temperature(self, temperature: float) -> None:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("PPO continue temperature must be finite and positive")
        self._ppo_continue_temperature = float(temperature)

    def set_inference_delay_offset_max_ms(self, maximum_ms: int | None) -> None:
        if maximum_ms is not None and maximum_ms not in self.config.delay_offset_ms:
            raise ValueError("inference delay limit must be a configured offset")
        self._inference_delay_offset_max_ms = maximum_ms
        self._inference_delay_offset_mask.copy_(
            torch.tensor(
                tuple(maximum_ms is None or offset <= maximum_ms for offset in self.config.delay_offset_ms),
                dtype=torch.bool,
                device=self._inference_delay_offset_mask.device,
            )
        )

    def set_ppo_rollout_sampling(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("PPO rollout sampling mode must be bool")
        self._ppo_rollout_sampling = enabled

    def set_ppo_location_head_fp32(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("PPO location-head precision mode must be bool")
        self._ppo_location_head_fp32 = enabled

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(
        self, batch_size: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> RecurrentPolicyStateV4:
        reference = next(self.parameters())
        actual_device = reference.device if device is None else device
        actual_dtype = reference.dtype if dtype is None else dtype
        shape = (batch_size, self.config.lstm_hidden_dim)
        return RecurrentPolicyStateV4(
            hidden=torch.zeros(shape, device=actual_device, dtype=actual_dtype),
            cell=torch.zeros(shape, device=actual_device, dtype=actual_dtype),
        )

    def encode_observation(self, batch: UniversalSemanticBatchV4, *, validate: bool = True) -> EncodedObservationV4:
        if validate:
            batch.validate(
                self.config,
                card_vocab_size=self.catalog.vocab_size,
                ability_vocab_size=self.ability_catalog.vocab_size,
                entity_archetype_vocab_size=(self.entity_archetype_catalog.vocab_size),
                effect_vocab_size=self.effect_catalog.vocab_size,
            )
        effect_memory = self.effect_encoder.all_effects()
        profile_memory = self.mechanic_profile_encoder(effect_memory)
        child_effects, tower_effects = self.active_effect_encoder(
            batch.active_effects,
            effect_memory=effect_memory,
            child_count=batch.groups.child_mask.shape[1],
            tower_count=batch.towers.mask.shape[1],
        )
        child_memory, local_slots = self.child_encoder(
            batch.groups, profile_memory=profile_memory, active_effect_memory=child_effects
        )
        group_tokens = self.group_encoder(batch.groups, local_slots, profile_memory=profile_memory)
        tower_tokens = self.tower_encoder(
            batch.towers, profile_memory=profile_memory, active_effect_memory=tower_effects
        )
        own_tokens = self.card_token_encoder(batch.own_cards, owner_type=0, profile_memory=profile_memory)
        opponent_tokens = self.card_token_encoder(batch.opponent_cards, owner_type=1, profile_memory=profile_memory)
        global_inputs = assemble_global_tokens(
            towers=batch.towers,
            tower_tokens=tower_tokens,
            own_cards=batch.own_cards,
            own_tokens=own_tokens,
            opponent_cards=batch.opponent_cards,
            opponent_tokens=opponent_tokens,
            groups=batch.groups,
            group_tokens=group_tokens,
        )
        (scene_state, token_memory, token_mask, own_card_memory, group_memory) = self.relation_transformer(
            global_inputs, batch.relation_edges
        )
        spatial_memory, spatial_summary = self.spatial_encoder(
            batch.explicit_spatial_planes,
            child_memory,
            batch.groups.child_position,
            batch.groups.child_extent,
            batch.groups.child_radius,
            batch.groups.child_mask,
        )
        scalar_summary = self.scalar_encoder(batch.match_scalars)
        event_summary = self.event_encoder(
            batch.events, group_memory=group_memory, token_memory=token_memory, profile_memory=profile_memory
        )
        previous_action_summary = self.previous_action_encoder(batch.previous_action, profile_memory=profile_memory)
        candidate_memory, candidate_summary = self.candidate_encoder(
            batch.candidates,
            own_card_memory=own_card_memory,
            group_memory=group_memory,
            child_memory=child_memory,
            profile_memory=profile_memory,
        )
        return EncodedObservationV4(
            scene_state=scene_state,
            token_memory=token_memory,
            token_mask=token_mask,
            own_card_memory=own_card_memory,
            group_memory=group_memory,
            child_memory=child_memory,
            spatial_memory=spatial_memory,
            spatial_summary=spatial_summary,
            scalar_summary=scalar_summary,
            event_summary=event_summary,
            previous_action_summary=previous_action_summary,
            candidate_memory=candidate_memory,
            candidate_summary=candidate_summary,
        )

    def advance_core(
        self, encoded: EncodedObservationV4, state: RecurrentPolicyStateV4, *, episode_start: Tensor | None = None
    ) -> PolicyContextV4:
        batch_size = encoded.scene_state.shape[0]
        expected_state = (batch_size, self.config.lstm_hidden_dim)
        if tuple(state.hidden.shape) != expected_state or tuple(state.cell.shape) != (expected_state):
            raise ValueError("V4 recurrent state has the wrong shape")
        if episode_start is None:
            keep = torch.ones(batch_size, 1, dtype=state.hidden.dtype, device=state.hidden.device)
        else:
            if tuple(episode_start.shape) != (batch_size,) or (episode_start.dtype != torch.bool):
                raise ValueError("episode_start must be bool [B]")
            keep = (~episode_start).unsqueeze(-1).to(state.hidden.dtype)
        hidden = state.hidden * keep
        cell = state.cell * keep
        next_hidden, next_cell = self.lstm_core(self._core_input(encoded), (hidden, cell))
        return self._policy_context(encoded, next_hidden, next_cell)

    def _core_input(self, encoded: EncodedObservationV4) -> Tensor:
        return self.input_projection(
            torch.cat(
                (
                    encoded.scene_state,
                    encoded.spatial_summary,
                    encoded.scalar_summary,
                    encoded.event_summary,
                    encoded.previous_action_summary,
                    encoded.candidate_summary,
                ),
                dim=-1,
            )
        )

    def _policy_context(self, encoded: EncodedObservationV4, hidden: Tensor, cell: Tensor) -> PolicyContextV4:
        normalized = self.core_output_norm(hidden)
        policy_context = self.policy_context_norm(
            normalized
            + self.scene_policy_skip(encoded.scene_state)
            + self.spatial_policy_skip(encoded.spatial_summary)
            + self.candidate_policy_skip(encoded.candidate_summary)
        )
        value_context = self.value_context_norm(
            normalized
            + self.scene_value_skip(encoded.scene_state)
            + self.spatial_value_skip(encoded.spatial_summary)
            + self.scalar_value_skip(encoded.scalar_summary)
            + self.event_value_skip(encoded.event_summary)
        )
        return PolicyContextV4(
            policy_context=policy_context,
            value_context=value_context,
            encoded=encoded,
            next_state=RecurrentPolicyStateV4(hidden=hidden, cell=cell),
            value=self.value_head(value_context).squeeze(-1),
        )

    def forward(
        self,
        batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        *,
        episode_start: Tensor | None = None,
        validate: bool = True,
    ) -> PolicyContextV4:
        return self.advance_core(self.encode_observation(batch, validate=validate), state, episode_start=episode_start)

    def _decoder_context(self, base: Tensor, encoded: EncodedObservationV4) -> Tensor:
        attended, _ = self.cross_attention(
            base.unsqueeze(1),
            encoded.token_memory,
            encoded.token_memory,
            key_padding_mask=~encoded.token_mask,
            need_weights=False,
        )
        return self.decoder_norm(base + attended[:, 0])

    def candidate_logits(self, decoder_context: Tensor, candidate_memory: Tensor) -> Tensor:
        query = self.candidate_query(decoder_context)
        key = self.candidate_key(candidate_memory)
        return torch.einsum("bd,bcd->bc", query, key) / math.sqrt(self.config.candidate_dim)

    def _selected_candidate_memory(self, memory: Tensor, candidate_index: Tensor) -> Tensor:
        return gather_memory(memory, candidate_index[:, None])[:, 0]

    def _location_logits(
        self, decoder_context: Tensor, candidate_memory: Tensor, spatial_memory: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self._ppo_location_head_fp32:
            with torch.autocast(device_type=decoder_context.device.type, enabled=False):
                condition_input = torch.cat((decoder_context.float(), candidate_memory.float()), dim=-1)
                condition = self.location_condition(condition_input)
                gamma, beta = self.location_film(condition).chunk(2, dim=-1)
                conditioned = spatial_memory.float() * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
                keys = self.location_key(conditioned)
                query = self.location_query(condition_input)
                logits = torch.einsum("bd,bdhw->bhw", query, keys) / math.sqrt(query.shape[-1])
            return logits, keys
        condition_input = torch.cat((decoder_context, candidate_memory), dim=-1)
        condition = self.location_condition(condition_input)
        gamma, beta = self.location_film(condition).chunk(2, dim=-1)
        conditioned = spatial_memory * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        keys = self.location_key(conditioned)
        query = self.location_query(condition_input)
        logits = torch.einsum("bd,bdhw->bhw", query, keys) / math.sqrt(query.shape[-1])
        return logits, keys

    def _target_feature(self, keys: Tensor, target_cell: Tensor, grid: Tensor) -> Tensor:
        safe = target_cell.clamp_min(0)
        gathered = keys.flatten(2).gather(2, safe[:, None, None].expand(-1, keys.shape[1], 1))[:, :, 0]
        return torch.where(grid[:, None], gathered, self.no_target_embedding[None].expand_as(gathered))

    def _delay_offset_logits(self, decoder_context: Tensor, candidate_memory: Tensor, target_feature: Tensor) -> Tensor:
        return self.delay_offset_head(torch.cat((decoder_context, candidate_memory, target_feature), dim=-1))

    def _action_embedding(self, candidate_memory: Tensor, target_feature: Tensor, delay_offset_bin: Tensor) -> Tensor:
        return self.action_embedding(
            torch.cat((candidate_memory, target_feature, self.delay_offset_embedding(delay_offset_bin)), dim=-1)
        )

    def _resolve_candidate_indices(self, batch: UniversalSemanticBatchV4, actions: ActionSequenceV4) -> Tensor:
        active = (
            torch.arange(self.config.max_micro_actions, device=actions.gate.device)[None]
            < actions.micro_action_count[:, None]
        )
        matches = (batch.candidates.uid[:, None, :] == actions.candidate_uid[:, :, None]) & batch.candidates.mask[
            :, None, :
        ]
        match_count = matches.sum(dim=-1)
        if torch.any(active & (match_count != 1)):
            raise ValueError("stored candidate UID is missing or ambiguous")
        resolved = matches.to(torch.long).argmax(dim=-1)
        return torch.where(active, resolved, torch.full_like(resolved, -1))

    @staticmethod
    def _choice(distribution: Categorical, *, sample: bool) -> Tensor:
        return distribution.sample() if sample else distribution.logits.argmax(dim=-1)

    def _decode(
        self,
        batch: UniversalSemanticBatchV4,
        context: PolicyContextV4,
        *,
        sample: bool,
        forced_actions: ActionSequenceV4 | None,
        gate_temperature: float,
        action_temperature: float,
        continue_temperature: float,
        validate: bool,
        preselected_gate: Tensor | None = None,
    ) -> PolicyOutputV4:
        config = self.config
        candidates = batch.candidates
        batch_size, candidate_count = candidates.mask.shape
        device = context.policy_context.device
        dtype = context.policy_context.dtype
        if forced_actions is not None:
            if validate:
                forced_actions.validate(config, candidate_count=candidate_count)
            resolved = self._resolve_candidate_indices(batch, forced_actions)
        else:
            resolved = None

        shadow = ShadowCandidateLegality(candidates, config)
        initial_legal = shadow.candidate_mask()
        gate_logits = self.gate_head(context.policy_context)
        gate_mask = torch.stack(
            (torch.ones(batch_size, dtype=torch.bool, device=device), initial_legal.any(dim=-1)), dim=-1
        )
        gate_distribution = _distribution(gate_logits, mask=gate_mask, temperature=gate_temperature, validate=validate)
        if forced_actions is not None and preselected_gate is not None:
            raise ValueError("forced actions and a preselected gate are mutually exclusive")
        gate = (
            forced_actions.gate
            if forced_actions is not None
            else preselected_gate
            if preselected_gate is not None
            else self._choice(gate_distribution, sample=sample)
        )
        if tuple(gate.shape) != (batch_size,):
            raise ValueError("preselected gate has the wrong batch shape")
        if validate and torch.any(~gate_mask.gather(1, gate[:, None])[:, 0]):
            raise ValueError("stored gate choice is illegal")
        gate_log_prob = gate_distribution.log_prob(gate)
        gate_entropy = gate_distribution.entropy()
        act = gate == GATE_ACT

        candidate_index = torch.full((batch_size, config.max_micro_actions), -1, dtype=torch.long, device=device)
        candidate_uid = torch.full_like(candidate_index, -1)
        target_cell = torch.full_like(candidate_index, -1)
        delay_offset_bin = torch.full_like(candidate_index, -1)
        candidate_log_prob = torch.zeros(batch_size, config.max_micro_actions, dtype=dtype, device=device)
        candidate_entropy = torch.zeros_like(candidate_log_prob)
        target_log_prob = torch.zeros_like(candidate_log_prob)
        target_entropy = torch.zeros_like(candidate_log_prob)
        delay_offset_log_prob = torch.zeros_like(candidate_log_prob)
        delay_offset_log_probs = torch.zeros(
            batch_size, config.max_micro_actions, len(config.delay_offset_ms), dtype=dtype, device=device
        )
        delay_offset_legal_mask = torch.zeros(
            batch_size, config.max_micro_actions, len(config.delay_offset_ms), dtype=torch.bool, device=device
        )
        delay_offset_entropy = torch.zeros_like(candidate_log_prob)
        continue_log_prob = torch.zeros(batch_size, dtype=dtype, device=device)
        continue_entropy = torch.zeros_like(continue_log_prob)

        decoder_base = self.decoder_initial(context.policy_context)
        decoder_context = self._decoder_context(decoder_base, context.encoded)
        active = act
        first_action_embedding = torch.zeros(batch_size, 128, dtype=dtype, device=device)
        second = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(config.max_micro_actions):
            legal_candidates = shadow.candidate_mask()
            if validate and torch.any(active & ~legal_candidates.any(dim=-1)):
                raise ValueError("active decoder step has no legal candidate")
            candidate_distribution = _distribution(
                self.candidate_logits(decoder_context, context.encoded.candidate_memory),
                mask=legal_candidates,
                temperature=action_temperature,
                validate=validate,
            )
            selected_candidate = (
                resolved[:, step].clamp_min(0)
                if resolved is not None
                else self._choice(candidate_distribution, sample=sample)
            )
            if validate and torch.any(active & ~legal_candidates.gather(1, selected_candidate[:, None])[:, 0]):
                raise ValueError("stored candidate is illegal under shadow state")
            selected_memory = self._selected_candidate_memory(context.encoded.candidate_memory, selected_candidate)
            selected_mode = candidates.target_mode.gather(1, selected_candidate[:, None])[:, 0]
            location_logits, location_keys = self._location_logits(
                decoder_context, selected_memory, context.encoded.spatial_memory
            )
            placement = shadow.selected_placement(selected_candidate)
            location_distribution = _distribution(
                location_logits.flatten(1), mask=placement.flatten(1), temperature=action_temperature, validate=validate
            )
            grid = active & (selected_mode == TARGET_GRID)
            selected_target = (
                forced_actions.target_cell[:, step].clamp_min(0)
                if forced_actions is not None
                else self._choice(location_distribution, sample=sample)
            )
            if validate and torch.any(grid & ~placement.flatten(1).gather(1, selected_target[:, None])[:, 0]):
                raise ValueError("stored target cell is illegal")
            if (
                validate
                and forced_actions is not None
                and torch.any(active & ~grid & (forced_actions.target_cell[:, step] != -1))
            ):
                raise ValueError("non-GRID candidate must use target cell -1")
            selected_target = torch.where(grid, selected_target, torch.full_like(selected_target, -1))
            target_feature = self._target_feature(location_keys, selected_target, grid)
            delay_offset_mask = shadow.delay_offset_mask(step=step)
            if forced_actions is None:
                delay_offset_mask &= self._inference_delay_offset_mask[None]
            delay_distribution = _distribution(
                self._delay_offset_logits(decoder_context, selected_memory, target_feature),
                mask=delay_offset_mask,
                temperature=action_temperature,
                validate=validate,
            )
            selected_delay = (
                forced_actions.delay_offset_bin[:, step].clamp_min(0)
                if forced_actions is not None
                else self._choice(delay_distribution, sample=sample)
            )
            if validate and torch.any(active & ~delay_offset_mask.gather(1, selected_delay[:, None])[:, 0]):
                raise ValueError("stored delay offsets must preserve action order")
            selected_action_embedding = self._action_embedding(selected_memory, target_feature, selected_delay)

            selected_log_prob = candidate_distribution.log_prob(selected_candidate)
            selected_target_log_prob = location_distribution.log_prob(selected_target.clamp_min(0))
            selected_delay_log_prob = delay_distribution.log_prob(selected_delay)
            candidate_log_prob[:, step] = selected_log_prob * active
            candidate_entropy[:, step] = candidate_distribution.entropy() * active
            target_log_prob[:, step] = selected_target_log_prob * grid
            target_entropy[:, step] = location_distribution.entropy() * grid
            delay_offset_log_prob[:, step] = selected_delay_log_prob * active
            delay_offset_log_probs[:, step] = delay_distribution.logits
            delay_offset_legal_mask[:, step] = delay_offset_mask
            delay_offset_entropy[:, step] = delay_distribution.entropy() * active
            candidate_index[:, step] = torch.where(active, selected_candidate, candidate_index[:, step])
            selected_uid = candidates.uid.gather(1, selected_candidate[:, None])[:, 0]
            candidate_uid[:, step] = torch.where(active, selected_uid, candidate_uid[:, step])
            target_cell[:, step] = torch.where(active, selected_target, target_cell[:, step])
            delay_offset_bin[:, step] = torch.where(active, selected_delay, delay_offset_bin[:, step])

            if step == 0:
                first_action_embedding = selected_action_embedding
                shadow.apply(active, selected_candidate, selected_target, selected_delay, step=step)
                remaining_mask = shadow.candidate_mask()
                remaining_summary = self.candidate_encoder.pool(context.encoded.candidate_memory, remaining_mask)
                shadow_features = shadow.shadow_features(step=1)
                autoregressive = torch.cat((first_action_embedding, shadow_features, remaining_summary), dim=-1)
                continue_distribution = _distribution(
                    self.continue_head(torch.cat((decoder_context, autoregressive), dim=-1)),
                    mask=torch.stack(
                        (torch.ones(batch_size, dtype=torch.bool, device=device), remaining_mask.any(dim=-1)), dim=-1
                    ),
                    temperature=continue_temperature,
                    validate=validate,
                )
                continue_choice = (
                    (forced_actions.micro_action_count == 2).to(torch.long)
                    if forced_actions is not None
                    else self._choice(continue_distribution, sample=sample)
                )
                if validate and torch.any(act & (continue_choice == 1) & ~remaining_mask.any(dim=-1)):
                    raise ValueError("stored SECOND choice has no legal candidate")
                continue_log_prob = continue_distribution.log_prob(continue_choice) * act
                continue_entropy = continue_distribution.entropy() * act
                second = act & (continue_choice == 1)
                active = second
                decoder_context = self._decoder_context(
                    decoder_base + self.decoder_step(autoregressive), context.encoded
                )

        micro_action_count = act.to(torch.long) + second.to(torch.long)
        actions = ActionSequenceV4(
            gate=gate,
            micro_action_count=micro_action_count,
            candidate_index=candidate_index,
            candidate_uid=candidate_uid,
            target_cell=target_cell,
            delay_offset_bin=delay_offset_bin,
        )
        if validate:
            actions.validate(config, candidate_count=candidate_count)
        components = ActionEvaluationComponentsV4(
            gate_log_prob=gate_log_prob,
            gate_entropy=gate_entropy,
            candidate_log_prob=candidate_log_prob,
            candidate_entropy=candidate_entropy,
            target_log_prob=target_log_prob,
            target_entropy=target_entropy,
            delay_offset_log_prob=delay_offset_log_prob,
            delay_offset_log_probs=delay_offset_log_probs,
            delay_offset_legal_mask=delay_offset_legal_mask,
            delay_offset_entropy=delay_offset_entropy,
            continue_log_prob=continue_log_prob,
            continue_entropy=continue_entropy,
        )
        total_log_prob = (
            gate_log_prob
            + candidate_log_prob.sum(dim=-1)
            + target_log_prob.sum(dim=-1)
            + delay_offset_log_prob.sum(dim=-1)
            + continue_log_prob
        )
        total_entropy = (
            gate_entropy
            + candidate_entropy.sum(dim=-1)
            + target_entropy.sum(dim=-1)
            + delay_offset_entropy.sum(dim=-1)
            + continue_entropy
        )
        return PolicyOutputV4(
            actions=actions,
            log_prob=total_log_prob,
            entropy=total_entropy,
            value=context.value,
            next_state=context.next_state,
            action_components=components,
        )

    def _select_action(
        self,
        batch: UniversalSemanticBatchV4,
        context: PolicyContextV4,
        *,
        sample: bool,
        validate: bool,
        preselected_act: bool = False,
    ) -> PolicyOutputV4:
        return self._decode(
            batch,
            context,
            sample=sample,
            forced_actions=None,
            gate_temperature=self.ppo_gate_temperature if sample else 1.0,
            action_temperature=self.ppo_action_temperature if sample else 1.0,
            continue_temperature=self.ppo_continue_temperature if sample else 1.0,
            validate=validate,
            preselected_gate=(
                torch.full((batch.batch_size,), GATE_ACT, dtype=torch.long, device=context.policy_context.device)
                if preselected_act
                else None
            ),
        )

    def act(
        self,
        batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        *,
        episode_start: Tensor | None = None,
        validate: bool = True,
    ) -> PolicyOutputV4:
        context = self.forward(batch, state, episode_start=episode_start, validate=validate)
        return self._select_action(batch, context, sample=False, validate=validate)

    def sample_for_ppo_rollout(
        self,
        batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        *,
        episode_start: Tensor | None = None,
        validate: bool = True,
    ) -> PolicyOutputV4:
        context = self.forward(batch, state, episode_start=episode_start, validate=validate)
        return self._select_action(batch, context, sample=self.ppo_rollout_sampling, validate=validate)

    def sample_after_preselected_act(
        self, batch: UniversalSemanticBatchV4, context: PolicyContextV4, *, validate: bool = True
    ) -> PolicyOutputV4:
        """Sample downstream heads for rows whose gate is already ACT."""
        return self._select_action(
            batch, context, sample=self.ppo_rollout_sampling, validate=validate, preselected_act=True
        )

    def act_after_preselected_act(
        self, batch: UniversalSemanticBatchV4, context: PolicyContextV4, *, validate: bool = True
    ) -> PolicyOutputV4:
        """Argmax downstream heads for rows whose gate is already ACT."""
        return self._select_action(batch, context, sample=False, validate=validate, preselected_act=True)

    def evaluate_actions(
        self,
        batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        actions: ActionSequenceV4,
        *,
        episode_start: Tensor | None = None,
        gate_temperature: float | None = None,
        action_temperature: float | None = None,
        continue_temperature: float | None = None,
        validate: bool = True,
    ) -> PolicyOutputV4:
        context = self.forward(batch, state, episode_start=episode_start, validate=validate)
        return self._decode(
            batch,
            context,
            sample=False,
            forced_actions=actions,
            gate_temperature=(self.ppo_gate_temperature if gate_temperature is None else gate_temperature),
            action_temperature=(self.ppo_action_temperature if action_temperature is None else action_temperature),
            continue_temperature=(
                self.ppo_continue_temperature if continue_temperature is None else continue_temperature
            ),
            validate=validate,
        )

    def _evaluate_sparse_forced_actions(
        self,
        batch: UniversalSemanticBatchV4,
        context: PolicyContextV4,
        actions: ActionSequenceV4,
        *,
        gate_temperature: float,
        action_temperature: float,
        continue_temperature: float,
        validate: bool,
        conditioned_gate_mask: Tensor | None = None,
        return_log_prob_components: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Evaluate WAIT rows with the gate alone and decode only ACT rows.

        The dense decoder masks every downstream action term to zero for WAIT,
        but still executes both autoregressive spatial decoders for those rows.
        PPO rollouts are overwhelmingly WAIT, so evaluating the exact same
        forced-action likelihood on the ACT subset avoids work that cannot
        contribute either a value or a gradient.
        """

        candidates = batch.candidates
        batch_size = batch.batch_size
        device = context.policy_context.device
        shadow = ShadowCandidateLegality(candidates, self.config)
        initial_legal = shadow.candidate_mask()
        gate_mask = torch.stack(
            (torch.ones(batch_size, dtype=torch.bool, device=device), initial_legal.any(dim=-1)), dim=-1
        )
        gate_distribution = _distribution(
            self.gate_head(context.policy_context), mask=gate_mask, temperature=gate_temperature, validate=validate
        )
        if validate and torch.any(~gate_mask.gather(1, actions.gate[:, None])[:, 0]):
            raise ValueError("stored gate choice is illegal")
        log_prob = gate_distribution.log_prob(actions.gate)
        entropy = gate_distribution.entropy()
        if conditioned_gate_mask is not None:
            if conditioned_gate_mask.dtype != torch.bool or tuple(conditioned_gate_mask.shape) != (batch_size,):
                raise ValueError("conditioned gate mask must be bool [B]")
            log_prob = torch.where(conditioned_gate_mask, torch.zeros_like(log_prob), log_prob)
            entropy = torch.where(conditioned_gate_mask, torch.zeros_like(entropy), entropy)
        zero_component = torch.zeros_like(log_prob)
        log_prob_components = torch.stack(
            (log_prob, zero_component, zero_component, zero_component, zero_component), dim=-1
        )

        active_indices = (actions.gate == GATE_ACT).nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            # The historical dense decoder touched every action-head
            # parameter and therefore materialized zero gradients for them.
            # Keep that optimizer-visible behavior without running the heads.
            zero_dependency = self.no_target_embedding.reshape(-1)[0].to(log_prob.dtype) * 0.0
            decoder_modules = (
                self.decoder_initial,
                self.cross_attention,
                self.decoder_norm,
                self.candidate_query,
                self.candidate_key,
                self.location_condition,
                self.location_film,
                self.location_key,
                self.location_query,
                self.delay_offset_head,
                self.delay_offset_embedding,
                self.action_embedding,
                self.decoder_step,
                self.continue_head,
            )
            for module in decoder_modules:
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        zero_dependency = zero_dependency + parameter.reshape(-1)[0].to(log_prob.dtype) * 0.0
            result = (log_prob + zero_dependency, entropy)
            if return_log_prob_components:
                return (*result, log_prob_components)
            return result

        active_output = self._decode(
            batch.index_select(active_indices),
            context.index_select(active_indices),
            sample=False,
            forced_actions=actions.index_select(active_indices),
            gate_temperature=gate_temperature,
            action_temperature=action_temperature,
            continue_temperature=continue_temperature,
            validate=validate,
        )
        active_log_prob = active_output.log_prob
        active_entropy = active_output.entropy
        if active_output.action_components is None:
            raise RuntimeError("sparse ACT evaluation lacks head components")
        active_components = _reduce_log_prob_components_v4(active_output.action_components)
        if conditioned_gate_mask is not None:
            active_conditioned = conditioned_gate_mask.index_select(0, active_indices)
            active_log_prob = active_log_prob - torch.where(
                active_conditioned,
                active_output.action_components.gate_log_prob,
                torch.zeros_like(active_output.action_components.gate_log_prob),
            )
            active_entropy = active_entropy - torch.where(
                active_conditioned,
                active_output.action_components.gate_entropy,
                torch.zeros_like(active_output.action_components.gate_entropy),
            )
            active_components[:, 0] = torch.where(
                active_conditioned, torch.zeros_like(active_components[:, 0]), active_components[:, 0]
            )
        result = (
            log_prob.index_copy(0, active_indices, active_log_prob),
            entropy.index_copy(0, active_indices, active_entropy),
        )
        if return_log_prob_components:
            return (*result, log_prob_components.index_copy(0, active_indices, active_components))
        return result

    def evaluate_encoded_actions(
        self,
        batch: UniversalSemanticBatchV4,
        encoded: EncodedObservationV4,
        state: RecurrentPolicyStateV4,
        actions: ActionSequenceV4,
        *,
        episode_start: Tensor | None = None,
        gate_temperature: float | None = None,
        action_temperature: float | None = None,
        continue_temperature: float | None = None,
        validate: bool = True,
    ) -> PolicyOutputV4:
        """Evaluate actions after a caller has batch-encoded observations."""

        context = self.advance_core(encoded, state, episode_start=episode_start)
        return self._decode(
            batch,
            context,
            sample=False,
            forced_actions=actions,
            gate_temperature=(self.ppo_gate_temperature if gate_temperature is None else gate_temperature),
            action_temperature=(self.ppo_action_temperature if action_temperature is None else action_temperature),
            continue_temperature=(
                self.ppo_continue_temperature if continue_temperature is None else continue_temperature
            ),
            validate=validate,
        )

    def evaluate_encoded_action_sequence(
        self,
        batch: UniversalSemanticBatchV4,
        encoded: EncodedObservationV4,
        state: RecurrentPolicyStateV4,
        actions: ActionSequenceV4,
        episode_start: Tensor,
        *,
        time_steps: int,
        gate_temperature: float | None = None,
        action_temperature: float | None = None,
        continue_temperature: float | None = None,
        conditioned_gate_mask: Tensor | None = None,
        return_log_prob_components: bool = False,
        validate: bool = True,
    ) -> (
        tuple[Tensor, Tensor, Tensor, RecurrentPolicyStateV4]
        | tuple[Tensor, Tensor, Tensor, RecurrentPolicyStateV4, Tensor]
    ):
        """Evaluate a one-episode sequence with the fused LSTM backend.

        PPO waves contain one episode per recurrent lane.  Flattening the
        independent action heads over ``T*B`` is exact, while the equivalent
        ``nn.LSTM`` kernel avoids launching one ``LSTMCell`` graph per frame.
        """

        if time_steps <= 0 or batch.batch_size % time_steps:
            raise ValueError("encoded action sequence has invalid T/B dimensions")
        batch_size = batch.batch_size // time_steps
        if tuple(encoded.scene_state.shape[:1]) != (batch.batch_size,):
            raise ValueError("encoded action sequence changed flattened rows")
        if tuple(episode_start.shape) != (time_steps, batch_size) or (episode_start.dtype != torch.bool):
            raise ValueError("encoded episode_start must be bool [T,B]")
        if conditioned_gate_mask is not None and (
            conditioned_gate_mask.dtype != torch.bool or tuple(conditioned_gate_mask.shape) != (time_steps, batch_size)
        ):
            raise ValueError("conditioned gate mask must be bool [T,B]")
        expected_state = (batch_size, self.config.lstm_hidden_dim)
        if tuple(state.hidden.shape) != expected_state or tuple(state.cell.shape) != (expected_state):
            raise ValueError("encoded action sequence state has the wrong shape")
        if validate and time_steps > 1 and torch.any(episode_start[1:]):
            raise ValueError("fused PPO sequence cannot contain mid-lane resets")

        keep = (~episode_start[0]).unsqueeze(-1).to(state.hidden.dtype)
        hidden = (state.hidden * keep).unsqueeze(0)
        cell = (state.cell * keep).unsqueeze(0)
        core_input = self._core_input(encoded).reshape(time_steps, batch_size, -1)
        recurrent, final_hidden, final_cell = torch._VF.lstm(  # type: ignore[attr-defined]
            core_input,
            (hidden, cell),
            (self.lstm_core.weight_ih, self.lstm_core.weight_hh, self.lstm_core.bias_ih, self.lstm_core.bias_hh),
            True,
            1,
            0.0,
            self.training,
            False,
            False,
        )
        recurrent_flat = recurrent.flatten(0, 1)
        # The decoder needs each hidden state, but never the intermediate cell states.
        context = self._policy_context(encoded, recurrent_flat, torch.zeros_like(recurrent_flat))
        actual_gate_temperature = self.ppo_gate_temperature if gate_temperature is None else gate_temperature
        actual_action_temperature = self.ppo_action_temperature if action_temperature is None else action_temperature
        actual_continue_temperature = (
            self.ppo_continue_temperature if continue_temperature is None else continue_temperature
        )
        evaluated = self._evaluate_sparse_forced_actions(
            batch,
            context,
            actions,
            gate_temperature=actual_gate_temperature,
            action_temperature=actual_action_temperature,
            continue_temperature=actual_continue_temperature,
            conditioned_gate_mask=(None if conditioned_gate_mask is None else conditioned_gate_mask.flatten()),
            return_log_prob_components=return_log_prob_components,
            validate=validate,
        )
        evaluated_log_prob, evaluated_entropy = evaluated[:2]
        shape = (time_steps, batch_size)
        result = (
            evaluated_log_prob.reshape(shape),
            evaluated_entropy.reshape(shape),
            context.value.reshape(shape),
            RecurrentPolicyStateV4(hidden=final_hidden[0], cell=final_cell[0]),
        )
        if return_log_prob_components:
            return (*result, evaluated[2].reshape(*shape, 5))
        return result
