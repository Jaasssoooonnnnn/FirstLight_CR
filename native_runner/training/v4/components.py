"""Neural components for :class:`UniversalCardPolicyV4`."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .catalog import CardCatalogV1, PAD_CARD_VOCAB_ID
from .config import ModelConfigV4
from .mechanics import (
    ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES,
    EFFECT_SEMANTIC_FEATURE_NAMES,
    MECHANIC_NUMERIC_FEATURE_NAMES,
    MECHANIC_TAG_NAMES,
    EffectSemanticCatalogV1,
    MechanicProfileCatalogV1,
    PAD_EFFECT_VOCAB_ID,
)
from .tensors import (
    ActiveEffectSetV4,
    ActionCandidatesV4,
    BattleGroupSetV4,
    CardSetV4,
    EventSetV4,
    EFFECT_PARENT_CHILD,
    EFFECT_PARENT_TOWER,
    PreviousActionV4,
    RelationEdgesV4,
    TowerSetV4,
)


def mlp(*dimensions: int, layer_norm: bool = False) -> nn.Sequential:
    modules: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
        modules.append(nn.Linear(source, target))
        if index + 2 < len(dimensions):
            modules.append(nn.SiLU())
            if layer_norm:
                modules.append(nn.LayerNorm(target))
    return nn.Sequential(*modules)


def masked_logits(logits: Tensor, mask: Tensor, *, validate: bool = True) -> Tensor:
    if validate:
        if logits.shape != mask.shape:
            raise ValueError(f"logit/mask mismatch: {tuple(logits.shape)} != {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            raise TypeError("categorical mask must be bool")
        if torch.any(~mask.any(dim=-1)):
            raise ValueError("every categorical row needs at least one legal choice")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def masked_pool_weights(logits: Tensor, mask: Tensor) -> Tensor:
    """Masked softmax that returns all-zero rows for empty sets."""

    if logits.shape != mask.shape:
        raise ValueError("masked pooling logits and mask must have equal shapes")
    valid = mask.any(dim=-1, keepdim=True)
    safe_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(safe_logits, dim=-1)
    weights = weights * mask.to(weights.dtype)
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return (weights / denominator) * valid.to(weights.dtype)


def gather_memory(memory: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B,N,D]`` rows with ``-1`` represented as an all-zero row."""

    if memory.ndim != 3 or indices.ndim != 2 or memory.shape[0] != indices.shape[0]:
        raise ValueError("memory gather requires [B,N,D] and [B,M]")
    batch, _, width = memory.shape
    if memory.shape[1] == 0:
        return memory.new_zeros(batch, indices.shape[1], width)
    valid = indices >= 0
    safe = indices.clamp(0, memory.shape[1] - 1)
    gathered = memory.gather(1, safe.unsqueeze(-1).expand(-1, -1, width))
    return gathered * valid.unsqueeze(-1)


def gather_scalar(values: Tensor, indices: Tensor, *, pad_value: int = 0) -> Tensor:
    if values.ndim != 2 or indices.ndim != 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("scalar gather requires [B,N] and [B,M]")
    if values.shape[1] == 0:
        return torch.full_like(indices, pad_value)
    valid = indices >= 0
    safe = indices.clamp(0, values.shape[1] - 1)
    gathered = values.gather(1, safe)
    return torch.where(valid, gathered, torch.full_like(gathered, pad_value))


class LearnedSetPool(nn.Module):
    """Permutation-invariant learned-query attention pooling."""

    def __init__(self, input_dim: int, output_dim: int, *, slots: int = 1) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(slots, input_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.output = nn.Linear(slots * input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, memory: Tensor, mask: Tensor) -> Tensor:
        scores = torch.einsum("bnd,qd->bqn", memory, self.queries) / math.sqrt(memory.shape[-1])
        weights = masked_pool_weights(scores, mask[:, None, :].expand_as(scores))
        pooled = torch.einsum("bqn,bnd->bqd", weights, memory)
        return self.norm(self.output(pooled.flatten(1)))


class EffectSemanticEncoder(nn.Module):
    """Shared effect semantics with a gated exact-Buff-ID residual."""

    def __init__(self, config: ModelConfigV4, catalog: EffectSemanticCatalogV1) -> None:
        super().__init__()
        features = catalog.feature_tensor()
        if features.shape != (catalog.vocab_size, len(EFFECT_SEMANTIC_FEATURE_NAMES)):
            raise ValueError("EffectSemanticCatalog tensor has the wrong shape")
        self.register_buffer("catalog_features", features, persistent=False)
        self.id_embedding = nn.Embedding(catalog.vocab_size, config.effect_dim, padding_idx=PAD_EFFECT_VOCAB_ID)
        self.semantic = mlp(len(EFFECT_SEMANTIC_FEATURE_NAMES), config.effect_dim, config.effect_dim)
        self.id_gate = mlp(config.effect_dim, config.effect_dim, 1)
        self.norm = nn.LayerNorm(config.effect_dim)
        final_gate = self.id_gate[-1]
        if not isinstance(final_gate, nn.Linear):
            raise TypeError("effect ID residual gate must end in Linear")
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, -2.0)

    def forward(self, effect_vocab_id: Tensor) -> Tensor:
        semantic = self.semantic(self.catalog_features[effect_vocab_id])
        gate = torch.sigmoid(self.id_gate(semantic))
        result = self.norm(semantic + gate * self.id_embedding(effect_vocab_id))
        return result * (effect_vocab_id != PAD_EFFECT_VOCAB_ID).unsqueeze(-1)

    def all_effects(self) -> Tensor:
        return self(torch.arange(self.catalog_features.shape[0], device=self.catalog_features.device))


class MechanicProfileEncoder(nn.Module):
    """Encode structured native operations, then pool them by static profile."""

    def __init__(self, config: ModelConfigV4, catalog: MechanicProfileCatalogV1) -> None:
        super().__init__()
        tables = catalog.tensor_tables()
        for name in (
            "profile_id",
            "trigger_id",
            "operation_id",
            "target_id",
            "source_id",
            "control_path_id",
            "condition_id",
            "amount_basis_id",
            "guard_archetype_id",
            "guard_polarity",
            "guard_numeric",
            "guard_mask",
            "tags",
            "effect_vocab_id",
            "produced_archetype_id",
            "numeric",
        ):
            value = tables[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"mechanic profile table {name} is not a Tensor")
            self.register_buffer(name, value, persistent=False)
        categorical = ("trigger", "operation", "target", "source", "control_path", "condition", "amount_basis", "guard")
        if not all(isinstance(tables[name + "_keys"], tuple) for name in categorical):
            raise TypeError("mechanic categorical tables are invalid")
        width = config.mechanic_profile_dim
        self.profile_count = catalog.profile_count
        for name in categorical:
            keys = tables[name + "_keys"]
            setattr(
                self,
                "guard_archetype" if name == "guard" else name,
                nn.Embedding(max(1, len(keys)), width, padding_idx=0 if name == "guard" else None),
            )
        self.guard_polarity_encoder = nn.Linear(1, width, bias=False)
        self.guard_numeric_encoder = mlp(2, width, width)
        self.tags_encoder = nn.Linear(len(MECHANIC_TAG_NAMES), width)
        self.numeric_encoder = mlp(len(MECHANIC_NUMERIC_FEATURE_NAMES), width, width)
        self.effect = nn.Linear(config.effect_dim, width)
        self.produced_archetype = nn.Embedding(config.child_archetype_count, width, padding_idx=0)
        self.row_norm = nn.LayerNorm(width)
        self.count = nn.Linear(1, width, bias=False)
        self.profile_norm = nn.LayerNorm(width)

    def forward(self, effect_memory: Tensor) -> Tensor:
        if self.profile_id.numel() == 0:
            return effect_memory.new_zeros(self.profile_count, self.tags_encoder.out_features)
        guard = (
            self.guard_archetype(self.guard_archetype_id)
            + self.guard_polarity_encoder(self.guard_polarity.unsqueeze(-1))
            + self.guard_numeric_encoder(self.guard_numeric)
        ) * self.guard_mask.unsqueeze(-1)
        guard_count = self.guard_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        guard = guard.sum(dim=-2) / guard_count.to(guard.dtype).sqrt()
        row = self.row_norm(
            self.trigger(self.trigger_id)
            + self.operation(self.operation_id)
            + self.target(self.target_id)
            + self.source(self.source_id)
            + self.control_path(self.control_path_id)
            + self.condition(self.condition_id)
            + self.amount_basis(self.amount_basis_id)
            + guard
            + self.tags_encoder(self.tags)
            + self.numeric_encoder(self.numeric)
            + self.effect(nn.functional.embedding(self.effect_vocab_id, effect_memory))
            + self.produced_archetype(self.produced_archetype_id)
        )
        width = row.shape[-1]
        pooled = row.new_zeros(self.profile_count, width)
        pooled.scatter_add_(0, self.profile_id[:, None].expand(-1, width), row)
        counts = row.new_zeros(self.profile_count, 1)
        counts.scatter_add_(0, self.profile_id[:, None], torch.ones_like(self.profile_id, dtype=row.dtype)[:, None])
        valid = counts > 0
        pooled = pooled / counts.clamp_min(1.0).sqrt()
        pooled = self.profile_norm(pooled + self.count(torch.log1p(counts)))
        return pooled * valid.to(pooled.dtype)


class ActiveEffectEncoder(nn.Module):
    """Encode exact live effects and scatter them only to their local parent."""

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.runtime = mlp(len(ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES), config.effect_dim, config.effect_dim)
        self.source_owner = nn.Embedding(config.owner_type_count, config.effect_dim)
        self.norm = nn.LayerNorm(config.effect_dim)
        self.count = nn.Linear(1, config.effect_dim, bias=False)

    def _scatter(self, memory: Tensor, effects: ActiveEffectSetV4, *, parent_type: int, parent_count: int) -> Tensor:
        batch, _, width = memory.shape
        output = memory.new_zeros(batch, parent_count, width)
        counts = memory.new_zeros(batch, parent_count, 1)
        valid = effects.mask & (effects.parent_type == parent_type)
        if parent_count == 0 or memory.shape[1] == 0:
            return output
        safe = effects.parent_index.clamp(0, parent_count - 1)
        source = memory * valid.unsqueeze(-1)
        output.scatter_add_(1, safe.unsqueeze(-1).expand(-1, -1, width), source)
        counts.scatter_add_(1, safe.unsqueeze(-1), valid.unsqueeze(-1).to(memory.dtype))
        present = counts > 0
        output = output / counts.clamp_min(1.0).sqrt()
        output = output + self.count(torch.log1p(counts))
        return output * present.to(output.dtype)

    def forward(
        self, effects: ActiveEffectSetV4, *, effect_memory: Tensor, child_count: int, tower_count: int
    ) -> tuple[Tensor, Tensor]:
        static = nn.functional.embedding(effects.effect_vocab_id, effect_memory)
        memory = self.norm(
            static + self.runtime(effects.runtime_features) + self.source_owner(effects.source_owner_type)
        )
        memory = memory * effects.mask.unsqueeze(-1)
        return (
            self._scatter(memory, effects, parent_type=EFFECT_PARENT_CHILD, parent_count=child_count),
            self._scatter(memory, effects, parent_type=EFFECT_PARENT_TOWER, parent_count=tower_count),
        )


class GlobalCardEncoder(nn.Module):
    """Global semantics with a gated card-ID residual."""

    def __init__(
        self, config: ModelConfigV4, catalog: CardCatalogV1, mechanic_catalog: MechanicProfileCatalogV1
    ) -> None:
        super().__init__()
        self.config = config
        self.catalog = catalog
        if catalog.static_dim != config.card_static_dim:
            raise ValueError("CardCatalog static width does not match ModelConfigV4")
        if catalog.mechanic_dim != config.card_mechanic_dim:
            raise ValueError("CardCatalog mechanic width does not match ModelConfigV4")
        static, mechanics = catalog.feature_tensors()
        # Catalog ordering is deterministic but deliberately not a persistent
        # checkpoint binding.  A caller reconstructs it from the current card
        # catalog before loading model weights.
        self.register_buffer("catalog_static", static, persistent=False)
        self.register_buffer("catalog_mechanics", mechanics, persistent=False)
        self.register_buffer(
            "card_form_profile_ids",
            torch.tensor(mechanic_catalog.card_form_profile_ids, dtype=torch.long),
            persistent=False,
        )
        self.id_embedding = nn.Embedding(catalog.vocab_size, config.card_semantic_dim)
        self.static_encoder = mlp(config.card_static_dim, config.card_semantic_dim, config.card_semantic_dim)
        self.mechanic_encoder = mlp(config.card_mechanic_dim, config.card_semantic_dim, config.card_semantic_dim)
        self.form_embedding = nn.Embedding(config.form_type_count, config.card_semantic_dim)
        self.mechanic_profile = nn.Linear(config.mechanic_profile_dim, config.card_semantic_dim)
        self.id_gate = mlp(2 * config.card_semantic_dim, config.card_semantic_dim, 1)
        self.output_norm = nn.LayerNorm(config.card_semantic_dim)
        final_gate = self.id_gate[-1]
        if not isinstance(final_gate, nn.Linear):
            raise TypeError("ID residual gate must end in a linear layer")
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, -2.0)

    def forward(self, card_vocab_id: Tensor, runtime_form: Tensor, profile_memory: Tensor) -> Tensor:
        static = self.catalog_static[card_vocab_id]
        mechanics = self.catalog_mechanics[card_vocab_id]
        static_memory = self.static_encoder(static)
        mechanic_memory = self.mechanic_encoder(mechanics)
        generic = (
            static_memory
            + mechanic_memory
            + self.form_embedding(runtime_form)
            + self.mechanic_profile(
                nn.functional.embedding(self.card_form_profile_ids[card_vocab_id, runtime_form], profile_memory)
            )
        )
        gate = torch.sigmoid(self.id_gate(torch.cat((static_memory, mechanic_memory), dim=-1)))
        result = self.output_norm(generic + gate * self.id_embedding(card_vocab_id))
        return result * (card_vocab_id != PAD_CARD_VOCAB_ID).unsqueeze(-1)


class LocalChildSetEncoder(nn.Module):
    """Encode retained children and pool four slots inside each causal group."""

    def __init__(
        self, config: ModelConfigV4, card_encoder: GlobalCardEncoder, mechanic_catalog: MechanicProfileCatalogV1
    ) -> None:
        super().__init__()
        self.config = config
        self.card_encoder = card_encoder
        width = config.child_dim
        self.archetype_embedding = nn.Embedding(config.child_archetype_count, width)
        self.child_type_embedding = nn.Embedding(config.child_type_count, width)
        self.owner_embedding = nn.Embedding(config.owner_type_count, width)
        self.source_card = nn.Linear(config.card_semantic_dim, width)
        self.archetype_mechanics = nn.Linear(config.mechanic_profile_dim, width)
        self.id_gate = nn.Linear(2 * width, 1)
        self.active_effects = nn.Linear(config.effect_dim, width)
        self.register_buffer(
            "archetype_profile_ids",
            torch.tensor(mechanic_catalog.archetype_profile_ids, dtype=torch.long),
            persistent=False,
        )
        self.dynamic = mlp(config.child_feature_dim, width, width)
        self.input_norm = nn.LayerNorm(width)
        nn.init.zeros_(self.id_gate.weight)
        nn.init.constant_(self.id_gate.bias, -2.0)
        self.pool_queries = nn.Parameter(torch.empty(config.local_pool_slots, config.child_dim))
        nn.init.normal_(self.pool_queries, std=0.02)

    def forward(
        self, groups: BattleGroupSetV4, *, profile_memory: Tensor, active_effect_memory: Tensor
    ) -> tuple[Tensor, Tensor]:
        group_index = groups.child_group_index
        source_card_id = gather_scalar(groups.source_card_vocab_id, group_index, pad_value=PAD_CARD_VOCAB_ID)
        runtime_form = gather_scalar(groups.runtime_form, group_index)
        owner_type = gather_scalar(groups.owner_type, group_index)
        source_semantics = self.card_encoder(source_card_id, runtime_form, profile_memory)
        source_memory = self.source_card(source_semantics)
        mechanic_memory = self.archetype_mechanics(
            nn.functional.embedding(self.archetype_profile_ids[groups.child_archetype_id], profile_memory)
        )
        generic = (
            self.child_type_embedding(groups.child_type)
            + self.owner_embedding(owner_type)
            + source_memory
            + mechanic_memory
            + self.active_effects(active_effect_memory)
            + self.dynamic(groups.child_features)
        )
        gate = torch.sigmoid(self.id_gate(torch.cat((source_memory, mechanic_memory), dim=-1)))
        child_memory = self.input_norm(generic + gate * self.archetype_embedding(groups.child_archetype_id))
        child_memory = child_memory * groups.child_mask.unsqueeze(-1)

        batch, child_count, width = child_memory.shape
        group_count = groups.mask.shape[1]
        scores = torch.einsum("bnd,qd->bqn", child_memory, self.pool_queries) / math.sqrt(width)
        belongs = group_index[:, None, :] == torch.arange(group_count, device=group_index.device)[None, :, None]
        local_mask = belongs[:, :, None, :] & groups.child_mask[:, None, None, :] & groups.mask[:, :, None, None]
        local_scores = scores[:, None].expand(batch, group_count, self.config.local_pool_slots, child_count)
        weights = masked_pool_weights(local_scores, local_mask.expand_as(local_scores))
        slots = torch.einsum("bgqn,bnd->bgqd", weights, child_memory)
        return child_memory, slots


class BattleGroupEncoder(nn.Module):
    def __init__(self, config: ModelConfigV4, card_encoder: GlobalCardEncoder) -> None:
        super().__init__()
        self.card_encoder = card_encoder
        self.features = mlp(config.group_feature_dim, 128, 128)
        self.group_type = nn.Embedding(config.group_type_count, 128)
        self.owner_type = nn.Embedding(config.owner_type_count, 128)
        input_width = config.card_semantic_dim + 128 + config.local_pool_slots * config.child_dim
        self.output = mlp(input_width, config.group_dim, config.group_dim)
        self.norm = nn.LayerNorm(config.group_dim)

    def forward(self, groups: BattleGroupSetV4, local_slots: Tensor, *, profile_memory: Tensor) -> Tensor:
        card = self.card_encoder(groups.source_card_vocab_id, groups.runtime_form, profile_memory)
        result = self.norm(
            self.output(
                torch.cat(
                    (
                        card,
                        self.features(groups.features)
                        + self.group_type(groups.group_type)
                        + self.owner_type(groups.owner_type),
                        local_slots.flatten(2),
                    ),
                    dim=-1,
                )
            )
        )
        return result * groups.mask.unsqueeze(-1)


class CardTokenEncoder(nn.Module):
    def __init__(self, config: ModelConfigV4, card_encoder: GlobalCardEncoder) -> None:
        super().__init__()
        self.card_encoder = card_encoder
        self.runtime = mlp(config.card_runtime_feature_dim, 128, config.token_dim)
        self.role = nn.Linear(2, config.token_dim)
        self.owner = nn.Embedding(config.owner_type_count, config.token_dim)
        self.card = nn.Linear(config.card_semantic_dim, config.token_dim)
        self.norm = nn.LayerNorm(config.token_dim)

    def forward(self, cards: CardSetV4, *, owner_type: int, profile_memory: Tensor) -> Tensor:
        owner = torch.full_like(cards.card_vocab_id, owner_type)
        result = self.norm(
            self.card(self.card_encoder(cards.card_vocab_id, cards.runtime_form, profile_memory))
            + self.runtime(cards.runtime_features)
            + self.role(cards.role_bits.to(cards.runtime_features.dtype))
            + self.owner(owner)
        )
        return result * cards.mask.unsqueeze(-1)


class TowerEncoder(nn.Module):
    def __init__(self, config: ModelConfigV4, mechanic_catalog: MechanicProfileCatalogV1) -> None:
        super().__init__()
        self.feature = mlp(config.tower_feature_dim, 128, config.token_dim)
        self.position = nn.Linear(6, config.token_dim)
        self.tower_type = nn.Embedding(config.tower_type_count, config.token_dim)
        self.tower_troop_type = nn.Embedding(config.tower_troop_type_count, config.token_dim, padding_idx=0)
        self.owner_type = nn.Embedding(config.owner_type_count, config.token_dim)
        self.mechanic_profile = nn.Linear(config.mechanic_profile_dim, config.token_dim)
        self.active_effects = nn.Linear(config.effect_dim, config.token_dim)
        self.register_buffer(
            "tower_troop_profile_ids",
            torch.tensor(mechanic_catalog.tower_troop_profile_ids, dtype=torch.long),
            persistent=False,
        )
        self.norm = nn.LayerNorm(config.token_dim)

    def forward(self, towers: TowerSetV4, *, profile_memory: Tensor, active_effect_memory: Tensor) -> Tensor:
        geometry = torch.cat((towers.position, towers.extent), dim=-1)
        result = self.norm(
            self.feature(towers.features)
            + self.position(geometry)
            + self.tower_type(towers.tower_type)
            + self.tower_troop_type(towers.tower_troop_type)
            + self.owner_type(towers.owner_type)
            + self.mechanic_profile(
                nn.functional.embedding(self.tower_troop_profile_ids[towers.tower_troop_type], profile_memory)
            )
            + self.active_effects(active_effect_memory)
        )
        return result * towers.mask.unsqueeze(-1)


@dataclass(slots=True)
class GlobalTokenInputs:
    tokens: Tensor
    mask: Tensor
    position: Tensor
    extent: Tensor
    spatial_mask: Tensor
    own_start: int
    own_count: int
    group_start: int
    group_count: int


def assemble_global_tokens(
    *,
    towers: TowerSetV4,
    tower_tokens: Tensor,
    own_cards: CardSetV4,
    own_tokens: Tensor,
    opponent_cards: CardSetV4,
    opponent_tokens: Tensor,
    groups: BattleGroupSetV4,
    group_tokens: Tensor,
) -> GlobalTokenInputs:
    batch = tower_tokens.shape[0]
    device = tower_tokens.device
    dtype = tower_tokens.dtype

    def nonspatial(count: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.zeros(batch, count, 2, device=device, dtype=dtype),
            torch.zeros(batch, count, 4, device=device, dtype=dtype),
            torch.zeros(batch, count, device=device, dtype=torch.bool),
        )

    own_position, own_extent, own_spatial = nonspatial(own_tokens.shape[1])
    opponent_position, opponent_extent, opponent_spatial = nonspatial(opponent_tokens.shape[1])
    tower_spatial = towers.mask
    group_spatial = groups.mask
    own_start = tower_tokens.shape[1]
    group_start = own_start + own_tokens.shape[1] + opponent_tokens.shape[1]
    return GlobalTokenInputs(
        tokens=torch.cat((tower_tokens, own_tokens, opponent_tokens, group_tokens), dim=1),
        mask=torch.cat((towers.mask, own_cards.mask, opponent_cards.mask, groups.mask), dim=1),
        position=torch.cat((towers.position, own_position, opponent_position, groups.position), dim=1),
        extent=torch.cat((towers.extent, own_extent, opponent_extent, groups.extent), dim=1),
        spatial_mask=torch.cat((tower_spatial, own_spatial, opponent_spatial, group_spatial), dim=1),
        own_start=own_start,
        own_count=own_tokens.shape[1],
        group_start=group_start,
        group_count=group_tokens.shape[1],
    )


def spatial_pair_features(position: Tensor, extent: Tensor, spatial_mask: Tensor, config: ModelConfigV4) -> Tensor:
    # Positions and extents are observation inputs in production.  Preserve
    # the original autograd graph for standalone callers that explicitly make
    # either input differentiable; the lower-peak path below is for the real
    # non-differentiable PPO observations.
    if position.requires_grad or extent.requires_grad:
        x = position[..., 0]
        y = position[..., 1]
        dx = (x[:, :, None] - x[:, None, :]) / float(config.board_width)
        dy = (y[:, :, None] - y[:, None, :]) / float(config.board_height)
        abs_dx = dx.abs()
        abs_dy = dy.abs()
        distance = torch.sqrt(dx.square() + dy.square() + 1e-12)
        x_overlap = (
            torch.minimum(extent[:, :, None, 2], extent[:, None, :, 2])
            - torch.maximum(extent[:, :, None, 0], extent[:, None, :, 0])
        ).clamp_min(0.0) / float(config.board_width)
        y_overlap = (
            torch.minimum(extent[:, :, None, 3], extent[:, None, :, 3])
            - torch.maximum(extent[:, :, None, 1], extent[:, None, :, 1])
        ).clamp_min(0.0) / float(config.board_height)
        source_spatial = spatial_mask[:, :, None].expand_as(dx)
        target_spatial = spatial_mask[:, None, :].expand_as(dx)
        both = source_spatial & target_spatial
        result = torch.stack((dx, dy, abs_dx, abs_dy, distance, x_overlap, y_overlap, both.to(dx.dtype)), dim=-1)
        return result * both.unsqueeze(-1)

    x = position[..., 0]
    y = position[..., 1]
    dx = (x[:, :, None] - x[:, None, :]) / float(config.board_width)
    dy = (y[:, :, None] - y[:, None, :]) / float(config.board_height)
    # Allocate the final feature tensor before materializing the derived
    # pairwise channels.  At PPO training scale every [B,N,N] temporary is
    # hundreds of MiB; retaining all seven inputs until torch.stack allocates
    # its copy doubles the peak and can exhaust device memory.
    result = dx.new_empty((*dx.shape, 8))
    result[..., 0] = dx
    result[..., 1] = dy
    result[..., 2] = dx.abs()
    result[..., 3] = dy.abs()
    result[..., 4] = torch.sqrt(dx.square() + dy.square() + 1e-12)
    del dx, dy
    x_overlap = (
        torch.minimum(extent[:, :, None, 2], extent[:, None, :, 2])
        - torch.maximum(extent[:, :, None, 0], extent[:, None, :, 0])
    ).clamp_min(0.0) / float(config.board_width)
    result[..., 5] = x_overlap
    del x_overlap
    y_overlap = (
        torch.minimum(extent[:, :, None, 3], extent[:, None, :, 3])
        - torch.maximum(extent[:, :, None, 1], extent[:, None, :, 1])
    ).clamp_min(0.0) / float(config.board_height)
    result[..., 6] = y_overlap
    del y_overlap
    source_spatial = spatial_mask[:, :, None].expand(result.shape[:-1])
    target_spatial = spatial_mask[:, None, :].expand(result.shape[:-1])
    both = source_spatial & target_spatial
    result[..., 7] = both
    return result.mul_(both.unsqueeze(-1))


class SparseRelationAttentionBlock(nn.Module):
    """Pre-norm attention with additive multi-relation and spatial bias."""

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.token_dim // config.attention_heads
        self.norm1 = nn.LayerNorm(config.token_dim)
        self.qkv = nn.Linear(config.token_dim, 3 * config.token_dim)
        self.output = nn.Linear(config.token_dim, config.token_dim)
        self.relation_bias = nn.Embedding(config.relation_type_count, config.attention_heads)
        self.spatial_bias = mlp(config.spatial_pair_dim, 32, config.attention_heads)
        self.norm2 = nn.LayerNorm(config.token_dim)
        self.ffn = mlp(config.token_dim, config.transformer_ffn_dim, config.token_dim)

    def _sparse_bias(self, edges: RelationEdgesV4, *, token_count: int, dtype: torch.dtype) -> Tensor:
        batch, edge_count = edges.mask.shape
        values = self.relation_bias(edges.relation_type).to(dtype)
        batch_index = torch.arange(batch, device=edges.mask.device)[:, None]
        # The learned STATE token occupies row zero, so external token addresses
        # are shifted by one only inside this neural attention implementation.
        source = edges.source + 1
        target = edges.target + 1
        flat_index = batch_index * token_count * token_count + source * token_count + target
        flat = torch.zeros(batch * token_count * token_count, self.heads, device=values.device, dtype=dtype)
        if edge_count:
            if values.is_cuda and torch.cuda.is_current_stream_capturing():
                # Boolean advanced indexing emits a data-dependent ``nonzero``
                # allocation. During graph capture, add fixed masked rows;
                # padded addresses are contractually zero and therefore neutral.
                flat.index_add_(
                    0,
                    flat_index.reshape(-1),
                    (values * edges.mask.unsqueeze(-1).to(values.dtype)).reshape(-1, self.heads),
                )
            else:
                # Preserve the established eager/training kernel and numerical
                # path outside the dedicated live CUDA Graph capture.
                flat.index_add_(0, flat_index[edges.mask], values[edges.mask])
        return flat.view(batch, token_count, token_count, self.heads).permute(0, 3, 1, 2)

    def forward(self, tokens: Tensor, token_mask: Tensor, edges: RelationEdgesV4, pair_features: Tensor) -> Tensor:
        batch, count, width = tokens.shape
        normalized = self.norm1(tokens)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)

        def split_heads(item: Tensor) -> Tensor:
            return item.view(batch, count, self.heads, self.head_dim).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self._sparse_bias(edges, token_count=count, dtype=scores.dtype)
        scores = scores + self.spatial_bias(pair_features).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~token_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, count, width)
        tokens = tokens + self.output(attended) * token_mask.unsqueeze(-1)
        tokens = tokens + self.ffn(self.norm2(tokens)) * token_mask.unsqueeze(-1)
        return tokens * token_mask.unsqueeze(-1)


class RelationTransformerV4(nn.Module):
    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.config = config
        self.state_token = nn.Parameter(torch.empty(1, 1, config.token_dim))
        nn.init.normal_(self.state_token, std=0.02)
        self.blocks = nn.ModuleList(SparseRelationAttentionBlock(config) for _ in range(config.transformer_layers))
        self.output_norm = nn.LayerNorm(config.token_dim)

    def forward(
        self, inputs: GlobalTokenInputs, edges: RelationEdgesV4
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = inputs.tokens.shape[0]
        state = self.state_token.expand(batch, -1, -1)
        tokens = torch.cat((state, inputs.tokens), dim=1)
        token_mask = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=inputs.mask.device), inputs.mask), dim=1)
        zero_position = torch.zeros(batch, 1, 2, dtype=inputs.position.dtype, device=inputs.position.device)
        zero_extent = torch.zeros(batch, 1, 4, dtype=inputs.extent.dtype, device=inputs.extent.device)
        zero_spatial = torch.zeros(batch, 1, dtype=torch.bool, device=inputs.spatial_mask.device)
        pair = spatial_pair_features(
            torch.cat((zero_position, inputs.position), dim=1),
            torch.cat((zero_extent, inputs.extent), dim=1),
            torch.cat((zero_spatial, inputs.spatial_mask), dim=1),
            self.config,
        )
        for block in self.blocks:
            tokens = block(tokens, token_mask, edges, pair)
        tokens = self.output_norm(tokens)
        own_start = 1 + inputs.own_start
        group_start = 1 + inputs.group_start
        return (
            tokens[:, 0],
            tokens,
            token_mask,
            tokens[:, own_start : own_start + inputs.own_count],
            tokens[:, group_start : group_start + inputs.group_count],
        )


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = torch.nn.functional.silu(self.norm1(inputs))
        residual = self.conv1(residual)
        residual = torch.nn.functional.silu(self.norm2(residual))
        return inputs + self.conv2(residual)


class SpatialScatterEncoder(nn.Module):
    """Bilinear child scatter plus an exact-resolution spatial ResNet."""

    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.config = config
        self.child_value = nn.Linear(config.child_dim, config.learned_scatter_channels)
        input_channels = config.explicit_spatial_channels + config.learned_scatter_channels + 3
        self.input = nn.Conv2d(input_channels, config.spatial_hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *(ResidualSpatialBlock(config.spatial_hidden_channels) for _ in range(config.spatial_res_blocks))
        )
        self.summary = mlp(2 * config.spatial_hidden_channels, config.spatial_summary_dim, config.spatial_summary_dim)

    def scatter(
        self,
        child_memory: Tensor,
        child_position: Tensor,
        child_extent: Tensor,
        child_radius: Tensor,
        child_mask: Tensor,
    ) -> Tensor:
        config = self.config
        values = self.child_value(child_memory)
        extent_width = child_extent[..., 2] - child_extent[..., 0]
        extent_height = child_extent[..., 3] - child_extent[..., 1]
        shaped = child_mask & ((child_radius > 0.0) | (extent_width > 0.0) | (extent_height > 0.0))
        point_mask = child_mask & ~shaped
        x = child_position[..., 0].clamp(0.0, config.board_width - 1.0)
        y = child_position[..., 1].clamp(0.0, config.board_height - 1.0)
        x0 = torch.floor(x).to(torch.long)
        y0 = torch.floor(y).to(torch.long)
        x1 = (x0 + 1).clamp_max(config.board_width - 1)
        y1 = (y0 + 1).clamp_max(config.board_height - 1)
        wx = x - x0.to(x.dtype)
        wy = y - y0.to(y.dtype)
        output = values.new_zeros(
            values.shape[0], config.learned_scatter_channels, config.board_height * config.board_width
        )
        for cell_x, cell_y, weight in (
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x1, y0, wx * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y1, wx * wy),
        ):
            flat_index = cell_y * config.board_width + cell_x
            source = (
                values * weight.to(values.dtype).unsqueeze(-1) * point_mask.unsqueeze(-1).to(values.dtype)
            ).transpose(1, 2)
            output.scatter_add_(2, flat_index[:, None, :].expand(-1, config.learned_scatter_channels, -1), source)
        result = output.view(values.shape[0], config.learned_scatter_channels, config.board_height, config.board_width)
        cell_x = torch.arange(config.board_width, device=values.device, dtype=values.dtype) + 0.5
        cell_y = torch.arange(config.board_height, device=values.device, dtype=values.dtype) + 0.5
        grid_y, grid_x = torch.meshgrid(cell_y, cell_x, indexing="ij")
        if not (values.is_cuda and torch.cuda.is_current_stream_capturing()):
            shaped_entries = shaped.nonzero(as_tuple=False)
            shaped_batch = shaped_entries[:, 0]
            shaped_child = shaped_entries[:, 1]
            shaped_position = child_position[shaped_batch, shaped_child]
            shaped_extent = child_extent[shaped_batch, shaped_child]
            shaped_radius = child_radius[shaped_batch, shaped_child]
            circle = (grid_x[None] - shaped_position[:, 0, None, None]).square() + (
                grid_y[None] - shaped_position[:, 1, None, None]
            ).square() <= shaped_radius[:, None, None].square() + 1e-6
            rectangle = (
                (grid_x[None] >= shaped_extent[:, 0, None, None])
                & (grid_x[None] < shaped_extent[:, 2, None, None])
                & (grid_y[None] >= shaped_extent[:, 1, None, None])
                & (grid_y[None] < shaped_extent[:, 3, None, None])
            )
            coverage = torch.where((shaped_radius > 0.0)[:, None, None], circle, rectangle)
            covered_cells = coverage.nonzero(as_tuple=False)
            shaped_row = covered_cells[:, 0]
            covered_batch = shaped_batch[shaped_row]
            covered_child = shaped_child[shaped_row]
            flat_cell = (
                covered_batch * config.board_height * config.board_width
                + covered_cells[:, 1] * config.board_width
                + covered_cells[:, 2]
            )
            coverage_output = values.new_zeros(
                values.shape[0] * config.board_height * config.board_width, config.learned_scatter_channels
            )
            coverage_output.index_add_(0, flat_cell, values[covered_batch, covered_child])
            coverage_output = coverage_output.view(
                values.shape[0], config.board_height, config.board_width, config.learned_scatter_channels
            ).permute(0, 3, 1, 2)
            return result + coverage_output

        # CUDA Graph capture needs a fixed [B,C,H,W] coverage tensor instead
        # of the established data-dependent sparse eager path above.
        circle = (grid_x[None, None] - child_position[..., 0, None, None]).square() + (
            grid_y[None, None] - child_position[..., 1, None, None]
        ).square() <= child_radius[..., None, None].square() + 1e-6
        rectangle = (
            (grid_x[None, None] >= child_extent[..., 0, None, None])
            & (grid_x[None, None] < child_extent[..., 2, None, None])
            & (grid_y[None, None] >= child_extent[..., 1, None, None])
            & (grid_y[None, None] < child_extent[..., 3, None, None])
        )
        coverage = torch.where((child_radius > 0.0)[..., None, None], circle, rectangle) & shaped[..., None, None]
        coverage_output = torch.einsum("bcyx,bck->bkyx", coverage.to(values.dtype), values)
        return result + coverage_output

    def coordinate_planes(self, reference: Tensor) -> Tensor:
        config = self.config
        y = torch.linspace(-1.0, 1.0, config.board_height, device=reference.device)
        x = torch.linspace(-1.0, 1.0, config.board_width, device=reference.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        river_side = torch.sign(grid_y)
        return torch.stack((grid_x, grid_y, river_side), dim=0).to(reference.dtype)

    def forward(
        self,
        explicit_planes: Tensor,
        child_memory: Tensor,
        child_position: Tensor,
        child_extent: Tensor,
        child_radius: Tensor,
        child_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        learned = self.scatter(child_memory, child_position, child_extent, child_radius, child_mask)
        coordinates = self.coordinate_planes(explicit_planes).unsqueeze(0).expand(explicit_planes.shape[0], -1, -1, -1)
        spatial = self.blocks(self.input(torch.cat((explicit_planes, learned, coordinates), dim=1)))
        average = spatial.mean(dim=(2, 3))
        maximum = spatial.amax(dim=(2, 3))
        return spatial, self.summary(torch.cat((average, maximum), dim=-1))


class EventSetEncoder(nn.Module):
    def __init__(self, config: ModelConfigV4, card_encoder: GlobalCardEncoder) -> None:
        super().__init__()
        self.card_encoder = card_encoder
        self.event_type = nn.Embedding(config.event_type_count, 128)
        self.owner_type = nn.Embedding(config.owner_type_count, 128)
        self.card = nn.Linear(config.card_semantic_dim, 128)
        self.source_group = nn.Linear(config.group_dim, 128)
        self.target_token = nn.Linear(config.token_dim, 128)
        self.numeric = mlp(config.event_feature_dim, 128, 128)
        self.norm = nn.LayerNorm(128)
        self.pool = LearnedSetPool(128, config.event_summary_dim, slots=2)

    def forward(
        self, events: EventSetV4, *, group_memory: Tensor, token_memory: Tensor, profile_memory: Tensor
    ) -> Tensor:
        source = gather_memory(group_memory, events.source_group_index)
        target_address = torch.where(
            events.target_token_index >= 0, events.target_token_index + 1, events.target_token_index
        )
        target = gather_memory(token_memory, target_address)
        memory = self.norm(
            self.event_type(events.event_type)
            + self.owner_type(events.owner_type)
            + self.card(self.card_encoder(events.source_card_vocab_id, events.source_form, profile_memory))
            + self.source_group(source)
            + self.target_token(target)
            + self.numeric(events.features)
        )
        memory = memory * events.mask.unsqueeze(-1)
        return self.pool(memory, events.mask)


class GenericScalarEncoder(nn.Module):
    def __init__(self, config: ModelConfigV4) -> None:
        super().__init__()
        self.network = mlp(config.generic_scalar_dim, 128, config.scalar_summary_dim)

    def forward(self, scalars: Tensor) -> Tensor:
        return self.network(scalars)


class PreviousActionEncoder(nn.Module):
    def __init__(
        self, config: ModelConfigV4, card_encoder: GlobalCardEncoder, mechanic_catalog: MechanicProfileCatalogV1
    ) -> None:
        super().__init__()
        self.config = config
        self.card_encoder = card_encoder
        self.variant = nn.Embedding(2, 32)
        self.visible = nn.Linear(config.card_semantic_dim, 128)
        self.effective = nn.Linear(config.card_semantic_dim, 128)
        self.ability = nn.Embedding(config.ability_vocab_size, 64)
        self.ability_mechanics = nn.Linear(config.mechanic_profile_dim, 64)
        ability_profile_ids = list(mechanic_catalog.ability_profile_ids)
        ability_profile_ids.extend([0] * (config.ability_vocab_size - len(ability_profile_ids)))
        self.register_buffer(
            "ability_profile_ids", torch.tensor(ability_profile_ids, dtype=torch.long), persistent=False
        )
        self.target = mlp(3, 64, 128)
        self.source = mlp(3, 64, 128)
        self.delay_offset = nn.Embedding(len(config.delay_offset_ms) + 1, 32)
        self.step = nn.Embedding(config.max_micro_actions, 32)
        self.action = mlp(32 + 128 + 128 + 64 + 128 + 128 + 32 + 32, 256, 128)
        self.pool = LearnedSetPool(128, config.previous_action_dim)
        self.count = nn.Embedding(config.max_micro_actions + 1, config.previous_action_dim)
        self.norm = nn.LayerNorm(config.previous_action_dim)

    def forward(self, previous: PreviousActionV4, *, profile_memory: Tensor) -> Tensor:
        normal_form = torch.zeros_like(previous.effective_form)
        target_valid = previous.target_cell >= 0
        safe_target = previous.target_cell.clamp_min(0)
        x = safe_target.remainder(self.config.board_width).to(torch.float32)
        y = torch.div(safe_target, self.config.board_width, rounding_mode="floor").to(torch.float32)
        target_features = torch.stack(
            (
                x / max(1, self.config.board_width - 1),
                y / max(1, self.config.board_height - 1),
                target_valid.to(x.dtype),
            ),
            dim=-1,
        ).to(previous.visible_card_vocab_id.device)
        source_features = torch.cat(
            (
                torch.stack(
                    (
                        previous.source_position[..., 0] / self.config.board_width,
                        previous.source_position[..., 1] / self.config.board_height,
                    ),
                    dim=-1,
                ),
                previous.source_mask.unsqueeze(-1).to(previous.source_position.dtype),
            ),
            dim=-1,
        )
        delay_offset_index = torch.where(
            previous.action_mask, previous.delay_offset_bin + 1, torch.zeros_like(previous.delay_offset_bin)
        )
        action = self.action(
            torch.cat(
                (
                    self.step(
                        torch.arange(self.config.max_micro_actions, device=previous.variant.device)[None].expand_as(
                            previous.variant
                        )
                    ),
                    self.variant(previous.variant.clamp_min(0)),
                    self.visible(self.card_encoder(previous.visible_card_vocab_id, normal_form, profile_memory)),
                    self.effective(
                        self.card_encoder(previous.effective_card_vocab_id, previous.effective_form, profile_memory)
                    ),
                    self.ability(previous.ability_vocab_id)
                    + self.ability_mechanics(
                        nn.functional.embedding(self.ability_profile_ids[previous.ability_vocab_id], profile_memory)
                    ),
                    self.target(target_features),
                    self.source(source_features),
                    self.delay_offset(delay_offset_index),
                ),
                dim=-1,
            )
        )
        action = action * previous.action_mask.unsqueeze(-1)
        return self.norm(self.pool(action, previous.action_mask) + self.count(previous.micro_action_count))


class CandidateEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfigV4,
        card_encoder: GlobalCardEncoder,
        *,
        ability_features: Tensor,
        mechanic_catalog: MechanicProfileCatalogV1,
    ) -> None:
        super().__init__()
        self.config = config
        self.card_encoder = card_encoder
        if tuple(ability_features.shape) != (config.ability_vocab_size, config.ability_feature_dim):
            raise ValueError("ability feature catalog has the wrong shape")
        self.register_buffer("ability_features", ability_features.to(torch.float32), persistent=False)
        self.variant = nn.Embedding(2, 32)
        self.ability_id = nn.Embedding(config.ability_vocab_size, 128)
        self.ability_static = nn.Linear(config.ability_feature_dim, 128)
        self.ability_mechanics = nn.Linear(config.mechanic_profile_dim, 128)
        ability_profile_ids = list(mechanic_catalog.ability_profile_ids)
        ability_profile_ids.extend([0] * (config.ability_vocab_size - len(ability_profile_ids)))
        self.register_buffer(
            "ability_profile_ids", torch.tensor(ability_profile_ids, dtype=torch.long), persistent=False
        )
        self.runtime = nn.Linear(config.candidate_runtime_feature_dim, 128)
        input_width = (
            config.token_dim + 2 * config.card_semantic_dim + 32 + 128 + config.group_dim + config.child_dim + 128
        )
        self.output = mlp(input_width, 512, config.candidate_dim)
        self.norm = nn.LayerNorm(config.candidate_dim)
        self.pool = LearnedSetPool(config.candidate_dim, config.candidate_summary_dim)

    def forward(
        self,
        candidates: ActionCandidatesV4,
        *,
        own_card_memory: Tensor,
        group_memory: Tensor,
        child_memory: Tensor,
        profile_memory: Tensor,
    ) -> tuple[Tensor, Tensor]:
        own = gather_memory(own_card_memory, candidates.own_card_row)
        group = gather_memory(group_memory, candidates.source_group_index)
        child = gather_memory(child_memory, candidates.source_child_index)
        visible = self.card_encoder(
            candidates.visible_card_vocab_id, torch.zeros_like(candidates.effective_form), profile_memory
        )
        effective = self.card_encoder(candidates.effective_card_vocab_id, candidates.effective_form, profile_memory)
        ability = (
            self.ability_id(candidates.ability_vocab_id)
            + self.ability_static(self.ability_features[candidates.ability_vocab_id])
            + self.ability_mechanics(
                nn.functional.embedding(self.ability_profile_ids[candidates.ability_vocab_id], profile_memory)
            )
        )
        memory = self.norm(
            self.output(
                torch.cat(
                    (
                        own,
                        visible,
                        effective,
                        self.variant(candidates.variant),
                        self.runtime(candidates.runtime_features),
                        group,
                        child,
                        ability,
                    ),
                    dim=-1,
                )
            )
        )
        memory = memory * candidates.mask.unsqueeze(-1)
        return memory, self.pool(memory, candidates.mask)
