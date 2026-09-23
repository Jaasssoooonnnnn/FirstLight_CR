from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from native_runner.contracts import ObservationTier, ObservationV1, PlayerStateV1
from native_runner.training.v4.policy_session import PolicySessionV4, resolve_v4_checkpoint
from native_runner.training.v4.native_actions import DecodedActionSequenceV4


def _observation(tick: int = 90) -> ObservationV1:
    return ObservationV1(
        tier=ObservationTier.FAIR,
        owner=0,
        tick=tick,
        episode_id="offline-v4-test",
        players=(
            PlayerStateV1(owner=0, elixir_exact=6.0, elixir_visible=6.0, private_state_visible=True),
            PlayerStateV1(owner=1),
        ),
    )


def test_checkpoint_directory_resolves_the_latest_step(tmp_path: Path) -> None:
    older = tmp_path / "checkpoint-step-00000010.pt"
    newer = tmp_path / "checkpoint-step-00000100.pt"
    older.touch()
    newer.touch()

    assert resolve_v4_checkpoint(tmp_path) == newer.resolve()
    assert resolve_v4_checkpoint(older) == older.resolve()


class _FakeActions:
    def __init__(self) -> None:
        self.validations = 0

    def validate(self, _config: object, *, candidate_count: int) -> None:
        assert candidate_count == 1
        self.validations += 1


class _FakeBatch:
    def __init__(self, *, ability: bool) -> None:
        self.candidates = SimpleNamespace(mask=torch.tensor([[True]]), variant=torch.tensor([[1 if ability else 0]]))

    def to_model_input(self, _device: torch.device) -> _FakeBatch:
        copied = _FakeBatch(ability=bool(self.candidates.variant[0, 0]))
        copied.candidates.mask = self.candidates.mask.clone()
        copied.candidates.variant = self.candidates.variant.clone()
        return copied


class _FakeTensorizer:
    actor_owner = 0
    config = object()
    catalog = object()
    deck = tuple(range(1, 9))
    card_costs: dict[int, float] = {}
    ability_id_by_vocab_id: dict[int, str] = {}
    perspective = SimpleNamespace(horizontal_mirror=False, hand_slot_permutation=(0, 1, 2, 3))

    def __init__(self) -> None:
        self.started = None
        self.ended = 0
        self.recorded: list[tuple[object, object]] = []
        self.batches: list[_FakeBatch] = []

    def start_episode(self, observation: ObservationV1, *, initial_elixir: dict[int, float]) -> None:
        self.started = (observation, initial_elixir)

    def end_episode(self) -> None:
        self.ended += 1

    def tensorize(self, _observation: ObservationV1, *, validate: bool) -> _FakeBatch:
        assert validate is False
        batch = _FakeBatch(ability=True)
        self.batches.append(batch)
        return batch

    def record_action(self, actions: object, batch: object, *, row: int, validate: bool) -> None:
        assert row == 0
        assert validate is False
        self.recorded.append((actions, batch))


class _FakeModel:
    def __init__(self) -> None:
        self.actions = _FakeActions()
        self.calls: list[tuple[object, bool, bool]] = []

    def initial_state(self, batch_size: int, *, device: torch.device) -> object:
        assert batch_size == 1
        assert device.type == "cpu"
        return "state-0"

    def act(self, batch: _FakeBatch, state: object, *, episode_start: torch.Tensor, validate: bool) -> object:
        # Offline inference must expose the same legal Ability candidate as IL.
        assert bool(batch.candidates.mask[0, 0])
        self.calls.append((state, bool(episode_start[0]), validate))
        return SimpleNamespace(actions=self.actions, next_state=f"state-{len(self.calls)}", value=torch.tensor([0.25]))


def test_offline_session_keeps_lstm_actions_and_ability_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    tensorizer = _FakeTensorizer()
    model = _FakeModel()
    decoded = DecodedActionSequenceV4(owner=0)

    def fake_decode(_actions: object, candidates: object, **_kwargs: object) -> DecodedActionSequenceV4:
        # Decoding uses the unmodified semantic candidate set.
        assert bool(candidates.mask[0, 0])
        return decoded

    monkeypatch.setattr("native_runner.training.v4.policy_session.decode_action_sequence_v4", fake_decode)
    session = PolicySessionV4(
        model,  # type: ignore[arg-type]
        tensorizer,  # type: ignore[arg-type]
        device="cpu",
    )
    session.start_episode(_observation(), initial_elixir={0: 6.0, 1: 6.0})

    first = session.decide(_observation())
    second = session.decide(_observation(95))
    session.end_episode()

    assert first.decoded is decoded
    assert second.decoded is decoded
    assert model.calls == [("state-0", True, False), ("state-1", False, False)]
    assert len(tensorizer.recorded) == 2
    assert all(
        recorded_batch is expected_batch
        for (_actions, recorded_batch), expected_batch in zip(tensorizer.recorded, tensorizer.batches, strict=True)
    )
    assert model.actions.validations == 2
    assert tensorizer.ended == 1


@pytest.mark.parametrize("can_deploy", [False, True])
def test_forced_play_only_overrides_gate_when_a_deploy_is_legal(monkeypatch, can_deploy: bool) -> None:
    @dataclass(frozen=True)
    class Candidates:
        mask: torch.Tensor
        variant: torch.Tensor

    @dataclass(frozen=True)
    class Batch:
        candidates: Candidates

        def to_model_input(self, _device):
            return self

    batch = Batch(Candidates(
        mask=torch.tensor([[True, True]]), variant=torch.tensor([[1, 0]]),
    ))

    class Actions:
        def validate(self, _config, *, candidate_count):
            assert candidate_count == 2

    class Tensorizer(_FakeTensorizer):
        def tensorize(self, _observation, *, validate):
            return batch

    class Model:
        def initial_state(self, *_args, **_kwargs):
            return "initial"

        def forward(self, selected, *_args, **_kwargs):
            assert selected.candidates.mask.tolist() == [[False, True]]
            return "context"

        def act_after_preselected_act(self, selected, context, *, validate):
            assert context == "context"
            return SimpleNamespace(actions=Actions(), next_state="forced")

        def act(self, selected, *_args, **_kwargs):
            assert selected is batch
            return SimpleNamespace(actions=Actions(), next_state="normal")

    class Legality:
        def __init__(self, *_args):
            pass

        def candidate_mask(self):
            return torch.tensor([[False, can_deploy]])

    monkeypatch.setattr("native_runner.training.v4.policy_session.ShadowCandidateLegality", Legality)
    monkeypatch.setattr(
        "native_runner.training.v4.policy_session.decode_action_sequence_v4",
        lambda *_args, **_kwargs: DecodedActionSequenceV4(owner=0),
    )
    session = PolicySessionV4(Model(), Tensorizer(), device="cpu")  # type: ignore[arg-type]
    session.start_episode(_observation(), initial_elixir={0: 6.0, 1: 6.0})

    session.decide(_observation(), force_act=True)

    assert session.state == ("forced" if can_deploy else "normal")
