from __future__ import annotations

import pytest

from native_runner.cr_native_env import (
    CARD_FORM_LABELS,
    HandAction,
    NativeClashEnv,
    ResidentNativeClashEnv,
    RunnerError,
    _observed_command_card_id,
    decode_native_card_parameter,
)


def _packed(*, form_code: int, deck_slot: int, cost: int, opaque: int = 0) -> int:
    return (
        (form_code & 0xF)
        | (opaque & 0x003FFFF0)
        | ((deck_slot + 1) << 22)
        | (cost << 28)
    )


@pytest.mark.parametrize(
    ("form_code", "form_name"),
    tuple(enumerate(CARD_FORM_LABELS)),
)
def test_decode_native_card_parameter_preserves_exact_form(
    form_code: int,
    form_name: str,
) -> None:
    packed = _packed(
        form_code=form_code,
        deck_slot=6,
        cost=9,
        opaque=0x00123450,
    )

    descriptor = decode_native_card_parameter(
        packed,
        expected_deck_slot=6,
        expected_cost=9,
    )

    assert descriptor.packed == packed
    assert descriptor.deck_slot == 6
    assert descriptor.cost == 9
    assert descriptor.form_code == form_code
    assert descriptor.form_name == form_name


@pytest.mark.parametrize(
    "packed",
    (
        -1,
        0x1_0000_0000,
        True,
        "0",
        _packed(form_code=6, deck_slot=0, cost=3),
        _packed(form_code=0, deck_slot=-1, cost=3),
    ),
)
def test_decode_native_card_parameter_rejects_invalid_words(packed: object) -> None:
    with pytest.raises(ValueError):
        decode_native_card_parameter(packed)


def test_decode_native_card_parameter_rejects_mismatched_observation() -> None:
    packed = _packed(form_code=1, deck_slot=3, cost=4)

    with pytest.raises(ValueError, match="deck slot"):
        decode_native_card_parameter(packed, expected_deck_slot=2)
    with pytest.raises(ValueError, match="cost"):
        decode_native_card_parameter(packed, expected_cost=5)


def test_native_command_card_id_prefers_exact_effective_selection() -> None:
    assert _observed_command_card_id(
        {
            "cardId": 28_000_006,
            "commandCardId": 26_000_014,
        }
    ) == 26_000_014
    assert _observed_command_card_id({"cardId": 26_000_014}) == 26_000_014


@pytest.mark.parametrize("value", (0, -1, True, None, "26000014"))
def test_native_command_card_id_rejects_invalid_effective_selection(
    value: object,
) -> None:
    with pytest.raises(RunnerError, match="commandCardId"):
        _observed_command_card_id(
            {
                "cardId": 28_000_006,
                "commandCardId": value,
            }
        )


def test_immediate_play_recovers_one_tick_hand_descriptor_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = NativeClashEnv()
    before = {
        "tick": 130,
        "queuedCommands": 0,
        "players": [
            {
                "owner": 0,
                "accountId": 123,
                "hand": [
                    {
                        "handIndex": 0,
                        "deckSlot": 0,
                        "cardId": 28_000_002,
                        "commandCardId": 28_000_002,
                        "cardParameter": _packed(
                            form_code=0,
                            deck_slot=0,
                            cost=2,
                        ),
                        "cost": 2,
                    }
                ],
            }
        ],
    }
    settled = {
        "tick": 132,
        "queuedCommands": 0,
        "players": [{"owner": 0, "accountId": 123, "hand": []}],
    }
    observations = iter((before, settled))
    injected: list[str] = []
    step_calls: list[int] = []

    monkeypatch.setattr(env, "status", lambda: {"mode": "headless"})
    monkeypatch.setattr(env, "observe", lambda: next(observations))
    monkeypatch.setattr(
        env,
        "_request",
        lambda command: injected.append(command) or {"ok": True},
    )

    def step(ticks: int) -> dict[str, object]:
        step_calls.append(ticks)
        if len(step_calls) < 4:
            raise RunnerError("native hand card is invalid")
        return settled

    monkeypatch.setattr(env, "step", step)

    result = env.play_immediate(HandAction(0, 0, 14_500, 17_500))

    assert result == settled
    assert step_calls == [1, 1, 1, 1]
    assert len(injected) == 1


def test_absolute_hand_queue_uses_the_native_injection_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = NativeClashEnv()
    observation = {
        "tick": 130,
        "players": [
            {
                "owner": 0,
                "accountId": 123,
                "hand": [
                    {
                        "handIndex": 0,
                        "deckSlot": 0,
                        "cardId": 26_000_001,
                        "commandCardId": 26_000_001,
                        "cardParameter": _packed(
                            form_code=0,
                            deck_slot=0,
                            cost=3,
                        ),
                        "cost": 3,
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(env, "observe", lambda: observation)
    monkeypatch.setattr(
        env,
        "inject_command",
        lambda _command: {"ok": True, "tick": 130},
    )

    receipt = env.queue_hand_action_at(
        HandAction(0, 0, 9_000, 10_000),
        execute_tick=131,
    )

    assert receipt["queuedAtTick"] == 130
    assert receipt["commandAgeBoundaryTick"] == 130
    assert receipt["executeTick"] == 131


def test_absolute_hand_queue_rejects_a_late_native_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = NativeClashEnv()
    observation = {
        "tick": 130,
        "players": [
            {
                "owner": 0,
                "accountId": 123,
                "hand": [
                    {
                        "handIndex": 0,
                        "deckSlot": 0,
                        "cardId": 26_000_001,
                        "commandCardId": 26_000_001,
                        "cardParameter": _packed(
                            form_code=0,
                            deck_slot=0,
                            cost=3,
                        ),
                        "cost": 3,
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(env, "observe", lambda: observation)
    monkeypatch.setattr(
        env,
        "inject_command",
        lambda _command: {"ok": True, "tick": 131},
    )

    with pytest.raises(RunnerError, match="not queued before"):
        env.queue_hand_action_at(
            HandAction(0, 0, 9_000, 10_000),
            execute_tick=131,
        )


def test_observation_anchored_hand_queue_falls_back_after_missed_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = NativeClashEnv()
    observation = {
        "tick": 130,
        "players": [
            {
                "owner": 0,
                "accountId": 123,
                "hand": [
                    {
                        "handIndex": 0,
                        "deckSlot": 0,
                        "cardId": 26_000_001,
                        "commandCardId": 26_000_001,
                        "cardParameter": _packed(form_code=0, deck_slot=0, cost=3),
                        "cost": 3,
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(env, "observe", lambda: observation)
    monkeypatch.setattr(env, "inject_command", lambda _command: {"ok": True, "tick": 130})

    receipt = env.queue_hand_action_at(
        HandAction(0, 0, 9_000, 10_000),
        execute_tick=129,
        clamp_late=True,
    )

    assert receipt["queuedAtTick"] == 130
    assert receipt["executeTick"] == 150


def test_resident_hand_queue_accepts_legacy_success_receipt_without_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = ResidentNativeClashEnv(env_id=3)
    observation = {
        "tick": 130,
        "players": [
            {
                "owner": 0,
                "accountId": 123,
                "hand": [
                    {
                        "handIndex": 0,
                        "deckSlot": 0,
                        "cardId": 26_000_001,
                        "commandCardId": 26_000_001,
                        "cardParameter": _packed(
                            form_code=0,
                            deck_slot=0,
                            cost=3,
                        ),
                        "cost": 3,
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(env, "observe", lambda: observation)
    monkeypatch.setattr(
        env,
        "inject_command",
        lambda _command: {
            "ok": True,
            "mode": "resident-headless",
            "injected": True,
        },
    )

    receipt = env.queue_hand_action_at(
        HandAction(0, 0, 9_000, 10_000),
        execute_tick=131,
    )

    assert receipt["queuedAtTick"] == 130
    assert receipt["executeTick"] == 131


def test_nonresident_hand_queue_still_requires_injection_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = NativeClashEnv()
    observation = {
        "tick": 130,
        "players": [
            {
                "owner": 0,
                "accountId": 123,
                "hand": [
                    {
                        "handIndex": 0,
                        "deckSlot": 0,
                        "cardId": 26_000_001,
                        "commandCardId": 26_000_001,
                        "cardParameter": _packed(
                            form_code=0,
                            deck_slot=0,
                            cost=3,
                        ),
                        "cost": 3,
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(env, "observe", lambda: observation)
    monkeypatch.setattr(
        env,
        "inject_command",
        lambda _command: {"ok": True, "injected": True},
    )

    with pytest.raises(RunnerError, match="invalid hand queued tick"):
        env.queue_hand_action_at(
            HandAction(0, 0, 9_000, 10_000),
            execute_tick=131,
        )


def test_host_replay_card_registration_defers_hand_form_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = NativeClashEnv()
    commands: list[str] = []
    monkeypatch.setattr(
        env,
        "_request",
        lambda command: commands.append(command)
        or {
            "ok": True,
            "sequence": 7,
            "generation": 3,
            "registeredAtTick": 40,
            "executeTick": 131,
        },
    )

    result = env.schedule_replay_card_at_tick(
        owner=0,
        card_id=26_000_001,
        x=9_000,
        y=10_000,
        execute_tick=131,
    )

    assert commands == ["replay-schedule-card 0 26000001 9000 10000 131"]
    assert result["sequence"] == 7
    assert result["registeredAtTick"] == 40
    assert result["cardId"] == 26_000_001
