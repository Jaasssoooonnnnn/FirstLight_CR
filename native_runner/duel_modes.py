"""Offline duel rules, separate from native event-deck/draft modes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DuelMode:
    name: str
    game_mode: int
    evolutions: bool = False
    random_decks: bool = False
    unavailable_reason: str = ""


DUEL_MODES = (
    DuelMode("常规天梯 (1v1)", 72000006),
    DuelMode("7倍圣水·随机全觉醒", 72000232, evolutions=True, random_decks=True),
    DuelMode("疯狂加速模式 (Ramp Up)", 72000033),
    DuelMode("全觉醒大战 (All Evolution)", 72000006, evolutions=True),
    DuelMode("圣水桶模式（暂不可用）", 72000384,
             unavailable_reason="圣水桶的场地事件与模型观测尚未验证，暂不可用。"),
)
DUEL_MODE_BY_NAME = {mode.name: mode for mode in DUEL_MODES}


def resolve_duel_mode(name: str) -> DuelMode:
    # Older saved/UI values must retain their intended custom rule.
    if name == "7倍圣水无限火力 (7x Elixir)":
        return DUEL_MODES[1]
    if name == "圣水桶模式 (Elixir Barrel)":
        name = DUEL_MODES[4].name
    try:
        mode = DUEL_MODE_BY_NAME[name]
    except KeyError:
        raise ValueError(f"未知对战模式：{name}") from None
    if mode.unavailable_reason:
        raise ValueError(mode.unavailable_reason)
    return mode
