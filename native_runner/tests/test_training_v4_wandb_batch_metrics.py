from __future__ import annotations

import math

import pytest

from native_runner.training.v4.matchmaking import POLICY_IL
from native_runner.training.v4.train_ppo_self_play_cluster import (
    _batch_wandb_metrics,
    _wandb_update_metrics,
)


def _segment(
    *,
    update: int,
    frames: int,
    complete: bool,
    return_mean: float,
    return_std: float,
    value_loss: float,
) -> dict[str, object]:
    return {
        "update": update,
        "batch_index": 7,
        "batch_complete": complete,
        "learner_owner_frames": frames,
        "return_mean": return_mean,
        "return_std": return_std,
        "reward_mean": return_mean,
        "reward_std": return_std,
        "advantage_mean": return_mean,
        "advantage_std": return_std,
        "behavior_value_mean": return_mean,
        "value_explained_variance": 1.0,
        "value": value_loss,
        "policy": value_loss + 0.1,
        "total": value_loss + 0.2,
        "entropy": value_loss + 0.3,
        "approx_kl": value_loss + 0.4,
        "clip_fraction": value_loss + 0.5,
        "act_eligible_ticks": frames * 10,
        "eligible_wait_ticks": frames * 6,
        "forced_opening_act_ticks": frames,
        "continue_stop_count": frames * 3,
        "continue_second_count": frames,
        "native_rejected_actions": 0,
        "collection_boundary_bootstrap_count": frames * 2,
        "collection_truncated_count": frames * 2,
        "collection_initial_hidden_abs_max": 0.5 + 0.1 * update,
        "delay_0ms_count": frames,
        "delay_50ms_count": frames * 2,
        "delay_100ms_count": frames * 3,
        "delay_150ms_count": frames * 4,
        "delay_200ms_count": 0,
    }


def _report(
    result: tuple[float, float],
    crowns: tuple[int, int],
) -> dict[str, object]:
    return {
        "terminal_tick": 6_000,
        "crowns": crowns,
        "policy_matchup": POLICY_IL,
        "current_owners": (0,),
        "result": result,
        "category": "focus-focus",
    }


def test_batch_wandb_metrics_emit_one_frame_weighted_match_wave_point() -> None:
    segments = (
        _segment(
            update=49,
            frames=3,
            complete=False,
            return_mean=0.0,
            return_std=1.0,
            value_loss=0.1,
        ),
        _segment(
            update=50,
            frames=1,
            complete=True,
            return_mean=2.0,
            return_std=0.0,
            value_loss=0.9,
        ),
    )
    reports = (
        _report((1.0, -1.0), (2, 1)),
        _report((-1.0, 1.0), (0, 1)),
    )

    metrics = _batch_wandb_metrics(segments, reports)

    assert metrics["batch/index"] == 7
    assert metrics["batch/end_update"] == 50
    assert metrics["batch/segments"] == 2
    assert metrics["batch_data_quality/learner_frames"] == 4
    assert metrics["batch_critic/value_loss"] == pytest.approx(0.3)
    assert metrics["batch_critic/target_return_mean"] == pytest.approx(0.5)
    assert metrics["batch_critic/target_return_std"] == pytest.approx(
        math.sqrt(1.5)
    )
    assert metrics["batch_critic/explained_variance"] == pytest.approx(1.0)
    assert metrics["batch_exploration/eligible_wait_rate"] == pytest.approx(
        0.6
    )
    assert metrics["batch_exploration/delay_150ms_fraction"] == pytest.approx(
        0.4
    )
    assert metrics["batch_performance/vs_il/games"] == 2
    assert metrics["batch_performance/vs_il/score"] == pytest.approx(0.5)


def test_batch_wandb_metrics_reject_incomplete_wave() -> None:
    segment = _segment(
        update=49,
        frames=3,
        complete=False,
        return_mean=0.0,
        return_std=1.0,
        value_loss=0.1,
    )

    with pytest.raises(ValueError, match="completed match wave"):
        _batch_wandb_metrics((segment,), ())


def test_segment_wandb_metrics_keep_only_hard_safety_guards() -> None:
    metrics = _wandb_update_metrics(
        {
            "update": 9,
            "collection_truncated_count": 4,
            "collection_boundary_bootstrap_count": 4,
            "learner_owner_frame_fraction": 1.0,
            "native_rejected_actions": 0,
            "league_history_snapshots": 3,
        }
    )

    assert metrics == {
        "update": 9,
        "safety/all_training_frames_used": 1.0,
        "safety/rejected_actions": 0,
        "safety/boundary_bootstrap_coverage": 1.0,
        "league/history_snapshots": 3,
    }
