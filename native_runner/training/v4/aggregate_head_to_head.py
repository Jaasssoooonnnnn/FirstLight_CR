"""Aggregate head-to-head game rows into Bradley-Terry ratings and pair tables.

Input rows are the ``games.jsonl`` lines written by ``evaluate_head_to_head``.
Ratings are a Bradley-Terry maximum-likelihood fit (Hunter's MM iteration;
draws count as half a win for each side) with ``--prior-draws`` virtual draws
added to every observed pair so that a perfect record stays finite.  Intervals
come from a bootstrap that resamples schedule configurations (seed, both decks,
battle level) with replacement: all games sharing one configuration -- the two
seat-swapped games of every model pair, which all pairs share under common
random numbers -- enter or leave a replicate together.

Usage:
    python -m native_runner.training.v4.aggregate_head_to_head \
        --games run/games.jsonl [more.jsonl ...] --output-dir run/aggregate \
        [--bootstrap 2000] [--seed 20261023] [--prior-draws 1.0] \
        [--tiebreak-tick 5995] [--reference NAME] [--include-sanity-in-bt]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TIEBREAK_TICK = 5995
WILSON_Z = 1.959963984540054
ELO_PER_NAT = 400.0 / math.log(10.0)


def wilson_interval(successes: float, games: int, *, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval; ``successes`` may be fractional (draws = 0.5)."""

    if games < 0 or successes < 0.0 or successes > games:
        raise ValueError("Wilson interval needs 0 <= successes <= games")
    if games == 0:
        return (0.0, 1.0)
    p = successes / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_game_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != "v4-head-to-head-game.v1":
                raise ValueError(f"{path}:{number} is not a head-to-head game row")
            rows.append(row)
    if not rows:
        raise ValueError("no head-to-head game rows were found")
    keys = [(row["pair"], int(row["schedule_row_id"])) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate game rows for (pair, schedule row): {duplicates[:5]}")
    return rows


def config_key(row: Mapping[str, Any]) -> tuple[int, str, str, int]:
    return (int(row["seed"]), str(row["deck0_id"]), str(row["deck1_id"]), int(row["battle_level"]))


def is_sanity_game(row: Mapping[str, Any]) -> bool:
    """A-vs-A: both seats run the identical checkpoint, offset, mode and temperatures."""

    return str(row["seat0_behavior"]) == str(row["seat1_behavior"])


def model_score(row: Mapping[str, Any], model: str) -> float:
    """Score in {0, 0.5, 1} for ``model`` in this game."""

    seat0_result = float(row["result_seat0"])
    if seat0_result not in (-1.0, 0.0, 1.0):
        raise ValueError("seat-0 result must be -1, 0 or 1")
    seat0_score = 0.5 * (seat0_result + 1.0)
    if row["seat0_model"] == model:
        return seat0_score
    if row["seat1_model"] == model:
        return 1.0 - seat0_score
    raise ValueError(f"model {model} did not play game {row.get('game_id')}")


def _seat_of(row: Mapping[str, Any], model: str) -> int:
    if row["seat0_model"] == model:
        return 0
    if row["seat1_model"] == model:
        return 1
    raise ValueError(f"model {model} did not play game {row.get('game_id')}")


def _tower_hitpoints(row: Mapping[str, Any], seat: int) -> float | None:
    towers = row.get("final_tower_hitpoints")
    if not towers:
        return None
    return float(sum(float(item["hitpoints"]) for item in towers if int(item["owner"]) == seat))


def _owner_counts(row: Mapping[str, Any], key: str) -> tuple[int, int]:
    """Per-owner counter of a game row; rows written before ``key`` existed count 0."""

    value = row.get(key)
    if value is None:
        return (0, 0)
    return (int(value[0]), int(value[1]))


def _second_action_rate(second_action_steps: int, act_steps: int) -> float:
    return second_action_steps / act_steps if act_steps else 0.0


def model_action_summaries_v4(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per model over all rows: ACT decision steps, steps with a second micro-action, and their ratio."""

    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        act = _owner_counts(row, "act_steps")
        second = _owner_counts(row, "second_action_steps")
        active = _owner_counts(row, "active_actions")
        for seat in (0, 1):
            model = str(row[f"seat{seat}_model"])
            totals[model]["seat_games"] += 1
            totals[model]["active_actions"] += active[seat]
            totals[model]["act_steps"] += act[seat]
            totals[model]["second_action_steps"] += second[seat]
    return {
        model: {
            "seat_games": int(counts["seat_games"]),
            "active_actions": int(counts["active_actions"]),
            "act_steps": int(counts["act_steps"]),
            "second_action_steps": int(counts["second_action_steps"]),
            "second_action_rate": _second_action_rate(counts["second_action_steps"], counts["act_steps"]),
        }
        for model, counts in sorted(totals.items())
    }


def _score_block(scores: Sequence[float]) -> dict[str, Any]:
    games = len(scores)
    total = float(sum(scores))
    low, high = wilson_interval(total, games)
    return {
        "games": games,
        "wins": sum(value == 1.0 for value in scores),
        "draws": sum(value == 0.5 for value in scores),
        "losses": sum(value == 0.0 for value in scores),
        "score_rate": total / games if games else None,
        "score_rate_wilson_95": [low, high],
    }


def pair_summaries_v4(rows: Sequence[Mapping[str, Any]], *, tiebreak_tick: int = TIEBREAK_TICK) -> dict[str, Any]:
    """Per-pair score, seat/role/level splits and secondary metrics (model A's view)."""

    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair"])].append(row)
    result: dict[str, Any] = {}
    for pair, pair_rows in sorted(by_pair.items()):
        model_a = str(pair_rows[0]["model_a"])
        model_b = str(pair_rows[0]["model_b"])
        if any(row["model_a"] != model_a or row["model_b"] != model_b for row in pair_rows):
            raise ValueError(f"pair {pair} rows disagree on model names")
        scores = [model_score(row, model_a) for row in pair_rows]
        splits: dict[str, dict[str, list[float]]] = {
            "a_seat": defaultdict(list),
            "a_role": defaultdict(list),
            "battle_level": defaultdict(list),
            "category": defaultdict(list),
            "forced_opening": defaultdict(list),
        }
        crown_diffs: list[int] = []
        tower_diffs: list[float] = []
        actions = {model_a: 0, model_b: 0}
        rejections = {model_a: 0, model_b: 0}
        act_steps = {model_a: 0, model_b: 0}
        second_steps = {model_a: 0, model_b: 0}
        forced_by_model: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        tiebreak = both_idle = truncated = 0
        for row, score in zip(pair_rows, scores, strict=True):
            seat_a = _seat_of(row, model_a)
            seat_b = 1 - seat_a
            splits["a_seat"][str(seat_a)].append(score)
            splits["a_role"]["current" if row["current_model"] == model_a else "frozen"].append(score)
            splits["battle_level"][str(row["battle_level"])].append(score)
            splits["category"][str(row["category"])].append(score)
            forced_model = row.get("forced_model")
            splits["forced_opening"]["forced" if forced_model is not None else "unforced"].append(score)
            if forced_model is not None:
                forced_by_model[str(forced_model)] += 1
            crowns = row["crowns"]
            crown_diffs.append(int(crowns[seat_a]) - int(crowns[seat_b]))
            hp_a, hp_b = _tower_hitpoints(row, seat_a), _tower_hitpoints(row, seat_b)
            if hp_a is not None and hp_b is not None:
                tower_diffs.append(hp_a - hp_b)
            active = row["active_actions"]
            rejected = row["rejected_actions"]
            actions[model_a] += int(active[seat_a])
            actions[model_b] += int(active[seat_b])
            rejections[model_a] += int(rejected[seat_a])
            rejections[model_b] += int(rejected[seat_b])
            act, second = _owner_counts(row, "act_steps"), _owner_counts(row, "second_action_steps")
            act_steps[model_a] += act[seat_a]
            act_steps[model_b] += act[seat_b]
            second_steps[model_a] += second[seat_a]
            second_steps[model_b] += second[seat_b]
            tiebreak += int(int(row["terminal_tick"]) >= tiebreak_tick)
            both_idle += int(int(active[0]) == 0 and int(active[1]) == 0)
            truncated += int(bool(row["hard_failure"]))
            reasons[str(row.get("terminal_reason"))] += 1
        games = len(pair_rows)
        result[pair] = {
            "model_a": model_a,
            "model_b": model_b,
            "sanity_a_vs_a": all(is_sanity_game(row) for row in pair_rows),
            **_score_block(scores),
            "splits": {
                name: {key: _score_block(values) for key, values in sorted(groups.items())}
                for name, groups in splits.items()
            },
            "crown_diff_mean": float(np.mean(crown_diffs)),
            "crown_diff_std": float(np.std(crown_diffs, ddof=1)) if games > 1 else None,
            "tower_hitpoint_diff_mean": float(np.mean(tower_diffs)) if len(tower_diffs) == games else None,
            "tiebreak_games": tiebreak,
            "tiebreak_share": tiebreak / games,
            "both_idle_games": both_idle,
            "forced_games_by_model": dict(sorted(forced_by_model.items())),
            "active_actions": actions,
            "rejected_actions": rejections,
            "rejected_action_rate": {
                model: rejections[model] / actions[model] if actions[model] else 0.0 for model in (model_a, model_b)
            },
            "act_steps": act_steps,
            "second_action_steps": second_steps,
            "second_action_rate": {
                model: _second_action_rate(second_steps[model], act_steps[model]) for model in (model_a, model_b)
            },
            "terminal_reasons": dict(sorted(reasons.items())),
            "hard_failures": truncated,
        }
    return result


def _comparison_matrices(rows: Sequence[Mapping[str, Any]], players: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (score[i, j], games[i, j]) for the given rows."""

    index = {name: position for position, name in enumerate(players)}
    score = np.zeros((len(players), len(players)))
    games = np.zeros((len(players), len(players)))
    for row in rows:
        first, second = index[str(row["seat0_model"])], index[str(row["seat1_model"])]
        seat0 = model_score(row, str(row["seat0_model"]))
        score[first, second] += seat0
        score[second, first] += 1.0 - seat0
        games[first, second] += 1.0
        games[second, first] += 1.0
    return score, games


def fit_bradley_terry_v4(
    score: np.ndarray, games: np.ndarray, *, prior_draws: float = 1.0, max_iterations: int = 100_000, tol: float = 1e-10
) -> np.ndarray:
    """Return mean-zero log-strengths theta with P(i beats j) = sigmoid(theta_i - theta_j)."""

    score = np.asarray(score, dtype=np.float64)
    games = np.asarray(games, dtype=np.float64)
    players = score.shape[0]
    if score.shape != (players, players) or games.shape != score.shape or players < 2:
        raise ValueError("Bradley-Terry needs square score/game matrices for at least two players")
    if prior_draws < 0.0 or not math.isfinite(prior_draws):
        raise ValueError("prior draws must be finite and nonnegative")
    if np.any(np.diag(games) != 0) or not np.allclose(games, games.T) or np.any(score < 0) or np.any(score > games):
        raise ValueError("Bradley-Terry matrices are inconsistent")
    observed = games > 0
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for other in np.flatnonzero(observed[node]):
            if int(other) not in reached:
                reached.add(int(other))
                frontier.append(int(other))
    if len(reached) != players:
        raise ValueError("Bradley-Terry comparison graph is disconnected")
    smoothed_score = score + observed * (0.5 * prior_draws)
    smoothed_games = games + observed * prior_draws
    wins = smoothed_score.sum(axis=1)
    if np.any(wins <= 0.0) or np.any(wins >= smoothed_games.sum(axis=1)):
        raise ValueError("a player won or lost every game; increase --prior-draws")
    strength = np.ones(players)
    for _ in range(max_iterations):
        denominator = (smoothed_games / (strength[:, None] + strength[None, :])).sum(axis=1)
        updated = wins / denominator
        updated /= math.exp(float(np.mean(np.log(updated))))
        if float(np.max(np.abs(np.log(updated) - np.log(strength)))) < tol:
            strength = updated
            break
        strength = updated
    else:
        raise RuntimeError("Bradley-Terry MM iteration did not converge")
    theta = np.log(strength)
    return theta - float(theta.mean())


def clustered_bootstrap_v4(
    rows: Sequence[Mapping[str, Any]], players: Sequence[str], *, replicates: int, seed: int, prior_draws: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample schedule configurations; return (thetas[R, P], scores[R, P, P], games[R, P, P])."""

    if replicates <= 0:
        raise ValueError("bootstrap needs at least one replicate")
    clusters: dict[tuple[int, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[config_key(row)].append(row)
    keys = sorted(clusters)
    cluster_scores = np.zeros((len(keys), len(players), len(players)))
    cluster_games = np.zeros_like(cluster_scores)
    for position, key in enumerate(keys):
        cluster_scores[position], cluster_games[position] = _comparison_matrices(clusters[key], players)
    rng = np.random.default_rng(seed)
    thetas = np.full((replicates, len(players)), np.nan)
    scores = np.zeros((replicates, len(players), len(players)))
    games = np.zeros_like(scores)
    for replicate in range(replicates):
        weights = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys))).astype(np.float64)
        scores[replicate] = np.tensordot(weights, cluster_scores, axes=1)
        games[replicate] = np.tensordot(weights, cluster_games, axes=1)
        try:
            thetas[replicate] = fit_bradley_terry_v4(scores[replicate], games[replicate], prior_draws=prior_draws)
        except ValueError:
            # A resample can disconnect a sparse comparison graph; leave NaN and report the count.
            continue
    return thetas, scores, games


def _interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def aggregate_head_to_head_v4(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 2000,
    seed: int = 20261023,
    prior_draws: float = 1.0,
    tiebreak_tick: int = TIEBREAK_TICK,
    reference: str | None = None,
    include_sanity_in_bt: bool = False,
) -> dict[str, Any]:
    pairs = pair_summaries_v4(rows, tiebreak_tick=tiebreak_tick)
    bt_rows = [row for row in rows if include_sanity_in_bt or not is_sanity_game(row)]
    players = sorted({str(row[key]) for row in bt_rows for key in ("seat0_model", "seat1_model")})
    report: dict[str, Any] = {
        "schema": "v4-head-to-head-aggregate.v1",
        "games": len(rows),
        "bt_games": len(bt_rows),
        "configurations": len({config_key(row) for row in rows}),
        "parameters": {
            "bootstrap_replicates": replicates,
            "seed": seed,
            "prior_draws": prior_draws,
            "tiebreak_tick": tiebreak_tick,
            "reference": reference,
            "include_sanity_in_bt": include_sanity_in_bt,
            "cluster": "schedule configuration (seed, deck0_id, deck1_id, battle_level) across all pairs",
        },
        "pairs": pairs,
        "sanity": {pair: summary for pair, summary in pairs.items() if summary["sanity_a_vs_a"]},
        "models": model_action_summaries_v4(rows),
        "seat0_score_rate": _score_block([model_score(row, str(row["seat0_model"])) for row in rows]),
        "hard_failures": sum(int(bool(row["hard_failure"])) for row in rows),
    }
    if len(players) < 2:
        report["ratings"] = None
        return report
    if reference is not None and reference not in players:
        raise ValueError(f"reference model {reference} has no rated games")
    score, games = _comparison_matrices(bt_rows, players)
    theta = fit_bradley_terry_v4(score, games, prior_draws=prior_draws)
    thetas, boot_scores, boot_games = clustered_bootstrap_v4(
        bt_rows, players, replicates=replicates, seed=seed, prior_draws=prior_draws
    )
    reference_index = players.index(reference) if reference is not None else None
    centered = theta - (theta[reference_index] if reference_index is not None else 0.0)
    boot_centered = thetas - (thetas[:, [reference_index]] if reference_index is not None else 0.0)
    ratings = []
    for position, name in enumerate(players):
        interval = _interval(boot_centered[:, position])
        ratings.append(
            {
                "model": name,
                "theta": float(centered[position]),
                "elo": float(ELO_PER_NAT * centered[position]),
                "theta_95": interval,
                "elo_95": None if interval is None else [ELO_PER_NAT * value for value in interval],
                "games": int(games[position].sum()),
                "score_rate": float(score[position].sum() / games[position].sum()),
            }
        )
    ratings.sort(key=lambda item: -item["theta"])
    differences = []
    for first in range(len(players)):
        for second in range(first + 1, len(players)):
            delta = theta[first] - theta[second]
            boot_delta = thetas[:, first] - thetas[:, second]
            interval = _interval(boot_delta)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot_pair_score = boot_scores[:, first, second] / boot_games[:, first, second]
            observed_games = int(games[first, second])
            differences.append(
                {
                    "model_i": players[first],
                    "model_j": players[second],
                    "theta_diff": float(delta),
                    "theta_diff_95": interval,
                    "elo_diff": float(ELO_PER_NAT * delta),
                    "elo_diff_95": None if interval is None else [ELO_PER_NAT * value for value in interval],
                    "expected_score_i": float(1.0 / (1.0 + math.exp(-delta))),
                    "excludes_zero": None if interval is None else bool(interval[0] > 0.0 or interval[1] < 0.0),
                    "direct_games": observed_games,
                    "direct_score_rate_i": float(score[first, second] / observed_games) if observed_games else None,
                    "direct_score_rate_i_cluster_95": _interval(boot_pair_score) if observed_games else None,
                }
            )
    report["ratings"] = {
        "players": players,
        "table": ratings,
        "differences": differences,
        "bootstrap_failed_replicates": int(np.isnan(thetas).any(axis=1).sum()),
    }
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"[{_fmt(value[0], digits)}, {_fmt(value[1], digits)}]"
    return str(value)


def render_markdown_v4(report: Mapping[str, Any]) -> str:
    lines = [
        "# Head-to-head aggregate",
        "",
        f"- games: {report['games']} (Bradley-Terry: {report['bt_games']}); configurations: {report['configurations']}",
        f"- hard failures (truncated / not ended): {report['hard_failures']}",
        f"- seat-0 score rate: {_fmt(report['seat0_score_rate']['score_rate'])} "
        f"{_fmt(report['seat0_score_rate']['score_rate_wilson_95'])}",
        f"- bootstrap: {report['parameters']['bootstrap_replicates']} replicates clustered by "
        f"{report['parameters']['cluster']}; prior draws per pair {report['parameters']['prior_draws']}",
        "",
    ]
    ratings = report.get("ratings")
    if ratings:
        reference = report["parameters"]["reference"]
        lines += [
            "## Bradley-Terry ratings" + (f" (relative to {reference})" if reference else " (mean zero)"),
            "",
            "| model | Elo | Elo 95% | theta | games | raw score |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for item in ratings["table"]:
            lines.append(
                f"| {item['model']} | {_fmt(item['elo'], 1)} | {_fmt(item['elo_95'], 1)} | {_fmt(item['theta'])} | "
                f"{item['games']} | {_fmt(item['score_rate'])} |"
            )
        lines += [
            "",
            f"Bootstrap replicates that failed to fit: {ratings['bootstrap_failed_replicates']}",
            "",
            "## Rating differences (i minus j)",
            "",
            "| i | j | Elo diff | 95% | excludes 0 | E[score i] | direct games | direct score i | 95% (cluster) |",
            "|---|---|---:|---|---|---:|---:|---:|---|",
        ]
        for item in ratings["differences"]:
            lines.append(
                f"| {item['model_i']} | {item['model_j']} | {_fmt(item['elo_diff'], 1)} | "
                f"{_fmt(item['elo_diff_95'], 1)} | {_fmt(item['excludes_zero'])} | {_fmt(item['expected_score_i'])} | "
                f"{item['direct_games']} | {_fmt(item['direct_score_rate_i'])} | "
                f"{_fmt(item['direct_score_rate_i_cluster_95'])} |"
            )
        lines.append("")
    lines += [
        "## Pairs (model A's view)",
        "",
        "| pair | games | A score | Wilson 95% (iid) | crown diff | tiebreak share | both idle | forced | "
        "reject rate A / B | hard failures |",
        "|---|---:|---:|---|---:|---:|---:|---|---|---:|",
    ]
    for pair, item in report["pairs"].items():
        rates = item["rejected_action_rate"]
        lines.append(
            f"| {pair}{' (A-vs-A)' if item['sanity_a_vs_a'] else ''} | {item['games']} | {_fmt(item['score_rate'])} | "
            f"{_fmt(item['score_rate_wilson_95'])} | {_fmt(item['crown_diff_mean'], 2)} | "
            f"{_fmt(item['tiebreak_share'])} | {item['both_idle_games']} | {item['forced_games_by_model'] or '-'} | "
            f"{_fmt(rates[item['model_a']], 4)} / {_fmt(rates[item['model_b']], 4)} | {item['hard_failures']} |"
        )
    models = report.get("models") or {}
    if models:
        lines += [
            "",
            "## Per-model action counts (all games)",
            "",
            "| model | seat-games | active actions | ACT steps | second-action steps | second_action_rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model, item in models.items():
            lines.append(
                f"| {model} | {item['seat_games']} | {item['active_actions']} | {item['act_steps']} | "
                f"{item['second_action_steps']} | {_fmt(item['second_action_rate'])} |"
            )
    lines += ["", "## Seat / role splits (A score)", "", "| pair | A seat 0 | A seat 1 | A current | A frozen |"]
    lines.append("|---|---|---|---|---|")
    for pair, item in report["pairs"].items():
        seat, role = item["splits"]["a_seat"], item["splits"]["a_role"]

        def cell(block: Mapping[str, Any] | None) -> str:
            if block is None:
                return "-"
            return f"{_fmt(block['score_rate'])} (n={block['games']})"

        lines.append(
            f"| {pair} | {cell(seat.get('0'))} | {cell(seat.get('1'))} | {cell(role.get('current'))} | "
            f"{cell(role.get('frozen'))} |"
        )
    if report["sanity"]:
        lines += ["", "## A-vs-A sanity", ""]
        for pair, item in report["sanity"].items():
            inside = item["score_rate_wilson_95"][0] <= 0.5 <= item["score_rate_wilson_95"][1]
            lines.append(
                f"- {pair}: score {_fmt(item['score_rate'])} {_fmt(item['score_rate_wilson_95'])}; "
                f"0.5 inside interval: {inside}"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--prior-draws", type=float, default=1.0)
    parser.add_argument("--tiebreak-tick", type=int, default=TIEBREAK_TICK)
    parser.add_argument("--reference")
    parser.add_argument("--include-sanity-in-bt", action="store_true")
    args = parser.parse_args(argv)
    rows = load_game_rows(args.games)
    report = aggregate_head_to_head_v4(
        rows,
        replicates=args.bootstrap,
        seed=args.seed,
        prior_draws=args.prior_draws,
        tiebreak_tick=args.tiebreak_tick,
        reference=args.reference,
        include_sanity_in_bt=args.include_sanity_in_bt,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    (args.output_dir / "aggregate.md").write_text(render_markdown_v4(report), encoding="utf-8")
    print(json.dumps({"event": "head_to_head_aggregate", "games": report["games"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
