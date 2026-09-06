from __future__ import annotations

import copy
import hashlib
import json
import unittest
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from native_runner.arena import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PlacementRule,
    cell_to_world,
    deployment_mask,
    terrain_walkable,
    world_to_cell,
)
from native_runner.battle_env import (
    BattleEnvError,
    BattleEnvV1,
    TOWER_LAYOUT,
)
from native_runner.card_specs import CardSpecCatalog, build_card_catalog
from native_runner.contracts import (
    ActionKind,
    ActionV1,
    ObservationTier,
)
from native_runner.cr_native_env import FIRST_PLAYABLE_TICK, HandAction
from native_runner.match_factory import MatchConfig
from native_runner.semantic_subset import SEMANTIC_BASELINE_DECK
from native_runner.timeline import load_game_mode_timeline


def _card(
    hand_index: int,
    card_id: int,
    cost: int,
    *,
    deck_slot: int | None = None,
) -> dict[str, Any]:
    resolved_deck_slot = min(hand_index, 7) if deck_slot is None else deck_slot
    return {
        "handIndex": hand_index,
        "cardId": card_id,
        "commandCardId": card_id,
        "cost": cost,
        "cardParameter": (cost << 28) | ((resolved_deck_slot + 1) << 22),
        "deckSlot": resolved_deck_slot,
    }


def _tower_object(
    entity_id: int,
    owner: int,
    kind: str,
    x: int,
    y: int,
    *,
    hp: int = 1000,
) -> dict[str, Any]:
    return {
        "slot": entity_id,
        "objectIndex": 100 + entity_id,
        "secondaryIndex": 0,
        "owner": owner,
        "cardId": 27000000 + entity_id,
        "kind": kind,
        "x": x,
        "y": y,
        "targetX": x,
        "targetY": y,
        "hp": hp,
        "maxHp": 1000,
    }


def _troop_object(
    *,
    object_index: int = 500,
    owner: int = 1,
    card_id: int = 26000001,
    x: int = 8500,
    y: int = 16500,
    hp: int = 100,
) -> dict[str, Any]:
    return {
        "slot": 20 + object_index,
        "objectIndex": object_index,
        "secondaryIndex": 0,
        "owner": owner,
        "cardId": card_id,
        "x": x,
        "y": y,
        "targetX": x,
        "targetY": y,
        "hp": hp,
        "maxHp": 100,
    }


class FakeNativeEnv:
    """In-memory substitute for the thin NativeClashEnv engine binding."""

    def __init__(self, *, initial_entity: bool = True) -> None:
        self.initial_entity = initial_entity
        self.state: dict[str, Any] = {}
        self.queued: list[dict[str, Any]] = []
        self.step_mutators: list[Callable[[dict[str, Any], int, int], None]] = []
        self.create_calls = 0
        self._last_match: MatchConfig | None = None

    @staticmethod
    def _player(owner: int, deck: tuple[int, ...]) -> dict[str, Any]:
        if owner == 0:
            # Slot 1 is intentionally unaffordable and slot 3 absent.  This
            # exercises affordability and not-in-hand mask reasons at once.
            hand = [
                _card(0, deck[0], 3, deck_slot=0),
                _card(1, deck[2], 7, deck_slot=2),
                _card(2, deck[6], 4, deck_slot=6),
            ]
            elixir = 60_000
        else:
            hand = [
                _card(0, deck[1], 2, deck_slot=1),
                _card(1, deck[3], 3, deck_slot=3),
                _card(2, deck[7], 6, deck_slot=7),
                _card(3, deck[5], 2, deck_slot=5),
            ]
            elixir = 50_000
        return {
            "owner": owner,
            "accountId": 70_000_000 + owner,
            "elixirRaw": elixir,
            "hand": hand,
            "nextCard": _card(
                4,
                deck[4],
                int(_test_catalog().by_id[deck[4]].elixir_cost or 0),
                deck_slot=4,
            ),
            "deck": [{"cardId": card_id} for card_id in deck],
            "cycle": [{"cardId": card_id} for card_id in deck[4:]],
        }

    def _initial_state(self, match: MatchConfig) -> dict[str, Any]:
        objects = [
            _tower_object(entity_id, owner, kind, x, y)
            for entity_id, owner, kind, x, y in TOWER_LAYOUT
        ]
        if self.initial_entity:
            objects.append(_troop_object())
        return {
            "tick": FIRST_PLAYABLE_TICK,
            "ended": False,
            "truncated": False,
            "generation": 1,
            "count": len(objects),
            "returned": len(objects),
            "queuedCommands": 0,
            "players": [
                self._player(0, match.deck0),
                self._player(1, match.deck1),
            ],
            "objects": objects,
        }

    def create_match(self, match: MatchConfig) -> dict[str, Any]:
        self.create_calls += 1
        self._last_match = match
        self.queued.clear()
        self.state = self._initial_state(match)
        return self.observe()
    def create_native_match(self, match: MatchConfig) -> dict[str, Any]:
        return self.create_match(match)

    def observe(self) -> dict[str, Any]:
        self.state.setdefault("stateEpoch", 1)
        for slot, item in enumerate(self.state["objects"]):
            item["slot"] = slot
            item.setdefault("nativeObjectId", 5_000_000 + int(item["objectIndex"]))
        return copy.deepcopy(self.state)

    def observe_atomic(self) -> dict[str, Any]:
        from native_runner.tests.test_battle_env_rich_telemetry import _rich

        ordinary = self.observe()
        return {"ordinary": ordinary, "rich": _rich(ordinary)}

    def digest(self) -> dict[str, str]:
        payload = json.dumps(self.state, sort_keys=True, separators=(",", ":"))
        return {"digest": hashlib.sha256(payload.encode("utf-8")).hexdigest()}

    def queue_hand_action_at(
        self,
        action: HandAction,
        *,
        execute_tick: int | None = None,
        execute_in_ticks: int | None = None,
    ) -> dict[str, Any]:
        if (execute_tick is None) == (execute_in_ticks is None):
            raise ValueError("pass exactly one execution time")
        actual_tick = (
            int(execute_tick)
            if execute_tick is not None
            else int(self.state["tick"]) + int(execute_in_ticks or 0)
        )
        record = {"action": action, "execute_tick": actual_tick}
        self.queued.append(record)
        self.state["queuedCommands"] = len(self.queued)
        player = next(
            item for item in self.state["players"] if item["owner"] == action.owner
        )
        card = next(
            item
            for item in player["hand"]
            if item["handIndex"] == action.hand_index
        )
        return {
            "executeTick": actual_tick,
            "queued": True,
            "cardId": card["cardId"],
            "commandCardId": card["commandCardId"],
            "cardParameter": card["cardParameter"],
            "deckSlot": card["deckSlot"],
            "cost": card["cost"],
            "formCode": 0,
            "formName": "Basic",
        }

    def _apply_queued(self, old_tick: int, new_tick: int) -> None:
        remaining: list[dict[str, Any]] = []
        for record in self.queued:
            if not (old_tick < int(record["execute_tick"]) <= new_tick):
                remaining.append(record)
                continue
            action: HandAction = record["action"]
            player = next(
                item for item in self.state["players"] if item["owner"] == action.owner
            )
            hand = list(player["hand"])
            old = next(
                item for item in hand if int(item["handIndex"]) == action.hand_index
            )
            replacement = dict(player["nextCard"])
            replacement["handIndex"] = action.hand_index
            player["hand"] = [replacement if item is old else item for item in hand]
            player["nextCard"] = _card(4, 26000005, 3)
            self.state["objects"].append(
                _troop_object(
                    object_index=700 + len(self.state["objects"]),
                    owner=action.owner,
                    card_id=int(old["cardId"]),
                    x=action.x,
                    y=action.y,
                )
            )
        self.queued = remaining
        self.state["queuedCommands"] = len(remaining)

    def step(self, ticks: int = 1) -> dict[str, Any]:
        if ticks < 1:
            raise ValueError("ticks must be positive")
        # Replace the full graph before every mutation: BattleEnv deliberately
        # keeps the previous raw state and must never observe our later writes.
        self.state = copy.deepcopy(self.state)
        old_tick = int(self.state["tick"])
        new_tick = old_tick + int(ticks)
        self.state["tick"] = new_tick
        self._apply_queued(old_tick, new_tick)
        if self.step_mutators:
            self.step_mutators.pop(0)(self.state, old_tick, new_tick)
        self.state["count"] = len(self.state["objects"])
        self.state["returned"] = len(self.state["objects"])
        return self.observe()




@lru_cache(maxsize=1)
def _test_catalog() -> CardSpecCatalog:
    """Use the production-bound catalog even with the fake native transport."""

    return build_card_catalog()


def _make_env(
    fake: FakeNativeEnv | None = None,
    *,
    shaping_beta: float = 0.0,
) -> tuple[BattleEnvV1, FakeNativeEnv]:
    native = fake or FakeNativeEnv()
    env = BattleEnvV1(
        native=native,  # type: ignore[arg-type]
        card_catalog=_test_catalog(),
        ruleset_id="a" * 64,
        warmup_ticks=0,
        shaping_beta=shaping_beta,
    )
    return env, native


def _reset(env: BattleEnvV1, *, seed: int = 7):
    return env.reset(match_config=MatchConfig(
        deck0=SEMANTIC_BASELINE_DECK,
        deck1=SEMANTIC_BASELINE_DECK,
        seed=seed,
    ))


class ArenaCoordinateTests(unittest.TestCase):
    def test_18_by_32_coordinate_contract_and_round_trip(self) -> None:
        self.assertEqual((GRID_WIDTH, GRID_HEIGHT), (18, 32))
        self.assertEqual(cell_to_world((0, 0)), (500, 500))
        self.assertEqual(cell_to_world((17, 31)), (17_500, 31_500))
        for cell in ((0, 0), (8, 10), (17, 31)):
            self.assertEqual(world_to_cell(*cell_to_world(cell)), cell)
        self.assertEqual(world_to_cell(-1, 32_001, clamp=True), (0, 31))
        with self.assertRaises(ValueError):
            cell_to_world((18, 0))
        with self.assertRaises(ValueError):
            cell_to_world((0, 32))
        with self.assertRaises(ValueError):
            world_to_cell(18_000, 0)

    def test_deployment_masks_are_row_major_and_owner_relative(self) -> None:
        bottom = deployment_mask(0, PlacementRule.OWN_TERRITORY)
        top = deployment_mask(1, PlacementRule.OWN_TERRITORY)
        anywhere = deployment_mask(0, PlacementRule.ANYWHERE)
        self.assertEqual((len(bottom), len(bottom[0])), (32, 18))
        self.assertTrue(terrain_walkable(8, 10))
        self.assertTrue(bottom[10][8])
        self.assertFalse(bottom[25][8])
        self.assertFalse(top[10][8])
        self.assertTrue(top[25][8])
        self.assertEqual(sum(sum(row) for row in anywhere), 18 * 32)


class BattleEnvironmentTests(unittest.TestCase):

    def test_destroyed_tower_position_does_not_rebind_to_a_troop(self) -> None:
        env, native = _make_env()
        _reset(env)
        collision = _troop_object(
            object_index=901,
            owner=0,
            card_id=26000001,
            x=3500,
            y=6500,
        )
        native.state["objects"] = [
            item
            for item in native.state["objects"]
            if int(item["slot"]) != 1
        ] + [collision]

        native.step(1)
        towers = env._extract_towers(native.state)
        destroyed = next(item for item in towers if item.entity_id == 1)
        entities = env._raw_entity_map(native.state)

        self.assertFalse(destroyed.active)
        self.assertEqual(destroyed.hitpoints, 0.0)
        self.assertEqual(destroyed.max_hitpoints, 1000.0)
        self.assertIn(26000001, {item["cardId"] for item in entities.values()})

    def test_actor_step_materializes_only_the_requested_fair_view(self) -> None:
        env, _ = _make_env()
        _reset(env)

        actor, reward, terminated, truncated, info = env.step_actor(
            0,
            {
                0: ActionV1.wait(0, ticks=2),
                1: ActionV1.wait(1, ticks=2),
            },
        )
        refreshed = env.observe(ObservationTier.FAIR, owner=0)

        self.assertEqual(actor.owner, 0)
        self.assertEqual(actor.tick, FIRST_PLAYABLE_TICK + 2)
        self.assertEqual(reward, env._last_rewards[0])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["native_tick"], FIRST_PLAYABLE_TICK + 2)
        self.assertEqual(actor.players, refreshed.players)
        self.assertEqual(actor.towers, refreshed.towers)
        self.assertEqual(actor.entities, refreshed.entities)
        self.assertEqual(actor.events, refreshed.events)
        self.assertEqual(actor.action_mask, refreshed.action_mask)
        self.assertEqual(actor.raster, refreshed.raster)
        self.assertEqual(actor.terminal, refreshed.terminal)

        with self.assertRaisesRegex(ValueError, "actor_owner"):
            env.step_actor(2, {})

    def test_independent_decision_deadlines_are_not_overwritten(self) -> None:
        env, _ = _make_env()
        _reset(env)
        observations, _, _, _, infos = env.step(
            {0: ActionV1.wait(0, ticks=5), 1: ActionV1.wait(1, ticks=2)}
        )
        self.assertEqual(observations[0].tick, FIRST_PLAYABLE_TICK + 2)
        self.assertFalse(infos[0]["decision_ready"])
        self.assertTrue(infos[1]["decision_ready"])
        self.assertFalse(observations[0].action_mask.kinds[ActionKind.PLAY_CARD.value])
        self.assertEqual(
            observations[0].action_mask.reasons["raw_elixir"],
            6.0,
        )
        self.assertEqual(
            observations[0].action_mask.reasons["reserved_elixir"],
            0.0,
        )
        self.assertEqual(
            observations[0].action_mask.reasons["effective_elixir"],
            6.0,
        )
        observations, _, _, _, infos = env.step(
            {1: ActionV1.wait(1, ticks=2)}
        )
        self.assertEqual(observations[0].tick, FIRST_PLAYABLE_TICK + 4)
        self.assertFalse(infos[0]["decision_ready"])
        observations, _, _, _, infos = env.step(
            {1: ActionV1.wait(1, ticks=1)}
        )
        self.assertEqual(observations[0].tick, FIRST_PLAYABLE_TICK + 5)
        self.assertTrue(infos[0]["decision_ready"])
        self.assertTrue(infos[1]["decision_ready"])

    def test_default_timeline_phase_and_elixir_rate(self) -> None:
        timeline = load_game_mode_timeline(72_000_006).timeline
        self.assertEqual(timeline.phase(3599), ("normal", 2.0))
        self.assertEqual(timeline.phase(3600), ("overtime", 2.0))
        self.assertEqual(timeline.phase(4800), ("overtime", 3.0))
        # At the 28-second full-bar rate, one elixir takes 56 native ticks.
        self.assertEqual(timeline.ticks_for_raw(130, 10_000), 56)
        self.assertEqual(timeline.ticks_for_raw(2400, 10_000), 28)


    def test_m1_fair_observation_does_not_leak_opponent_private_state(self) -> None:
        env, _ = _make_env()
        observations, _ = _reset(env)
        fair = observations[0]
        own = next(player for player in fair.players if player.owner == 0)
        opponent = next(player for player in fair.players if player.owner == 1)
        self.assertEqual(fair.tier, ObservationTier.FAIR)
        self.assertEqual(fair.owner, 0)
        self.assertTrue(own.private_state_visible)
        self.assertTrue(own.hand)
        self.assertFalse(opponent.private_state_visible)
        self.assertIsNone(opponent.elixir_exact)
        self.assertIsNone(opponent.elixir_visible)
        self.assertEqual(opponent.hand, ())
        self.assertIsNone(opponent.next_card)
        self.assertEqual(opponent.deck, ())
        self.assertEqual(opponent.cycle, ())
        self.assertIsNone(opponent.metadata.get("native_account_id"))
        self.assertIsNone(fair.native_digest)
        self.assertTrue(all(not tower.internal for tower in fair.towers))
        self.assertTrue(all(not entity.internal for entity in fair.entities))
        self.assertTrue(
            all(pending.action.owner == fair.owner for pending in fair.pending_actions)
        )

    def test_dynamic_mask_covers_affordability_sources_and_deployment(self) -> None:
        env, _ = _make_env()
        observations, _ = _reset(env)
        mask = observations[0].action_mask
        self.assertTrue(mask.kinds[ActionKind.WAIT.value])
        self.assertTrue(mask.kinds[ActionKind.WAIT_UNTIL_AFFORDABLE.value])
        self.assertTrue(mask.kinds[ActionKind.PLAY_CARD.value])
        self.assertFalse(mask.kinds[ActionKind.ACTIVATE_ABILITY.value])
        self.assertEqual(mask.hand_slots, (True, False, True, False))
        self.assertEqual(mask.reasons["slots"]["1"], "insufficient_elixir")
        self.assertEqual(mask.reasons["slots"]["3"], "not_in_hand")

        troop = mask.placement_masks["0"]
        spell = mask.placement_masks["2"]
        self.assertEqual(troop["shape"], (32, 18))
        self.assertTrue(troop["row_major"][10][8])
        self.assertFalse(troop["row_major"][25][8])
        self.assertEqual(spell["placement_rule"], PlacementRule.ANYWHERE.value)
        self.assertTrue(spell["row_major"][0][0])
        self.assertTrue(spell["row_major"][31][17])

    def test_destroyed_princess_tower_opens_only_its_live_lane_pocket(self) -> None:
        env, native = _make_env()
        observations, _ = _reset(env)
        opening = observations[0].action_mask.placement_masks["0"]["row_major"]
        self.assertFalse(opening[17][2])
        self.assertFalse(opening[17][13])

        def destroy_left_princess(
            state: dict[str, Any],
            _old: int,
            _new: int,
        ) -> None:
            tower = next(
                item
                for item in state["objects"]
                if item["x"] == 3500 and item["y"] == 25500
            )
            tower["hp"] = 0

        native.step_mutators.append(destroy_left_princess)
        observations, *_ = env.step(
            {0: ActionV1.wait(0), 1: ActionV1.wait(1)}
        )
        expanded = observations[0].action_mask.placement_masks["0"]["row_major"]
        self.assertTrue(expanded[17][2])
        self.assertTrue(expanded[17][8])
        self.assertFalse(expanded[17][9])
        self.assertFalse(expanded[17][13])

    def test_mirror_hand_runtime_binds_effective_identity_cost_and_form(self) -> None:
        env, _native = _make_env()
        runtime = env._hand_runtime_contract(
            {
                "handIndex": 2,
                "cardId": 28_000_006,
                "commandCardId": 26_000_014,
                "cost": 5,
                "cardParameter": (5 << 28) | (3 << 22),
                "formCode": 0,
            }
        )

        self.assertEqual(runtime["visible_card_id"], 28_000_006)
        self.assertEqual(runtime["effective_card_id"], 26_000_014)
        self.assertEqual(runtime["native_effective_card_id"], 26_000_014)
        self.assertEqual(runtime["effective_cost"], 5.0)
        self.assertEqual(runtime["form_code"], 0)
        self.assertEqual(runtime["native_form_code"], 0)
        hero_runtime = env._hand_runtime_contract(
            {
                "handIndex": 2,
                "cardId": 28_000_006,
                "commandCardId": 203_000_034,
                "cost": 6,
                "cardParameter": (6 << 28) | (3 << 22),
                "formCode": 0,
            }
        )
        self.assertEqual(hero_runtime["effective_card_id"], 26_000_034)
        self.assertEqual(
            hero_runtime["native_effective_card_id"],
            203_000_034,
        )
        self.assertEqual(hero_runtime["form_code"], 2)
        self.assertEqual(hero_runtime["native_form_code"], 0)
        self.assertEqual(
            env._policy_effective_identity(26_000_034, 2),
            (26_000_034, 2),
        )
        ordinary_hero_runtime = env._hand_runtime_contract(
            {
                "handIndex": 1,
                "cardId": 26_000_034,
                "commandCardId": 203_000_034,
                "cost": 5,
                "cardParameter": (5 << 28) | (2 << 22),
                "formCode": 0,
            }
        )
        self.assertEqual(
            ordinary_hero_runtime["effective_card_id"],
            26_000_034,
        )
        self.assertEqual(ordinary_hero_runtime["form_code"], 2)
        self.assertEqual(ordinary_hero_runtime["native_form_code"], 0)
        evolution_runtime = env._hand_runtime_contract(
            {
                "handIndex": 3,
                "cardId": 28_000_006,
                "commandCardId": 202_000_006,
                "cost": 5,
                "cardParameter": (5 << 28) | (4 << 22) | 1,
                "formCode": 1,
            },
            form_aliases={202_000_006: 27_000_006},
        )
        self.assertEqual(evolution_runtime["effective_card_id"], 27_000_006)
        self.assertEqual(evolution_runtime["form_code"], 1)
        self.assertEqual(evolution_runtime["native_form_code"], 1)
        with self.assertRaisesRegex(BattleEnvError, "source cost plus one"):
            env._hand_runtime_contract(
                {
                    "handIndex": 2,
                    "cardId": 28_000_006,
                    "commandCardId": 26_000_014,
                    "cost": 1,
                    "cardParameter": (1 << 28) | (3 << 22),
                    "formCode": 0,
                }
            )
        with self.assertRaisesRegex(BattleEnvError, "conflicting native form code"):
            env._hand_runtime_contract(
                {
                    "handIndex": 2,
                    "cardId": 28_000_006,
                    "commandCardId": 203_000_034,
                    "cost": 6,
                    "cardParameter": (6 << 28) | (3 << 22) | 1,
                    "formCode": 1,
                }
            )
        with self.assertRaisesRegex(BattleEnvError, "effective card identity"):
            env._hand_runtime_contract(
                {
                    "handIndex": 2,
                    "cardId": 28_000_006,
                    "commandCardId": 28_000_006,
                    "cost": 1,
                    "cardParameter": (1 << 28) | (3 << 22),
                    "formCode": 0,
                }
            )

    def test_policy_form_requires_static_and_episode_slot_evidence(self) -> None:
        env, _native = _make_env()

        with self.assertRaisesRegex(BattleEnvError, "unsupported policy form"):
            env._policy_effective_identity(26_000_005, 1)

        _reset(env)
        raw = copy.deepcopy(env.raw_observation)
        knight = raw["players"][0]["hand"][0]
        knight.update(
            {
                "commandCardId": 203_000_000,
                "formCode": 0,
            }
        )
        with self.assertRaisesRegex(BattleEnvError, "not enabled by the episode"):
            env._hand_runtime_contract(knight, owner=0)

        forms = (2, 0, 0, 0, 0, 0, 0, 0)
        env.reset(
            match_config=MatchConfig(
                deck0=SEMANTIC_BASELINE_DECK,
                deck1=SEMANTIC_BASELINE_DECK,
                deck0_form_availability=forms,
            )
        )
        raw = copy.deepcopy(env.raw_observation)
        raw["players"][0]["hand"][0].update(
            {
                "commandCardId": 203_000_000,
                "formCode": 0,
            }
        )
        runtime = env._hand_runtime_contract(
            raw["players"][0]["hand"][0],
            owner=0,
        )
        self.assertEqual(runtime["effective_card_id"], 26_000_000)
        self.assertEqual(runtime["form_code"], 2)

    def test_private_mirror_visible_card_requires_owner_deck_membership(self) -> None:
        env, _native = _make_env()
        _reset(env)
        raw = copy.deepcopy(env.raw_observation)
        raw["tick"] = int(raw["tick"]) + 1
        raw["players"][0]["hand"][0].update(
            {
                "cardId": 28_000_006,
                "commandCardId": 26_000_000,
                "cost": 4,
                "cardParameter": (4 << 28) | (1 << 22),
                "deckSlot": 0,
                "formCode": 0,
            }
        )

        _native.state = copy.deepcopy(raw)
        with self.assertRaisesRegex(BattleEnvError, "episode deck"):
            env.action_mask(0, raw=raw)

    def test_pending_action_keeps_exact_tick_and_is_actor_private(self) -> None:
        env, native = _make_env()
        _reset(env)
        play = ActionV1.play(
            owner=0,
            hand_slot=0,
            grid=(8, 10),
            execute_offset_ticks=3,
            next_decision_ticks=1,
            action_id="owner0-play",
        )
        observations, _, _, _, infos = env.step(
            {0: play, 1: ActionV1.wait(1, ticks=1)}
        )
        self.assertEqual(len(native.queued), 1)
        self.assertEqual(native.queued[0]["execute_tick"], FIRST_PLAYABLE_TICK + 3)
        hand_action: HandAction = native.queued[0]["action"]
        self.assertEqual((hand_action.x, hand_action.y), cell_to_world((8, 10)))
        self.assertEqual(len(observations[0].pending_actions), 1)
        pending = observations[0].pending_actions[0]
        self.assertEqual(pending.requested_tick, FIRST_PLAYABLE_TICK)
        self.assertEqual(pending.expected_execution_tick, FIRST_PLAYABLE_TICK + 3)
        self.assertEqual(pending.status, "queued")
        self.assertFalse(
            observations[0].action_mask.hand_slots[0]
        )
        self.assertEqual(
            observations[0].action_mask.reasons["slots"]["0"],
            "pending_execution",
        )
        self.assertEqual(
            observations[0].action_mask.reasons["reserved_elixir"],
            3.0,
        )
        self.assertEqual(observations[1].pending_actions, ())
        self.assertEqual(infos[0]["pending_count"], 1)
        owner0_requests = [
            event for event in observations[0].events if event.event_type == "action_requested"
        ]
        owner1_requests = [
            event for event in observations[1].events if event.event_type == "action_requested"
        ]
        self.assertEqual([event.owner for event in owner0_requests], [0])
        self.assertEqual([event.owner for event in owner1_requests], [1])

        observations, _, _, _, infos = env.step(
            {0: ActionV1.wait(0, ticks=2), 1: ActionV1.wait(1, ticks=2)}
        )
        self.assertEqual(observations[0].pending_actions, ())
        self.assertEqual(infos[0]["pending_count"], 0)
        executed = [
            event for event in observations[0].events if event.event_type == "action_executed"
        ]
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0].tick, FIRST_PLAYABLE_TICK + 3)
        self.assertEqual(
            executed[0].data["expected_execution_tick"], FIRST_PLAYABLE_TICK + 3
        )

    def test_five_tick_offset_executes_before_next_policy_observation(self) -> None:
        env, native = _make_env()
        _reset(env)
        play = ActionV1.play(
            owner=0,
            hand_slot=0,
            grid=(8, 10),
            execute_offset_ticks=5,
            next_decision_ticks=5,
            action_id="policy-boundary-play",
        )

        observations, _, _, _, infos = env.step(
            {0: play, 1: ActionV1.wait(1, ticks=5)}
        )

        self.assertEqual(native.queued, [])
        self.assertEqual(observations[0].pending_actions, ())
        self.assertEqual(infos[0]["pending_count"], 0)
        executed = [
            event
            for event in observations[0].events
            if event.event_type == "action_executed"
            and event.data.get("action_id") == "policy-boundary-play"
        ]
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0].tick, FIRST_PLAYABLE_TICK + 5)

    def test_terminal_reward_is_zero_sum_and_crowns_follow_dead_king(self) -> None:
        env, native = _make_env()
        _reset(env)

        def destroy_owner1_king(state: dict[str, Any], _old: int, _new: int) -> None:
            king = next(
                item
                for item in state["objects"]
                if item["x"] == 9000 and item["y"] == 29000
            )
            king["hp"] = 0
            state["ended"] = True

        native.step_mutators.append(destroy_owner1_king)
        observations, rewards, terminations, truncations, infos = env.step(
            {0: ActionV1.wait(0), 1: ActionV1.wait(1)}
        )
        self.assertEqual(rewards, {0: 1.0, 1: -1.0})
        self.assertAlmostEqual(rewards[0] + rewards[1], 0.0)
        self.assertEqual(terminations, {0: True, 1: True, "__all__": True})
        self.assertEqual(truncations, {0: False, 1: False, "__all__": False})
        terminal = observations[0].terminal
        self.assertTrue(terminal.ended)
        self.assertEqual(terminal.winner, 0)
        self.assertEqual(terminal.result_by_owner, (1.0, -1.0))
        self.assertEqual(terminal.reason, "king_tower_destroyed")
        self.assertEqual(infos[0]["terminal_reason"], "king_tower_destroyed")
        players = {player.owner: player for player in observations[0].players}
        self.assertEqual(players[0].crowns, 3)
        self.assertEqual(players[1].crowns, 0)

    def test_event_history_derives_spawn_damage_death_and_tower_damage(self) -> None:
        env, native = _make_env(FakeNativeEnv(initial_entity=True))
        _reset(env)

        def damage_and_spawn(state: dict[str, Any], _old: int, _new: int) -> None:
            troop = next(item for item in state["objects"] if item.get("objectIndex") == 500)
            troop["hp"] = 70
            state["objects"].append(
                _troop_object(object_index=501, owner=0, card_id=26000005, x=9500, y=14500)
            )
            tower = next(
                item for item in state["objects"] if item["x"] == 3500 and item["y"] == 25500
            )
            tower["hp"] = 800

        def remove_original(state: dict[str, Any], _old: int, _new: int) -> None:
            state["objects"] = [
                item for item in state["objects"] if item.get("objectIndex") != 500
            ]

        native.step_mutators.extend((damage_and_spawn, remove_original))
        observations, *_ = env.step(
            {0: ActionV1.wait(0), 1: ActionV1.wait(1)}
        )
        public_types = {event.event_type for event in observations[0].events}
        self.assertTrue({"damage", "spawn", "tower_damage"}.issubset(public_types))
        observations, *_ = env.step(
            {0: ActionV1.wait(0), 1: ActionV1.wait(1)}
        )
        history_types = [event.event_type for event in observations[0].events]
        self.assertIn("death_or_despawn", history_types)
        self.assertGreaterEqual(len(history_types), 4)

    def test_same_seed_and_actions_produce_identical_replay_trace(self) -> None:
        def run_once() -> tuple:
            env, _ = _make_env()
            _reset(env, seed=99)
            env.step({0: ActionV1.wait(0, ticks=3), 1: ActionV1.wait(1, ticks=2)})
            env.step({0: ActionV1.wait(0, ticks=1), 1: ActionV1.wait(1, ticks=1)})
            return env.trace

        first = run_once()
        second = run_once()
        self.assertEqual(first, second)
        self.assertEqual(
            [(item.start_native_tick, item.end_native_tick, item.advance_ticks) for item in first],
            [(90, 92, 2), (92, 93, 1)],
        )

    def test_illegal_enemy_side_deployment_is_rejected_before_native_queue(self) -> None:
        env, native = _make_env()
        _reset(env)
        action = ActionV1.play(owner=0, hand_slot=0, grid=(8, 25))
        with self.assertRaises(BattleEnvError):
            env.step({0: action, 1: ActionV1.wait(1)})
        self.assertEqual(native.queued, [])

    def test_subcell_offset_is_revalidated_against_final_world_terrain(self) -> None:
        env, native = _make_env()
        observations, _ = _reset(env)
        self.assertTrue(observations[0].action_mask.placement_masks["0"]["row_major"][13][0])
        action = ActionV1.play(
            owner=0,
            hand_slot=0,
            grid=(0, 13),
            subcell_offset=(0.0, 0.5),
        )

        with self.assertRaisesRegex(BattleEnvError, "after subcell offset"):
            env.step({0: action, 1: ActionV1.wait(1)})

        self.assertEqual(native.queued, [])


if __name__ == "__main__":
    unittest.main()
