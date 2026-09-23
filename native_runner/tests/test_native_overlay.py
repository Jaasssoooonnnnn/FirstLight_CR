from __future__ import annotations

import unittest
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

from native_runner.arena import OccupiedFootprintV1, card_placement_legal_world
from native_runner.battle_env import TOWER_LAYOUT
from native_runner.card_specs import build_card_catalog
from native_runner.native_overlay import (
    ARENA_RECT,
    NativeOverlayApp,
    PlayResult,
    Rect,
    Selection,
    WORLD_PROJECTION_RECT,
    ability_control,
    all_card_hits,
    arena_source_to_world,
    hit_card_source,
    largest_portrait_viewport,
    model_hud_mask_rects,
    nearest_legal_placement,
    native_speed,
    source_rect_to_viewport,
    viewport_to_source,
)


class NativeOverlayGeometryTests(unittest.TestCase):
    @staticmethod
    def _pointer_app() -> tuple[NativeOverlayApp, list[tuple[Any, ...]]]:
        app = object.__new__(NativeOverlayApp)
        app.viewport = Rect(0, 0, 1080, 1920)
        app.latest_status = {"mode": "native-render"}
        app.latest_observation = {"players": [{"owner": 1, "elixirRaw": 100_000, "hand": [
            {"handIndex": 0, "cardId": 26_000_001, "cost": 3},
            {"handIndex": 1, "cardId": 26_000_002, "cost": 3},
        ]}]}
        app.owner = None
        app.card_names = {26_000_001: "Archers", 26_000_002: "Knight"}
        app.selection = None
        app.drag_selection = None
        app.pending_plays = deque()
        app.action_future = None
        app.last_crosshair = None
        app._set_message = lambda *_args, **_kwargs: None
        app._draw_feedback = lambda: None
        submitted: list[tuple[Any, ...]] = []
        app._submit_action = lambda *args: submitted.append(args)
        return app, submitted

    def test_drag_from_card_releases_one_play_on_arena(self) -> None:
        app, submitted = self._pointer_app()
        app._on_viewport_click(SimpleNamespace(x=211, y=1749))
        self.assertIsNotNone(app.drag_selection)
        app._on_viewport_drag(SimpleNamespace(x=540, y=1000))
        self.assertIsNotNone(app.last_crosshair)
        app._on_viewport_release(SimpleNamespace(x=540, y=1000))
        app._on_viewport_release(SimpleNamespace(x=540, y=1000))

        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0][0], "play")
        self.assertEqual(submitted[0][2].card_id, 26_000_001)
        self.assertEqual(submitted[0][3:], arena_source_to_world(540, 1000))

    def test_click_then_click_still_plays_and_release_outside_cancels_drag(self) -> None:
        app, submitted = self._pointer_app()
        app._on_viewport_click(SimpleNamespace(x=211, y=1749))
        app._on_viewport_release(SimpleNamespace(x=211, y=1749))
        self.assertEqual(submitted, [])
        app._on_viewport_click(SimpleNamespace(x=540, y=1000))
        app._on_viewport_release(SimpleNamespace(x=540, y=1000))
        self.assertEqual(len(submitted), 1)

        app, submitted = self._pointer_app()
        app._on_viewport_click(SimpleNamespace(x=211, y=1749))
        app._on_viewport_release(SimpleNamespace(x=540, y=1800))
        self.assertEqual(submitted, [])
        self.assertIsNotNone(app.selection)

    def test_second_drag_waits_for_first_play_then_submits(self) -> None:
        app, submitted = self._pointer_app()
        app.last_poll = 1.0
        app._on_viewport_click(SimpleNamespace(x=211, y=1749))
        first = app.selection
        app._on_viewport_release(SimpleNamespace(x=540, y=1000))
        self.assertEqual(len(submitted), 1)
        first_future: Future[PlayResult] = Future()
        app.action_future = first_future
        app.action_kind = "play"

        app._on_viewport_click(SimpleNamespace(x=354, y=1749))
        second = app.selection
        app._on_viewport_release(SimpleNamespace(x=540, y=1200))
        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(app.pending_plays), 1)
        self.assertEqual(app.pending_plays[0][0], second)

        first_future.set_result(PlayResult("accepted", first, 9_500, 15_500, "native hand rotated"))
        app._consume_action()
        self.assertEqual(app.selection, second)
        fresh = {"players": [{"owner": 1, "elixirRaw": 70_000, "hand": [
            {"handIndex": 0, "cardId": 26_000_003, "cost": 3},
            {"handIndex": 1, "cardId": 26_000_002, "cost": 3},
        ]}], "queuedCommands": 1}
        app.latest_observation = fresh
        app._dispatch_pending_play()
        self.assertEqual(len(submitted), 1)
        fresh["queuedCommands"] = 0
        app._dispatch_pending_play()
        self.assertEqual(len(submitted), 2)
        self.assertEqual(submitted[1][2], second)
        self.assertEqual(submitted[1][3:], arena_source_to_world(540, 1200))
        self.assertEqual(len(app.pending_plays), 0)

    @staticmethod
    def _tower_observation() -> dict[str, Any]:
        return {
            "objects": [
                {"owner": owner, "x": x, "y": y, "hp": 1000, "maxHp": 1000, "cardId": 0}
                for _id, owner, _kind, x, y in TOWER_LAYOUT
            ]
        }

    def test_manual_placement_snaps_from_tower_and_occupied_building(self) -> None:
        specs = build_card_catalog().by_id
        observation = self._tower_observation()
        troop = {"cardId": 26_000_001, "commandCardId": 26_000_001, "formCode": 0}
        troop_target = nearest_legal_placement(observation, troop, 0, 9_500, 3_500, specs)
        self.assertNotEqual(troop_target, (9_500, 3_500))
        self.assertTrue(card_placement_legal_world(specs[26_000_001], 0, troop_target))
        enemy_side = nearest_legal_placement(observation, troop, 0, 9_500, 24_500, specs)
        self.assertEqual(enemy_side[0], 9_500)
        self.assertLessEqual(enemy_side[1], 16_500)
        self.assertTrue(card_placement_legal_world(specs[26_000_001], 0, enemy_side))
        top_side = nearest_legal_placement(observation, troop, 1, 9_500, 6_500, specs)
        self.assertEqual(top_side[0], 9_500)
        self.assertGreaterEqual(top_side[1], 15_500)
        self.assertEqual(nearest_legal_placement(observation, troop, 0, 500, 24_500, specs)[0], 500)
        self.assertEqual(nearest_legal_placement(observation, troop, 1, 17_500, 6_500, specs)[0], 17_500)
        fireball = {"cardId": 28_000_000, "commandCardId": 28_000_000, "formCode": 0}
        self.assertEqual(nearest_legal_placement(observation, fireball, 0, 9_500, 24_500, specs), (9_500, 24_500))

        observation["objects"].append(
            {"owner": 0, "x": 9_000, "y": 6_000, "hp": 1000, "maxHp": 1000, "cardId": 27_000_006}
        )
        building = {"cardId": 27_000_006, "commandCardId": 27_000_006, "formCode": 0}
        target = nearest_legal_placement(observation, building, 0, 9_500, 6_500, specs)
        self.assertNotEqual(target, (9_000, 6_000))
        self.assertTrue(card_placement_legal_world(
            specs[27_000_006], 0, target,
            occupied_building_footprints=(OccupiedFootprintV1(9_000, 6_000, 2, 2, "existing"),),
        ))

    def test_clicks_behind_own_towers_stay_behind_them(self) -> None:
        specs = build_card_catalog().by_id
        observation = self._tower_observation()
        troop = {"cardId": 26_000_001}
        for source_x, source_y, tower_y in ((275, 1390, 25_500), (540, 1530, 29_000)):
            requested = arena_source_to_world(source_x, source_y)
            actual = nearest_legal_placement(observation, troop, 1, *requested, specs)
            self.assertGreater(actual[1], tower_y)
            self.assertEqual(actual[0], requested[0])
        for tower_x, tower_y, requested_y in ((3_500, 6_500, 5_500), (9_500, 3_000, 2_500)):
            actual = nearest_legal_placement(observation, troop, 0, tower_x, requested_y, specs)
            self.assertLess(actual[1], tower_y)
            self.assertEqual(actual[0], tower_x)

    def test_manual_play_submits_snapped_native_coordinates(self) -> None:
        specs = build_card_catalog().by_id
        before = self._tower_observation()
        before["players"] = [{"owner": 0, "elixirRaw": 100_000, "hand": [
            {"handIndex": 0, "cardId": 27_000_006, "commandCardId": 27_000_006, "formCode": 0, "cost": 4}
        ]}]
        after = {"players": [{"owner": 0, "hand": [{"handIndex": 0, "cardId": 26_000_001}]}]}

        class StubEnv:
            action = None

            def observe(self):
                return after if self.action is not None else before

            def play_immediate(self, action):
                self.action = action

        env = StubEnv()
        app = object.__new__(NativeOverlayApp)
        app.env = env
        app.owner = None
        app.card_specs = specs
        result = app._run_play(Selection(0, 0, 0, 27_000_006, 4, "Tesla"), 9_500, 6_500)

        self.assertEqual(result.outcome, "accepted")
        self.assertEqual((env.action.x, env.action.y), (result.world_x, result.world_y))
        self.assertNotEqual((env.action.x, env.action.y), (9_500, 6_500))
        self.assertTrue(card_placement_legal_world(specs[27_000_006], 0, (env.action.x, env.action.y)))

    def test_native_speed_treats_null_and_invalid_status_as_unavailable(self) -> None:
        self.assertIsNone(native_speed({"speed": None}))
        self.assertIsNone(native_speed({"speed": "not-a-number"}))
        self.assertIsNone(native_speed({"speed": float("nan")}))
        self.assertEqual(native_speed({"speed": "2"}), 2.0)

    def test_bottom_anchored_portrait_fit_removes_top_toolbar(self) -> None:
        viewport = largest_portrait_viewport(Rect(100, 50, 640, 1050))
        self.assertEqual(viewport, Rect(100, 90, 640, 1050))

    def test_wide_container_is_centered(self) -> None:
        viewport = largest_portrait_viewport(Rect(0, 0, 1000, 960))
        self.assertEqual(viewport, Rect(230, 0, 770, 960))

    def test_both_visible_hands_map_to_native_selector_indices(self) -> None:
        upper_left = hit_card_source(211, 170)
        upper_right = hit_card_source(640, 170)
        lower_left = hit_card_source(211, 1749)
        lower_right = hit_card_source(640, 1749)
        self.assertEqual((upper_left.owner, upper_left.hand_index), (0, 3))
        self.assertEqual((upper_right.owner, upper_right.hand_index), (0, 0))
        self.assertEqual((lower_left.owner, lower_left.hand_index), (1, 0))
        self.assertEqual((lower_right.owner, lower_right.hand_index), (1, 3))

    def test_model_hud_mask_covers_cards_and_elixir_on_either_side(self) -> None:
        for human_owner, model_owner, elixir_y in ((1, 0, 40), (0, 1, 1880)):
            masks = model_hud_mask_rects(human_owner)
            for hit in (item for item in all_card_hits() if item.owner == model_owner):
                center_x = (hit.source_rect.left + hit.source_rect.right) / 2
                center_y = (hit.source_rect.top + hit.source_rect.bottom) / 2
                self.assertTrue(any(rect.contains(center_x, center_y) for rect in masks))
            self.assertTrue(any(rect.contains(900, elixir_y) for rect in masks))
            self.assertFalse(any(rect.contains(900, 250) for rect in masks))

    def test_feedback_draws_model_mask_only_when_switched_on(self) -> None:
        app = object.__new__(NativeOverlayApp)
        app.viewport = Rect(0, 0, 1080, 1920)
        app.owner = 1
        app.selection = None
        app.last_crosshair = None
        app._mask_items = []
        app._mask_viewport_size = None
        drawn: dict[int, tuple[int, int, int, int]] = {}
        created = 0

        class Canvas:
            def delete(self, tag):
                if tag == "model-mask":
                    drawn.clear()

            def create_rectangle(self, left, top, right, bottom, **options):
                nonlocal created
                if options.get("fill") == "#10151f":
                    created += 1
                    drawn[created] = (left, top, right, bottom)
                    return created
                return 0

            def coords(self, item, left, top, right, bottom):
                drawn[item] = (left, top, right, bottom)

            def tag_raise(self, _tag):
                pass

        app.canvas = Canvas()
        app.mask_model_hud_var = SimpleNamespace(get=lambda: True)
        app._draw_feedback()
        self.assertEqual(list(drawn.values()), [(0, 0, 1080, 85), (0, 80, 715, 270)])
        app._draw_feedback()
        self.assertEqual(created, 2)
        self.assertEqual(len(drawn), 2)
        app.viewport = Rect(0, 0, 540, 960)
        app._draw_feedback()
        self.assertEqual(list(drawn.values()), [(0, 0, 540, 42), (0, 40, 358, 135)])
        app.mask_model_hud_var = SimpleNamespace(get=lambda: False)
        app._draw_feedback()
        self.assertEqual(drawn, {})

    def test_measured_grid_cell_centers_round_trip(self) -> None:
        def source_for_cell(column: int, row: int) -> tuple[float, float]:
            return (
                WORLD_PROJECTION_RECT.left
                + (column + 0.5) / 18 * WORLD_PROJECTION_RECT.width,
                WORLD_PROJECTION_RECT.top
                + (row + 0.5) / 32 * WORLD_PROJECTION_RECT.height,
            )

        self.assertEqual(arena_source_to_world(*source_for_cell(0, 0)), (500, 500))
        self.assertEqual(
            arena_source_to_world(*source_for_cell(17, 31)),
            (17_500, 31_500),
        )
        self.assertEqual(
            arena_source_to_world(*source_for_cell(9, 16)),
            (9_500, 16_500),
        )

    def test_visual_padding_clamps_to_outer_cells(self) -> None:
        self.assertEqual(arena_source_to_world(ARENA_RECT.left, 1350), (500, 24_500))
        self.assertEqual(
            arena_source_to_world(ARENA_RECT.right - 0.001, 1350),
            (17_500, 24_500),
        )

    def test_back_center_click_reaches_last_row(self) -> None:
        self.assertEqual(arena_source_to_world(540, 1647), (9_500, 31_500))

    def test_viewport_scaling_round_trip(self) -> None:
        viewport = Rect(0, 0, 540, 960)
        source = viewport_to_source(viewport, 270, 480)
        self.assertEqual(source, (540.0, 960.0))
        scaled_arena = source_rect_to_viewport(viewport, ARENA_RECT)
        self.assertEqual(scaled_arena, Rect(27, 141, 513, 833))

    def test_ready_ability_resolves_exact_native_entity_key(self) -> None:
        rich = {
            "players": [
                {
                    "owner": 0,
                    "abilityRuntime": [
                        {
                            "actionDataName": "Knight_hero_Ability",
                            "available": True,
                            "buttonStateLabel": "Ready",
                            "remainingCooldownMs": 0,
                            "championEntityKeys": [[0, -2, 5_000_028]],
                        }
                    ],
                }
            ]
        }

        control = ability_control(rich, 0)

        self.assertTrue(control.ready)
        self.assertEqual(control.source_entity_key, (0, -2, 5_000_028))
        self.assertEqual(control.ability_name, "Knight_hero_Ability")

    def test_ability_control_disables_cooldown_and_ambiguous_sources(self) -> None:
        cooldown = {
            "players": [
                {
                    "owner": 1,
                    "abilityRuntime": [
                        {
                            "actionDataName": "Knight_hero_Ability",
                            "available": False,
                            "buttonStateLabel": "Cooldown",
                            "remainingCooldownMs": 12_400,
                            "championEntityKeys": [[1, -2, 5_000_029]],
                        }
                    ],
                }
            ]
        }
        ambiguous = {
            "players": [
                {
                    "owner": 0,
                    "abilityRuntime": [
                        {
                            "actionDataName": "Knight_hero_Ability",
                            "available": True,
                            "buttonStateLabel": "Ready",
                            "championEntityKeys": [[0, -2, 5_000_028]],
                        },
                        {
                            "actionDataName": "Other_Ability",
                            "available": True,
                            "buttonStateLabel": "Ready",
                            "championEntityKeys": [[0, -2, 5_000_030]],
                        },
                    ],
                }
            ]
        }

        cooldown_control = ability_control(cooldown, 1)
        ambiguous_control = ability_control(ambiguous, 0)

        self.assertFalse(cooldown_control.ready)
        self.assertIn("12.4s", cooldown_control.text)
        self.assertFalse(ambiguous_control.ready)
        self.assertIn("来源不唯一", ambiguous_control.text)

    def test_limited_availability_resolves_exact_native_entity_key(self) -> None:
        rich = {
            "players": [
                {
                    "owner": 0,
                    "abilityRuntime": [
                        {
                            "actionDataName": "Tombstone_hero_Ability",
                            "available": True,
                            "buttonStateLabel": "LimitedAvailability",
                            "remainingCooldownMs": 0,
                            "championEntityKeys": [[0, -2, 5_000_191]],
                        }
                    ],
                }
            ]
        }

        control = ability_control(rich, 0)

        self.assertTrue(control.ready)
        self.assertEqual(control.source_entity_key, (0, -2, 5_000_191))
        self.assertEqual(control.ability_name, "Tombstone_hero_Ability")

    def test_run_ability_rechecks_runtime_and_calls_native_api(self) -> None:
        class StubEnv:
            def __init__(self) -> None:
                self.action = None

            def observe_rich(self):
                return {
                    "players": [
                        {
                            "owner": 0,
                            "abilityRuntime": [
                                {
                                    "actionDataName": "Knight_hero_Ability",
                                    "available": True,
                                    "buttonStateLabel": "Ready",
                                    "championEntityKeys": [[0, -2, 5_000_028]],
                                }
                            ],
                        }
                    ]
                }

            def activate_ability(self, action):
                self.action = action
                return {"ok": True, "tick": 42}

        env = StubEnv()
        app = object.__new__(NativeOverlayApp)
        app.env = env
        app.owner = None

        result = app._run_ability(0)

        self.assertEqual(
            (env.action.owner, env.action.object_index, env.action.secondary_index),
            (0, -2, 5_000_028),
        )
        self.assertEqual(result["abilityName"], "Knight_hero_Ability")


if __name__ == "__main__":
    unittest.main()
