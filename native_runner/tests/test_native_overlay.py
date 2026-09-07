from __future__ import annotations

import unittest

from native_runner.native_overlay import (
    ARENA_RECT,
    NativeOverlayApp,
    Rect,
    WORLD_PROJECTION_RECT,
    ability_control,
    arena_source_to_world,
    hit_card_source,
    largest_portrait_viewport,
    native_speed,
    source_rect_to_viewport,
    viewport_to_source,
)


class NativeOverlayGeometryTests(unittest.TestCase):
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
        self.assertEqual(arena_source_to_world(ARENA_RECT.left, 1350), (500, 23_500))
        self.assertEqual(
            arena_source_to_world(ARENA_RECT.right - 0.001, 1350),
            (17_500, 23_500),
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
