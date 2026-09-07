from __future__ import annotations

import unittest
from dataclasses import replace

from native_runner.contracts import (
    ActionKind,
    ActionMaskV1,
    ActionV1,
    EntityStateV1,
    EventV1,
    ObservationTier,
    ObservationV1,
    PlayerStateV1,
    TargetKind,
    TowerStateV1,
)
from native_runner.perspective import PerspectiveError, PerspectiveTransformV1


def _rows(marker_x: int, marker_y: int) -> list[list[bool]]:
    rows = [[False for _ in range(18)] for _ in range(32)]
    rows[marker_y][marker_x] = True
    return rows


def _observation() -> ObservationV1:
    action_mask = ActionMaskV1(
        kinds={"wait": True, "play_card": True},
        hand_slots=(True, False, True, False),
        placement_masks={
            str(slot): {
                "card_id": 11 + slot,
                "shape": [32, 18],
                "row_major": _rows(slot, 31 - slot),
            }
            for slot in range(4)
        },
        reasons={"slots": {str(slot): f"slot-{slot}" for slot in range(4)}},
    )
    return ObservationV1(
        tier=ObservationTier.FAIR,
        owner=1,
        tick=9,
        players=(
            PlayerStateV1(owner=0),
            PlayerStateV1(
                owner=1,
                hand=(11, 12, 13, 14),
                deck=(11, 12, 13, 14, 15, 16, 17, 18),
                private_state_visible=True,
            ),
        ),
        towers=(
            TowerStateV1(
                entity_id=1,
                owner=1,
                tower_kind="princess-left",
                position=(500.0, 31_500.0),
                hitpoints=100.0,
                max_hitpoints=100.0,
                tower_troop_id=159_000_000,
            ),
        ),
        entities=(
            EntityStateV1(
                entity_id=2,
                owner=1,
                card_id=11,
                entity_kind="troop",
                position=(1_500.0, 30_500.0),
                velocity=(2.0, 3.0),
                movement_target=(2_500.0, 29_500.0),
            ),
        ),
        events=(
            EventV1(
                tick=9,
                event_type="spawn",
                owner=1,
                position=(500.0, 31_500.0),
            ),
        ),
        action_mask=action_mask,

    )


class PerspectiveTransformTests(unittest.TestCase):
    def test_owner_one_is_rotated_to_own_side_bottom_with_all_masks(self) -> None:
        transform = PerspectiveTransformV1(
            actor_owner=1,
            hand_slot_permutation=(2, 0, 3, 1),
            deck_permutation=(7, 6, 5, 4, 3, 2, 1, 0),
        )
        model = transform.observation_policy_metadata_to_model(_observation())
        own = model.players[0]
        self.assertEqual(own.owner, 1)
        self.assertEqual(own.hand, (13, 11, 14, 12))
        self.assertEqual(own.deck, (18, 17, 16, 15, 14, 13, 12, 11))
        # Scene coordinates stay native; the V4 tensorizer applies flip signs.
        self.assertEqual(model.towers[0].position, (500.0, 31_500.0))
        self.assertEqual(model.entities[0].velocity, (2.0, 3.0))
        self.assertTrue(transform.flip_x)
        self.assertTrue(transform.flip_y)

        mask = model.action_mask
        self.assertEqual(mask.hand_slots, (True, True, False, False))
        slot_zero = mask.placement_masks["0"]
        self.assertEqual(slot_zero["source_native_hand_slot"], 2)
        self.assertTrue(slot_zero["row_major"][2][15])
        self.assertIsNone(model.raster)

    def test_action_spatial_and_slot_transform_is_reversible(self) -> None:
        transform = PerspectiveTransformV1(
            actor_owner=1,
            hand_slot_permutation=(2, 0, 3, 1),
        )
        model_action = ActionV1(
            owner=1,
            kind=ActionKind.PLAY_CARD,
            hand_slot=0,
            target_kind=TargetKind.GRID,
            target_grid=(1, 2),
            subcell_offset=(0.25, -0.5),
        )
        native = transform.action_to_native(model_action)
        self.assertEqual(native.hand_slot, 2)
        self.assertEqual(native.target_grid, (16, 29))
        self.assertEqual(native.subcell_offset, (-0.25, 0.5))
        round_trip = transform.action_to_model(native)
        self.assertEqual(round_trip.hand_slot, model_action.hand_slot)
        self.assertEqual(round_trip.target_grid, model_action.target_grid)
        self.assertEqual(round_trip.subcell_offset, model_action.subcell_offset)

    def test_partial_native_hand_preserves_exact_slot_mapping(self) -> None:
        observation = _observation()
        observation = replace(observation, players=tuple(
            replace(player, hand=(11, 13, 14), metadata={
                "hand_slot_by_card": {"11": 0, "13": 2, "14": 3},
                "native_hand_capacity": 4,
            }) if player.owner == 1 else player
            for player in observation.players
        ))
        transform = PerspectiveTransformV1(
            actor_owner=1,
            hand_slot_permutation=(2, 0, 3, 1),
        )

        model = transform.observation_policy_metadata_to_model(observation)
        own = model.players[0]

        self.assertEqual(own.hand, (13, 11, 14))
        self.assertEqual(
            dict(own.metadata["hand_slot_by_card"]),
            {"11": 1, "13": 0, "14": 2},
        )

    def test_horizontal_mirror_is_composed_after_owner_normalization(self) -> None:
        transform = PerspectiveTransformV1(actor_owner=1, horizontal_mirror=True)
        model = transform.observation_policy_metadata_to_model(_observation())
        self.assertFalse(transform.flip_x)
        self.assertTrue(transform.flip_y)
        self.assertTrue(model.action_mask.placement_masks["0"]["row_major"][0][0])
        self.assertIsNone(model.raster)

    def test_none_raster_remains_explicitly_unavailable(self) -> None:
        model = PerspectiveTransformV1(actor_owner=1).observation_policy_metadata_to_model(_observation())
        self.assertIsNone(model.raster)

    def test_rejects_wrong_owner_and_non_permutations(self) -> None:
        with self.assertRaisesRegex(PerspectiveError, "permutation"):
            PerspectiveTransformV1(actor_owner=0, hand_slot_permutation=(0, 0, 1, 2))
        with self.assertRaisesRegex(PerspectiveError, "does not match"):
            PerspectiveTransformV1(actor_owner=0).observation_policy_metadata_to_model(_observation())


if __name__ == "__main__":
    unittest.main()
