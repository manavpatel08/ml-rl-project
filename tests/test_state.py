"""Unit tests for state encoding and relative action mapping.

Uses Python's standard unittest library.
"""

import unittest

from snake_rl.environment import (
    Action,
    Direction,
    Point,
    SnakeEnvironment,
    SnakeObservation,
    get_next_direction,
)
from snake_rl.state import (
    FEATURE_NAMES,
    NUM_STATES,
    TOTAL_FEATURES,
    StateEncoder,
    decode_int_to_features,
    encode_features_to_int,
    encode_observation,
    encode_observation_to_int,
    is_action_dangerous,
)


class TestStateEncoding(unittest.TestCase):
    """Test suite for 11-bit state encoding, danger detection, and integer conversion."""

    def setUp(self) -> None:
        self.default_obs = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(9, 10), Point(8, 10)),
            food=Point(15, 10),
            direction=Direction.RIGHT,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )

    def test_01_feature_count_and_binary_values(self) -> None:
        """Requirement: Exactly 11 binary features (0 or 1) are returned."""
        self.assertEqual(len(FEATURE_NAMES), 11)
        self.assertEqual(TOTAL_FEATURES, 11)

        features = encode_observation(self.default_obs)
        self.assertEqual(len(features), 11)
        for i, val in enumerate(features):
            self.assertIn(val, (0, 1), f"Feature {FEATURE_NAMES[i]} is not binary: {val}")

    def test_02_heading_one_hot_all_four_directions(self) -> None:
        """Requirement: Heading left/right/up/down are correctly one-hot encoded."""
        # Feature indices 3, 4, 5, 6 correspond to heading left, right, up, down
        headings = [
            (Direction.LEFT, (1, 0, 0, 0)),
            (Direction.RIGHT, (0, 1, 0, 0)),
            (Direction.UP, (0, 0, 1, 0)),
            (Direction.DOWN, (0, 0, 0, 1)),
        ]

        for direction, expected_one_hot in headings:
            obs = SnakeObservation(
                head=Point(10, 10),
                body=(Point(10, 10), Point(10, 11)),
                food=Point(15, 15),
                direction=direction,
                grid_size=(20, 20),
                steps=0,
                score=0,
            )
            features = encode_observation(obs)
            self.assertEqual(
                features[3:7],
                expected_one_hot,
                f"Failed for direction {direction.name}",
            )

    def test_03_danger_features_for_walls(self) -> None:
        """Requirement: Danger straight/right/left accurately detect wall boundaries."""
        # Top-Right corner (19, 0) facing RIGHT:
        # Straight (20, 0) -> Wall!
        # Turn Right (DOWN -> (19, 1)) -> In bounds (Safe)
        # Turn Left (UP -> (19, -1)) -> Wall!
        obs_tr = SnakeObservation(
            head=Point(19, 0),
            body=(Point(19, 0), Point(18, 0)),
            food=Point(5, 5),
            direction=Direction.RIGHT,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        features_tr = encode_observation(obs_tr)
        self.assertEqual(features_tr[0], 1, "Danger straight should be 1 (right wall)")
        self.assertEqual(features_tr[1], 0, "Danger right should be 0 (down is safe)")
        self.assertEqual(features_tr[2], 1, "Danger left should be 1 (top wall)")

        # Bottom-Left corner (0, 19) facing LEFT:
        # Straight (-1, 19) -> Wall!
        # Turn Right (UP -> (0, 18)) -> In bounds (Safe)
        # Turn Left (DOWN -> (0, 20)) -> Wall!
        obs_bl = SnakeObservation(
            head=Point(0, 19),
            body=(Point(0, 19), Point(1, 19)),
            food=Point(5, 5),
            direction=Direction.LEFT,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        features_bl = encode_observation(obs_bl)
        self.assertEqual(features_bl[0], 1, "Danger straight should be 1 (left wall)")
        self.assertEqual(features_bl[1], 0, "Danger right should be 0 (up is safe)")
        self.assertEqual(features_bl[2], 1, "Danger left should be 1 (bottom wall)")

    def test_04_danger_features_for_body_and_tail_vacating(self) -> None:
        """Requirement: Danger reflects self-collision and respects tail vacating."""
        # Self-collision setup:
        # Snake body with head at (2, 3) facing DOWN. Body segment at (1, 3).
        # Turn RIGHT from DOWN points LEFT to (1, 3) (Body segment occupied -> Danger!).
        # Turn LEFT from DOWN points RIGHT to (3, 3) (Safe).
        obs_body = SnakeObservation(
            head=Point(2, 3),
            body=(
                Point(2, 3),
                Point(2, 2),
                Point(1, 2),
                Point(1, 3),
                Point(1, 4),
            ),
            food=Point(10, 10),
            direction=Direction.DOWN,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        features_body = encode_observation(obs_body)
        self.assertEqual(features_body[0], 0, "Danger straight should be 0 (down to (2, 4) is safe)")
        self.assertEqual(features_body[1], 1, "Danger right should be 1 (hits body at (1, 3))")
        self.assertEqual(features_body[2], 0, "Danger left should be 0 (right to (3, 3) is safe)")

        # Tail-vacating rule:
        # 4-cell loop: head at (1, 1), tail at (0, 1), facing UP.
        # Turning LEFT points to (0, 1), which is currently the tail.
        # Since no food is eaten, tail vacates, so moving into (0, 1) is LEGAL and NOT dangerous.
        obs_tail = SnakeObservation(
            head=Point(1, 1),
            body=(Point(1, 1), Point(1, 0), Point(0, 0), Point(0, 1)),
            food=Point(15, 15),
            direction=Direction.UP,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        features_tail = encode_observation(obs_tail)
        self.assertEqual(
            features_tail[2],
            0,
            "Danger left must be 0 because tail at (0, 1) vacates on non-growth move",
        )

    def test_05_relative_turns_all_four_headings(self) -> None:
        """Requirement: get_next_direction maps relative actions correctly for all headings."""
        cases = [
            # UP
            (Direction.UP, Action.STRAIGHT, Direction.UP),
            (Direction.UP, Action.TURN_RIGHT, Direction.RIGHT),
            (Direction.UP, Action.TURN_LEFT, Direction.LEFT),
            # RIGHT
            (Direction.RIGHT, Action.STRAIGHT, Direction.RIGHT),
            (Direction.RIGHT, Action.TURN_RIGHT, Direction.DOWN),
            (Direction.RIGHT, Action.TURN_LEFT, Direction.UP),
            # DOWN
            (Direction.DOWN, Action.STRAIGHT, Direction.DOWN),
            (Direction.DOWN, Action.TURN_RIGHT, Direction.LEFT),
            (Direction.DOWN, Action.TURN_LEFT, Direction.RIGHT),
            # LEFT
            (Direction.LEFT, Action.STRAIGHT, Direction.LEFT),
            (Direction.LEFT, Action.TURN_RIGHT, Direction.UP),
            (Direction.LEFT, Action.TURN_LEFT, Direction.DOWN),
        ]

        for current_dir, action, expected_dir in cases:
            res = get_next_direction(current_dir, action)
            self.assertEqual(
                res,
                expected_dir,
                f"Failed: {current_dir.name} + {action.name} -> expected {expected_dir.name}, got {res.name}",
            )

    def test_06_food_direction_features(self) -> None:
        """Requirement: Food left, right, up, down are computed properly."""
        # Head at (10, 10)
        # Food at top-left (5, 5) -> food_left=1, food_up=1
        obs_tl = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(10, 11)),
            food=Point(5, 5),
            direction=Direction.UP,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        f_tl = encode_observation(obs_tl)
        # food_left, food_right, food_up, food_down are indices 7, 8, 9, 10
        self.assertEqual(f_tl[7:11], (1, 0, 1, 0))

        # Food at bottom-right (15, 15) -> food_right=1, food_down=1
        obs_br = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(10, 11)),
            food=Point(15, 15),
            direction=Direction.UP,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        f_br = encode_observation(obs_br)
        self.assertEqual(f_br[7:11], (0, 1, 0, 1))

    def test_07_food_aligned_on_axis_does_not_set_feature(self) -> None:
        """Requirement: Food aligned with head on an axis yields 0 for that axis."""
        # Same horizontal row: food at (15, 10), head at (10, 10)
        # food_y == head_y, so food_up=0, food_down=0
        obs_row = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(9, 10)),
            food=Point(15, 10),
            direction=Direction.RIGHT,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        f_row = encode_observation(obs_row)
        self.assertEqual(f_row[7:11], (0, 1, 0, 0))

        # Same vertical column: food at (10, 5), head at (10, 10)
        # food_x == head_x, so food_left=0, food_right=0
        obs_col = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(10, 11)),
            food=Point(10, 5),
            direction=Direction.UP,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        f_col = encode_observation(obs_col)
        self.assertEqual(f_col[7:11], (0, 0, 1, 0))

        # Food is None (board full): all food features should be 0
        obs_none = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(10, 11)),
            food=None,
            direction=Direction.UP,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        f_none = encode_observation(obs_none)
        self.assertEqual(f_none[7:11], (0, 0, 0, 0))

    def test_08_integer_encoding_range_and_roundtrip(self) -> None:
        """Requirement: All 2,048 integers decode and re-encode identically."""
        self.assertEqual(NUM_STATES, 2048)

        for state_int in range(NUM_STATES):
            features = decode_int_to_features(state_int)
            self.assertEqual(len(features), 11)
            self.assertTrue(all(bit in (0, 1) for bit in features))

            re_encoded = encode_features_to_int(features)
            self.assertEqual(re_encoded, state_int)

            # Check StateEncoder class methods match
            self.assertEqual(StateEncoder.index_to_features(state_int), features)
            self.assertEqual(StateEncoder.features_to_index(features), state_int)

        # Boundary checks
        zeros = (0,) * 11
        ones = (1,) * 11
        self.assertEqual(encode_features_to_int(zeros), 0)
        self.assertEqual(encode_features_to_int(ones), 2047)
        self.assertEqual(decode_int_to_features(0), zeros)
        self.assertEqual(decode_int_to_features(2047), ones)

    def test_09_invalid_feature_lengths_and_values_raise_exceptions(self) -> None:
        """Requirement: Malformed feature vectors and out-of-range state integers raise clear exceptions."""
        # Wrong feature length
        with self.assertRaises(ValueError) as ctx:
            encode_features_to_int((0,) * 10)
        self.assertIn("Expected exactly 11 features", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            encode_features_to_int((0,) * 12)
        self.assertIn("Expected exactly 11 features", str(ctx.exception))

        # Non-binary feature values
        with self.assertRaises(ValueError) as ctx:
            encode_features_to_int((0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertIn("must be 0 or 1", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            encode_features_to_int((0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertIn("must be 0 or 1", str(ctx.exception))

        # Out-of-range integer states
        with self.assertRaises(ValueError):
            decode_int_to_features(-1)

        with self.assertRaises(ValueError):
            decode_int_to_features(2048)

        # Invalid type
        with self.assertRaises(TypeError):
            decode_int_to_features(True)  # type: ignore

        with self.assertRaises(TypeError):
            decode_int_to_features("42")  # type: ignore

    def test_10_state_encoding_does_not_mutate_observation_or_env(self) -> None:
        """Requirement: State encoding does not mutate the observation or environment."""
        env = SnakeEnvironment(seed=42)
        obs = env.reset(seed=42)

        head_before = obs.head
        body_before = obs.body
        food_before = obs.food
        direction_before = obs.direction
        score_before = obs.score
        steps_before = obs.steps

        features = encode_observation(obs)
        state_int = encode_observation_to_int(obs)

        # Verify observation properties did not mutate
        self.assertEqual(obs.head, head_before)
        self.assertEqual(obs.body, body_before)
        self.assertEqual(obs.food, food_before)
        self.assertEqual(obs.direction, direction_before)
        self.assertEqual(obs.score, score_before)
        self.assertEqual(obs.steps, steps_before)

        # Verify environment properties did not mutate
        self.assertEqual(env.head, head_before)
        self.assertEqual(env.food, food_before)
        self.assertEqual(env.direction, direction_before)
        self.assertEqual(env.steps, steps_before)


if __name__ == "__main__":
    unittest.main()
