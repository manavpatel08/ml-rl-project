"""Unit tests for the headless SnakeEnvironment.

Uses Python's standard unittest library.
"""

import unittest

from snake_rl.config import EnvironmentConfig
from snake_rl.environment import Action, Direction, Point, SnakeEnvironment


class TestSnakeEnvironment(unittest.TestCase):
    """Test suite for SnakeEnvironment game mechanics, collision, and reward rules."""

    def test_01_initial_reset_creates_valid_snake_and_food(self) -> None:
        """Requirement 1: Initial reset creates a valid snake and food."""
        config = EnvironmentConfig(grid_width=20, grid_height=20, initial_snake_length=3)
        env = SnakeEnvironment(config=config, seed=42)
        obs = env.reset(seed=42)

        # Check snake length, head, and body coordinates
        self.assertEqual(len(obs.body), 3)
        self.assertEqual(obs.head, Point(10, 10))
        self.assertEqual(obs.body, (Point(10, 10), Point(9, 10), Point(8, 10)))
        self.assertEqual(obs.direction, Direction.RIGHT)
        self.assertEqual(obs.score, 0)
        self.assertEqual(obs.steps, 0)
        self.assertFalse(env.done)

        # Check food validity
        self.assertIsNotNone(obs.food)
        assert obs.food is not None
        self.assertTrue(0 <= obs.food.x < config.grid_width)
        self.assertTrue(0 <= obs.food.y < config.grid_height)
        self.assertNotIn(obs.food, obs.body)

    def test_02_reset_with_same_seed_is_reproducible(self) -> None:
        """Requirement 2: Reset with the same seed is reproducible."""
        env1 = SnakeEnvironment(seed=12345)
        env2 = SnakeEnvironment(seed=12345)

        self.assertEqual(env1.food, env2.food)
        self.assertEqual(env1.head, env2.head)
        self.assertEqual(env1.body, env2.body)

        actions = [
            Action.STRAIGHT,
            Action.TURN_LEFT,
            Action.STRAIGHT,
            Action.TURN_RIGHT,
            Action.STRAIGHT,
        ]

        for act in actions:
            res1 = env1.step(act)
            res2 = env2.step(act)
            self.assertEqual(res1.observation, res2.observation)
            self.assertEqual(res1.reward, res2.reward)
            self.assertEqual(res1.done, res2.done)
            self.assertEqual(res1.score, res2.score)
            self.assertEqual(res1.info, res2.info)

    def test_03_straight_movement_works(self) -> None:
        """Requirement 3: Straight movement advances head and shifts body."""
        env = SnakeEnvironment(seed=1)
        env._set_state_for_testing(
            body=[Point(10, 10), Point(9, 10), Point(8, 10)],
            direction=Direction.RIGHT,
            food=Point(15, 15),
        )

        res = env.step(Action.STRAIGHT)
        self.assertEqual(res.observation.head, Point(11, 10))
        self.assertEqual(
            res.observation.body,
            (Point(11, 10), Point(10, 10), Point(9, 10)),
        )
        self.assertEqual(res.reward, -0.1)
        self.assertEqual(res.score, 0)
        self.assertFalse(res.done)
        self.assertEqual(res.observation.steps, 1)

    def test_04_relative_turns_relative_to_current_heading(self) -> None:
        """Requirement 4: Left and right turns work relative to current heading."""
        env = SnakeEnvironment()

        # From RIGHT -> TURN_RIGHT should face DOWN
        env._set_state_for_testing(
            body=[Point(10, 10), Point(9, 10), Point(8, 10)],
            direction=Direction.RIGHT,
            food=Point(18, 18),
        )
        res1 = env.step(Action.TURN_RIGHT)
        self.assertEqual(env.direction, Direction.DOWN)
        self.assertEqual(res1.observation.head, Point(10, 11))

        # From DOWN -> TURN_RIGHT should face LEFT
        res2 = env.step(Action.TURN_RIGHT)
        self.assertEqual(env.direction, Direction.LEFT)
        self.assertEqual(res2.observation.head, Point(9, 11))

        # From LEFT -> TURN_RIGHT should face UP
        res3 = env.step(Action.TURN_RIGHT)
        self.assertEqual(env.direction, Direction.UP)
        self.assertEqual(res3.observation.head, Point(9, 10))

        # From UP -> TURN_LEFT should face LEFT
        res4 = env.step(Action.TURN_LEFT)
        self.assertEqual(env.direction, Direction.LEFT)
        self.assertEqual(res4.observation.head, Point(8, 10))

    def test_05_eating_food_increases_score_and_length(self) -> None:
        """Requirement 5: Eating food increases score and length, keeping tail."""
        env = SnakeEnvironment()
        # Head at (10, 10), facing RIGHT, food at (11, 10)
        env._set_state_for_testing(
            body=[Point(10, 10), Point(9, 10), Point(8, 10)],
            direction=Direction.RIGHT,
            food=Point(11, 10),
        )

        res = env.step(Action.STRAIGHT)
        self.assertEqual(res.score, 1)
        self.assertEqual(len(res.observation.body), 4)
        self.assertEqual(
            res.observation.body,
            (Point(11, 10), Point(10, 10), Point(9, 10), Point(8, 10)),
        )
        self.assertEqual(res.reward, 10.0)
        self.assertTrue(res.info.get("ate_food"))
        self.assertFalse(res.done)
        self.assertIsNotNone(env.food)
        self.assertNotIn(env.food, env.body)

    def test_06_food_never_spawns_inside_snake(self) -> None:
        """Requirement 6: Food never spawns inside the snake."""
        env = SnakeEnvironment(seed=999)
        for _ in range(100):
            food = env._place_food()
            self.assertIsNotNone(food)
            self.assertNotIn(food, env.body)

    def test_07_wall_collision_terminates(self) -> None:
        """Requirement 7: Wall collision terminates the episode with penalty."""
        env = SnakeEnvironment()

        # Right wall collision
        env._set_state_for_testing(
            body=[Point(19, 10), Point(18, 10), Point(17, 10)],
            direction=Direction.RIGHT,
            food=Point(5, 5),
        )
        res = env.step(Action.STRAIGHT)
        self.assertTrue(res.done)
        self.assertEqual(res.reward, -10.0)
        self.assertEqual(res.info["reason"], "wall_collision")

        # Top wall collision
        env.reset()
        env._set_state_for_testing(
            body=[Point(10, 0), Point(10, 1), Point(10, 2)],
            direction=Direction.UP,
            food=Point(5, 5),
        )
        res = env.step(Action.STRAIGHT)
        self.assertTrue(res.done)
        self.assertEqual(res.reward, -10.0)
        self.assertEqual(res.info["reason"], "wall_collision")

        # Left wall collision
        env.reset()
        env._set_state_for_testing(
            body=[Point(0, 10), Point(1, 10), Point(2, 10)],
            direction=Direction.LEFT,
            food=Point(5, 5),
        )
        res = env.step(Action.STRAIGHT)
        self.assertTrue(res.done)
        self.assertEqual(res.reward, -10.0)
        self.assertEqual(res.info["reason"], "wall_collision")

        # Bottom wall collision
        env.reset()
        env._set_state_for_testing(
            body=[Point(10, 19), Point(10, 18), Point(10, 17)],
            direction=Direction.DOWN,
            food=Point(5, 5),
        )
        res = env.step(Action.STRAIGHT)
        self.assertTrue(res.done)
        self.assertEqual(res.reward, -10.0)
        self.assertEqual(res.info["reason"], "wall_collision")

    def test_08_self_collision_terminates(self) -> None:
        """Requirement 8: Self-collision terminates the episode."""
        env = SnakeEnvironment()
        # Snake in a 'C' shape where moving straight collides with body segment (2, 2)
        # Head at (2, 3) facing UP. Body: (2, 2) is right above head!
        env._set_state_for_testing(
            body=[
                Point(2, 3),  # Head
                Point(1, 3),
                Point(1, 2),
                Point(2, 2),  # Target of STRAIGHT move
                Point(3, 2),
                Point(3, 3),  # Tail
            ],
            direction=Direction.UP,
            food=Point(10, 10),
        )

        res = env.step(Action.STRAIGHT)
        self.assertTrue(res.done)
        self.assertEqual(res.reward, -10.0)
        self.assertEqual(res.info["reason"], "self_collision")

    def test_09_moving_into_vacated_tail_cell_is_legal(self) -> None:
        """Requirement 9: Moving into the cell vacated by tail on normal move is legal."""
        env = SnakeEnvironment()
        # 4-segment loop:
        # (0, 0) -- (1, 0)
        #   |         |
        # (0, 1) -- (1, 1) [head at (1, 1), tail at (0, 1)]
        # Head at (1, 1) facing UP.
        # Action TURN_LEFT makes new head (0, 1), which is the current tail.
        # Since no food is eaten, tail vacates (0, 1) on this step -> valid move!
        env._set_state_for_testing(
            body=[Point(1, 1), Point(1, 0), Point(0, 0), Point(0, 1)],
            direction=Direction.UP,
            food=Point(15, 15),
        )

        res = env.step(Action.TURN_LEFT)
        self.assertFalse(res.done)
        self.assertEqual(res.observation.head, Point(0, 1))
        self.assertEqual(
            res.observation.body,
            (Point(0, 1), Point(1, 1), Point(1, 0), Point(0, 0)),
        )
        self.assertEqual(res.reward, -0.1)

    def test_10_step_limit_terminates_episode(self) -> None:
        """Requirement 10: Step limit terminates the episode correctly."""
        config = EnvironmentConfig(max_steps=3)
        env = SnakeEnvironment(config=config)
        env._set_state_for_testing(
            body=[Point(5, 5), Point(4, 5), Point(3, 5)],
            direction=Direction.RIGHT,
            food=Point(15, 15),
        )

        res1 = env.step(Action.STRAIGHT)
        self.assertFalse(res1.done)
        res2 = env.step(Action.STRAIGHT)
        self.assertFalse(res2.done)
        res3 = env.step(Action.STRAIGHT)
        self.assertTrue(res3.done)
        self.assertEqual(res3.info["reason"], "step_limit")

    def test_11_stepping_after_termination_raises_runtime_error(self) -> None:
        """Requirement 11: Stepping after termination has clearly defined behavior."""
        env = SnakeEnvironment()
        # Trigger wall collision
        env._set_state_for_testing(
            body=[Point(19, 10), Point(18, 10), Point(17, 10)],
            direction=Direction.RIGHT,
            food=Point(5, 5),
        )
        res = env.step(Action.STRAIGHT)
        self.assertTrue(res.done)

        # Attempt to step again without reset
        with self.assertRaises(RuntimeError) as ctx:
            env.step(Action.STRAIGHT)
        self.assertIn("Episode is already done", str(ctx.exception))

    def test_12_nearly_full_and_full_board_handled_cleanly(self) -> None:
        """Requirement 12: Nearly full and full board do not infinite loop."""
        # 3x3 board = 9 cells total
        config = EnvironmentConfig(grid_width=3, grid_height=3, initial_snake_length=3)
        env = SnakeEnvironment(config=config)

        # Snake occupies 8 of the 9 cells:
        # (0,0), (1,0), (2,0), (2,1), (2,2), (1,2), (0,2), (0,1)
        # Empty cell is only (1,1).
        snake_body = [
            Point(0, 1),  # Head
            Point(0, 2),
            Point(1, 2),
            Point(2, 2),
            Point(2, 1),
            Point(2, 0),
            Point(1, 0),
            Point(0, 0),  # Tail
        ]
        env._set_state_for_testing(
            body=snake_body,
            direction=Direction.RIGHT,
            food=Point(1, 1),
        )

        # Food placement should identify (1, 1) cleanly without infinite loop
        food = env._place_food()
        self.assertEqual(food, Point(1, 1))

        # Snake turns right or moves into food: from (0, 1) facing RIGHT, moving STRAIGHT hits (1, 1)
        res = env.step(Action.STRAIGHT)
        self.assertEqual(res.score, 1)
        self.assertEqual(len(res.observation.body), 9)  # Board 100% full!
        self.assertTrue(res.done)
        self.assertEqual(res.info["reason"], "board_full")
        self.assertTrue(res.info["won"])
        self.assertIsNone(env.food)

    def test_invalid_action_raises_value_error(self) -> None:
        """Verify invalid actions raise a helpful ValueError."""
        env = SnakeEnvironment()
        with self.assertRaises(ValueError):
            env.step(99)  # type: ignore


if __name__ == "__main__":
    unittest.main()
