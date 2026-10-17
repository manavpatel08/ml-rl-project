"""Unit tests for rendering, HUD formatting, and visual playback runner.

Uses Python's standard unittest library.
"""

from pathlib import Path
import tempfile
import unittest

from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.checkpoints import load_q_table, save_q_table
from snake_rl.config import EnvironmentConfig, QLearningConfig
from snake_rl.environment import Action, Direction, Point, SnakeEnvironment, SnakeObservation
from snake_rl.rendering import format_hud, render_ascii, render_grid
from snake_rl.viewer import VisualPlaybackRunner


class TestRenderingAndViewer(unittest.TestCase):
    """Test suite for rendering utilities, HUD formatting, and VisualPlaybackRunner."""

    def setUp(self) -> None:
        self.obs = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(9, 10), Point(8, 10)),
            food=Point(15, 10),
            direction=Direction.RIGHT,
            grid_size=(20, 20),
            steps=5,
            score=0,
        )

    def test_01_render_grid_matches_observation(self) -> None:
        """Requirement: 2D grid matrix accurately matches observation coordinates."""
        grid = render_grid(self.obs)
        self.assertEqual(len(grid), 20)
        self.assertEqual(len(grid[0]), 20)

        # Head at (10, 10)
        self.assertEqual(grid[10][10], "H")
        # Body segments at (9, 10) and (8, 10)
        self.assertEqual(grid[10][9], "o")
        self.assertEqual(grid[10][8], "o")
        # Food at (15, 10)
        self.assertEqual(grid[10][15], "★")
        # Empty cells
        self.assertEqual(grid[0][0], "·")
        self.assertEqual(grid[19][19], "·")

    def test_02_render_ascii_structure(self) -> None:
        """Requirement: ASCII render produces expected box borders and dimensions."""
        text = render_ascii(self.obs, use_colors=False)
        lines = text.split("\n")
        # Top border + 20 rows + bottom border = 22 lines
        self.assertEqual(len(lines), 22)
        self.assertTrue(lines[0].startswith("┌") and lines[0].endswith("┐"))
        self.assertTrue(lines[-1].startswith("└") and lines[-1].endswith("┘"))
        self.assertIn("H", text)
        self.assertIn("★", text)
        self.assertIn("o", text)

    def test_03_format_hud_displays_live_info(self) -> None:
        """Requirement: format_hud displays episode, score, steps, epsilon, reward, and mode."""
        hud_exploring = format_hud(
            episode=3,
            total_episodes=10,
            score=2,
            steps=45,
            epsilon=0.6543,
            reward=12.5,
            rolling_avg=1.8,
            is_exploring=True,
            mode="TRAINING",
            action_name="STRAIGHT",
            use_colors=False,
        )
        self.assertIn("Episode: 3/10", hud_exploring)
        self.assertIn("Score: 2", hud_exploring)
        self.assertIn("Steps: 45", hud_exploring)
        self.assertIn("Reward:  +12.5", hud_exploring)
        self.assertIn("Epsilon: 0.6543", hud_exploring)
        self.assertIn("EXPLORING", hud_exploring)

        hud_exploiting = format_hud(
            episode=5,
            total_episodes=10,
            score=3,
            steps=50,
            epsilon=0.1,
            reward=20.0,
            rolling_avg=2.5,
            is_exploring=False,
            mode="TRAINING",
            action_name="TURN_RIGHT",
            use_colors=False,
        )
        self.assertIn("EXPLOITING", hud_exploiting)

    def test_04_select_action_with_mode_distinguishes_decisions(self) -> None:
        """Requirement: select_action_with_mode indicates exploration vs exploitation."""
        agent = QLearningAgent(seed=42)

        # High epsilon -> exploratory choices
        agent.epsilon = 1.0
        action_exp, is_exp = agent.select_action_with_mode(self.obs)
        self.assertTrue(is_exp)
        self.assertIn(action_exp, (Action.STRAIGHT, Action.TURN_RIGHT, Action.TURN_LEFT))

        # Zero epsilon -> greedy choices
        agent.epsilon = 0.0
        action_greedy, is_greedy = agent.select_action_with_mode(self.obs)
        self.assertFalse(is_greedy)

    def test_05_visual_runner_training_session_bounded(self) -> None:
        """Requirement: VisualPlaybackRunner runs bounded training and updates Q-values."""
        env = SnakeEnvironment()
        agent = QLearningAgent(seed=42)
        initial_eps = agent.epsilon

        runner = VisualPlaybackRunner(
            env=env,
            agent=agent,
            speed_delay=0.0,
            interactive=False,
        )

        history = runner.run_training_session(num_episodes=2, base_seed=10)
        self.assertEqual(len(history), 2)
        self.assertEqual(history.total_episodes, 2)
        # Epsilon decayed twice
        self.assertLess(agent.epsilon, initial_eps)
        # Q-values were updated (at least some non-zero values)
        has_nonzero_q = any(any(val != 0.0 for val in row) for row in agent.q_table)
        self.assertTrue(has_nonzero_q)

    def test_06_visual_runner_evaluation_session_neutrality(self) -> None:
        """Requirement: Evaluation session does not modify Q-values or epsilon."""
        env = SnakeEnvironment()
        agent = QLearningAgent()
        agent.set_q_value(10, Action.STRAIGHT, 5.0)
        agent.epsilon = 0.25

        q_snapshot = [list(r) for r in agent.q_table]
        eps_snapshot = agent.epsilon

        runner = VisualPlaybackRunner(
            env=env,
            agent=agent,
            speed_delay=0.0,
            interactive=False,
        )

        summaries = runner.run_evaluation_session(num_episodes=2, base_seed=500)
        self.assertEqual(len(summaries), 2)
        # Q-table and epsilon must remain completely unchanged
        self.assertEqual(agent.q_table, q_snapshot)
        self.assertEqual(agent.epsilon, eps_snapshot)

    def test_07_visual_runner_stop_prevents_further_updates(self) -> None:
        """Requirement: Stopping runner terminates immediately without running more episodes."""
        env = SnakeEnvironment()
        agent = QLearningAgent()
        runner = VisualPlaybackRunner(
            env=env,
            agent=agent,
            speed_delay=0.0,
            interactive=False,
        )

        runner.stop()
        history = runner.run_training_session(num_episodes=5)
        # Stopped before starting -> 0 completed episodes
        self.assertEqual(len(history), 0)

    def test_08_checkpoint_verification_and_isolated_roundtrip(self) -> None:
        """Requirement: models/q_table_500ep.json loads safely and re-saves identically."""
        real_ckpt = Path("models/q_table_500ep.json")
        self.assertTrue(real_ckpt.is_file(), "models/q_table_500ep.json must exist")

        loaded_agent = load_q_table(real_ckpt)
        self.assertEqual(len(loaded_agent.q_table), 2048)

        # Isolated roundtrip into temporary directory (does NOT touch real checkpoint)
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_copy.json"
            save_q_table(loaded_agent, test_path)
            self.assertTrue(test_path.is_file())

            reloaded_agent = load_q_table(test_path)
            self.assertEqual(loaded_agent.q_table, reloaded_agent.q_table)
            self.assertEqual(loaded_agent.epsilon, reloaded_agent.epsilon)
            self.assertEqual(loaded_agent.config.alpha, reloaded_agent.config.alpha)


if __name__ == "__main__":
    unittest.main()
