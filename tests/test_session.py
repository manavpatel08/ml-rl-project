"""Unit tests for interactive session managers, safe checkpointing, and graphical rendering.

Uses Python's standard unittest library.
"""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from snake_rl.agents import DQNAgent, QLearningAgent, RandomAgent
from snake_rl.config import DQNConfig, QLearningConfig
from snake_rl.environment import Action, Direction, Point, SnakeEnvironment, SnakeObservation
from snake_rl.rendering import render_image, render_svg
from snake_rl.session import (
    EvaluationSession,
    TrainingSession,
    list_available_checkpoints,
    safe_save_checkpoint,
)


class TestSessionAndRendering(unittest.TestCase):
    """Test suite covering non-UI interactive session logic and graphical rendering."""

    def setUp(self) -> None:
        self.dummy_obs = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(9, 10), Point(8, 10)),
            food=Point(15, 10),
            direction=Direction.RIGHT,
            grid_size=(20, 20),
            steps=5,
            score=1,
        )

    # -------------------------------------------------------------------------
    # TrainingSession Tests
    # -------------------------------------------------------------------------

    def test_01_training_session_initialization(self) -> None:
        """Requirement: TrainingSession initializes with valid defaults and validates inputs."""
        agent = QLearningAgent(seed=42)
        session = TrainingSession(agent=agent, total_episodes=5, base_seed=100)

        self.assertEqual(session.current_episode, 1)
        self.assertEqual(session.total_episodes, 5)
        self.assertEqual(session.current_score, 0)
        self.assertEqual(session.current_steps, 0)
        self.assertFalse(session.is_paused)
        self.assertFalse(session.is_stopped)
        self.assertFalse(session.is_episode_done)
        self.assertTrue(session.can_step())

        # Invalid total_episodes raises ValueError
        with self.assertRaises(ValueError):
            TrainingSession(agent=agent, total_episodes=0)
        with self.assertRaises(ValueError):
            TrainingSession(agent=agent, total_episodes=-2)

    def test_02_training_session_q_learning_stepping_and_learning(self) -> None:
        """Requirement: step() advances environment, updates Q-values, and tracks metrics."""
        agent = QLearningAgent(seed=42)
        session = TrainingSession(agent=agent, total_episodes=2, base_seed=42)

        # Take 3 steps
        for _ in range(3):
            obs, done, info = session.step()
            self.assertIn("action", info)
            self.assertIn("reward", info)
            self.assertIn("td_error", info)

        self.assertEqual(session.current_steps, 3)
        self.assertGreater(session.total_steps, 0)
        # Verify that agent's Q-table received updates (at least one non-zero entry)
        has_nonzero_q = any(any(v != 0.0 for v in row) for row in agent.q_table)
        self.assertTrue(has_nonzero_q)

    def test_03_training_session_pause_prevents_updates(self) -> None:
        """Requirement: Pausing stops stepping and learning without corrupting state."""
        agent = QLearningAgent(seed=42)
        session = TrainingSession(agent=agent, total_episodes=2, base_seed=42)

        # Step once
        session.step()
        steps_before = session.current_steps
        q_snapshot = [list(r) for r in agent.q_table]

        # Pause
        session.pause()
        self.assertTrue(session.is_paused)
        self.assertFalse(session.can_step())

        # Attempt to step while paused -> no changes allowed
        obs, done, info = session.step()
        self.assertEqual(session.current_steps, steps_before)
        self.assertEqual(agent.q_table, q_snapshot)
        self.assertEqual(info["status"], "paused")

        # Resume
        session.resume()
        self.assertFalse(session.is_paused)
        self.assertTrue(session.can_step())

        # Can step again
        session.step()
        self.assertEqual(session.current_steps, steps_before + 1)

    def test_04_training_session_stop_prevents_updates(self) -> None:
        """Requirement: Stopping permanently halts execution."""
        agent = QLearningAgent(seed=42)
        session = TrainingSession(agent=agent, total_episodes=5, base_seed=42)

        session.stop()
        self.assertTrue(session.is_stopped)
        self.assertFalse(session.can_step())

        obs, done, info = session.step()
        self.assertEqual(session.current_steps, 0)
        self.assertEqual(info["status"], "stopped")
        self.assertFalse(session.next_episode())
        self.assertTrue(session.is_finished())

    def test_05_training_session_dqn_learning_and_loss(self) -> None:
        """Requirement: TrainingSession supports DQNAgent and records replay transitions and loss."""
        config = DQNConfig(warmup_steps=2, batch_size=2)
        agent = DQNAgent(config=config, seed=42)
        session = TrainingSession(agent=agent, total_episodes=2, base_seed=10)

        # Take 5 steps to pass warmup and trigger gradient descent
        for _ in range(5):
            session.step()

        self.assertGreater(len(agent.replay_buffer), 0)
        self.assertEqual(session.current_steps, 5)

        summary = session.get_metrics_summary()
        self.assertIn("train_step_count", summary)
        self.assertIn("recent_loss_avg", summary)
        self.assertTrue(summary["is_dqn"])
        self.assertEqual(summary["buffer_size"], 5)
        self.assertEqual(summary["buffer_capacity"], config.buffer_capacity)
        self.assertAlmostEqual(summary["buffer_occupancy_pct"], 5 / config.buffer_capacity * 100.0, places=2)
        self.assertEqual(summary["gradient_updates"], agent.train_step_count)

    def test_05b_dqn_telemetry_and_sync_progress(self) -> None:
        """Requirement: DQN telemetry computes exact sync progress, steps remaining, and buffer occupancy."""
        config = DQNConfig(target_update_interval=500, buffer_capacity=1000)
        agent = DQNAgent(config=config, seed=42)
        session = TrainingSession(agent=agent, total_episodes=5, base_seed=42)

        # 1. At 0 gradient updates:
        agent.train_step_count = 0
        summary = session.get_metrics_summary()
        self.assertTrue(summary["is_dqn"])
        self.assertEqual(summary["updates_since_sync"], 0)
        self.assertEqual(summary["steps_until_sync"], 500)
        self.assertEqual(summary["target_sync_count"], 0)

        # 2. At 460 gradient updates:
        agent.train_step_count = 460
        summary = session.get_metrics_summary()
        self.assertEqual(summary["updates_since_sync"], 460)
        self.assertEqual(summary["steps_until_sync"], 40)
        self.assertEqual(summary["target_sync_count"], 0)

        # 3. At 500 gradient updates (just synchronized):
        agent.train_step_count = 500
        summary = session.get_metrics_summary()
        self.assertEqual(summary["updates_since_sync"], 0)
        self.assertEqual(summary["steps_until_sync"], 500)
        self.assertEqual(summary["target_sync_count"], 1)

        # 4. At 501 gradient updates:
        agent.train_step_count = 501
        summary = session.get_metrics_summary()
        self.assertEqual(summary["updates_since_sync"], 1)
        self.assertEqual(summary["steps_until_sync"], 499)
        self.assertEqual(summary["target_sync_count"], 1)

        # 5. Non-DQN agent (QLearningAgent) has is_dqn=False and zero/empty DQN fields
        q_agent = QLearningAgent(seed=42)
        q_session = TrainingSession(agent=q_agent, total_episodes=2)
        q_summary = q_session.get_metrics_summary()
        self.assertFalse(q_summary["is_dqn"])
        self.assertEqual(q_summary["buffer_size"], 0)
        self.assertEqual(q_summary["buffer_capacity"], 0)
        self.assertEqual(q_summary["buffer_occupancy_str"], "N/A")

    def test_06_training_session_episode_completion_and_decay(self) -> None:
        """Requirement: Episode conclusion decays epsilon once and records EpisodeMetrics."""
        config = QLearningConfig(epsilon_start=1.0, epsilon_decay=0.9, epsilon_min=0.01)
        agent = QLearningAgent(config=config, seed=42)
        session = TrainingSession(agent=agent, total_episodes=2, base_seed=10)

        # Run until episode finishes
        max_safety = 500
        while not session.is_episode_done and max_safety > 0:
            session.step()
            max_safety -= 1

        self.assertTrue(session.is_episode_done)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.history.episodes[0].cumulative_steps, session.total_steps)
        self.assertEqual(session.history.cumulative_steps[0], session.total_steps)
        # Epsilon decayed once
        self.assertAlmostEqual(agent.epsilon, 0.9)

        # Advancing to next episode succeeds
        has_next = session.next_episode()
        self.assertTrue(has_next)
        self.assertEqual(session.current_episode, 2)
        self.assertEqual(session.current_steps, 0)
        self.assertFalse(session.is_episode_done)

    # -------------------------------------------------------------------------
    # EvaluationSession Tests
    # -------------------------------------------------------------------------

    def test_07_evaluation_session_neutrality(self) -> None:
        """Requirement: EvaluationSession selects greedy actions and never modifies weights or epsilon."""
        agent = QLearningAgent(seed=42)
        agent.set_q_value(10, Action.STRAIGHT, 5.0)
        agent.epsilon = 0.5

        q_snapshot = [list(r) for r in agent.q_table]
        eps_snapshot = agent.epsilon

        session = EvaluationSession(agent=agent, base_seed=500)

        for _ in range(10):
            session.step()

        # Q-table and epsilon must be strictly identical
        self.assertEqual(agent.q_table, q_snapshot)
        self.assertEqual(agent.epsilon, eps_snapshot)
        self.assertEqual(session.current_steps, 10)

    # -------------------------------------------------------------------------
    # Checkpoint Management Tests
    # -------------------------------------------------------------------------

    def test_08_list_available_checkpoints(self) -> None:
        """Requirement: list_available_checkpoints scans directory and identifies algorithms."""
        ckpts = list_available_checkpoints("models")
        self.assertIsInstance(ckpts, list)

        # If models/q_table_500ep.json exists, verify its metadata
        q_ckpt = next((c for c in ckpts if "q_table_500ep.json" in c["name"]), None)
        if q_ckpt:
            self.assertEqual(q_ckpt["algorithm"], "q_learning")
            self.assertGreater(q_ckpt["size_kb"], 0)

    def test_09_safe_save_checkpoint_protection(self) -> None:
        """Requirement: safe_save_checkpoint rejects overwrite unless explicitly confirmed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = QLearningAgent(seed=42)
            dest = Path(tmp_dir) / "test_model.json"

            # 1. First save succeeds
            success, msg = safe_save_checkpoint(agent, dest, overwrite=False)
            self.assertTrue(success)
            self.assertTrue(dest.is_file())

            # 2. Duplicate save without overwrite fails with warning
            success2, msg2 = safe_save_checkpoint(agent, dest, overwrite=False)
            self.assertFalse(success2)
            self.assertIn("already exists", msg2)

            # 3. Duplicate save with overwrite=True succeeds
            success3, msg3 = safe_save_checkpoint(agent, dest, overwrite=True)
            self.assertTrue(success3)

    # -------------------------------------------------------------------------
    # Graphical Rendering Tests
    # -------------------------------------------------------------------------

    def test_10_render_svg_validity_and_structure(self) -> None:
        """Requirement: render_svg produces valid XML with snake head, body, eyes, and food."""
        svg_str = render_svg(self.dummy_obs, cell_size=24, show_eyes=True, status_overlay="TEST BANNER")

        # Must parse as valid XML
        root = ET.fromstring(svg_str)
        self.assertIn("svg", root.tag)

        # Check for key SVG elements
        rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
        circles = root.findall(".//{http://www.w3.org/2000/svg}circle")
        texts = root.findall(".//{http://www.w3.org/2000/svg}text")

        self.assertGreater(len(rects), 3)  # background, body, head
        self.assertGreater(len(circles), 1)  # food, eyes
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0].text, "TEST BANNER")

    def test_11_render_image_dimensions_and_mode(self) -> None:
        """Requirement: render_image outputs an RGB PIL Image with matching dimensions."""
        img = render_image(self.dummy_obs, cell_size=20)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (400, 400))  # 20 * 20


if __name__ == "__main__":
    unittest.main()
