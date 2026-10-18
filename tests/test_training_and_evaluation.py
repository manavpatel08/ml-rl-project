"""Unit tests for training infrastructure, baselines, evaluation, and checkpoints.

Uses Python's standard unittest library.
"""

from pathlib import Path
import tempfile
import unittest

from snake_rl.agents import HeuristicAgent, QLearningAgent, RandomAgent
from snake_rl.checkpoints import load_q_table, save_q_table
from snake_rl.config import EnvironmentConfig, QLearningConfig
from snake_rl.environment import Action, Direction, Point, SnakeEnvironment, SnakeObservation
from snake_rl.evaluation import EvaluationResult, compare_agents, evaluate_agent
from snake_rl.state import NUM_STATES
from snake_rl.training import EpisodeMetrics, TrainingHistory, train_q_learning


class TestTrainingAndEvaluation(unittest.TestCase):
    """Test suite covering Step 4 infrastructure, baselines, evaluation, and checkpoints."""

    def test_01_bounded_training_run_and_metrics(self) -> None:
        """Requirement: train_q_learning runs exact episode count and records structured metrics."""
        env = SnakeEnvironment(seed=42)
        agent = QLearningAgent(seed=42)

        history = train_q_learning(env, agent, num_episodes=4, base_seed=100)

        self.assertEqual(len(history), 4)
        self.assertEqual(history.total_episodes, 4)
        self.assertEqual(len(history.scores), 4)
        self.assertEqual(len(history.rewards), 4)
        self.assertEqual(len(history.lengths), 4)
        self.assertEqual(len(history.epsilons), 4)
        self.assertGreater(history.total_steps, 0)
        self.assertEqual(len(history.cumulative_steps), 4)
        self.assertEqual(history.cumulative_steps[-1], history.total_steps)
        self.assertTrue(all(history.cumulative_steps[i] <= history.cumulative_steps[i+1] for i in range(3)))

        # Verify moving averages
        ma_scores = history.moving_average_scores(window=2)
        self.assertEqual(len(ma_scores), 4)
        ma_rewards = history.moving_average_rewards(window=2)
        self.assertEqual(len(ma_rewards), 4)

        # Invalid episode counts raise ValueError
        with self.assertRaises(ValueError):
            train_q_learning(env, agent, num_episodes=0)
        with self.assertRaises(ValueError):
            train_q_learning(env, agent, num_episodes=-5)

    def test_02_epsilon_decay_once_per_episode(self) -> None:
        """Requirement: Epsilon decays exactly once per episode at episode completion."""
        env = SnakeEnvironment()
        config = QLearningConfig(
            epsilon_start=1.0,
            epsilon_decay=0.8,
            epsilon_min=0.01,
        )
        agent = QLearningAgent(config=config)

        history = train_q_learning(env, agent, num_episodes=3, base_seed=10)

        # Expected:
        # After ep 1: 1.0 * 0.8 = 0.8
        # After ep 2: 0.8 * 0.8 = 0.64
        # After ep 3: 0.64 * 0.8 = 0.512
        self.assertAlmostEqual(history.epsilons[0], 0.8, places=4)
        self.assertAlmostEqual(history.epsilons[1], 0.64, places=4)
        self.assertAlmostEqual(history.epsilons[2], 0.512, places=4)

    def test_03_training_reproducibility_with_fixed_seed(self) -> None:
        """Requirement: Training with the same base_seed and agent seed is reproducible."""
        env1 = SnakeEnvironment()
        agent1 = QLearningAgent(seed=777)
        history1 = train_q_learning(env1, agent1, num_episodes=3, base_seed=50)

        env2 = SnakeEnvironment()
        agent2 = QLearningAgent(seed=777)
        history2 = train_q_learning(env2, agent2, num_episodes=3, base_seed=50)

        self.assertEqual(history1.scores, history2.scores)
        self.assertEqual(history1.rewards, history2.rewards)
        self.assertEqual(history1.lengths, history2.lengths)
        self.assertEqual(agent1.q_table, agent2.q_table)

    def test_04_evaluation_does_not_modify_q_values_or_epsilon(self) -> None:
        """Requirement: Evaluation mode does not update Q-values or modify epsilon."""
        env = SnakeEnvironment()
        agent = QLearningAgent()
        agent.set_q_value(42, Action.TURN_RIGHT, 15.5)
        agent.set_q_value(100, Action.STRAIGHT, -3.2)
        agent.epsilon = 0.35

        q_table_snapshot = [list(row) for row in agent.q_table]
        initial_epsilon = agent.epsilon

        result = evaluate_agent(env, agent, num_episodes=5, base_seed=200)

        self.assertEqual(result.num_episodes, 5)
        self.assertEqual(agent.q_table, q_table_snapshot)
        self.assertEqual(agent.epsilon, initial_epsilon)

    def test_05_greedy_evaluation_never_explores(self) -> None:
        """Requirement: Evaluation always uses greedy policy even if agent epsilon is high."""
        env = SnakeEnvironment()
        agent = QLearningAgent()
        # Set high epsilon on agent that would cause random actions during training
        agent.epsilon = 1.0

        # State 10 has a clear best action: TURN_LEFT
        agent.set_q_value(10, Action.STRAIGHT, 0.0)
        agent.set_q_value(10, Action.TURN_RIGHT, 0.0)
        agent.set_q_value(10, Action.TURN_LEFT, 50.0)

        # evaluate_agent calls select_greedy_action, which must pick TURN_LEFT
        chosen = agent.select_greedy_action(10)
        self.assertEqual(chosen, Action.TURN_LEFT)

    def test_06_baseline_agents_action_validity(self) -> None:
        """Requirement: Baseline agents produce valid actions and sensible behaviors."""
        obs = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(9, 10), Point(8, 10)),
            food=Point(15, 10),
            direction=Direction.RIGHT,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )

        # RandomAgent
        random_agent = RandomAgent(seed=42)
        for _ in range(30):
            action = random_agent.select_action(obs)
            self.assertIn(action, (Action.STRAIGHT, Action.TURN_RIGHT, Action.TURN_LEFT))

        # HeuristicAgent
        # Head at (10, 10) facing RIGHT, food at (15, 10) (straight ahead).
        # STRAIGHT gets closer to food (dist = 4).
        # TURN_RIGHT goes to (10, 11) (dist = 5 + 1 = 6).
        # TURN_LEFT goes to (10, 9) (dist = 5 + 1 = 6).
        heuristic_agent = HeuristicAgent(seed=42)
        action = heuristic_agent.select_action(obs)
        self.assertEqual(action, Action.STRAIGHT)

        # Test danger avoidance: head at (19, 10) facing RIGHT (wall directly ahead)
        obs_wall = SnakeObservation(
            head=Point(19, 10),
            body=(Point(19, 10), Point(18, 10), Point(17, 10)),
            food=Point(19, 5),  # Food is UP
            direction=Direction.RIGHT,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )
        # STRAIGHT hits wall. Safe actions: TURN_RIGHT (DOWN), TURN_LEFT (UP).
        # Food is at (19, 5) (UP), so TURN_LEFT moves closer to food and avoids wall!
        action_wall = heuristic_agent.select_action(obs_wall)
        self.assertEqual(action_wall, Action.TURN_LEFT)

    def test_07_fair_seed_reuse_across_policies(self) -> None:
        """Requirement: compare_agents evaluates all agents across identical seeds."""
        env = SnakeEnvironment()
        agents = {
            "random": RandomAgent(seed=1),
            "heuristic": HeuristicAgent(seed=1),
            "q_learning": QLearningAgent(seed=1),
        }

        results = compare_agents(env, agents, num_episodes=3, base_seed=900)

        self.assertEqual(set(results.keys()), {"random", "heuristic", "q_learning"})
        for name, res in results.items():
            self.assertEqual(res.num_episodes, 3)
            self.assertEqual(len(res.scores), 3)
            self.assertEqual(len(res.rewards), 3)
            self.assertEqual(len(res.steps), 3)
            self.assertIsInstance(res.mean_score, float)
            self.assertIsInstance(res.std_score, float)

    def test_08_checkpoint_save_and_load_roundtrip(self) -> None:
        """Requirement: Q-table and config serialize to JSON and reload identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "models" / "q_table.json"

            agent = QLearningAgent(config=QLearningConfig(alpha=0.2, gamma=0.85))
            agent.set_q_value(12, Action.TURN_LEFT, 9.99)
            agent.set_q_value(2047, Action.STRAIGHT, -4.5)
            agent.epsilon = 0.42

            saved_path = save_q_table(
                agent,
                ckpt_path,
                metadata={"test_run": True, "episodes": 25},
            )
            self.assertTrue(saved_path.is_file())

            # Load into a new agent
            loaded_agent = load_q_table(saved_path)
            self.assertEqual(loaded_agent.config.alpha, 0.2)
            self.assertEqual(loaded_agent.config.gamma, 0.85)
            self.assertAlmostEqual(loaded_agent.epsilon, 0.42)
            self.assertAlmostEqual(loaded_agent.get_q_value(12, Action.TURN_LEFT), 9.99)
            self.assertAlmostEqual(loaded_agent.get_q_value(2047, Action.STRAIGHT), -4.5)
            self.assertEqual(loaded_agent.q_table, agent.q_table)

            # Load into an existing agent instance
            existing_agent = QLearningAgent()
            load_q_table(saved_path, agent=existing_agent)
            self.assertEqual(existing_agent.q_table, agent.q_table)
            self.assertAlmostEqual(existing_agent.epsilon, 0.42)

    def test_09_checkpoint_invalid_handling(self) -> None:
        """Requirement: Corrupted or invalid checkpoints raise clear exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Non-existent file
            with self.assertRaises(FileNotFoundError):
                load_q_table(base / "missing.json")

            # Malformed JSON
            corrupt_file = base / "corrupt.json"
            corrupt_file.write_text("{invalid json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_q_table(corrupt_file)

            # Wrong algorithm
            wrong_algo = base / "wrong_algo.json"
            wrong_algo.write_text('{"algorithm": "dqn"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_q_table(wrong_algo)

            # Non-finite values during save
            bad_agent = QLearningAgent()
            bad_agent.q_table[0][0] = float("nan")
            with self.assertRaises(ValueError):
                save_q_table(bad_agent, base / "bad.json")


if __name__ == "__main__":
    unittest.main()
