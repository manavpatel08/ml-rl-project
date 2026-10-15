"""Unit tests for Tabular Q-learning agent.

Uses Python's standard unittest library.
"""

import math
import unittest

from snake_rl.agents.q_learning import NUM_ACTIONS, QLearningAgent
from snake_rl.config import EnvironmentConfig, QLearningConfig
from snake_rl.environment import Action, Direction, Point, SnakeEnvironment, SnakeObservation
from snake_rl.state import NUM_STATES, StateEncoder


class TestQLearningAgent(unittest.TestCase):
    """Test suite for QLearningAgent initialization, Bellman arithmetic, and exploration."""

    def test_01_q_table_dimensions_and_initialization(self) -> None:
        """Requirement: Q-table has shape (2048, 3) initialized with zeros."""
        agent = QLearningAgent()
        self.assertEqual(len(agent.q_table), NUM_STATES)
        self.assertEqual(NUM_STATES, 2048)
        self.assertEqual(NUM_ACTIONS, 3)

        for state_idx in range(NUM_STATES):
            self.assertEqual(len(agent.q_table[state_idx]), NUM_ACTIONS)
            self.assertEqual(agent.q_table[state_idx], [0.0, 0.0, 0.0])

    def test_02_greedy_action_selection(self) -> None:
        """Requirement: select_greedy_action picks argmax Q without exploration."""
        agent = QLearningAgent()
        state = 100

        # Set specific Q-values: [1.2, 5.5, 3.1] -> Action.TURN_RIGHT (index 1) is highest
        agent.set_q_value(state, Action.STRAIGHT, 1.2)
        agent.set_q_value(state, Action.TURN_RIGHT, 5.5)
        agent.set_q_value(state, Action.TURN_LEFT, 3.1)

        # Even with high exploration rate, select_greedy_action must be purely greedy
        agent.epsilon = 1.0
        chosen_action = agent.select_greedy_action(state)
        self.assertEqual(chosen_action, Action.TURN_RIGHT)

        # Verify get_q_values
        self.assertEqual(agent.get_q_values(state), (1.2, 5.5, 3.1))

    def test_03_epsilon_greedy_behavior(self) -> None:
        """Requirement: Epsilon controls exploration vs exploitation."""
        agent = QLearningAgent(seed=123)
        state = 50
        agent.set_q_value(state, Action.STRAIGHT, 10.0)
        agent.set_q_value(state, Action.TURN_RIGHT, 1.0)
        agent.set_q_value(state, Action.TURN_LEFT, 1.0)

        # When epsilon = 0.0, always exploits (greedy)
        agent.epsilon = 0.0
        for _ in range(50):
            action = agent.select_action(state)
            self.assertEqual(action, Action.STRAIGHT)

        # When epsilon = 1.0, explores all actions uniformly
        agent.epsilon = 1.0
        action_counts = {Action.STRAIGHT: 0, Action.TURN_RIGHT: 0, Action.TURN_LEFT: 0}
        for _ in range(300):
            action = agent.select_action(state)
            action_counts[action] += 1

        # Each of the 3 actions should be picked roughly ~100 times
        for act, count in action_counts.items():
            self.assertGreater(count, 50, f"Action {act.name} picked only {count} times")

    def test_04_hand_calculated_bellman_update_non_terminal(self) -> None:
        """Requirement: Correct numerical Bellman update for non-terminal transition.

        Hand calculation:
          alpha = 0.1, gamma = 0.9
          s = 42, a = STRAIGHT (0), Q(s, a) = 0.5
          reward = 10.0
          next_state s' = 100 with Q(s', :) = [1.0, 4.0, 2.0] -> max_a' Q(s', a') = 4.0
          target = reward + gamma * max_a' Q(s', a') = 10.0 + 0.9 * 4.0 = 13.6
          td_error = target - Q(s, a) = 13.6 - 0.5 = 13.1
          new_Q(s, a) = 0.5 + 0.1 * 13.1 = 1.81
        """
        config = QLearningConfig(alpha=0.1, gamma=0.9)
        agent = QLearningAgent(config=config)

        state = 42
        action = Action.STRAIGHT
        agent.set_q_value(state, action, 0.5)

        next_state = 100
        agent.set_q_value(next_state, Action.STRAIGHT, 1.0)
        agent.set_q_value(next_state, Action.TURN_RIGHT, 4.0)
        agent.set_q_value(next_state, Action.TURN_LEFT, 2.0)

        td_error = agent.update(
            state=state,
            action=action,
            reward=10.0,
            next_state=next_state,
            done=False,
        )

        self.assertAlmostEqual(td_error, 13.1, places=5)
        self.assertAlmostEqual(agent.get_q_value(state, action), 1.81, places=5)

    def test_05_hand_calculated_bellman_update_terminal_no_bootstrap(self) -> None:
        """Requirement: Terminal transitions do not bootstrap from next state.

        Hand calculation:
          alpha = 0.1, gamma = 0.9
          s = 42, a = TURN_RIGHT (1), Q(s, a) = 2.0
          reward = -10.0
          done = True
          next_state s' = 100 with high Q(s', :) = [100.0, 100.0, 100.0]
          Since done is True, target = reward = -10.0 (no gamma * max Q!)
          td_error = -10.0 - 2.0 = -12.0
          new_Q(s, a) = 2.0 + 0.1 * (-12.0) = 0.8
        """
        config = QLearningConfig(alpha=0.1, gamma=0.9)
        agent = QLearningAgent(config=config)

        state = 42
        action = Action.TURN_RIGHT
        agent.set_q_value(state, action, 2.0)

        next_state = 100
        # High next state values that MUST be ignored
        agent.set_q_value(next_state, Action.STRAIGHT, 100.0)
        agent.set_q_value(next_state, Action.TURN_RIGHT, 100.0)
        agent.set_q_value(next_state, Action.TURN_LEFT, 100.0)

        td_error = agent.update(
            state=state,
            action=action,
            reward=-10.0,
            next_state=next_state,
            done=True,
        )

        self.assertAlmostEqual(td_error, -12.0, places=5)
        self.assertAlmostEqual(agent.get_q_value(state, action), 0.8, places=5)

    def test_06_epsilon_decay_and_minimum_clamping(self) -> None:
        """Requirement: Epsilon decays per episode and clamps to epsilon_min."""
        config = QLearningConfig(
            epsilon_start=1.0,
            epsilon_decay=0.5,
            epsilon_min=0.15,
        )
        agent = QLearningAgent(config=config)
        self.assertEqual(agent.epsilon, 1.0)

        # Episode 1: 1.0 * 0.5 = 0.5
        eps1 = agent.decay_epsilon()
        self.assertAlmostEqual(eps1, 0.5)

        # Episode 2: 0.5 * 0.5 = 0.25
        eps2 = agent.decay_epsilon()
        self.assertAlmostEqual(eps2, 0.25)

        # Episode 3: 0.25 * 0.5 = 0.125 -> clamped to epsilon_min (0.15)
        eps3 = agent.decay_epsilon()
        self.assertAlmostEqual(eps3, 0.15)

        # Episode 4: clamped at 0.15
        eps4 = agent.decay_epsilon()
        self.assertAlmostEqual(eps4, 0.15)

        # Reset epsilon restores epsilon_start
        agent.reset_epsilon()
        self.assertEqual(agent.epsilon, 1.0)

    def test_07_reproducibility_with_seed(self) -> None:
        """Requirement: Seeding produces identical exploration trajectories."""
        agent1 = QLearningAgent(seed=999)
        agent2 = QLearningAgent(seed=999)

        agent1.epsilon = 0.5
        agent2.epsilon = 0.5

        actions1 = [agent1.select_action(s % NUM_STATES) for s in range(100)]
        actions2 = [agent2.select_action(s % NUM_STATES) for s in range(100)]
        self.assertEqual(actions1, actions2)

    def test_08_integration_with_snake_observation(self) -> None:
        """Requirement: Agent accepts raw SnakeObservation transparently."""
        agent = QLearningAgent()
        obs = SnakeObservation(
            head=Point(10, 10),
            body=(Point(10, 10), Point(9, 10), Point(8, 10)),
            food=Point(15, 10),
            direction=Direction.RIGHT,
            grid_size=(20, 20),
            steps=0,
            score=0,
        )

        expected_idx = StateEncoder.to_index(obs)
        agent.set_q_value(obs, Action.TURN_LEFT, 7.7)

        self.assertEqual(agent.get_q_value(expected_idx, Action.TURN_LEFT), 7.7)
        self.assertEqual(agent.select_greedy_action(obs), Action.TURN_LEFT)

    def test_09_invalid_inputs_raise_clear_exceptions(self) -> None:
        """Requirement: Out-of-range states, actions, or non-finite rewards raise errors."""
        agent = QLearningAgent()

        # Invalid state indices
        with self.assertRaises(ValueError):
            agent.get_q_value(-1, Action.STRAIGHT)
        with self.assertRaises(ValueError):
            agent.get_q_value(2048, Action.STRAIGHT)
        with self.assertRaises(TypeError):
            agent.get_q_value("invalid", Action.STRAIGHT)  # type: ignore

        # Invalid actions
        with self.assertRaises(ValueError):
            agent.get_q_value(0, -1)
        with self.assertRaises(ValueError):
            agent.get_q_value(0, 3)
        with self.assertRaises(TypeError):
            agent.get_q_value(0, "STRAIGHT")  # type: ignore

        # Non-finite rewards
        with self.assertRaises(ValueError):
            agent.update(0, Action.STRAIGHT, float("nan"), 1, False)
        with self.assertRaises(ValueError):
            agent.update(0, Action.STRAIGHT, float("inf"), 1, False)

        # Config validations
        with self.assertRaises(ValueError):
            QLearningConfig(alpha=0.0)
        with self.assertRaises(ValueError):
            QLearningConfig(alpha=1.5)
        with self.assertRaises(ValueError):
            QLearningConfig(gamma=1.1)
        with self.assertRaises(ValueError):
            QLearningConfig(epsilon_min=0.5, epsilon_start=0.2)

    def test_10_bounded_environment_learning_loop(self) -> None:
        """Integration check: Run a bounded 1-episode loop with environment."""
        env = SnakeEnvironment(seed=42)
        agent = QLearningAgent(seed=42)
        obs = env.reset(seed=42)

        steps = 0
        total_reward = 0.0

        while not env.done and steps < 20:
            action = agent.select_action(obs)
            result = env.step(action)
            agent.update(
                state=obs,
                action=action,
                reward=result.reward,
                next_state=result.observation,
                done=result.done,
            )
            total_reward += result.reward
            obs = result.observation
            steps += 1

        agent.decay_epsilon()
        self.assertGreater(steps, 0)
        self.assertFalse(math.isnan(total_reward))


if __name__ == "__main__":
    unittest.main()
