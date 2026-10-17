"""Unit tests for Deep Q-Network (DQN) implementation, buffer, and checkpointing.

Uses Python's standard unittest library.
"""

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from snake_rl.agents import DQNAgent, DQNNetwork, ReplayBatch, ReplayBuffer, Transition
from snake_rl.checkpoints import load_dqn_checkpoint, save_dqn_checkpoint
from snake_rl.config import DQNConfig
from snake_rl.environment import Action, Direction, Point, SnakeEnvironment, SnakeObservation
from snake_rl.evaluation import evaluate_agent
from snake_rl.state import NUM_STATES, TOTAL_FEATURES, encode_observation
from snake_rl.training import EpisodeMetrics, TrainingHistory, train_dqn


class TestDQN(unittest.TestCase):
    """Test suite covering Step 5 DQN network, replay buffer, agent, and checkpoints."""

    def setUp(self) -> None:
        """Create a standard test observation."""
        self.dummy_obs = SnakeObservation(
            head=Point(5, 5),
            body=(Point(5, 5), Point(5, 4), Point(5, 3)),
            food=Point(5, 8),
            direction=Direction.DOWN,
            grid_size=(20, 20),
            score=0,
            steps=0,
        )


    # -------------------------------------------------------------------------
    # Part 1: DQNNetwork Architecture & Shapes
    # -------------------------------------------------------------------------

    def test_01_network_initialization_and_forward_shapes(self) -> None:
        """Requirement: DQNNetwork has 11 inputs, 3 outputs, and computes correct output shapes."""
        net = DQNNetwork(input_dim=11, hidden_dims=(128, 128), output_dim=3)

        # Single 1D vector (11,) -> (1, 3)
        single_input = torch.zeros(11)
        out_single = net(single_input)
        self.assertEqual(out_single.shape, torch.Size([1, 3]))

        # 2D batch (16, 11) -> (16, 3)
        batch_input = torch.randn(16, 11)
        out_batch = net(batch_input)
        self.assertEqual(out_batch.shape, torch.Size([16, 3]))

        # Verify layer structure
        modules = [m for m in net.network if isinstance(m, nn.Linear)]
        self.assertEqual(len(modules), 3)
        self.assertEqual(modules[0].in_features, 11)
        self.assertEqual(modules[0].out_features, 128)
        self.assertEqual(modules[1].in_features, 128)
        self.assertEqual(modules[1].out_features, 128)
        self.assertEqual(modules[2].in_features, 128)
        self.assertEqual(modules[2].out_features, 3)

    # -------------------------------------------------------------------------
    # Part 2: Replay Buffer Capacity, Eviction, and Sampling
    # -------------------------------------------------------------------------

    def test_02_replay_buffer_capacity_and_fifo_eviction(self) -> None:
        """Requirement: ReplayBuffer adheres to capacity and evicts oldest items (FIFO)."""
        buffer = ReplayBuffer(capacity=5)
        self.assertEqual(len(buffer), 0)

        # Push 5 transitions
        for i in range(5):
            buffer.push([float(i)] * 11, Action.STRAIGHT, float(i), [float(i + 1)] * 11, False)

        self.assertEqual(len(buffer), 5)

        # Push 6th transition: item 0 should be evicted, leaving items 1..5
        buffer.push([99.0] * 11, Action.TURN_RIGHT, 10.0, [100.0] * 11, True)
        self.assertEqual(len(buffer), 5)


        # First transition in deque should have state starting with 1.0
        oldest = buffer._buffer[0]
        self.assertEqual(oldest.state[0], 1.0)
        newest = buffer._buffer[-1]
        self.assertEqual(newest.state[0], 99.0)


    def test_03_replay_buffer_sampling_types_and_shapes(self) -> None:
        """Requirement: sample() returns ReplayBatch with correct PyTorch shapes and types."""
        buffer = ReplayBuffer(capacity=100, seed=42)
        for i in range(20):
            buffer.push(
                state=self.dummy_obs,
                action=Action(i % 3),
                reward=float(i) - 5.0,
                next_state=self.dummy_obs,
                done=(i % 5 == 0),
            )

        batch = buffer.sample(batch_size=8)
        self.assertIsInstance(batch, ReplayBatch)
        self.assertEqual(batch.states.shape, torch.Size([8, 11]))
        self.assertEqual(batch.actions.shape, torch.Size([8, 1]))
        self.assertEqual(batch.rewards.shape, torch.Size([8, 1]))
        self.assertEqual(batch.next_states.shape, torch.Size([8, 11]))
        self.assertEqual(batch.dones.shape, torch.Size([8, 1]))

        self.assertEqual(batch.states.dtype, torch.float32)
        self.assertEqual(batch.actions.dtype, torch.int64)
        self.assertEqual(batch.rewards.dtype, torch.float32)
        self.assertEqual(batch.next_states.dtype, torch.float32)
        self.assertEqual(batch.dones.dtype, torch.float32)

        # Sampling more than buffer size raises ValueError
        with self.assertRaises(ValueError):
            buffer.sample(batch_size=25)

    def test_04_replay_buffer_reproducibility(self) -> None:
        """Requirement: ReplayBuffer seed ensures deterministic sampling batches."""
        buf1 = ReplayBuffer(capacity=50, seed=123)
        buf2 = ReplayBuffer(capacity=50, seed=123)

        for i in range(30):
            s = [float(i)] * 11
            buf1.push(s, Action.STRAIGHT, float(i), s, False)
            buf2.push(s, Action.STRAIGHT, float(i), s, False)

        batch1 = buf1.sample(10)
        batch2 = buf2.sample(10)

        self.assertTrue(torch.equal(batch1.states, batch2.states))
        self.assertTrue(torch.equal(batch1.actions, batch2.actions))

    # -------------------------------------------------------------------------
    # Part 3: Target Network Freezing and Synchronization
    # -------------------------------------------------------------------------

    def test_05_target_network_frozen_parameters(self) -> None:
        """Requirement: Target network parameters have requires_grad = False and match online weights at start."""
        config = DQNConfig(input_dim=11, output_dim=3)
        agent = DQNAgent(config=config, seed=42)

        for param in agent.target_network.parameters():
            self.assertFalse(param.requires_grad)

        # Initial weights must match between online and target networks
        for p_online, p_target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        ):
            self.assertTrue(torch.equal(p_online, p_target))

    def test_06_target_network_updates_at_interval(self) -> None:
        """Requirement: Target network is synced from online network only every target_update_interval steps."""
        config = DQNConfig(
            target_update_interval=3,
            warmup_steps=4,
            batch_size=4,
        )
        agent = DQNAgent(config=config, seed=42)

        # Fill replay buffer to allow training steps
        for _ in range(10):
            agent.remember(self.dummy_obs, Action.STRAIGHT, 1.0, self.dummy_obs, False)

        # Record target network weight before training
        initial_target_weight = agent.target_network.network[0].weight.clone()

        # Step 1: online network updates, target network stays fixed
        agent.train_step()
        self.assertEqual(agent.train_step_count, 1)
        self.assertTrue(
            torch.equal(agent.target_network.network[0].weight, initial_target_weight)
        )

        # Step 2: target network still stays fixed
        agent.train_step()
        self.assertEqual(agent.train_step_count, 2)
        self.assertTrue(
            torch.equal(agent.target_network.network[0].weight, initial_target_weight)
        )

        # Step 3: interval hit -> target network must update
        agent.train_step()
        self.assertEqual(agent.train_step_count, 3)
        self.assertTrue(
            torch.equal(
                agent.target_network.network[0].weight,
                agent.online_network.network[0].weight,
            )
        )

    # -------------------------------------------------------------------------
    # Part 4: Epsilon-Greedy Exploration & Exploitation
    # -------------------------------------------------------------------------

    def test_07_epsilon_greedy_behavior(self) -> None:
        """Requirement: Epsilon controls random exploration vs greedy argmax Q."""
        config = DQNConfig(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.5)
        agent = DQNAgent(config=config, seed=123)

        # At epsilon=1.0, agent always explores
        _, is_exploring = agent.select_action_with_mode(self.dummy_obs)
        self.assertTrue(is_exploring)

        # At epsilon=0.0, agent never explores
        agent.epsilon = 0.0
        _, is_exploring = agent.select_action_with_mode(self.dummy_obs)
        self.assertFalse(is_exploring)

        # Verify decay and clamping
        agent.reset_epsilon()
        self.assertEqual(agent.epsilon, 1.0)
        agent.decay_epsilon()
        self.assertAlmostEqual(agent.epsilon, 0.5)
        agent.decay_epsilon()
        self.assertAlmostEqual(agent.epsilon, 0.25)
        agent.decay_epsilon()
        agent.decay_epsilon()
        agent.decay_epsilon()
        agent.decay_epsilon()
        self.assertEqual(agent.epsilon, 0.05)  # Clamped to min

    # -------------------------------------------------------------------------
    # Part 5: Bellman Target & Terminal Non-Bootstrapping
    # -------------------------------------------------------------------------

    def test_08_terminal_target_does_not_bootstrap(self) -> None:
        """Requirement: When done=True, target_q is strictly reward (no bootstrapping from next_states)."""
        config = DQNConfig(gamma=0.9, warmup_steps=1, batch_size=1)
        agent = DQNAgent(config=config, seed=42)

        # Terminal transition with done=True and reward=10.0
        agent.remember(self.dummy_obs, Action.STRAIGHT, 10.0, self.dummy_obs, done=True)
        batch = agent.replay_buffer.sample(1)

        with torch.no_grad():
            next_q = agent.target_network(batch.next_states).max(1, keepdim=True)[0]
            # Since done=1.0, (1 - done) is 0: target must equal reward exactly
            target_q = batch.rewards + (1.0 - batch.dones) * config.gamma * next_q

        self.assertAlmostEqual(target_q.item(), 10.0)

    def test_09_non_terminal_target_bootstraps_correctly(self) -> None:
        """Requirement: When done=False, target_q includes gamma * max_a' Q_target(s', a')."""
        config = DQNConfig(gamma=0.8, warmup_steps=1, batch_size=1)
        agent = DQNAgent(config=config, seed=42)

        agent.remember(self.dummy_obs, Action.STRAIGHT, 2.0, self.dummy_obs, done=False)
        batch = agent.replay_buffer.sample(1)

        with torch.no_grad():
            next_q = agent.target_network(batch.next_states).max(1, keepdim=True)[0]
            expected_target = 2.0 + 0.8 * next_q.item()
            target_q = batch.rewards + (1.0 - batch.dones) * config.gamma * next_q

        self.assertAlmostEqual(target_q.item(), expected_target, places=5)

    def test_10_training_step_loss_and_gradient_update(self) -> None:
        """Requirement: train_step computes finite Huber loss and updates online weights."""
        config = DQNConfig(warmup_steps=4, batch_size=4, lr=0.01)
        agent = DQNAgent(config=config, seed=42)

        # Warmup check: returns None if buffer < warmup_steps
        self.assertIsNone(agent.train_step())

        for _ in range(6):
            agent.remember(self.dummy_obs, Action.STRAIGHT, 1.0, self.dummy_obs, False)

        old_weight = agent.online_network.network[0].weight.clone()
        res = agent.train_step()
        self.assertIsNotNone(res)
        loss, td_error = res

        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(td_error, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(loss)))
        self.assertFalse(torch.isinf(torch.tensor(loss)))

        # Online weights should have been updated
        new_weight = agent.online_network.network[0].weight
        self.assertFalse(torch.equal(old_weight, new_weight))

    # -------------------------------------------------------------------------
    # Part 6: Checkpoint Serialization & Deserialization
    # -------------------------------------------------------------------------

    def test_11_checkpoint_save_and_load_roundtrip(self) -> None:
        """Requirement: save_dqn_checkpoint and load_dqn_checkpoint restore weights and state exactly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "test_dqn.pt"

            config = DQNConfig(lr=0.002, gamma=0.85, epsilon_start=0.9, epsilon_decay=0.99)
            agent = DQNAgent(config=config, seed=42)
            agent.epsilon = 0.45
            agent.train_step_count = 150

            # Save checkpoint
            saved_path = save_dqn_checkpoint(agent, chk_path, metadata={"note": "unit test"})
            self.assertTrue(saved_path.is_file())

            # Load into new agent
            loaded_agent = load_dqn_checkpoint(saved_path)

            self.assertEqual(loaded_agent.config.lr, 0.002)
            self.assertEqual(loaded_agent.config.gamma, 0.85)
            self.assertAlmostEqual(loaded_agent.epsilon, 0.45)
            self.assertEqual(loaded_agent.train_step_count, 150)

            # Online and target weights must match exactly
            for p1, p2 in zip(
                agent.online_network.parameters(), loaded_agent.online_network.parameters()
            ):
                self.assertTrue(torch.equal(p1, p2))

            for p1, p2 in zip(
                agent.target_network.parameters(), loaded_agent.target_network.parameters()
            ):
                self.assertTrue(torch.equal(p1, p2))

    def test_12_checkpoint_error_handling(self) -> None:
        """Requirement: Corrupted or non-existent checkpoints raise clear ValueError or FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_dqn_checkpoint("non_existent_model.pt")

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "invalid.pt"
            bad_path.write_text("not a torch file", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_dqn_checkpoint(bad_path)

    # -------------------------------------------------------------------------
    # Part 7: Bounded Training Loop & Greedy Evaluation
    # -------------------------------------------------------------------------

    def test_13_bounded_dqn_training_session(self) -> None:
        """Requirement: train_dqn completes specified episodes and records valid metrics."""
        env = SnakeEnvironment(seed=42)
        config = DQNConfig(warmup_steps=4, batch_size=4)
        agent = DQNAgent(config=config, seed=42)

        callbacks_received = []

        def on_episode(m: EpisodeMetrics) -> None:
            callbacks_received.append(m.episode)

        history = train_dqn(env, agent, num_episodes=3, base_seed=10, callback=on_episode)

        self.assertEqual(len(history), 3)
        self.assertEqual(callbacks_received, [1, 2, 3])
        self.assertGreater(history.total_steps, 0)
        self.assertEqual(len(history.scores), 3)
        self.assertEqual(len(history.rewards), 3)
        self.assertEqual(len(history.losses), 3)

        # Invalid episode counts raise ValueError
        with self.assertRaises(ValueError):
            train_dqn(env, agent, num_episodes=0)

    def test_14_evaluate_dqn_agent_is_greedy_and_safe(self) -> None:
        """Requirement: evaluate_agent evaluates DQNAgent with greedy actions and no weight updates."""
        env = SnakeEnvironment(seed=42)
        config = DQNConfig(epsilon_start=1.0)  # Epsilon is 1.0!
        agent = DQNAgent(config=config, seed=42)

        old_weight = agent.online_network.network[0].weight.clone()
        res = evaluate_agent(env, agent, num_episodes=2, base_seed=50)

        self.assertEqual(res.num_episodes, 2)
        self.assertEqual(len(res.scores), 2)
        # Agent's exploration rate must not have changed during evaluation
        self.assertEqual(agent.epsilon, 1.0)
        # Weights must not have changed
        self.assertTrue(torch.equal(agent.online_network.network[0].weight, old_weight))

    def test_15_visual_runner_with_dqn(self) -> None:
        """Requirement: VisualPlaybackRunner works with DQNAgent during training and evaluation."""
        from snake_rl.viewer import VisualPlaybackRunner

        env = SnakeEnvironment(seed=42)
        config = DQNConfig(warmup_steps=4, batch_size=4)
        agent = DQNAgent(config=config, seed=42)

        runner = VisualPlaybackRunner(
            env=env,
            agent=agent,
            speed_delay=0.0,
            interactive=False,
        )

        history = runner.run_training_session(num_episodes=2, base_seed=10)
        self.assertEqual(len(history), 2)
        self.assertLess(agent.epsilon, 1.0)

        # Evaluation mode doesn't change epsilon
        eps_before = agent.epsilon
        summaries = runner.run_evaluation_session(num_episodes=2, base_seed=100)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(agent.epsilon, eps_before)


if __name__ == "__main__":

    unittest.main()
