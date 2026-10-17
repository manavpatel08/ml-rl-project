"""Deep Q-Network (DQN) implementation for Snake RL.

Implements an MLP Q-network, target network synchronization, experience replay
integration, Huber loss optimization, and epsilon-greedy exploration on CPU.
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from snake_rl.agents.replay_buffer import ReplayBuffer
from snake_rl.config import DQNConfig
from snake_rl.environment import Action, SnakeObservation
from snake_rl.state import TOTAL_FEATURES, encode_observation

NUM_ACTIONS = len(Action)  # 3 relative actions


class DQNNetwork(nn.Module):
    """Multi-Layer Perceptron (MLP) for Q-value approximation.

    Architecture:
      Input (11) -> Linear(11, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 3)
    """

    def __init__(
        self,
        input_dim: int = TOTAL_FEATURES,
        hidden_dims: Tuple[int, ...] = (128, 128),
        output_dim: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing Q-values for each relative action.

        Args:
            x: Tensor of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            Tensor of shape (batch_size, output_dim) containing Q-values.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)


class DQNAgent:
    """Deep Q-Network agent with online and target networks."""

    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the DQNAgent.

        Args:
            config: Hyperparameters for network, replay buffer, and optimizer.
            seed: Optional integer seed for reproducibility.
        """
        self.config: DQNConfig = config if config is not None else DQNConfig()
        self.device = torch.device(self.config.device)

        if seed is not None:
            torch.manual_seed(seed)
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        # 1. Initialize Online and Target networks
        self.online_network = DQNNetwork(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
        ).to(self.device)

        self.target_network = DQNNetwork(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
        ).to(self.device)

        # Synchronize target network and exclude it from gradient updates
        self.update_target_network()
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False

        # 2. Experience replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            seed=seed,
        )

        # 3. Optimizer and Huber Loss (SmoothL1Loss)
        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=self.config.lr,
        )
        self.loss_fn = nn.SmoothL1Loss()

        # 4. Telemetry and exploration
        self.train_step_count: int = 0
        self.epsilon: float = self.config.epsilon_start

    def seed(self, seed: Optional[int] = None) -> None:
        """Re-seed the agent's RNG and replay buffer."""
        self._rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.replay_buffer.seed(seed)

    def select_action(self, state: Union[Sequence[float], SnakeObservation]) -> Action:
        """Select an action using epsilon-greedy exploration."""
        action, _ = self.select_action_with_mode(state)
        return action

    def select_action_with_mode(
        self,
        state: Union[Sequence[float], SnakeObservation],
    ) -> Tuple[Action, bool]:
        """Select an action using epsilon-greedy exploration, returning whether it explored.

        Returns:
            Tuple of (Action, is_exploring: bool).
        """
        if self._rng.random() < self.epsilon:
            random_int = self._rng.randint(0, self.config.output_dim - 1)
            return Action(random_int), True

        return self.select_greedy_action(state), False

    def select_greedy_action(
        self,
        state: Union[Sequence[float], SnakeObservation],
    ) -> Action:
        """Select the greedy action (argmax Q) without exploration."""
        tensor_state = self._state_to_tensor(state)
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(tensor_state).squeeze(0)

        # Deterministic tie-breaking if multiple Q-values are identical
        max_q = q_values.max().item()
        best_indices = [
            i for i, q in enumerate(q_values.tolist())
            if math.isclose(q, max_q, abs_tol=1e-7)
        ]

        if len(best_indices) == 1:
            return Action(best_indices[0])

        return Action(self._rng.choice(best_indices))

    def remember(
        self,
        state: Union[Sequence[float], SnakeObservation],
        action: Union[Action, int],
        reward: float,
        next_state: Union[Sequence[float], SnakeObservation],
        done: bool,
    ) -> None:
        """Store an environment transition in the replay buffer."""
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def train_step(self) -> Optional[Tuple[float, float]]:
        """Perform a single minibatch gradient descent update on the online network.

        Returns:
            Tuple of (loss_value, mean_td_error) if a step was taken, or None if
            the buffer contains fewer transitions than warmup_steps or batch_size.
        """
        min_required = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay_buffer) < min_required:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size, device=self.device)

        self.online_network.train()

        # 1. Current Q-values: Q_online(s, a)
        all_q_values = self.online_network(batch.states)
        current_q = all_q_values.gather(1, batch.actions)

        # 2. Target Q-values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_target_q = self.target_network(batch.next_states).max(1, keepdim=True)[0]
            # When done == 1.0 (terminal), next_target_q is zeroed out: target = reward
            target_q = batch.rewards + (1.0 - batch.dones) * self.config.gamma * next_target_q

        # 3. Loss computation and backpropagation
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms to stabilize neural-network training
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1

        # 4. Synchronize target network on configured interval
        if self.train_step_count % self.config.target_update_interval == 0:
            self.update_target_network()

        td_error = (target_q - current_q).abs().mean().item()
        return (loss.item(), td_error)

    def update_target_network(self) -> None:
        """Copy weights from the online network to the target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self) -> float:
        """Decay exploration rate per episode, clamped to epsilon_min."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        return self.epsilon

    def reset_epsilon(self) -> None:
        """Reset exploration rate to epsilon_start."""
        self.epsilon = self.config.epsilon_start

    def _state_to_tensor(
        self,
        state: Union[Sequence[float], SnakeObservation],
    ) -> torch.Tensor:
        """Convert a state representation into a PyTorch float32 tensor."""
        if isinstance(state, SnakeObservation):
            features = [float(f) for f in encode_observation(state)]
        elif isinstance(state, (list, tuple)):
            features = [float(f) for f in state]
        else:
            raise TypeError(f"Unsupported state type: {type(state).__name__}")

        if len(features) != self.config.input_dim:
            raise ValueError(
                f"State must have {self.config.input_dim} features, got {len(features)}"
            )

        return torch.tensor(features, dtype=torch.float32, device=self.device)
