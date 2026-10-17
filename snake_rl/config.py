"""Configuration dataclasses for Snake RL."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for the headless Snake environment.

    Attributes:
        grid_width: Width of the grid in cells.
        grid_height: Height of the grid in cells.
        reward_food: Reward obtained when eating food. Default +10.0.
        reward_collision: Penalty (negative reward) on collision. Default -10.0.
        reward_step: Penalty (negative reward) on each normal non-food move. Default -0.1.
        max_steps: Maximum step limit before episode terminates. Default 1000.
        initial_snake_length: Starting length of the snake. Default 3.
    """

    grid_width: int = 20
    grid_height: int = 20
    reward_food: float = 10.0
    reward_collision: float = -10.0
    reward_step: float = -0.1
    max_steps: Optional[int] = 1000
    initial_snake_length: int = 3


@dataclass(frozen=True)
class QLearningConfig:
    """Hyperparameters for Tabular Q-learning.

    Attributes:
        alpha: Learning rate (0.0 < alpha <= 1.0). Default 0.1.
        gamma: Discount factor for future rewards (0.0 <= gamma <= 1.0). Default 0.9.
        epsilon_start: Initial exploration rate. Default 1.0.
        epsilon_min: Minimum exploration rate floor. Default 0.01.
        epsilon_decay: Multiplicative decay factor applied per episode. Default 0.995.
    """

    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {self.alpha}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0.0, 1.0], got {self.gamma}")
        if not (0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0):
            raise ValueError(
                f"Invalid epsilon configuration: 0.0 <= epsilon_min ({self.epsilon_min}) "
                f"<= epsilon_start ({self.epsilon_start}) <= 1.0"
            )
        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError(f"epsilon_decay must be in (0.0, 1.0], got {self.epsilon_decay}")


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for Deep Q-Network (DQN).

    Attributes:
        input_dim: Dimension of input state feature vector. Default 11.
        hidden_dims: Tuple of neuron counts for hidden layers. Default (128, 128).
        output_dim: Dimension of action outputs (Q-values). Default 3.
        lr: Learning rate for Adam optimizer. Default 0.001.
        gamma: Discount factor for future rewards. Default 0.9.
        epsilon_start: Initial exploration rate. Default 1.0.
        epsilon_min: Minimum exploration rate floor. Default 0.01.
        epsilon_decay: Multiplicative decay factor per episode. Default 0.995.
        buffer_capacity: Maximum transitions stored in replay buffer. Default 10000.
        batch_size: Number of transitions sampled per gradient update. Default 64.
        target_update_interval: Frequency (in gradient update steps) to sync target network. Default 500.
        warmup_steps: Minimum buffer transitions before training steps begin. Default 64.
        device: PyTorch device ('cpu' by default).
    """

    input_dim: int = 11
    hidden_dims: tuple = (128, 128)
    output_dim: int = 3
    lr: float = 0.001
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    buffer_capacity: int = 10_000
    batch_size: int = 64
    target_update_interval: int = 500
    warmup_steps: int = 64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if not (0.0 < self.lr <= 1.0):
            raise ValueError(f"lr must be in (0.0, 1.0], got {self.lr}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0.0, 1.0], got {self.gamma}")
        if not (0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0):
            raise ValueError(
                f"Invalid epsilon configuration: 0.0 <= epsilon_min ({self.epsilon_min}) "
                f"<= epsilon_start ({self.epsilon_start}) <= 1.0"
            )
        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError(f"epsilon_decay must be in (0.0, 1.0], got {self.epsilon_decay}")
        if self.buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be positive, got {self.buffer_capacity}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.batch_size > self.buffer_capacity:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed buffer_capacity ({self.buffer_capacity})"
            )
        if self.target_update_interval <= 0:
            raise ValueError(
                f"target_update_interval must be positive, got {self.target_update_interval}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
