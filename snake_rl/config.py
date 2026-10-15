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
