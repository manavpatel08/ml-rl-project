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
