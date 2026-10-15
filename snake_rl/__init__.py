"""Snake RL package.

An educational Reinforcement Learning project comparing tabular Q-learning
and Deep Q-Networks (DQN) on Snake.
"""

from snake_rl.config import EnvironmentConfig
from snake_rl.environment import Action, Direction, Point, SnakeEnvironment, SnakeObservation, StepResult

__all__ = [
    "Action",
    "Direction",
    "EnvironmentConfig",
    "Point",
    "SnakeEnvironment",
    "SnakeObservation",
    "StepResult",
]
