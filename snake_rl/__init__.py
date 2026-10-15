"""Snake RL package.

An educational Reinforcement Learning project comparing tabular Q-learning
and Deep Q-Networks (DQN) on Snake.
"""

from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.config import EnvironmentConfig, QLearningConfig
from snake_rl.environment import (
    Action,
    Direction,
    Point,
    SnakeEnvironment,
    SnakeObservation,
    StepResult,
    get_next_direction,
)
from snake_rl.state import (
    FEATURE_NAMES,
    NUM_STATES,
    TOTAL_FEATURES,
    StateEncoder,
    decode_int_to_features,
    encode_features_to_int,
    encode_observation,
    encode_observation_to_int,
    is_action_dangerous,
)

__all__ = [
    "Action",
    "Direction",
    "EnvironmentConfig",
    "FEATURE_NAMES",
    "NUM_STATES",
    "Point",
    "QLearningAgent",
    "QLearningConfig",
    "SnakeEnvironment",
    "SnakeObservation",
    "StateEncoder",
    "StepResult",
    "TOTAL_FEATURES",
    "decode_int_to_features",
    "encode_features_to_int",
    "encode_observation",
    "encode_observation_to_int",
    "get_next_direction",
    "is_action_dangerous",
]
