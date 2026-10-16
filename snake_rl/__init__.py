"""Snake RL package.

An educational Reinforcement Learning project comparing tabular Q-learning
and Deep Q-Networks (DQN) on Snake.
"""

from snake_rl.agents import HeuristicAgent, QLearningAgent, RandomAgent
from snake_rl.checkpoints import load_q_table, save_q_table
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
from snake_rl.evaluation import EvaluationResult, compare_agents, evaluate_agent
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
from snake_rl.training import EpisodeMetrics, TrainingHistory, train_q_learning

__all__ = [
    "Action",
    "Direction",
    "EnvironmentConfig",
    "EpisodeMetrics",
    "EvaluationResult",
    "FEATURE_NAMES",
    "HeuristicAgent",
    "NUM_STATES",
    "Point",
    "QLearningAgent",
    "QLearningConfig",
    "RandomAgent",
    "SnakeEnvironment",
    "SnakeObservation",
    "StateEncoder",
    "StepResult",
    "TOTAL_FEATURES",
    "TrainingHistory",
    "compare_agents",
    "decode_int_to_features",
    "encode_features_to_int",
    "encode_observation",
    "encode_observation_to_int",
    "evaluate_agent",
    "get_next_direction",
    "is_action_dangerous",
    "load_q_table",
    "save_q_table",
    "train_q_learning",
]
