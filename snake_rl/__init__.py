"""Snake RL package.

An educational Reinforcement Learning project comparing tabular Q-learning
and Deep Q-Networks (DQN) on Snake.
"""

from snake_rl.agents import (
    DQNAgent,
    DQNNetwork,
    HeuristicAgent,
    QLearningAgent,
    RandomAgent,
    ReplayBatch,
    ReplayBuffer,
    Transition,
)
from snake_rl.checkpoints import (
    load_dqn_checkpoint,
    load_q_table,
    save_dqn_checkpoint,
    save_q_table,
)
from snake_rl.config import DQNConfig, EnvironmentConfig, QLearningConfig
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
from snake_rl.rendering import RenderedFrame, format_hud, render_ascii, render_grid
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
from snake_rl.training import (
    EpisodeMetrics,
    TrainingHistory,
    train_dqn,
    train_q_learning,
)
from snake_rl.viewer import VisualPlaybackRunner

__all__ = [
    "Action",
    "DQNConfig",
    "DQNAgent",
    "DQNNetwork",
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
    "RenderedFrame",
    "ReplayBatch",
    "ReplayBuffer",
    "SnakeEnvironment",
    "SnakeObservation",
    "StateEncoder",
    "StepResult",
    "TOTAL_FEATURES",
    "TrainingHistory",
    "Transition",
    "VisualPlaybackRunner",
    "compare_agents",
    "decode_int_to_features",
    "encode_features_to_int",
    "encode_observation",
    "encode_observation_to_int",
    "evaluate_agent",
    "format_hud",
    "get_next_direction",
    "is_action_dangerous",
    "load_dqn_checkpoint",
    "load_q_table",
    "render_ascii",
    "render_grid",
    "save_dqn_checkpoint",
    "save_q_table",
    "train_dqn",
    "train_q_learning",
]

