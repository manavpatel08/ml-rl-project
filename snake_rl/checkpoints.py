"""Checkpoint saving and loading utilities for Snake RL agents.

Provides safe, JSON-based serialization for Q-tables and agent configurations,
ensuring portability and avoiding unsafe binary pickle formats.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from snake_rl.agents.dqn import DQNAgent
from snake_rl.agents.q_learning import NUM_ACTIONS, QLearningAgent
from snake_rl.config import DQNConfig, QLearningConfig
from snake_rl.state import NUM_STATES

CHECKPOINT_FORMAT_VERSION: str = "1.0"
ALGORITHM_NAME: str = "q_learning"
DQN_ALGORITHM_NAME: str = "dqn"



def save_q_table(
    agent: QLearningAgent,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save an agent's Q-table, configuration, and metadata to a JSON file.

    Args:
        agent: Trained QLearningAgent to serialize.
        filepath: Destination file path (e.g., 'models/q_table.json').
        metadata: Optional dictionary of training stats or timestamps.

    Returns:
        Path object pointing to the saved file.

    Raises:
        ValueError: If the Q-table contains invalid or non-finite values.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate Q-table values before saving
    for s_idx, row in enumerate(agent.q_table):
        for a_idx, val in enumerate(row):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Cannot serialize non-finite Q-value at state {s_idx}, action {a_idx}: {val}"
                )

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "algorithm": ALGORITHM_NAME,
        "num_states": NUM_STATES,
        "num_actions": NUM_ACTIONS,
        "config": {
            "alpha": agent.config.alpha,
            "gamma": agent.config.gamma,
            "epsilon_start": agent.config.epsilon_start,
            "epsilon_min": agent.config.epsilon_min,
            "epsilon_decay": agent.config.epsilon_decay,
        },
        "epsilon": agent.epsilon,
        "metadata": metadata or {},
        "q_table": agent.q_table,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def load_q_table(
    filepath: Union[str, Path],
    agent: Optional[QLearningAgent] = None,
) -> QLearningAgent:
    """Load a saved Q-table from a JSON checkpoint into an existing or new agent.

    Args:
        filepath: Source checkpoint path.
        agent: Optional existing agent to populate. If None, a new agent is created.

    Returns:
        QLearningAgent initialized with the loaded Q-table and configuration.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the checkpoint format, dimensions, or values are invalid.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Checkpoint file is not valid JSON: {exc}") from exc

    # Validate header and algorithm
    if data.get("algorithm") != ALGORITHM_NAME:
        raise ValueError(
            f"Expected algorithm '{ALGORITHM_NAME}', but found '{data.get('algorithm')}'."
        )

    num_states = data.get("num_states")
    num_actions = data.get("num_actions")
    if num_states != NUM_STATES or num_actions != NUM_ACTIONS:
        raise ValueError(
            f"Shape mismatch: expected ({NUM_STATES}, {NUM_ACTIONS}), "
            f"got ({num_states}, {num_actions})."
        )

    q_table_data = data.get("q_table")
    if not isinstance(q_table_data, list) or len(q_table_data) != NUM_STATES:
        raise ValueError(
            f"Q-table must be a list of {NUM_STATES} rows, got {type(q_table_data).__name__}."
        )

    # Validate row lengths and value finiteness
    validated_table = []
    for s_idx, row in enumerate(q_table_data):
        if not isinstance(row, list) or len(row) != NUM_ACTIONS:
            raise ValueError(
                f"Row {s_idx} must have length {NUM_ACTIONS}, got {len(row) if isinstance(row, list) else 'non-list'}."
            )
        for a_idx, val in enumerate(row):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Non-finite value found at state {s_idx}, action {a_idx}: {val}"
                )
        validated_table.append([float(v) for v in row])

    # Construct or update the agent
    cfg_dict = data.get("config", {})
    config = QLearningConfig(
        alpha=cfg_dict.get("alpha", 0.1),
        gamma=cfg_dict.get("gamma", 0.9),
        epsilon_start=cfg_dict.get("epsilon_start", 1.0),
        epsilon_min=cfg_dict.get("epsilon_min", 0.01),
        epsilon_decay=cfg_dict.get("epsilon_decay", 0.995),
    )

    if agent is None:
        agent = QLearningAgent(config=config)
    else:
        # Update existing agent config
        object.__setattr__(agent, "config", config)

    # Populate Q-table and current epsilon
    agent._q_table = validated_table
    if "epsilon" in data and isinstance(data["epsilon"], (int, float)):
        agent.epsilon = float(data["epsilon"])

    return agent


def save_dqn_checkpoint(
    agent: DQNAgent,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a DQN agent's network weights, configuration, and metadata.

    Args:
        agent: Trained DQNAgent to serialize.
        filepath: Destination file path (e.g., 'models/dqn_model.pt').
        metadata: Optional dictionary of training stats or timestamps.

    Returns:
        Path object pointing to the saved file.

    Raises:
        ValueError: If network weights contain non-finite values (NaN or Inf).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate weights finiteness
    for name, param in agent.online_network.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(
                f"Cannot serialize non-finite weights in online network parameter '{name}'"
            )

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "algorithm": DQN_ALGORITHM_NAME,
        "config": {
            "input_dim": agent.config.input_dim,
            "hidden_dims": list(agent.config.hidden_dims),
            "output_dim": agent.config.output_dim,
            "lr": agent.config.lr,
            "gamma": agent.config.gamma,
            "epsilon_start": agent.config.epsilon_start,
            "epsilon_min": agent.config.epsilon_min,
            "epsilon_decay": agent.config.epsilon_decay,
            "buffer_capacity": agent.config.buffer_capacity,
            "batch_size": agent.config.batch_size,
            "target_update_interval": agent.config.target_update_interval,
            "warmup_steps": agent.config.warmup_steps,
            "device": str(agent.device),
        },
        "online_state_dict": agent.online_network.state_dict(),
        "target_state_dict": agent.target_network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "train_step_count": agent.train_step_count,
        "epsilon": float(agent.epsilon),
        "metadata": metadata or {},
    }

    torch.save(payload, path)
    return path


def load_dqn_checkpoint(
    filepath: Union[str, Path],
    agent: Optional[DQNAgent] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> DQNAgent:
    """Load a saved DQN checkpoint into an existing or new agent.

    Args:
        filepath: Source checkpoint path.
        agent: Optional existing agent to populate. If None, a new agent is created.
        device: Device to map tensors to (defaults to CPU).

    Returns:
        DQNAgent initialized with loaded weights and configuration.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If checkpoint format or algorithm is invalid.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path.resolve()}")

    map_dev = torch.device(device) if device is not None else torch.device("cpu")

    try:
        data = torch.load(path, map_location=map_dev, weights_only=True)
    except Exception as exc:
        raise ValueError(f"Failed to load PyTorch checkpoint: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint must be a dictionary payload, got {type(data).__name__}.")

    if data.get("algorithm") != DQN_ALGORITHM_NAME:
        raise ValueError(
            f"Expected algorithm '{DQN_ALGORITHM_NAME}', but found '{data.get('algorithm')}'."
        )

    if "online_state_dict" not in data:
        raise ValueError("Missing 'online_state_dict' in checkpoint payload.")

    cfg_dict = data.get("config", {})
    hidden_dims = tuple(cfg_dict.get("hidden_dims", (128, 128)))
    config = DQNConfig(
        input_dim=cfg_dict.get("input_dim", 11),
        hidden_dims=hidden_dims,
        output_dim=cfg_dict.get("output_dim", 3),
        lr=cfg_dict.get("lr", 0.001),
        gamma=cfg_dict.get("gamma", 0.9),
        epsilon_start=cfg_dict.get("epsilon_start", 1.0),
        epsilon_min=cfg_dict.get("epsilon_min", 0.01),
        epsilon_decay=cfg_dict.get("epsilon_decay", 0.995),
        buffer_capacity=cfg_dict.get("buffer_capacity", 10_000),
        batch_size=cfg_dict.get("batch_size", 64),
        target_update_interval=cfg_dict.get("target_update_interval", 500),
        warmup_steps=cfg_dict.get("warmup_steps", 64),
        device=str(map_dev),
    )

    if agent is None:
        agent = DQNAgent(config=config)
    else:
        object.__setattr__(agent, "config", config)

    agent.online_network.load_state_dict(data["online_state_dict"])
    if "target_state_dict" in data:
        agent.target_network.load_state_dict(data["target_state_dict"])
    else:
        agent.update_target_network()

    agent.target_network.eval()
    for param in agent.target_network.parameters():
        param.requires_grad = False

    if "optimizer_state_dict" in data and hasattr(agent, "optimizer"):
        try:
            agent.optimizer.load_state_dict(data["optimizer_state_dict"])
        except Exception:
            pass

    if "train_step_count" in data and isinstance(data["train_step_count"], int):
        agent.train_step_count = data["train_step_count"]

    if "epsilon" in data and isinstance(data["epsilon"], (int, float)):
        agent.epsilon = float(data["epsilon"])

    return agent

