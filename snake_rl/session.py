"""Session state managers and checkpoint discovery for interactive playback and training.

Decouples interactive stepping, pause/resume, and checkpoint handling from UI frameworks,
allowing rigorous unit testing without a browser or Streamlit context.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from snake_rl.agents.baselines import HeuristicAgent, RandomAgent
from snake_rl.agents.dqn import DQNAgent
from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.checkpoints import (
    load_dqn_checkpoint,
    load_q_table,
    save_dqn_checkpoint,
    save_q_table,
)
from snake_rl.environment import Action, Direction, SnakeEnvironment, SnakeObservation
from snake_rl.training import EpisodeMetrics, TrainingHistory


class TrainingSession:
    """Orchestrates an interactive training run across discrete steps or episodes.

    Guarantees:
      - State persists across UI reruns without resetting the agent or environment.
      - Stepping advances the real environment and calls real learning updates.
      - Stepping when paused or stopped performs zero learning or environment updates.
      - Epsilon decays exactly once per episode at episode completion.
      - Metrics (score, steps, reward, epsilon, td_error, loss) are recorded faithfully.
    """

    def __init__(
        self,
        agent: Union[QLearningAgent, DQNAgent],
        env: Optional[SnakeEnvironment] = None,
        total_episodes: int = 10,
        base_seed: Optional[int] = 42,
    ) -> None:
        if total_episodes <= 0:
            raise ValueError(f"total_episodes must be positive, got {total_episodes}")

        self.agent = agent
        self.env = env if env is not None else SnakeEnvironment()
        self.total_episodes: int = total_episodes
        self.base_seed: Optional[int] = base_seed

        self.current_episode: int = 1
        self.current_score: int = 0
        self.current_steps: int = 0
        self.current_reward: float = 0.0

        self.last_action: Optional[Action] = None
        self.last_is_exploring: Optional[bool] = None
        self.last_td_error: float = 0.0
        self.last_loss: float = 0.0
        self.last_termination_reason: str = "running"

        self.total_steps: int = 0
        self.history: TrainingHistory = TrainingHistory()

        self.is_paused: bool = False
        self.is_stopped: bool = False
        self.is_episode_done: bool = False

        self._ep_td_errors: List[float] = []
        self._ep_losses: List[float] = []

        # Initialize first episode
        self.current_obs: SnakeObservation = self.reset_episode()

    def reset_episode(self, seed: Optional[int] = None) -> SnakeObservation:
        """Reset the environment for the current episode."""
        ep_seed = (
            seed
            if seed is not None
            else (self.base_seed + self.current_episode if self.base_seed is not None else None)
        )
        self.current_obs = self.env.reset(seed=ep_seed)
        self.current_score = 0
        self.current_steps = 0
        self.current_reward = 0.0
        self.is_episode_done = False
        self.last_termination_reason = "running"
        self._ep_td_errors = []
        self._ep_losses = []
        return self.current_obs

    def can_step(self) -> bool:
        """Return True if the session can advance a step."""
        return not self.is_paused and not self.is_stopped and not self.is_episode_done

    def pause(self) -> None:
        """Pause session execution."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume session execution."""
        self.is_paused = False

    def stop(self) -> None:
        """Stop session execution permanently."""
        self.is_stopped = True

    def step(self) -> Tuple[SnakeObservation, bool, Dict[str, Any]]:
        """Advance the training environment and agent by a single real action.

        Returns:
            Tuple of (current_observation, is_done, step_info_dict).
        """
        if not self.can_step():
            return (
                self.current_obs,
                self.is_episode_done,
                {
                    "action": self.last_action,
                    "is_exploring": self.last_is_exploring,
                    "reward": 0.0,
                    "td_error": self.last_td_error,
                    "loss": self.last_loss,
                    "status": "paused" if self.is_paused else ("stopped" if self.is_stopped else "done"),
                },
            )

        # 1. Action selection with exploration tracking
        action, is_exploring = self.agent.select_action_with_mode(self.current_obs)
        self.last_action = action
        self.last_is_exploring = is_exploring

        # 2. Step the real environment
        step_res = self.env.step(action)
        self.current_score = self.env.score
        self.current_steps = self.env.steps
        self.current_reward += step_res.reward
        self.total_steps += 1

        step_td = 0.0
        step_loss = 0.0

        # 3. Apply algorithm-specific learning updates
        if isinstance(self.agent, QLearningAgent):
            step_td = abs(
                self.agent.update(
                    state=self.current_obs,
                    action=action,
                    reward=step_res.reward,
                    next_state=step_res.observation,
                    done=step_res.done,
                )
            )
            self._ep_td_errors.append(step_td)
            self.last_td_error = step_td
        elif isinstance(self.agent, DQNAgent):
            self.agent.remember(
                state=self.current_obs,
                action=action,
                reward=step_res.reward,
                next_state=step_res.observation,
                done=step_res.done,
            )
            train_res = self.agent.train_step()
            if train_res is not None:
                step_loss, step_td = train_res
                self._ep_losses.append(step_loss)
                self._ep_td_errors.append(step_td)
                self.last_loss = step_loss
                self.last_td_error = step_td

        self.current_obs = step_res.observation

        # 4. Handle episode conclusion
        if step_res.done:
            self.is_episode_done = True
            self.last_termination_reason = step_res.info.get("reason", "unknown")

            # Epsilon decay occurs exactly once at episode end
            new_epsilon = self.agent.decay_epsilon()

            mean_td = (
                sum(self._ep_td_errors) / len(self._ep_td_errors)
                if self._ep_td_errors
                else 0.0
            )
            mean_loss = (
                sum(self._ep_losses) / len(self._ep_losses)
                if self._ep_losses
                else 0.0
            )

            metrics = EpisodeMetrics(
                episode=self.current_episode,
                total_reward=round(self.current_reward, 4),
                score=self.current_score,
                steps=self.current_steps,
                epsilon=round(new_epsilon, 6),
                mean_td_error=round(mean_td, 6),
                termination_reason=self.last_termination_reason,
                loss=round(mean_loss, 6),
                cumulative_steps=self.total_steps,
                cumulative_gradient_updates=getattr(self.agent, "train_step_count", 0),
            )
            self.history.episodes.append(metrics)

        info = {
            "action": action,
            "is_exploring": is_exploring,
            "reward": step_res.reward,
            "td_error": step_td,
            "loss": step_loss,
            "reason": step_res.info.get("reason", ""),
            "status": "done" if step_res.done else "active",
        }
        return (self.current_obs, step_res.done, info)

    def next_episode(self) -> bool:
        """Advance to the next episode if the current one is done and total limit not reached.

        Returns:
            True if a new episode was successfully started, False if session is complete.
        """
        if self.is_stopped:
            return False

        if self.current_episode < self.total_episodes:
            self.current_episode += 1
            self.reset_episode()
            return True

        return False

    def is_finished(self) -> bool:
        """Return True if all planned episodes are completed or session is stopped."""
        return self.is_stopped or (
            self.current_episode >= self.total_episodes and self.is_episode_done
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return structured summary metrics for UI presentation."""
        rolling_avg = (
            sum(self.history.scores[-10:]) / len(self.history.scores[-10:])
            if self.history.scores
            else float(self.current_score)
        )
        recent_loss_avg = (
            sum(self.history.losses[-10:]) / len(self.history.losses[-10:])
            if self.history.losses
            else self.last_loss
        )

        is_dqn = isinstance(self.agent, DQNAgent)
        buffer_size = len(self.agent.replay_buffer) if is_dqn and hasattr(self.agent, "replay_buffer") else 0
        buffer_capacity = (
            self.agent.replay_buffer.capacity
            if is_dqn and hasattr(self.agent, "replay_buffer") and hasattr(self.agent.replay_buffer, "capacity")
            else 0
        )
        buffer_occupancy_pct = (
            (buffer_size / buffer_capacity * 100.0) if is_dqn and buffer_capacity > 0 else 0.0
        )
        gradient_updates = getattr(self.agent, "train_step_count", 0)
        target_sync_interval = (
            getattr(self.agent.config, "target_update_interval", 0) if is_dqn else 0
        )
        if is_dqn and target_sync_interval > 0:
            target_sync_count = gradient_updates // target_sync_interval
            updates_since_sync = gradient_updates % target_sync_interval
            steps_until_sync = (
                target_sync_interval - updates_since_sync
                if updates_since_sync > 0
                else target_sync_interval
            )
        else:
            target_sync_count = 0
            updates_since_sync = 0
            steps_until_sync = 0

        return {
            "episode": self.current_episode,
            "total_episodes": self.total_episodes,
            "score": self.current_score,
            "steps": self.current_steps,
            "reward": round(self.current_reward, 2),
            "epsilon": getattr(self.agent, "epsilon", 0.0),
            "rolling_avg_score": round(rolling_avg, 2),
            "last_action_name": self.last_action.name if self.last_action is not None else "START",
            "is_exploring": self.last_is_exploring,
            "last_loss": round(self.last_loss, 5),
            "recent_loss_avg": round(recent_loss_avg, 5),
            "last_td_error": round(self.last_td_error, 5),
            "total_steps": self.total_steps,
            "train_step_count": gradient_updates,
            "is_paused": self.is_paused,
            "is_stopped": self.is_stopped,
            "is_episode_done": self.is_episode_done,
            "termination_reason": self.last_termination_reason,
            "completed_episodes": len(self.history),
            "is_dqn": is_dqn,
            "buffer_size": buffer_size,
            "buffer_capacity": buffer_capacity,
            "buffer_occupancy_pct": round(buffer_occupancy_pct, 2),
            "buffer_occupancy_str": f"{buffer_size} / {buffer_capacity} ({buffer_occupancy_pct:.1f}%)" if is_dqn else "N/A",
            "gradient_updates": gradient_updates,
            "target_sync_interval": target_sync_interval,
            "updates_since_sync": updates_since_sync,
            "steps_until_sync": steps_until_sync,
            "target_sync_count": target_sync_count,
            "target_sync_str": f"{updates_since_sync} / {target_sync_interval} ({steps_until_sync} remaining)" if is_dqn else "N/A",
        }


class EvaluationSession:
    """Manages an evaluation playback session under pure exploitation.

    Guarantees:
      - Actions are chosen greedily without exploration (epsilon = 0.0).
      - Zero Q-values, neural-network weights, or epsilon values are modified.
      - Replay buffer and optimizer states remain completely untouched.
    """

    def __init__(
        self,
        agent: Any,
        env: Optional[SnakeEnvironment] = None,
        base_seed: int = 1000,
    ) -> None:
        self.agent = agent
        self.env = env if env is not None else SnakeEnvironment()
        self.base_seed: int = base_seed

        self.current_episode: int = 1
        self.current_score: int = 0
        self.current_steps: int = 0
        self.current_reward: float = 0.0
        self.last_action: Optional[Action] = None
        self.last_termination_reason: str = "running"

        self.is_paused: bool = False
        self.is_stopped: bool = False
        self.is_episode_done: bool = False

        self.scores: List[int] = []
        self.rewards: List[float] = []
        self.steps_list: List[int] = []

        self.current_obs: SnakeObservation = self.reset_episode()

    def reset_episode(self, seed: Optional[int] = None) -> SnakeObservation:
        """Reset the evaluation environment for a new episode."""
        ep_seed = seed if seed is not None else (self.base_seed + self.current_episode)
        self.current_obs = self.env.reset(seed=ep_seed)
        self.current_score = 0
        self.current_steps = 0
        self.current_reward = 0.0
        self.is_episode_done = False
        self.last_termination_reason = "running"
        return self.current_obs

    def can_step(self) -> bool:
        """Return True if the evaluation session can advance a step."""
        return not self.is_paused and not self.is_stopped and not self.is_episode_done

    def pause(self) -> None:
        """Pause playback."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume playback."""
        self.is_paused = False

    def stop(self) -> None:
        """Stop playback."""
        self.is_stopped = True

    def step(self) -> Tuple[SnakeObservation, bool, Dict[str, Any]]:
        """Advance evaluation by a single greedy action without weight updates."""
        if not self.can_step():
            return (
                self.current_obs,
                self.is_episode_done,
                {
                    "action": self.last_action,
                    "reward": 0.0,
                    "status": "paused" if self.is_paused else ("stopped" if self.is_stopped else "done"),
                },
            )

        # Greedy action selection (pure exploitation)
        if hasattr(self.agent, "select_greedy_action"):
            action = self.agent.select_greedy_action(self.current_obs)
        else:
            action = self.agent.select_action(self.current_obs)

        self.last_action = action
        step_res = self.env.step(action)
        self.current_score = self.env.score
        self.current_steps = self.env.steps
        self.current_reward += step_res.reward
        self.current_obs = step_res.observation

        if step_res.done:
            self.is_episode_done = True
            self.last_termination_reason = step_res.info.get("reason", "unknown")
            self.scores.append(self.current_score)
            self.rewards.append(round(self.current_reward, 2))
            self.steps_list.append(self.current_steps)

        info = {
            "action": action,
            "reward": step_res.reward,
            "reason": step_res.info.get("reason", ""),
            "status": "done" if step_res.done else "active",
        }
        return (self.current_obs, step_res.done, info)

    def next_episode(self) -> None:
        """Advance to a subsequent evaluation episode."""
        self.current_episode += 1
        self.reset_episode()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return evaluation metrics summary."""
        mean_score = sum(self.scores) / len(self.scores) if self.scores else float(self.current_score)
        max_score = max(self.scores) if self.scores else self.current_score
        return {
            "episode": self.current_episode,
            "score": self.current_score,
            "steps": self.current_steps,
            "reward": round(self.current_reward, 2),
            "mean_score": round(mean_score, 2),
            "max_score": max_score,
            "last_action_name": self.last_action.name if self.last_action is not None else "START",
            "is_paused": self.is_paused,
            "is_stopped": self.is_stopped,
            "is_episode_done": self.is_episode_done,
            "termination_reason": self.last_termination_reason,
            "completed_episodes": len(self.scores),
        }


def list_available_checkpoints(models_dir: Union[str, Path] = "models") -> List[Dict[str, Any]]:
    """Scan the models directory for valid checkpoints without mutating files.

    Returns:
        List of dictionaries with file metadata (name, path, algorithm, size_kb, modified).
    """
    path = Path(models_dir)
    if not path.is_dir():
        return []

    checkpoints = []
    for file in sorted(path.iterdir()):
        if file.suffix.lower() in (".json", ".pt") and file.is_file():
            stat = file.stat()
            algo = "unknown"
            if file.suffix.lower() == ".json":
                algo = "q_learning"
            elif file.suffix.lower() == ".pt":
                algo = "dqn"

            checkpoints.append(
                {
                    "name": file.name,
                    "path": str(file.resolve()),
                    "algorithm": algo,
                    "size_kb": round(stat.st_size / 1024.0, 1),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                }
            )
    return checkpoints


def safe_save_checkpoint(
    agent: Any,
    filepath: Union[str, Path],
    overwrite: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Safely serialize an agent checkpoint with explicit overwrite protection.

    Args:
        agent: Trained QLearningAgent or DQNAgent instance.
        filepath: Destination path string or Path object.
        overwrite: If False, saving will abort if the file already exists.
        metadata: Optional dictionary of training stats.

    Returns:
        Tuple of (success: bool, message: str).
    """
    dest = Path(filepath)
    if dest.exists() and not overwrite:
        return (
            False,
            f"File '{dest.name}' already exists. Please check 'Overwrite existing checkpoint' to proceed.",
        )

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(agent, DQNAgent):
            saved = save_dqn_checkpoint(agent, dest, metadata=metadata)
            return (True, f"DQN checkpoint successfully saved to {saved.name}")
        elif isinstance(agent, QLearningAgent):
            saved = save_q_table(agent, dest, metadata=metadata)
            return (True, f"Q-table checkpoint successfully saved to {saved.name}")
        else:
            return (False, f"Unsupported agent type for checkpoint serialization: {type(agent).__name__}")
    except Exception as exc:
        return (False, f"Error saving checkpoint: {exc}")
