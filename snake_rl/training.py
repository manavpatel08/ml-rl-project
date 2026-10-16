"""Training infrastructure and metrics tracking for Snake RL agents.

Provides a clean, bounded training orchestrator that connects the SnakeEnvironment
with learning agents while recording structured per-episode metrics.
"""

from dataclasses import dataclass, field
import statistics
from typing import Callable, Dict, List, Optional, Sequence

from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.environment import SnakeEnvironment, SnakeObservation


@dataclass(frozen=True)
class EpisodeMetrics:
    """Performance metrics captured from a single training episode.

    Attributes:
        episode: 1-based episode index.
        total_reward: Cumulative scalar reward earned in the episode.
        score: Food items eaten in the episode.
        steps: Total actions taken before episode termination.
        epsilon: Exploration rate at the conclusion of the episode.
        mean_td_error: Average absolute Temporal Difference error during training steps.
        termination_reason: Why the episode ended (e.g., 'wall_collision', 'self_collision', 'step_limit').
    """

    episode: int
    total_reward: float
    score: int
    steps: int
    epsilon: float
    mean_td_error: float
    termination_reason: str


@dataclass
class TrainingHistory:
    """Structured history of metrics across a complete training run.

    Provides utility methods for computing rolling moving averages
    suitable for visualization and UI displays.
    """

    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def total_episodes(self) -> int:
        """Total number of completed episodes recorded."""
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        """Total environment steps across all episodes."""
        return sum(ep.steps for ep in self.episodes)

    @property
    def scores(self) -> List[int]:
        """List of final scores for each episode."""
        return [ep.score for ep in self.episodes]

    @property
    def rewards(self) -> List[float]:
        """List of cumulative rewards for each episode."""
        return [ep.total_reward for ep in self.episodes]

    @property
    def lengths(self) -> List[int]:
        """List of step counts (durations) for each episode."""
        return [ep.steps for ep in self.episodes]

    @property
    def epsilons(self) -> List[float]:
        """List of epsilon values after each episode."""
        return [ep.epsilon for ep in self.episodes]

    def moving_average_scores(self, window: int = 10) -> List[float]:
        """Compute the trailing rolling moving average of scores.

        Args:
            window: Number of episodes to average over (default 10).

        Returns:
            List of moving average values matching the length of episodes.
        """
        return self._compute_moving_average([float(s) for s in self.scores], window)

    def moving_average_rewards(self, window: int = 10) -> List[float]:
        """Compute the trailing rolling moving average of rewards."""
        return self._compute_moving_average(self.rewards, window)

    @staticmethod
    def _compute_moving_average(values: Sequence[float], window: int) -> List[float]:
        if not values:
            return []
        window = max(1, window)
        averages = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            subset = values[start_idx : i + 1]
            averages.append(sum(subset) / len(subset))
        return averages


def train_q_learning(
    env: SnakeEnvironment,
    agent: QLearningAgent,
    num_episodes: int,
    base_seed: Optional[int] = None,
    callback: Optional[Callable[[EpisodeMetrics], None]] = None,
) -> TrainingHistory:
    """Run a bounded training session for a Tabular Q-learning agent.

    Args:
        env: Headless Snake environment.
        agent: QLearningAgent to train.
        num_episodes: Exact number of episodes to train (must be positive).
        base_seed: Optional base seed. If supplied, each episode uses seed (base_seed + episode).
        callback: Optional callback invoked after each episode with EpisodeMetrics.

    Returns:
        TrainingHistory containing structured metrics for all completed episodes.

    Raises:
        ValueError: If num_episodes <= 0.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")

    history = TrainingHistory()

    for ep in range(1, num_episodes + 1):
        # 1. Reset environment with deterministic seed if provided
        ep_seed = (base_seed + ep) if base_seed is not None else None
        obs = env.reset(seed=ep_seed)

        total_reward = 0.0
        td_errors: List[float] = []
        last_reason = "unknown"

        # 2. Run episode steps
        while not env.done:
            action = agent.select_action(obs)
            step_result = env.step(action)

            # Update Q-values using the Bellman equation
            td_error = agent.update(
                state=obs,
                action=action,
                reward=step_result.reward,
                next_state=step_result.observation,
                done=step_result.done,
            )

            td_errors.append(abs(td_error))
            total_reward += step_result.reward
            obs = step_result.observation
            last_reason = step_result.info.get("reason", "unknown")

        # 3. Decay epsilon once per episode
        current_epsilon = agent.decay_epsilon()

        # 4. Record episode metrics
        mean_td = (sum(td_errors) / len(td_errors)) if td_errors else 0.0
        metrics = EpisodeMetrics(
            episode=ep,
            total_reward=round(total_reward, 4),
            score=env.score,
            steps=env.steps,
            epsilon=round(current_epsilon, 6),
            mean_td_error=round(mean_td, 6),
            termination_reason=last_reason,
        )

        history.episodes.append(metrics)

        # 5. Invoke optional per-episode progress callback
        if callback is not None:
            callback(metrics)

    return history
