"""Evaluation pipeline for benchmark comparisons and policy assessment.

Evaluates trained agents and baseline policies without updating weights or
decaying exploration parameters, using fixed reproducible seeds.
"""

from dataclasses import dataclass
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from snake_rl.environment import Action, SnakeEnvironment, SnakeObservation


@dataclass(frozen=True)
class EvaluationResult:
    """Summary statistics for an agent across an evaluation session.

    Attributes:
        agent_name: Descriptive name or label of the evaluated agent.
        num_episodes: Number of evaluation episodes completed.
        mean_score: Average food items consumed per episode.
        std_score: Standard deviation of scores across episodes.
        max_score: Maximum score achieved in any single episode.
        min_score: Minimum score in any single episode.
        mean_reward: Average cumulative reward per episode.
        mean_steps: Average number of steps per episode.
        scores: Tuple of final scores for every evaluated episode.
        rewards: Tuple of cumulative rewards for every evaluated episode.
        steps: Tuple of step counts for every evaluated episode.
        termination_reasons: Dictionary mapping termination reasons to counts.
    """

    agent_name: str
    num_episodes: int
    mean_score: float
    std_score: float
    max_score: int
    min_score: int
    mean_reward: float
    mean_steps: float
    scores: Tuple[int, ...]
    rewards: Tuple[float, ...]
    steps: Tuple[int, ...]
    termination_reasons: Dict[str, int]


def evaluate_agent(
    env: SnakeEnvironment,
    agent: Any,
    num_episodes: int = 20,
    base_seed: int = 1000,
    agent_name: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate an agent's policy under pure exploitation across fixed seeds.

    Guarantees:
      - For QLearningAgent, actions are chosen greedily (select_greedy_action).
      - No model weights / Q-values are updated.
      - Epsilon is NOT decayed or modified.
      - Each episode uses a deterministic seed (base_seed + i).

    Args:
        env: Headless Snake environment.
        agent: Agent instance (QLearningAgent, RandomAgent, HeuristicAgent, etc.).
        num_episodes: Number of test episodes to run (must be > 0).
        base_seed: Starting seed for reproducible evaluation.
        agent_name: Optional display name for the agent.

    Returns:
        EvaluationResult containing aggregate metrics and per-episode outcomes.

    Raises:
        ValueError: If num_episodes <= 0.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")

    name = agent_name or agent.__class__.__name__

    scores: List[int] = []
    rewards: List[float] = []
    steps: List[int] = []
    reasons: Dict[str, int] = {}

    for i in range(num_episodes):
        ep_seed = base_seed + i
        obs = env.reset(seed=ep_seed)

        total_reward = 0.0
        last_reason = "unknown"

        while not env.done:
            # Prefer select_greedy_action if available (e.g. QLearningAgent)
            # Otherwise use select_action (e.g. RandomAgent, HeuristicAgent)
            if hasattr(agent, "select_greedy_action"):
                action = agent.select_greedy_action(obs)
            else:
                action = agent.select_action(obs)

            step_result = env.step(action)
            total_reward += step_result.reward
            obs = step_result.observation
            last_reason = step_result.info.get("reason", "unknown")

        scores.append(env.score)
        rewards.append(round(total_reward, 4))
        steps.append(env.steps)
        reasons[last_reason] = reasons.get(last_reason, 0) + 1

    mean_score = sum(scores) / len(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mean_reward = sum(rewards) / len(rewards)
    mean_steps = sum(steps) / len(steps)

    return EvaluationResult(
        agent_name=name,
        num_episodes=num_episodes,
        mean_score=round(mean_score, 3),
        std_score=round(std_score, 3),
        max_score=max(scores),
        min_score=min(scores),
        mean_reward=round(mean_reward, 3),
        mean_steps=round(mean_steps, 2),
        scores=tuple(scores),
        rewards=tuple(rewards),
        steps=tuple(steps),
        termination_reasons=reasons,
    )


def compare_agents(
    env: SnakeEnvironment,
    agents: Dict[str, Any],
    num_episodes: int = 20,
    base_seed: int = 1000,
) -> Dict[str, EvaluationResult]:
    """Fairly evaluate multiple agents across the exact same evaluation seeds.

    Args:
        env: Headless Snake environment.
        agents: Mapping of display name to agent instance.
        num_episodes: Number of evaluation episodes for each agent.
        base_seed: Common base seed ensuring identical board and food layouts.

    Returns:
        Mapping of agent name to its EvaluationResult.
    """
    results: Dict[str, EvaluationResult] = {}
    for name, agent in agents.items():
        results[name] = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            base_seed=base_seed,
            agent_name=name,
        )
    return results
