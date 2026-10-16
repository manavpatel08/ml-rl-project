"""Baseline agents for Snake benchmark comparisons.

Provides lightweight reference policies:
  - RandomAgent: Selects among the 3 relative actions uniformly.
  - HeuristicAgent: Uses local 1-step danger checks and Manhattan distance to food.
"""

import random
from typing import Optional, Sequence, Union

from snake_rl.environment import (
    Action,
    Direction,
    DIRECTION_VECTORS,
    Point,
    SnakeObservation,
    get_next_direction,
)
from snake_rl.state import is_action_dangerous

ALL_ACTIONS: Sequence[Action] = (Action.STRAIGHT, Action.TURN_RIGHT, Action.TURN_LEFT)


class RandomAgent:
    """Baseline agent that selects actions uniformly at random."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the RandomAgent.

        Args:
            seed: Optional random seed for reproducible random choices.
        """
        self._rng = random.Random(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        """Re-seed the random generator."""
        self._rng = random.Random(seed)

    def select_action(self, observation: SnakeObservation) -> Action:
        """Choose one of the 3 relative actions uniformly at random."""
        return self._rng.choice(ALL_ACTIONS)

    def select_greedy_action(self, observation: SnakeObservation) -> Action:
        """For compatibility with evaluation loops: selects an action uniformly at random."""
        return self.select_action(observation)


class HeuristicAgent:
    """Rule-based heuristic agent.

    Strategy:
      1. Evaluates all 3 relative actions to filter out immediately dangerous moves
         (those that cause wall or self-collision on the next step).
      2. If one or more safe actions exist, chooses the safe action whose resulting
         head position minimizes Manhattan distance to the food.
      3. If all actions are dangerous, chooses STRAIGHT as a default fallback.

    Known Limitations:
      - 1-step greedy horizon: Does not look ahead multiple steps.
      - Prone to self-trapping: Cannot detect dead ends, coils, or pockets where
        the snake will inevitably run out of space after several steps.
      - Manhattan distance ignores obstacles: It may guide the snake straight into
        a trapped pocket if the food is just beyond a body wall.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the HeuristicAgent.

        Args:
            seed: Optional seed for breaking ties among equally optimal moves.
        """
        self._rng = random.Random(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        """Re-seed the tie-breaking generator."""
        self._rng = random.Random(seed)

    def select_action(self, observation: SnakeObservation) -> Action:
        """Select the best safe action that minimizes Manhattan distance to food."""
        safe_actions = [
            action
            for action in ALL_ACTIONS
            if is_action_dangerous(observation, action) == 0
        ]

        # If trapped with no safe moves, default to STRAIGHT
        if not safe_actions:
            return Action.STRAIGHT

        # If food is absent (e.g. board full), pick any safe action
        if observation.food is None:
            return safe_actions[0]

        food = observation.food
        head = observation.head

        # Evaluate Manhattan distance for each safe action
        def action_distance(act: Action) -> int:
            cand_dir = get_next_direction(observation.direction, act)
            dx, dy = DIRECTION_VECTORS[cand_dir]
            next_x = head.x + dx
            next_y = head.y + dy
            return abs(next_x - food.x) + abs(next_y - food.y)

        best_distance = min(action_distance(act) for act in safe_actions)
        best_actions = [
            act for act in safe_actions if action_distance(act) == best_distance
        ]

        if len(best_actions) == 1:
            return best_actions[0]

        return self._rng.choice(best_actions)

    def select_greedy_action(self, observation: SnakeObservation) -> Action:
        """For compatibility with evaluation loops: selects the heuristic action."""
        return self.select_action(observation)
