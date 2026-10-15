"""Tabular Q-learning agent for Snake.

This module implements tabular Q-learning with an epsilon-greedy exploration
schedule, Bellman update arithmetic, and terminal state handling.
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from snake_rl.config import QLearningConfig
from snake_rl.environment import Action, SnakeObservation
from snake_rl.state import NUM_STATES, StateEncoder, encode_observation_to_int

NUM_ACTIONS: int = len(Action)  # 3 actions: STRAIGHT, TURN_RIGHT, TURN_LEFT


class QLearningAgent:
    """Tabular Q-learning agent.

    Maintains a Q-table of size (NUM_STATES, NUM_ACTIONS) = (2048, 3) where:
      - Rows represent the 11-bit encoded game state indices (0 through 2047).
      - Columns represent the three relative actions:
          Column 0: Action.STRAIGHT
          Column 1: Action.TURN_RIGHT
          Column 2: Action.TURN_LEFT
    """

    def __init__(
        self,
        config: Optional[QLearningConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the Q-learning agent.

        Args:
            config: Agent hyperparameters (alpha, gamma, epsilon schedule).
            seed: Optional random seed for reproducible exploration and tie-breaking.
        """
        self.config = config if config is not None else QLearningConfig()
        self._rng = random.Random(seed)
        self.epsilon: float = self.config.epsilon_start

        # Initialize Q-table with zeros: shape (2048, 3)
        self._q_table: List[List[float]] = [
            [0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)
        ]

    @property
    def q_table(self) -> List[List[float]]:
        """Access the underlying Q-table as a 2D list of shape (2048, 3)."""
        return self._q_table

    def seed(self, seed: Optional[int] = None) -> None:
        """Re-seed the agent's internal random number generator."""
        self._rng = random.Random(seed)

    def select_action(self, state: Union[int, SnakeObservation]) -> Action:
        """Select an action using epsilon-greedy exploration.

        With probability epsilon, chooses a random action uniformly.
        With probability (1 - epsilon), chooses the greedy action with highest Q-value.

        Args:
            state: Either an integer state index [0, 2047] or a SnakeObservation.

        Returns:
            The chosen Action.
        """
        if self._rng.random() < self.epsilon:
            random_action_int = self._rng.randint(0, NUM_ACTIONS - 1)
            return Action(random_action_int)

        return self.select_greedy_action(state)

    def select_greedy_action(self, state: Union[int, SnakeObservation]) -> Action:
        """Select the greedy action (argmax Q) without exploration.

        If multiple actions have the identical maximum Q-value, breaks ties
        deterministically using the agent's seeded random generator.

        Args:
            state: Either an integer state index [0, 2047] or a SnakeObservation.

        Returns:
            The greedy Action.
        """
        state_idx = self._resolve_state_index(state)
        q_values = self._q_table[state_idx]
        max_q = max(q_values)

        best_actions = [Action(a) for a, q in enumerate(q_values) if q == max_q]
        if len(best_actions) == 1:
            return best_actions[0]

        return self._rng.choice(best_actions)

    def update(
        self,
        state: Union[int, SnakeObservation],
        action: Union[int, Action],
        reward: float,
        next_state: Union[int, SnakeObservation],
        done: bool,
    ) -> float:
        """Perform a single tabular Bellman Q-learning update.

        Update formula:
          If done (terminal transition):
            target = reward
          Else:
            target = reward + gamma * max_a' Q(next_state, a')

          td_error = target - Q(state, action)
          Q(state, action) <- Q(state, action) + alpha * td_error

        Args:
            state: Starting state (index or observation).
            action: Action taken (Action enum or index).
            reward: Scalar reward received.
            next_state: Resulting next state (index or observation).
            done: Whether the episode terminated on this transition.

        Returns:
            The calculated temporal difference error (td_error).

        Raises:
            ValueError: If reward is non-finite or state/action indices are invalid.
            TypeError: If input types are incorrect.
        """
        if not isinstance(reward, (int, float)) or math.isnan(reward) or math.isinf(reward):
            raise ValueError(f"Reward must be a finite real number, got {reward!r}")

        s = self._resolve_state_index(state)
        a = self._resolve_action(action)

        current_q = self._q_table[s][a]

        if done:
            target = float(reward)
        else:
            s_next = self._resolve_state_index(next_state)
            max_next_q = max(self._q_table[s_next])
            target = float(reward) + self.config.gamma * max_next_q

        td_error = target - current_q
        self._q_table[s][a] = current_q + self.config.alpha * td_error

        return td_error

    def decay_epsilon(self) -> float:
        """Decay epsilon by the configured decay factor, clamped to epsilon_min.

        Should be called once per episode at episode completion.

        Returns:
            The new epsilon value.
        """
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return self.epsilon

    def reset_epsilon(self) -> None:
        """Reset epsilon back to the configured epsilon_start."""
        self.epsilon = self.config.epsilon_start

    def get_q_value(
        self,
        state: Union[int, SnakeObservation],
        action: Union[int, Action],
    ) -> float:
        """Get the Q-value for a specific (state, action) pair."""
        s = self._resolve_state_index(state)
        a = self._resolve_action(action)
        return self._q_table[s][a]

    def set_q_value(
        self,
        state: Union[int, SnakeObservation],
        action: Union[int, Action],
        value: float,
    ) -> None:
        """Set the Q-value for a specific (state, action) pair."""
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            raise ValueError(f"Q-value must be a finite real number, got {value!r}")
        s = self._resolve_state_index(state)
        a = self._resolve_action(action)
        self._q_table[s][a] = float(value)

    def get_q_values(
        self,
        state: Union[int, SnakeObservation],
    ) -> Tuple[float, ...]:
        """Get all Q-values for a given state across all 3 actions."""
        s = self._resolve_state_index(state)
        return tuple(self._q_table[s])

    def _resolve_state_index(self, state: Union[int, SnakeObservation]) -> int:
        """Validate and resolve state input into an integer index [0, 2047]."""
        if isinstance(state, SnakeObservation):
            return encode_observation_to_int(state)

        if isinstance(state, int) and not isinstance(state, bool):
            if not (0 <= state < NUM_STATES):
                raise ValueError(
                    f"State index out of range: must be in [0, {NUM_STATES - 1}], got {state}"
                )
            return state

        raise TypeError(
            f"State must be an int in [0, {NUM_STATES - 1}] or a SnakeObservation, "
            f"got {type(state).__name__}"
        )

    def _resolve_action(self, action: Union[int, Action]) -> int:
        """Validate and resolve action input into an action integer [0, 2]."""
        if isinstance(action, Action):
            return action.value

        if isinstance(action, int) and not isinstance(action, bool):
            if not (0 <= action < NUM_ACTIONS):
                raise ValueError(
                    f"Action out of range: must be in [0, {NUM_ACTIONS - 1}], got {action}"
                )
            return action

        raise TypeError(
            f"Action must be an Action enum or int in [0, {NUM_ACTIONS - 1}], "
            f"got {type(action).__name__}"
        )
