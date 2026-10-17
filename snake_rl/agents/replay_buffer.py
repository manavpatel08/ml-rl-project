"""Experience Replay Buffer for Deep Q-Networks (DQN).

Provides a fixed-capacity FIFO circular buffer that stores environment transitions
and samples random minibatches as PyTorch tensors for gradient updates.
"""

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from snake_rl.environment import Action, SnakeObservation
from snake_rl.state import TOTAL_FEATURES, encode_observation


class Transition(NamedTuple):
    """Single transition tuple stored in the replay buffer."""

    state: Tuple[float, ...]
    action: int
    reward: float
    next_state: Tuple[float, ...]
    done: bool


@dataclass(frozen=True)
class ReplayBatch:
    """Minibatch of sampled transitions converted to PyTorch tensors.

    Attributes:
        states: Float32 tensor of shape (batch_size, input_dim).
        actions: Int64 tensor of shape (batch_size, 1).
        rewards: Float32 tensor of shape (batch_size, 1).
        next_states: Float32 tensor of shape (batch_size, input_dim).
        dones: Float32 tensor of shape (batch_size, 1) where 1.0 represents terminal.
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity experience replay buffer with FIFO eviction."""

    def __init__(
        self,
        capacity: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the ReplayBuffer.

        Args:
            capacity: Maximum number of transitions stored before older ones are evicted.
            seed: Optional seed for reproducible minibatch sampling.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.capacity: int = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng: random.Random = random.Random(seed)

    def __len__(self) -> int:
        """Current number of transitions in the buffer."""
        return len(self._buffer)

    def seed(self, seed: Optional[int] = None) -> None:
        """Re-seed the random sampling generator."""
        self._rng = random.Random(seed)

    def push(
        self,
        state: Union[Sequence[float], SnakeObservation],
        action: Union[Action, int],
        reward: float,
        next_state: Union[Sequence[float], SnakeObservation],
        done: bool,
    ) -> None:
        """Add a validated transition to the buffer.

        Automatically evicts the oldest transition if capacity is exceeded.

        Args:
            state: Current 11-feature state or SnakeObservation.
            action: Action taken (Action enum or int 0..2).
            reward: Scalar reward received.
            next_state: Next 11-feature state or SnakeObservation.
            done: Whether the episode terminated on this transition.

        Raises:
            ValueError: If inputs have incorrect dimensions or non-finite values.
            TypeError: If input types are invalid.
        """
        resolved_s = self._validate_state(state, "state")
        resolved_a = self._validate_action(action)
        resolved_r = self._validate_reward(reward)
        resolved_s_next = self._validate_state(next_state, "next_state")
        resolved_done = bool(done)

        transition = Transition(
            state=resolved_s,
            action=resolved_a,
            reward=resolved_r,
            next_state=resolved_s_next,
            done=resolved_done,
        )
        self._buffer.append(transition)

    def can_sample(self, batch_size: int) -> bool:
        """Check whether the buffer contains at least batch_size transitions."""
        return len(self._buffer) >= batch_size

    def sample(
        self,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
    ) -> ReplayBatch:
        """Sample a random minibatch of transitions without replacement.

        Args:
            batch_size: Number of transitions to sample.
            device: Target PyTorch device ('cpu' by default).

        Returns:
            ReplayBatch with tensors placed on the specified device.

        Raises:
            ValueError: If batch_size <= 0 or len(self) < batch_size.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Cannot sample batch of size {batch_size} from buffer with {len(self._buffer)} transitions."
            )

        indices = self._rng.sample(range(len(self._buffer)), batch_size)
        sampled = [self._buffer[i] for i in indices]

        states = torch.tensor(
            [t.state for t in sampled],
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [[t.action] for t in sampled],
            dtype=torch.int64,
            device=device,
        )
        rewards = torch.tensor(
            [[t.reward] for t in sampled],
            dtype=torch.float32,
            device=device,
        )
        next_states = torch.tensor(
            [t.next_state for t in sampled],
            dtype=torch.float32,
            device=device,
        )
        dones = torch.tensor(
            [[1.0 if t.done else 0.0] for t in sampled],
            dtype=torch.float32,
            device=device,
        )

        return ReplayBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

    def clear(self) -> None:
        """Empty the replay buffer."""
        self._buffer.clear()

    @staticmethod
    def _validate_state(
        state: Union[Sequence[float], SnakeObservation],
        field_name: str,
    ) -> Tuple[float, ...]:
        if isinstance(state, SnakeObservation):
            raw_features = encode_observation(state)
            return tuple(float(f) for f in raw_features)

        if isinstance(state, (list, tuple)):
            if len(state) != TOTAL_FEATURES:
                raise ValueError(
                    f"{field_name} must have exactly {TOTAL_FEATURES} features, got {len(state)}"
                )
            for i, val in enumerate(state):
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"Non-finite value at {field_name}[{i}]: {val!r}"
                    )
            return tuple(float(v) for v in state)

        raise TypeError(
            f"{field_name} must be a sequence of {TOTAL_FEATURES} numbers or SnakeObservation, "
            f"got {type(state).__name__}"
        )

    @staticmethod
    def _validate_action(action: Union[Action, int]) -> int:
        if isinstance(action, Action):
            return action.value
        if isinstance(action, int) and not isinstance(action, bool):
            if not (0 <= action < 3):
                raise ValueError(f"action must be in range [0, 2], got {action}")
            return action
        raise TypeError(f"action must be Action enum or int, got {type(action).__name__}")

    @staticmethod
    def _validate_reward(reward: float) -> float:
        if not isinstance(reward, (int, float)) or math.isnan(reward) or math.isinf(reward):
            raise ValueError(f"reward must be a finite real number, got {reward!r}")
        return float(reward)
