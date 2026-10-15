"""Headless Snake environment for Reinforcement Learning.

This module implements the core game logic of Snake independently of any
rendering engine, UI framework, or RL training loop.
"""

from dataclasses import dataclass
from enum import IntEnum
import random
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from snake_rl.config import EnvironmentConfig


class Point(NamedTuple):
    """Grid coordinate representation (x, y).

    x: horizontal index (0 <= x < grid_width), increasing to the right.
    y: vertical index (0 <= y < grid_height), increasing downwards.
    """

    x: int
    y: int


class Direction(IntEnum):
    """Absolute directions on the grid ordered clockwise."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# Coordinate deltas for each direction: (dx, dy)
DIRECTION_VECTORS: Dict[Direction, Tuple[int, int]] = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}


class Action(IntEnum):
    """Relative movement actions for the Snake agent.

    Actions are relative to the snake's current heading to prevent
    instantaneous 180-degree reversals.
    """

    STRAIGHT = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2


def get_next_direction(current: Direction, action: Action) -> Direction:
    """Compute the new absolute direction given current heading and relative action.

    Args:
        current: Current heading of the snake.
        action: Relative movement action (STRAIGHT, TURN_RIGHT, TURN_LEFT).

    Returns:
        New absolute Direction.
    """
    if not isinstance(action, Action):
        try:
            action = Action(action)
        except ValueError:
            raise ValueError(
                f"Invalid action '{action}'. Must be Action.STRAIGHT (0), "
                "Action.TURN_RIGHT (1), or Action.TURN_LEFT (2)."
            )

    if action == Action.STRAIGHT:
        return current
    if action == Action.TURN_RIGHT:
        return Direction((current.value + 1) % 4)
    if action == Action.TURN_LEFT:
        return Direction((current.value - 1) % 4)
    raise ValueError(f"Unknown action: {action}")


@dataclass(frozen=True)
class SnakeObservation:
    """Immutable snapshot of the game state returned to callers.

    Attributes:
        head: Coordinate of the snake's head.
        body: Ordered sequence of coordinates from head to tail.
        food: Coordinate of the food, or None if board is full.
        direction: Current movement direction.
        grid_size: (width, height) of the board.
        steps: Steps taken in the current episode.
        score: Number of food items consumed.
    """

    head: Point
    body: Tuple[Point, ...]
    food: Optional[Point]
    direction: Direction
    grid_size: Tuple[int, int]
    steps: int
    score: int


@dataclass(frozen=True)
class StepResult:
    """Result of taking a single action in the environment.

    Attributes:
        observation: New state snapshot after the action.
        reward: Scalar feedback signal for this step.
        done: Whether the episode has terminated.
        score: Current score (total food eaten).
        info: Diagnostic metadata (e.g., termination reason).
    """

    observation: SnakeObservation
    reward: float
    done: bool
    score: int
    info: Dict[str, Any]


class SnakeEnvironment:
    """Headless 2D Snake game environment.

    Game Rules:
      - The snake advances by 1 grid cell per step.
      - Relative actions (STRAIGHT, TURN_RIGHT, TURN_LEFT) update the heading.
      - Eating food increases score by 1 and extends snake length by 1.
        The tail does not move on the step when food is eaten.
      - Normal movement advances the head and vacates the tail cell.
      - Moving into the tail cell is legal if the tail vacates on that step.
      - Moving out of bounds (wall) or into snake body (self) terminates the game.
      - A configurable step limit terminates endless loops.
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the Snake environment.

        Args:
            config: Environment configuration. Defaults to 20x20 with standard rewards.
            seed: Optional integer seed for reproducible random number generation.
        """
        self.config = config if config is not None else EnvironmentConfig()
        self._rng = random.Random(seed)

        self._body: List[Point] = []
        self._direction: Direction = Direction.RIGHT
        self._food: Optional[Point] = None
        self._score: int = 0
        self._steps: int = 0
        self._done: bool = False

        self.reset(seed=seed)

    @property
    def head(self) -> Point:
        """Current coordinate of the snake's head."""
        return self._body[0]

    @property
    def body(self) -> List[Point]:
        """Copy of the current snake body coordinates from head to tail."""
        return list(self._body)

    @property
    def food(self) -> Optional[Point]:
        """Current coordinate of the food."""
        return self._food

    @property
    def direction(self) -> Direction:
        """Current absolute movement direction."""
        return self._direction

    @property
    def score(self) -> int:
        """Total food items eaten in the current episode."""
        return self._score

    @property
    def steps(self) -> int:
        """Total steps taken in the current episode."""
        return self._steps

    @property
    def done(self) -> bool:
        """Whether the current episode has terminated."""
        return self._done

    def reset(self, seed: Optional[int] = None) -> SnakeObservation:
        """Reset the environment to the initial state.

        Args:
            seed: Optional seed to re-seed the internal random number generator.

        Returns:
            Initial SnakeObservation.
        """
        if seed is not None:
            self._rng = random.Random(seed)

        self._score = 0
        self._steps = 0
        self._done = False
        self._direction = Direction.RIGHT

        # Place initial snake horizontally near the center facing RIGHT.
        center_x = self.config.grid_width // 2
        center_y = self.config.grid_height // 2

        length = max(1, self.config.initial_snake_length)
        self._body = [
            Point(center_x - i, center_y)
            for i in range(length)
        ]

        self._food = self._place_food()
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        """Advance the environment by one step given a relative action.

        Args:
            action: Relative action (STRAIGHT, TURN_RIGHT, TURN_LEFT).

        Returns:
            StepResult containing observation, reward, done flag, score, and info dict.

        Raises:
            RuntimeError: If called after the episode has terminated.
            ValueError: If an invalid action is provided.
        """
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() before calling step()."
            )

        if not isinstance(action, Action):
            try:
                action = Action(action)
            except ValueError:
                raise ValueError(
                    f"Invalid action '{action}'. Must be Action.STRAIGHT (0), "
                    "Action.TURN_RIGHT (1), or Action.TURN_LEFT (2)."
                )

        # 1. Update direction based on relative action
        self._direction = self._get_next_direction(self._direction, action)
        dx, dy = DIRECTION_VECTORS[self._direction]
        new_head = Point(self.head.x + dx, self.head.y + dy)
        self._steps += 1

        # 2. Check wall collision
        if not (0 <= new_head.x < self.config.grid_width and 0 <= new_head.y < self.config.grid_height):
            self._done = True
            return StepResult(
                observation=self._get_observation(),
                reward=self.config.reward_collision,
                done=True,
                score=self._score,
                info={"reason": "wall_collision"},
            )

        # 3. Check food consumption
        ate_food = (self._food is not None and new_head == self._food)

        if ate_food:
            self._score += 1
            # When eating food, the snake grows: new head is added, tail remains
            self._body.insert(0, new_head)
            self._food = self._place_food()

            # Check if board became full after eating
            if self._food is None:
                self._done = True
                return StepResult(
                    observation=self._get_observation(),
                    reward=self.config.reward_food,
                    done=True,
                    score=self._score,
                    info={"reason": "board_full", "won": True, "ate_food": True},
                )

            # Check step limit
            if self.config.max_steps is not None and self._steps >= self.config.max_steps:
                self._done = True
                return StepResult(
                    observation=self._get_observation(),
                    reward=self.config.reward_food,
                    done=True,
                    score=self._score,
                    info={"reason": "step_limit", "ate_food": True},
                )

            return StepResult(
                observation=self._get_observation(),
                reward=self.config.reward_food,
                done=False,
                score=self._score,
                info={"reason": "running", "ate_food": True},
            )

        # 4. Normal move (not eating food):
        # The tail cell will vacate this step. Any collision must be with body segments
        # that remain occupied, i.e., body[:-1].
        body_obstacles = set(self._body[:-1])
        if new_head in body_obstacles:
            self._done = True
            return StepResult(
                observation=self._get_observation(),
                reward=self.config.reward_collision,
                done=True,
                score=self._score,
                info={"reason": "self_collision"},
            )

        # Move snake forward: add new head and remove old tail
        self._body.insert(0, new_head)
        self._body.pop()

        # 5. Check step limit
        if self.config.max_steps is not None and self._steps >= self.config.max_steps:
            self._done = True
            return StepResult(
                observation=self._get_observation(),
                reward=self.config.reward_step,
                done=True,
                score=self._score,
                info={"reason": "step_limit", "ate_food": False},
            )

        return StepResult(
            observation=self._get_observation(),
            reward=self.config.reward_step,
            done=False,
            score=self._score,
            info={"reason": "running", "ate_food": False},
        )

    def _get_next_direction(self, current: Direction, action: Action) -> Direction:
        """Compute the new absolute direction given current heading and relative action."""
        return get_next_direction(current, action)

    def _place_food(self) -> Optional[Point]:
        """Place food uniformly at random on an unoccupied cell.

        Returns:
            A Point on an empty cell, or None if the board is completely filled.
        """
        occupied = set(self._body)
        total_cells = self.config.grid_width * self.config.grid_height

        if len(occupied) >= total_cells:
            return None

        # Rejection sampling is fast when board has plenty of free space (< 75% full)
        if len(occupied) < total_cells * 0.75:
            for _ in range(50):
                x = self._rng.randint(0, self.config.grid_width - 1)
                y = self._rng.randint(0, self.config.grid_height - 1)
                candidate = Point(x, y)
                if candidate not in occupied:
                    return candidate

        # Guaranteed deterministic collection of unoccupied cells when crowded
        unoccupied = [
            Point(x, y)
            for y in range(self.config.grid_height)
            for x in range(self.config.grid_width)
            if Point(x, y) not in occupied
        ]
        if not unoccupied:
            return None

        return self._rng.choice(unoccupied)

    def _get_observation(self) -> SnakeObservation:
        """Create an immutable snapshot of the current environment state."""
        return SnakeObservation(
            head=self.head,
            body=tuple(self._body),
            food=self._food,
            direction=self._direction,
            grid_size=(self.config.grid_width, self.config.grid_height),
            steps=self._steps,
            score=self._score,
        )

    def _set_state_for_testing(
        self,
        body: Sequence[Point],
        direction: Direction,
        food: Optional[Point] = None,
        score: int = 0,
        steps: int = 0,
        done: bool = False,
    ) -> None:
        """Helper to inject a specific game state directly for deterministic unit tests."""
        self._body = list(body)
        self._direction = direction
        self._food = food
        self._score = score
        self._steps = steps
        self._done = done
