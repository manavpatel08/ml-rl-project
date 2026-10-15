"""State representation and feature encoding for Snake RL.

This module converts a raw SnakeObservation into an 11-bit compact binary
feature representation suitable for tabular Q-learning and initial DQN inputs.
"""

from typing import Dict, Optional, Sequence, Tuple

from snake_rl.environment import (
    Action,
    Direction,
    DIRECTION_VECTORS,
    Point,
    SnakeObservation,
    get_next_direction,
)

# Canonical ordering and descriptions of the 11 binary state features
FEATURE_NAMES: Tuple[str, ...] = (
    "danger_straight",  # 0: Collision ahead if moving straight
    "danger_right",     # 1: Collision to the right if turning right
    "danger_left",      # 2: Collision to the left if turning left
    "heading_left",     # 3: Snake is currently moving left
    "heading_right",    # 4: Snake is currently moving right
    "heading_up",       # 5: Snake is currently moving up
    "heading_down",     # 6: Snake is currently moving down
    "food_left",        # 7: Food x < Head x
    "food_right",       # 8: Food x > Head x
    "food_up",          # 9: Food y < Head y (above head)
    "food_down",        # 10: Food y > Head y (below head)
)

TOTAL_FEATURES: int = 11
NUM_STATES: int = 2 ** TOTAL_FEATURES  # 2048 distinct states


def is_action_dangerous(observation: SnakeObservation, action: Action) -> int:
    """Determine whether taking a relative action results in an immediate collision.

    A candidate cell is dangerous if it hits a wall or any part of the snake body
    that will remain occupied on that step.

    Tail vacating rule:
      On a normal move (food is not eaten), the tail cell vacates simultaneously.
      Therefore, moving into the current tail cell on a normal move is legal and
      NOT marked as dangerous.
      If the candidate cell contains the food, the candidate cell is by definition
      unoccupied and safe (food never spawns on the snake body or wall).

    Args:
        observation: Current game observation snapshot.
        action: Relative action to evaluate (STRAIGHT, TURN_RIGHT, TURN_LEFT).

    Returns:
        1 if the action causes immediate collision (wall or self), 0 if safe.
    """
    next_dir = get_next_direction(observation.direction, action)
    dx, dy = DIRECTION_VECTORS[next_dir]
    next_coord = Point(observation.head.x + dx, observation.head.y + dy)

    # 1. Wall collision check
    grid_w, grid_h = observation.grid_size
    if not (0 <= next_coord.x < grid_w and 0 <= next_coord.y < grid_h):
        return 1

    # 2. Food cell check: food is always on an unoccupied cell
    if observation.food is not None and next_coord == observation.food:
        return 0

    # 3. Snake body collision check:
    # On a normal move, the tail cell (body[-1]) vacates on this step.
    # The obstacle cells that remain occupied are body[:-1].
    body_obstacles = set(observation.body[:-1])
    if next_coord in body_obstacles:
        return 1

    return 0


def encode_observation(observation: SnakeObservation) -> Tuple[int, ...]:
    """Convert a SnakeObservation into an ordered tuple of 11 binary features.

    Feature order:
      1. Danger straight (0 or 1)
      2. Danger right (0 or 1)
      3. Danger left (0 or 1)
      4. Heading left (0 or 1)
      5. Heading right (0 or 1)
      6. Heading up (0 or 1)
      7. Heading down (0 or 1)
      8. Food left (0 or 1)
      9. Food right (0 or 1)
      10. Food up (above) (0 or 1)
      11. Food down (below) (0 or 1)

    Args:
        observation: Immutable snapshot of the game state.

    Returns:
        Tuple of exactly 11 integers, each either 0 or 1.
    """
    # 1-3. Danger features relative to current heading
    danger_straight = is_action_dangerous(observation, Action.STRAIGHT)
    danger_right = is_action_dangerous(observation, Action.TURN_RIGHT)
    danger_left = is_action_dangerous(observation, Action.TURN_LEFT)

    # 4-7. Current heading (one-hot)
    heading_left = 1 if observation.direction == Direction.LEFT else 0
    heading_right = 1 if observation.direction == Direction.RIGHT else 0
    heading_up = 1 if observation.direction == Direction.UP else 0
    heading_down = 1 if observation.direction == Direction.DOWN else 0

    # 8-11. Food position relative to head
    food = observation.food
    head = observation.head

    if food is None:
        food_left = 0
        food_right = 0
        food_up = 0
        food_down = 0
    else:
        food_left = 1 if food.x < head.x else 0
        food_right = 1 if food.x > head.x else 0
        food_up = 1 if food.y < head.y else 0
        food_down = 1 if food.y > head.y else 0

    return (
        danger_straight,
        danger_right,
        danger_left,
        heading_left,
        heading_right,
        heading_up,
        heading_down,
        food_left,
        food_right,
        food_up,
        food_down,
    )


def encode_features_to_int(features: Sequence[int]) -> int:
    """Convert an 11-element binary feature sequence into a unique integer [0, 2047].

    Encoding convention (Big-Endian):
      Bit 10 (value 1024): features[0]  (danger_straight)
      Bit  9 (value  512): features[1]  (danger_right)
      Bit  8 (value  256): features[2]  (danger_left)
      Bit  7 (value  128): features[3]  (heading_left)
      Bit  6 (value   64): features[4]  (heading_right)
      Bit  5 (value   32): features[5]  (heading_up)
      Bit  4 (value   16): features[6]  (heading_down)
      Bit  3 (value    8): features[7]  (food_left)
      Bit  2 (value    4): features[8]  (food_right)
      Bit  1 (value    2): features[9]  (food_up)
      Bit  0 (value    1): features[10] (food_down)

    Args:
        features: Sequence of length 11 with elements in {0, 1}.

    Returns:
        Integer index in range [0, 2047].

    Raises:
        ValueError: If features length != 11 or any element is not 0 or 1.
    """
    if len(features) != TOTAL_FEATURES:
        raise ValueError(
            f"Expected exactly {TOTAL_FEATURES} features, got {len(features)}."
        )

    encoded = 0
    for i, val in enumerate(features):
        if val not in (0, 1):
            raise ValueError(
                f"Feature at index {i} must be 0 or 1, got {val!r}."
            )
        encoded = (encoded << 1) | val

    return encoded


def decode_int_to_features(state_int: int) -> Tuple[int, ...]:
    """Reconstruct the 11-element binary feature tuple from an integer in [0, 2047].

    Args:
        state_int: State integer in range [0, 2047].

    Returns:
        Tuple of 11 binary values (each 0 or 1).

    Raises:
        TypeError: If state_int is not an integer or is a boolean.
        ValueError: If state_int is outside [0, 2047].
    """
    if not isinstance(state_int, int) or isinstance(state_int, bool):
        raise TypeError(
            f"State index must be an integer, got {type(state_int).__name__}."
        )
    if not (0 <= state_int < NUM_STATES):
        raise ValueError(
            f"State integer must be in range [0, {NUM_STATES - 1}], got {state_int}."
        )

    return tuple((state_int >> (TOTAL_FEATURES - 1 - i)) & 1 for i in range(TOTAL_FEATURES))


def encode_observation_to_int(observation: SnakeObservation) -> int:
    """Convenience function: encode an observation directly to an integer [0, 2047]."""
    return encode_features_to_int(encode_observation(observation))


class StateEncoder:
    """StateEncoder class matching ARCHITECTURE.md specification.

    Encapsulates encoding of observations into 11-bit feature representations
    and integer Q-table indices.
    """

    @staticmethod
    def encode(observation: SnakeObservation) -> Tuple[int, ...]:
        """Convert observation to 11 binary features."""
        return encode_observation(observation)

    @staticmethod
    def to_index(observation: SnakeObservation) -> int:
        """Convert observation directly to an integer index in [0, 2047]."""
        return encode_observation_to_int(observation)

    @staticmethod
    def features_to_index(features: Sequence[int]) -> int:
        """Convert 11 binary features to integer index in [0, 2047]."""
        return encode_features_to_int(features)

    @staticmethod
    def index_to_features(index: int) -> Tuple[int, ...]:
        """Convert integer index in [0, 2047] back to 11 binary features."""
        return decode_int_to_features(index)
