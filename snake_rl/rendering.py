"""Rendering utilities for the Snake environment.

Transforms raw SnakeObservations into text grids, formatted frames, and
terminal visual representations independently of external GUI/display servers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from snake_rl.environment import Direction, Point, SnakeObservation

# Standard ANSI escape codes for optional terminal styling
COLOR_RESET = "\033[0m"
COLOR_HEAD = "\033[1;92m"       # Bold bright green
COLOR_BODY = "\033[32m"         # Green
COLOR_FOOD = "\033[1;91m"       # Bold bright red
COLOR_BORDER = "\033[36m"       # Cyan
COLOR_EMPTY = "\033[90m"        # Dim gray
COLOR_EXPLORE = "\033[1;93m"     # Bold yellow for exploration
COLOR_EXPLOIT = "\033[1;94m"     # Bold blue for exploitation
COLOR_HEADER = "\033[1;97m"      # Bold white


@dataclass(frozen=True)
class RenderedFrame:
    """Snapshot of a rendered visual game frame with accompanying metadata."""

    grid_text: str
    score: int
    steps: int
    direction: Direction
    head: Point
    food: Optional[Point]
    status: str


def render_grid(
    observation: SnakeObservation,
    char_head: str = "H",
    char_body: str = "o",
    char_food: str = "★",
    char_empty: str = "·",
) -> List[List[str]]:
    """Render a 2D matrix of characters representing each cell of the grid.

    Args:
        observation: Current game observation snapshot.
        char_head: Character symbol for the snake head.
        char_body: Character symbol for snake body segments.
        char_food: Character symbol for food.
        char_empty: Character symbol for empty cells.

    Returns:
        2D list of strings of shape (grid_height, grid_width).
    """
    width, height = observation.grid_size
    grid = [[char_empty for _ in range(width)] for _ in range(height)]

    # Draw body segments (from neck to tail)
    for segment in observation.body[1:]:
        if 0 <= segment.x < width and 0 <= segment.y < height:
            grid[segment.y][segment.x] = char_body

    # Draw head
    if 0 <= observation.head.x < width and 0 <= observation.head.y < height:
        grid[observation.head.y][observation.head.x] = char_head

    # Draw food
    if observation.food is not None:
        fx, fy = observation.food.x, observation.food.y
        if 0 <= fx < width and 0 <= fy < height:
            grid[fy][fx] = char_food

    return grid


def render_ascii(
    observation: SnakeObservation,
    use_colors: bool = False,
    char_head: str = "H",
    char_body: str = "o",
    char_food: str = "★",
    char_empty: str = "·",
) -> str:
    """Render the game board surrounded by box-drawing border characters.

    Args:
        observation: Current game observation snapshot.
        use_colors: If True, wraps elements with ANSI color escape codes.
        char_head: Head symbol.
        char_body: Body symbol.
        char_food: Food symbol.
        char_empty: Empty cell symbol.

    Returns:
        Formatted multi-line string of the board.
    """
    width, height = observation.grid_size
    grid = render_grid(
        observation,
        char_head=char_head,
        char_body=char_body,
        char_food=char_food,
        char_empty=char_empty,
    )

    border_color = COLOR_BORDER if use_colors else ""
    reset_color = COLOR_RESET if use_colors else ""

    top_border = f"{border_color}┌{'──' * width}┐{reset_color}"
    bottom_border = f"{border_color}└{'──' * width}┘{reset_color}"

    formatted_rows = []
    for row in grid:
        colored_cells = []
        for cell in row:
            if not use_colors:
                colored_cells.append(f"{cell} ")
            elif cell == char_head:
                colored_cells.append(f"{COLOR_HEAD}{cell}{COLOR_RESET} ")
            elif cell == char_body:
                colored_cells.append(f"{COLOR_BODY}{cell}{COLOR_RESET} ")
            elif cell == char_food:
                colored_cells.append(f"{COLOR_FOOD}{cell}{COLOR_RESET} ")
            else:
                colored_cells.append(f"{COLOR_EMPTY}{cell}{COLOR_RESET} ")

        formatted_rows.append(f"{border_color}│{reset_color}{''.join(colored_cells)}{border_color}│{reset_color}")

    return "\n".join([top_border] + formatted_rows + [bottom_border])


def format_hud(
    episode: int,
    total_episodes: int,
    score: int,
    steps: int,
    epsilon: float,
    reward: float,
    rolling_avg: float,
    is_exploring: Optional[bool] = None,
    mode: str = "TRAINING",
    action_name: str = "",
    use_colors: bool = False,
    speed_ms: int = 50,
    is_paused: bool = False,
) -> str:
    """Generate a clean, structured Heads-Up Display (HUD) string.

    Args:
        episode: Current episode index (1-based).
        total_episodes: Total planned episodes.
        score: Food items eaten in the current episode.
        steps: Step count in the current episode.
        epsilon: Current exploration rate.
        reward: Total cumulative reward in the current episode.
        rolling_avg: Trailing rolling average score.
        is_exploring: True if last action was exploratory (random), False if greedy.
        mode: Display mode ('TRAINING' or 'EVALUATION').
        action_name: Name of the last action taken.
        use_colors: If True, uses ANSI color escape sequences.
        speed_ms: Playback delay in milliseconds.
        is_paused: Whether playback is paused.

    Returns:
        Formatted multi-line HUD string.
    """
    reset = COLOR_RESET if use_colors else ""
    header_color = COLOR_HEADER if use_colors else ""

    # Mode and status indicator
    status_str = "PAUSED (Press Space/Enter to resume)" if is_paused else "RUNNING"
    title_line = f"{header_color}═══ Snake AI Visualizer [{mode}] — {status_str} ═══{reset}"

    # Exploration vs Exploitation badge
    if is_exploring is None:
        policy_badge = "GREEDY (Exploitation)"
    elif is_exploring:
        exp_color = COLOR_EXPLORE if use_colors else ""
        policy_badge = f"{exp_color}🎲 EXPLORING (Random Action){reset}"
    else:
        exp_color = COLOR_EXPLOIT if use_colors else ""
        policy_badge = f"{exp_color}🧠 EXPLOITING (Greedy Q-Value){reset}"

    # Progress bar calculation
    pct = (episode / total_episodes) if total_episodes > 0 else 1.0
    bar_len = 20
    filled = int(round(bar_len * pct))
    progress_bar = f"[{'=' * filled}{' ' * (bar_len - filled)}] {int(pct * 100)}%"

    hud_lines = [
        title_line,
        f"Episode: {episode}/{total_episodes} {progress_bar} | Speed: {speed_ms}ms",
        f"Score: {score:<3} | Steps: {steps:<4} | Reward: {reward:+6.1f} | Rolling Avg Score: {rolling_avg:5.2f}",
        f"Epsilon: {epsilon:.4f} | Action: {action_name:<10} | Decision: {policy_badge}",
    ]

    return "\n".join(hud_lines)
