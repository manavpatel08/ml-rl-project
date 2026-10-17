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


def render_svg(
    observation: SnakeObservation,
    cell_size: int = 24,
    show_eyes: bool = True,
    status_overlay: Optional[str] = None,
) -> str:
    """Render a scalable vector graphic (SVG) representation of the Snake board.

    Produces a crisp, modern 2D grid visualization with rounded snake body segments,
    directional eye orientation, glowing food, and subtle grid lines.

    Args:
        observation: Current game observation snapshot.
        cell_size: Width/height of each grid cell in pixels (default 24).
        show_eyes: Whether to render directional eye pupils on the snake's head.
        status_overlay: Optional overlay text banner (e.g. 'Game Over: Wall Collision').

    Returns:
        Valid SVG XML string suitable for direct HTML/Streamlit embedding.
    """
    width, height = observation.grid_size
    svg_w, svg_h = width * cell_size, height * cell_size

    parts: List[str] = [
        f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.4);">'
    ]

    # Background
    parts.append(
        f'<rect width="{svg_w}" height="{svg_h}" rx="12" fill="#0b0f19" stroke="#1e293b" stroke-width="2"/>'
    )

    # Grid mesh lines
    for x in range(cell_size, svg_w, cell_size):
        parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{svg_h}" stroke="#141c2e" stroke-width="0.75"/>'
        )
    for y in range(cell_size, svg_h, cell_size):
        parts.append(
            f'<line x1="0" y1="{y}" x2="{svg_w}" y2="{y}" stroke="#141c2e" stroke-width="0.75"/>'
        )

    # Food item (glowing ruby circle with specular dot)
    if observation.food is not None:
        fx = observation.food.x * cell_size + cell_size / 2.0
        fy = observation.food.y * cell_size + cell_size / 2.0
        r_food = max(3.0, (cell_size / 2.0) - 3.0)
        parts.append(
            f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="{r_food:.1f}" fill="#ef4444" stroke="#fca5a5" stroke-width="1.5"/>'
        )
        parts.append(
            f'<circle cx="{fx - 2.5:.1f}" cy="{fy - 2.5:.1f}" r="{max(1.0, r_food / 4.0):.1f}" fill="#ffffff" opacity="0.8"/>'
        )

    # Snake body segments (from tail to neck)
    body_len = len(observation.body)
    for idx, seg in enumerate(observation.body[1:], start=1):
        bx = seg.x * cell_size + 1.5
        by = seg.y * cell_size + 1.5
        bw = cell_size - 3.0
        # Color gradient: closer to neck is brighter, closer to tail is deeper
        t = idx / max(1, body_len)
        color = "#10b981" if t < 0.5 else "#059669"
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bw:.1f}" rx="5" '
            f'fill="{color}" stroke="#047857" stroke-width="1"/>'
        )

    # Snake head
    hx = observation.head.x * cell_size + 1.0
    hy = observation.head.y * cell_size + 1.0
    hw = cell_size - 2.0
    parts.append(
        f'<rect x="{hx:.1f}" y="{hy:.1f}" width="{hw:.1f}" height="{hw:.1f}" rx="6" '
        f'fill="#34d399" stroke="#10b981" stroke-width="1.5"/>'
    )

    # Directional eyes
    if show_eyes:
        d = observation.direction
        r_eye = max(1.5, cell_size * 0.11)
        r_pupil = max(0.8, cell_size * 0.055)
        offset = cell_size * 0.24
        edge_offset = cell_size * 0.20

        if d == Direction.RIGHT:
            e1 = (hx + hw - edge_offset, hy + offset)
            e2 = (hx + hw - edge_offset, hy + hw - offset)
            p1 = (e1[0] + 0.8, e1[1])
            p2 = (e2[0] + 0.8, e2[1])
        elif d == Direction.LEFT:
            e1 = (hx + edge_offset, hy + offset)
            e2 = (hx + edge_offset, hy + hw - offset)
            p1 = (e1[0] - 0.8, e1[1])
            p2 = (e2[0] - 0.8, e2[1])
        elif d == Direction.UP:
            e1 = (hx + offset, hy + edge_offset)
            e2 = (hx + hw - offset, hy + edge_offset)
            p1 = (e1[0], e1[1] - 0.8)
            p2 = (e2[0], e2[1] - 0.8)
        else:  # DOWN
            e1 = (hx + offset, hy + hw - edge_offset)
            e2 = (hx + hw - offset, hy + hw - edge_offset)
            p1 = (e1[0], e1[1] + 0.8)
            p2 = (e2[0], e2[1] + 0.8)

        parts.append(
            f'<circle cx="{e1[0]:.1f}" cy="{e1[1]:.1f}" r="{r_eye:.1f}" fill="#ffffff"/>'
            f'<circle cx="{p1[0]:.1f}" cy="{p1[1]:.1f}" r="{r_pupil:.1f}" fill="#000000"/>'
            f'<circle cx="{e2[0]:.1f}" cy="{e2[1]:.1f}" r="{r_eye:.1f}" fill="#ffffff"/>'
            f'<circle cx="{p2[0]:.1f}" cy="{p2[1]:.1f}" r="{r_pupil:.1f}" fill="#000000"/>'
        )

    # Optional status overlay card (e.g. for game over)
    if status_overlay:
        card_w = min(svg_w - 40, 360)
        card_h = 50
        cx = (svg_w - card_w) / 2.0
        cy = (svg_h - card_h) / 2.0
        parts.append(
            f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{card_w}" height="{card_h}" rx="8" '
            f'fill="#0f172a" fill-opacity="0.9" stroke="#ef4444" stroke-width="2"/>'
            f'<text x="{svg_w / 2.0:.1f}" y="{(cy + 31):.1f}" text-anchor="middle" fill="#f87171" '
            f'font-family="system-ui, -apple-system, sans-serif" font-size="15" font-weight="bold">'
            f'{status_overlay}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def render_image(
    observation: SnakeObservation,
    cell_size: int = 24,
    show_eyes: bool = True,
) -> Any:
    """Render a Pillow Image representation of the Snake board.

    Args:
        observation: Current game observation snapshot.
        cell_size: Pixel dimension for each cell.
        show_eyes: Whether to draw eyes on the head.

    Returns:
        PIL.Image.Image instance.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for render_image. Install with: pip install pillow"
        ) from exc

    width, height = observation.grid_size
    img_w, img_h = width * cell_size, height * cell_size
    img = Image.new("RGB", (img_w, img_h), color="#0b0f19")
    draw = ImageDraw.Draw(img)

    # Grid mesh lines
    for x in range(cell_size, img_w, cell_size):
        draw.line([(x, 0), (x, img_h)], fill="#141c2e", width=1)
    for y in range(cell_size, img_h, cell_size):
        draw.line([(0, y), (img_w, y)], fill="#141c2e", width=1)

    # Food
    if observation.food is not None:
        fx = observation.food.x * cell_size
        fy = observation.food.y * cell_size
        draw.ellipse(
            [fx + 3, fy + 3, fx + cell_size - 3, fy + cell_size - 3],
            fill="#ef4444",
            outline="#fca5a5",
        )

    # Snake body
    for seg in observation.body[1:]:
        bx = seg.x * cell_size
        by = seg.y * cell_size
        draw.rectangle(
            [bx + 2, by + 2, bx + cell_size - 2, by + cell_size - 2],
            fill="#10b981",
            outline="#047857",
        )

    # Snake head
    hx = observation.head.x * cell_size
    hy = observation.head.y * cell_size
    draw.rectangle(
        [hx + 1, hy + 1, hx + cell_size - 1, hy + cell_size - 1],
        fill="#34d399",
        outline="#10b981",
    )

    # Eyes
    if show_eyes:
        d = observation.direction
        s = cell_size
        if d == Direction.RIGHT:
            eyes = [(hx + s - 6, hy + 5), (hx + s - 6, hy + s - 7)]
        elif d == Direction.LEFT:
            eyes = [(hx + 4, hy + 5), (hx + 4, hy + s - 7)]
        elif d == Direction.UP:
            eyes = [(hx + 5, hy + 4), (hx + s - 7, hy + 4)]
        else:
            eyes = [(hx + 5, hy + s - 6), (hx + s - 7, hy + s - 6)]

        for ex, ey in eyes:
            draw.ellipse([ex, ey, ex + 2, ey + 2], fill="#ffffff")

    return img

