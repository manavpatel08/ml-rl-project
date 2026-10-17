"""Visual interactive training and playback engine for Snake RL.

Provides real-time terminal-based visual playback of the actual SnakeEnvironment
and QLearningAgent during training or evaluation runs.
"""

from dataclasses import dataclass, field
from pathlib import Path
import select
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.checkpoints import load_q_table, save_q_table
from snake_rl.config import EnvironmentConfig, QLearningConfig
from snake_rl.environment import Action, SnakeEnvironment, SnakeObservation
from snake_rl.rendering import format_hud, render_ascii
from snake_rl.training import EpisodeMetrics, TrainingHistory


class VisualPlaybackRunner:
    """Orchestrates interactive visual training and evaluation playback.

    Guarantees:
      - During training, the real agent receives environment rewards and updates its Q-table.
      - During evaluation, Q-values and epsilon remain strictly untouched.
      - Pausing, resuming, or stopping does not duplicate training updates.
      - Checkpoints are never overwritten without explicit user confirmation.
    """

    def __init__(
        self,
        env: Optional[SnakeEnvironment] = None,
        agent: Optional[Any] = None,
        speed_delay: float = 0.04,
        use_colors: bool = True,
        interactive: bool = True,
        on_frame: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize the visual runner.

        Args:
            env: SnakeEnvironment instance. If None, creates default 20x20.
            agent: Agent instance (QLearningAgent, DQNAgent, etc.). If None, creates fresh QLearningAgent.

            speed_delay: Delay between frames in seconds (default 0.04s = ~25 fps).
            use_colors: Whether to output ANSI color escape sequences.
            interactive: Whether to check keyboard input for pause/resume/quit.
            on_frame: Optional callback invoked with the rendered string for each frame.
        """
        self.env = env if env is not None else SnakeEnvironment()
        self.agent = agent if agent is not None else QLearningAgent()
        self.speed_delay = speed_delay
        self.use_colors = use_colors
        self.interactive = interactive
        self.on_frame = on_frame

        self.is_paused: bool = False
        self.is_stopped: bool = False
        self.history: TrainingHistory = TrainingHistory()

    def set_speed(self, delay_seconds: float) -> None:
        """Set playback delay between frames (clamped between 0.005s and 1.0s)."""
        self.speed_delay = max(0.005, min(1.0, delay_seconds))

    def pause(self) -> None:
        """Pause execution."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume execution."""
        self.is_paused = False

    def stop(self) -> None:
        """Stop the session safely."""
        self.is_stopped = True

    def run_training_session(
        self,
        num_episodes: int = 10,
        base_seed: Optional[int] = None,
        reset_stop: bool = False,
    ) -> TrainingHistory:
        """Run a visually rendered training session.

        Updates Q-values on each step and decays epsilon once per episode.

        Args:
            num_episodes: Exact number of training episodes to run.
            base_seed: Optional base seed for reproducible food/spawn layouts.
            reset_stop: If True, resets the is_stopped flag to False before starting.

        Returns:
            TrainingHistory containing metrics for all completed episodes.
        """
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        if self.is_stopped and not reset_stop:
            return self.history

        self.is_stopped = False
        self.is_paused = False

        for ep in range(1, num_episodes + 1):
            if self.is_stopped:
                break

            ep_seed = (base_seed + ep) if base_seed is not None else None
            obs = self.env.reset(seed=ep_seed)

            total_reward = 0.0
            td_errors: List[float] = []
            last_reason = "unknown"
            last_action_name = "START"
            last_exploring: Optional[bool] = None

            # Render initial board
            self._render_step(
                obs=obs,
                ep=ep,
                total_episodes=num_episodes,
                reward=0.0,
                rolling_avg=self._get_rolling_avg(),
                action_name=last_action_name,
                is_exploring=last_exploring,
                mode="TRAINING",
            )

            # Step-by-step game loop
            while not self.env.done and not self.is_stopped:
                self._handle_input()
                if self.is_stopped:
                    break

                # Real Q-learning action selection
                action, is_exploring = self.agent.select_action_with_mode(obs)
                last_exploring = is_exploring
                last_action_name = action.name

                step_res = self.env.step(action)

                # Real learning update (QLearningAgent or DQNAgent)
                if hasattr(self.agent, "update"):
                    td_error = self.agent.update(
                        state=obs,
                        action=action,
                        reward=step_res.reward,
                        next_state=step_res.observation,
                        done=step_res.done,
                    )
                    td_errors.append(abs(td_error))
                elif hasattr(self.agent, "remember") and hasattr(self.agent, "train_step"):
                    self.agent.remember(
                        state=obs,
                        action=action,
                        reward=step_res.reward,
                        next_state=step_res.observation,
                        done=step_res.done,
                    )
                    step_res_train = self.agent.train_step()
                    if step_res_train is not None:
                        td_errors.append(abs(step_res_train[1]))

                total_reward += step_res.reward
                obs = step_res.observation
                last_reason = step_res.info.get("reason", "unknown")

                # Render step frame
                self._render_step(
                    obs=obs,
                    ep=ep,
                    total_episodes=num_episodes,
                    reward=total_reward,
                    rolling_avg=self._get_rolling_avg(),
                    action_name=last_action_name,
                    is_exploring=last_exploring,
                    mode="TRAINING",
                )

                # Frame delay
                if self.speed_delay > 0 and self.interactive:
                    time.sleep(self.speed_delay)

            # Episode completion: decay epsilon once per episode
            if not self.is_stopped:
                current_eps = self.agent.decay_epsilon()
                mean_td = (sum(td_errors) / len(td_errors)) if td_errors else 0.0
                metrics = EpisodeMetrics(
                    episode=ep,
                    total_reward=round(total_reward, 4),
                    score=self.env.score,
                    steps=self.env.steps,
                    epsilon=round(current_eps, 6),
                    mean_td_error=round(mean_td, 6),
                    termination_reason=last_reason,
                )
                self.history.episodes.append(metrics)

        return self.history

    def run_evaluation_session(
        self,
        num_episodes: int = 5,
        base_seed: int = 1000,
        reset_stop: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run a visually rendered evaluation session under pure greedy exploitation.

        Guarantees:
          - Does NOT update Q-values.
          - Does NOT decay or modify epsilon.
          - Strictly selects greedy action (select_greedy_action).

        Args:
            num_episodes: Number of evaluation episodes to watch.
            base_seed: Fixed evaluation seed base.
            reset_stop: If True, resets the is_stopped flag to False before starting.

        Returns:
            List of summary dicts for each completed episode.
        """
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        if self.is_stopped and not reset_stop:
            return []

        self.is_stopped = False
        self.is_paused = False
        eval_summaries = []

        for ep in range(1, num_episodes + 1):
            if self.is_stopped:
                break

            ep_seed = base_seed + ep - 1
            obs = self.env.reset(seed=ep_seed)

            total_reward = 0.0
            last_reason = "unknown"
            last_action_name = "START"

            self._render_step(
                obs=obs,
                ep=ep,
                total_episodes=num_episodes,
                reward=0.0,
                rolling_avg=self._compute_eval_avg(eval_summaries),
                action_name=last_action_name,
                is_exploring=False,
                mode="EVALUATION (GREEDY)",
            )

            while not self.env.done and not self.is_stopped:
                self._handle_input()
                if self.is_stopped:
                    break

                action = self.agent.select_greedy_action(obs)
                last_action_name = action.name

                step_res = self.env.step(action)
                total_reward += step_res.reward
                obs = step_res.observation
                last_reason = step_res.info.get("reason", "unknown")

                self._render_step(
                    obs=obs,
                    ep=ep,
                    total_episodes=num_episodes,
                    reward=total_reward,
                    rolling_avg=self._compute_eval_avg(eval_summaries),
                    action_name=last_action_name,
                    is_exploring=False,
                    mode="EVALUATION (GREEDY)",
                )

                if self.speed_delay > 0 and self.interactive:
                    time.sleep(self.speed_delay)

            if not self.is_stopped:
                eval_summaries.append({
                    "episode": ep,
                    "seed": ep_seed,
                    "score": self.env.score,
                    "steps": self.env.steps,
                    "reward": round(total_reward, 2),
                    "reason": last_reason,
                })

        return eval_summaries

    def _render_step(
        self,
        obs: SnakeObservation,
        ep: int,
        total_episodes: int,
        reward: float,
        rolling_avg: float,
        action_name: str,
        is_exploring: Optional[bool],
        mode: str,
    ) -> None:
        """Render a single frame consisting of the HUD and ASCII board."""
        speed_ms = int(self.speed_delay * 1000)
        hud = format_hud(
            episode=ep,
            total_episodes=total_episodes,
            score=obs.score,
            steps=obs.steps,
            epsilon=self.agent.epsilon,
            reward=reward,
            rolling_avg=rolling_avg,
            is_exploring=is_exploring,
            mode=mode,
            action_name=action_name,
            use_colors=self.use_colors,
            speed_ms=speed_ms,
            is_paused=self.is_paused,
        )

        board = render_ascii(obs, use_colors=self.use_colors)
        controls = (
            "Controls: [Space/Enter: Pause/Resume]  [1: Fast (10ms)]  "
            "[2: Normal (40ms)]  [3: Slow (120ms)]  [q: Stop]"
        )

        full_frame = f"{hud}\n{board}\n{controls}\n"

        if self.on_frame is not None:
            self.on_frame(full_frame)

        if self.interactive and sys.stdout.isatty():
            # Clear screen and place cursor at top-left
            sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(full_frame)
            sys.stdout.flush()

    def _handle_input(self) -> None:
        """Check for interactive keyboard commands (pause, speed, quit)."""
        if not self.interactive or not sys.stdin.isatty():
            return

        # Handle pause loop
        while self.is_paused and not self.is_stopped:
            # Poll stdin for resume key
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                char = sys.stdin.read(1)
                if char in (" ", "\n", "\r", "p", "P"):
                    self.is_paused = False
                    break
                elif char in ("q", "Q"):
                    self.is_stopped = True
                    break

        if self.is_stopped:
            return

        # Non-blocking poll for key press during active play
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            char = sys.stdin.read(1)
            if char in (" ", "p", "P"):
                self.is_paused = True
            elif char in ("q", "Q"):
                self.is_stopped = True
            elif char in ("1", "f", "F"):
                self.set_speed(0.01)
            elif char in ("2", "n", "N"):
                self.set_speed(0.04)
            elif char in ("3", "s", "S"):
                self.set_speed(0.12)

    def _get_rolling_avg(self) -> float:
        if not self.history.episodes:
            return 0.0
        scores = self.history.scores[-10:]
        return sum(scores) / len(scores)

    @staticmethod
    def _compute_eval_avg(eval_summaries: List[Dict[str, Any]]) -> float:
        if not eval_summaries:
            return 0.0
        recent = [s["score"] for s in eval_summaries[-10:]]
        return sum(recent) / len(recent)
