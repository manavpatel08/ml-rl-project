"""Interactive CLI Visualizer for Snake RL.

Usage:
  .venv/bin/python viewer.py
  .venv/bin/python viewer.py --mode train --episodes 10
  .venv/bin/python viewer.py --mode eval --episodes 5 --checkpoint models/q_table_500ep.json
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

from snake_rl.agents.dqn import DQNAgent

from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.checkpoints import (
    load_dqn_checkpoint,
    load_q_table,
    save_dqn_checkpoint,
    save_q_table,
)
from snake_rl.config import DQNConfig, QLearningConfig
from snake_rl.environment import SnakeEnvironment
from snake_rl.viewer import VisualPlaybackRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual training and playback viewer for Snake AI."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "menu"],
        default="menu",
        help="Mode to run: 'train' (watch live training), 'eval' (watch greedy playback), or 'menu' (interactive menu).",
    )
    parser.add_argument(
        "--agent",
        choices=["q_learning", "dqn"],
        default="q_learning",
        help="Agent algorithm to train: 'q_learning' or 'dqn' (default: 'q_learning').",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/q_table_500ep.json",
        help="Path to checkpoint (JSON for Q-table, .pt for DQN).",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.04,
        help="Playback delay in seconds per step (default: 0.04s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducible random seed (default: 42).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI terminal colors.",
    )
    return parser.parse_args()


def interactive_menu() -> None:
    """Prompt the user for interactive choices if no CLI flags were provided."""
    print("=" * 64)
    print("           SNAKE RL — VISUAL PLAYBACK & TRAINING VIEWER         ")
    print("=" * 64)
    print("1. Watch Live Q-Learning Training (Fresh Agent, default 10 episodes)")
    print("2. Watch Live DQN Training (Fresh Agent, default 10 episodes)")
    print("3. Watch Evaluation Playback (models/q_table_500ep.json)")
    print("4. Continue Training from Existing Checkpoint (.json or .pt)")
    print("5. Exit")
    print("-" * 64)

    choice = input("Select an option [1-5] (default: 3): ").strip()
    if choice == "1":
        run_training(episodes=10, checkpoint=None, agent_type="q_learning")
    elif choice == "2":
        run_training(episodes=10, checkpoint=None, agent_type="dqn")
    elif choice == "4":
        ckpt = input("Enter checkpoint path [models/q_table_500ep.json]: ").strip() or "models/q_table_500ep.json"
        eps = input("Enter number of episodes to train [10]: ").strip() or "10"
        run_training(episodes=int(eps), checkpoint=ckpt)
    elif choice == "5":
        print("Goodbye!")
        sys.exit(0)
    else:
        # Default: Watch evaluation playback
        run_eval(episodes=5, checkpoint="models/q_table_500ep.json")


def run_training(
    episodes: int = 10,
    checkpoint: Optional[str] = None,
    agent_type: str = "q_learning",
    speed: float = 0.04,
    seed: int = 42,
    use_colors: bool = True,
) -> None:
    """Launch visual training session."""
    env = SnakeEnvironment()

    if checkpoint and Path(checkpoint).is_file():
        print(f"\n[INFO] Loading existing checkpoint: {checkpoint}")
        if checkpoint.endswith(".pt"):
            agent = load_dqn_checkpoint(checkpoint)
        else:
            agent = load_q_table(checkpoint)
        print(f"[INFO] Continuing training from epsilon = {agent.epsilon:.4f}")
    else:
        print(f"\n[INFO] Starting training with fresh untrained {agent_type} agent (epsilon = 1.0)")
        if agent_type == "dqn":
            config = DQNConfig()
            agent = DQNAgent(config=config, seed=seed)
        else:
            config = QLearningConfig(
                alpha=0.1,
                gamma=0.9,
                epsilon_start=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
            )
            agent = QLearningAgent(config=config, seed=seed)

    runner = VisualPlaybackRunner(
        env=env,
        agent=agent,
        speed_delay=speed,
        use_colors=use_colors,
        interactive=True,
    )

    history = runner.run_training_session(num_episodes=episodes, base_seed=seed)

    print("\n" + "=" * 64)
    print("                 TRAINING SESSION SUMMARY                       ")
    print("=" * 64)
    print(f"Completed Episodes: {len(history)}")
    if history.scores:
        print(f"Scores:             {history.scores}")
        print(f"Mean Score:         {sum(history.scores) / len(history.scores):.2f}")
        print(f"Max Score:          {max(history.scores)}")
        print(f"Final Epsilon:      {agent.epsilon:.4f}")
    print("=" * 64)

    # Explicit, separate save action (Rule: Never overwrite automatically)
    save_choice = input("\nSave trained agent checkpoint? (y/n) [n]: ").strip().lower()
    if save_choice == "y":
        is_dqn = isinstance(agent, DQNAgent)
        default_dest = "models/dqn_interactive.pt" if is_dqn else "models/q_table_interactive.json"
        dest = input(f"Enter destination file path [{default_dest}]: ").strip()
        dest_path = Path(dest or default_dest)
        if dest_path.is_file():
            confirm = input(f"WARNING: '{dest_path}' already exists. Overwrite? (y/n) [n]: ").strip().lower()
            if confirm != "y":
                print("Save aborted. File was NOT overwritten.")
                return
        if is_dqn:
            save_dqn_checkpoint(agent, dest_path, metadata={"episodes_trained": len(history)})
        else:
            save_q_table(agent, dest_path, metadata={"episodes_trained": len(history)})
        print(f"Checkpoint successfully saved to: {dest_path}")


def run_eval(
    episodes: int = 5,
    checkpoint: str = "models/q_table_500ep.json",
    speed: float = 0.04,
    seed: int = 1000,
    use_colors: bool = True,
) -> None:
    """Launch visual evaluation session (pure exploitation, no updates)."""
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        print(f"[ERROR] Checkpoint file not found: {ckpt_path.resolve()}")
        sys.exit(1)

    print(f"\n[INFO] Loading checkpoint for greedy evaluation: {ckpt_path}")
    if str(ckpt_path).endswith(".pt"):
        agent = load_dqn_checkpoint(ckpt_path)
    else:
        agent = load_q_table(ckpt_path)

    env = SnakeEnvironment()
    runner = VisualPlaybackRunner(
        env=env,
        agent=agent,
        speed_delay=speed,
        use_colors=use_colors,
        interactive=True,
    )

    summaries = runner.run_evaluation_session(num_episodes=episodes, base_seed=seed)

    print("\n" + "=" * 64)
    print("                EVALUATION SESSION SUMMARY                      ")
    print("=" * 64)
    print(f"Completed Episodes: {len(summaries)}")
    if summaries:
        scores = [s["score"] for s in summaries]
        print(f"Scores:             {scores}")
        print(f"Mean Score:         {sum(scores) / len(scores):.2f}")
        print(f"Max Score:          {max(scores)}")
    print("=" * 64)


def main() -> None:
    args = parse_args()
    use_colors = not args.no_color

    if args.mode == "train":
        run_training(
            episodes=args.episodes,
            checkpoint=args.checkpoint if Path(args.checkpoint).is_file() else None,
            agent_type=args.agent,
            speed=args.speed,
            seed=args.seed,
            use_colors=use_colors,
        )
    elif args.mode == "eval":
        run_eval(
            episodes=args.episodes,
            checkpoint=args.checkpoint,
            speed=args.speed,
            seed=args.seed,
            use_colors=use_colors,
        )
    else:
        interactive_menu()


if __name__ == "__main__":
    main()

