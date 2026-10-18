"""Controlled 200-Episode DQN Training Experiment Runner.

Orchestrates:
1. Controlled training of DQNAgent for 200 episodes using project defaults.
2. Per-episode recording of score, reward, steps, epsilon, loss, cumulative steps,
   and cumulative gradient updates.
3. Safe checkpoint preservation to models/dqn_200ep_seed42.pt with overwrite protection.
4. Export of machine-readable metrics (JSON & CSV) and high-resolution training curves (PNG).
5. Fair evaluation against untrained DQN, RandomAgent, and HeuristicAgent under fixed seeds.
6. Checkpoint verification and reload validation.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import torch

from snake_rl.agents import DQNAgent, HeuristicAgent, RandomAgent
from snake_rl.checkpoints import load_dqn_checkpoint
from snake_rl.config import DQNConfig, EnvironmentConfig
from snake_rl.environment import SnakeEnvironment
from snake_rl.evaluation import compare_agents
from snake_rl.session import safe_save_checkpoint
from snake_rl.training import EpisodeMetrics, TrainingHistory, train_dqn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled 200-episode DQN experiment.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--gamma", type=float, default=0.90, help="Discount factor gamma (default: 0.90)")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device (default: 'cpu')")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/dqn_200ep_seed42.pt",
        help="Destination path for trained checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dqn_200ep_seed42",
        help="Directory to save metrics and plots",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per agent (default: 20)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1000,
        help="Base random seed for benchmark evaluation (default: 1000)",
    )
    return parser.parse_args()


def export_metrics(history: TrainingHistory, output_dir: Path) -> None:
    """Save machine-readable per-episode metrics in JSON and CSV formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_list: List[Dict[str, Any]] = [
        {
            "episode": ep.episode,
            "score": ep.score,
            "total_reward": ep.total_reward,
            "steps": ep.steps,
            "epsilon": ep.epsilon,
            "mean_td_error": ep.mean_td_error,
            "loss": ep.loss,
            "cumulative_steps": ep.cumulative_steps,
            "cumulative_gradient_updates": ep.cumulative_gradient_updates,
            "termination_reason": ep.termination_reason,
        }
        for ep in history.episodes
    ]

    # 1. JSON Export
    json_path = output_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)

    # 2. CSV Export
    csv_path = output_dir / "metrics.csv"
    if metrics_list:
        keys = list(metrics_list[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics_list)


def generate_training_plots(history: TrainingHistory, output_dir: Path) -> Path:
    """Generate high-resolution multi-panel training plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "training_curves.png"

    episodes = [ep.episode for ep in history.episodes]
    scores = history.scores
    rolling_scores = history.moving_average_scores(window=10)
    rewards = history.rewards
    rolling_rewards = history.moving_average_rewards(window=10)
    losses = history.losses
    rolling_losses = history.moving_average_losses(window=10)
    epsilons = history.epsilons
    cum_steps = history.cumulative_steps
    cum_updates = history.cumulative_gradient_updates

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("DQN 200-Episode Controlled Training Experiment (Seed 42, Gamma 0.90)", fontsize=14, fontweight="bold")

    # Panel 1: Scores
    axes[0, 0].plot(episodes, scores, color="#94a3b8", alpha=0.5, label="Raw Score")
    axes[0, 0].plot(episodes, rolling_scores, color="#38bdf8", linewidth=2.0, label="10-Ep Rolling Avg")
    axes[0, 0].set_title("Food Score per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    # Panel 2: Rewards
    axes[0, 1].plot(episodes, rewards, color="#fca5a5", alpha=0.5, label="Raw Reward")
    axes[0, 1].plot(episodes, rolling_rewards, color="#ef4444", linewidth=2.0, label="10-Ep Rolling Avg")
    axes[0, 1].set_title("Total Episode Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    # Panel 3: Huber Loss
    axes[1, 0].plot(episodes, losses, color="#fdba74", alpha=0.5, label="Mean Loss")
    axes[1, 0].plot(episodes, rolling_losses, color="#f97316", linewidth=2.0, label="10-Ep Rolling Avg")
    axes[1, 0].set_title("DQN Huber Loss (SmoothL1)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    # Panel 4: Epsilon Decay
    axes[1, 1].plot(episodes, epsilons, color="#a855f7", linewidth=2.0, label="Epsilon (ε)")
    axes[1, 1].axhline(y=0.01, color="#64748b", linestyle=":", label="Floor (0.01)")
    axes[1, 1].set_title("Exploration Rate Schedule (Decay: 0.995)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    # Panel 5: Cumulative Steps vs Cumulative Gradient Updates
    axes[2, 0].plot(episodes, cum_steps, color="#10b981", linewidth=2.0, label="Cumulative Env Steps")
    axes[2, 0].plot(episodes, cum_updates, color="#6366f1", linewidth=2.0, label="Cumulative Gradient Updates")
    axes[2, 0].set_title("Cumulative Throughput: Steps & Updates")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylabel("Count")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    axes[2, 0].legend()

    # Panel 6: Episode Durations (Steps)
    durations = history.lengths
    axes[2, 1].bar(episodes, durations, color="#0ea5e9", alpha=0.6, width=1.0)
    axes[2, 1].set_title("Episode Length (Steps)")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_ylabel("Steps")
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("      CONTROLLED 200-EPISODE DEEP Q-NETWORK (DQN) EXPERIMENT      ")
    print("=" * 72)
    print(f"Algorithm:                   Deep Q-Network (DQN)")
    print(f"Network Architecture:        11 -> 128 -> 128 -> 3 (ReLU)")
    print(f"Optimizer:                   Adam (lr=0.001)")
    print(f"Loss Function:               SmoothL1Loss (Huber loss)")
    print(f"Discount Factor (gamma):     {args.gamma}")
    print(f"Epsilon Schedule:            start=1.0, min=0.01, decay=0.995")
    print(f"Replay Buffer:               capacity=10,000, batch_size=64, warmup=64")
    print(f"Target Sync Interval:        500 gradient updates")
    print(f"Training Episodes:           {args.episodes}")
    print(f"Base Random Seed:            {args.seed}")
    print(f"Execution Device:            {args.device}")
    print(f"Checkpoint Destination:      {args.checkpoint_path}")
    print(f"Output Directory:            {args.output_dir}")
    print("-" * 72)

    # 1. Initialize configuration and agent
    config = DQNConfig(
        gamma=args.gamma,
        device=args.device,
    )
    agent = DQNAgent(config=config, seed=args.seed)
    env = SnakeEnvironment()

    print(f"[INFO] Initialized fresh DQNAgent (Device: {config.device}, Initial Epsilon: {agent.epsilon:.4f})")
    print("[INFO] Starting bounded training run...")
    start_time = time.time()

    def progress_callback(m: EpisodeMetrics) -> None:
        if m.episode == 1 or m.episode % 25 == 0 or m.episode == args.episodes:
            print(
                f"  Ep {m.episode:3d}/{args.episodes} | "
                f"Score: {m.score:2d} | "
                f"Steps: {m.steps:3d} | "
                f"Reward: {m.total_reward:+6.1f} | "
                f"Eps: {m.epsilon:.4f} | "
                f"Loss: {m.loss:.5f} | "
                f"Total Steps: {m.cumulative_steps:5d} | "
                f"Updates: {m.cumulative_gradient_updates:5d}"
            )

    history = train_dqn(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        base_seed=args.seed,
        callback=progress_callback,
    )
    elapsed = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed:.2f}s ({elapsed / args.episodes:.3f}s/ep)")

    # 2. Save Checkpoint with Overwrite Protection
    print("\n[STEP 1/4] Saving Trained Model Checkpoint...")
    ckpt_path = Path(args.checkpoint_path)
    success, msg = safe_save_checkpoint(
        agent=agent,
        filepath=ckpt_path,
        overwrite=False,
        metadata={
            "episodes": args.episodes,
            "seed": args.seed,
            "gamma": args.gamma,
            "elapsed_seconds": round(elapsed, 2),
            "final_epsilon": round(agent.epsilon, 6),
            "total_steps": history.total_steps,
            "total_gradient_updates": agent.train_step_count,
            "mean_score_last_20": round(sum(history.scores[-20:]) / 20.0, 2),
        },
    )
    if not success:
        print(f"[ERROR] Failed to save checkpoint: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ {msg} ({ckpt_path.stat().st_size / 1024:.1f} KB)")

    # 3. Export Machine-Readable Metrics and Plots
    print("\n[STEP 2/4] Exporting Metrics & Plots...")
    out_dir = Path(args.output_dir)
    export_metrics(history, out_dir)
    print(f"  ✓ Saved JSON: {out_dir / 'metrics.json'}")
    print(f"  ✓ Saved CSV:  {out_dir / 'metrics.csv'}")

    plot_file = generate_training_plots(history, out_dir)
    print(f"  ✓ Saved Plots: {plot_file}")

    # 4. Fair Benchmark Evaluation
    print(f"\n[STEP 3/4] Benchmark Evaluation across {args.eval_episodes} Fixed Seeds (Base Seed: {args.eval_seed})...")
    eval_agents = {
        "DQN (Trained 200ep)": agent,
        "DQN (Untrained Baseline)": DQNAgent(config=config, seed=args.eval_seed),
        "Random Baseline": RandomAgent(seed=args.eval_seed),
        "Heuristic Baseline": HeuristicAgent(),
    }

    eval_results = compare_agents(
        env=SnakeEnvironment(),
        agents=eval_agents,
        num_episodes=args.eval_episodes,
        base_seed=args.eval_seed,
    )

    eval_summary_path = out_dir / "evaluation_summary.json"
    eval_data = {}
    print("\n" + "-" * 78)
    print(f"{'Agent':<26} | {'Mean Score':<12} | {'Max Score':<10} | {'Mean Steps':<11} | {'Mean Reward'}")
    print("-" * 78)
    for name, res in eval_results.items():
        eval_data[name] = {
            "mean_score": res.mean_score,
            "std_score": res.std_score,
            "max_score": res.max_score,
            "min_score": res.min_score,
            "mean_steps": res.mean_steps,
            "mean_reward": res.mean_reward,
            "termination_reasons": res.termination_reasons,
        }
        print(
            f"{name:<26} | {res.mean_score:5.2f} ± {res.std_score:4.2f} | "
            f"{res.max_score:<10d} | {res.mean_steps:8.1f}   | {res.mean_reward:+8.2f}"
        )
    print("-" * 78)

    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"  ✓ Saved Evaluation Summary: {eval_summary_path}")

    # 5. Checkpoint Verification
    print("\n[STEP 4/4] Verifying Checkpoint Reload & Parameter Equality...")
    reloaded_agent = load_dqn_checkpoint(ckpt_path)
    for (n1, p1), (n2, p2) in zip(
        agent.online_network.named_parameters(),
        reloaded_agent.online_network.named_parameters(),
    ):
        if not torch.equal(p1, p2):
            print(f"[ERROR] Parameter mismatch in {n1}!", file=sys.stderr)
            sys.exit(1)

    print("  ✓ Checkpoint reloads cleanly; all online network weights match exactly.")
    print("=" * 72)
    print("                   EXPERIMENT COMPLETED SUCCESSFULLY                   ")
    print("=" * 72)


if __name__ == "__main__":
    main()
