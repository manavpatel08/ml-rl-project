"""Streamlit Dashboard for Snake RL: From Q-Learning to DQN.

Provides an interactive educational interface for:
  - Play: Real-time graphical playback of trained agents under pure exploitation.
  - Train: Live observable visual training for Tabular Q-Learning and DQN with real-time HUD and learning updates.
  - Compare: Multi-agent benchmarking across identical fixed evaluation seeds.
  - How It Works: Beginner-friendly explanations of environments, encodings, and algorithms.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from snake_rl.agents.baselines import HeuristicAgent, RandomAgent
from snake_rl.agents.dqn import DQNAgent
from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.checkpoints import load_dqn_checkpoint, load_q_table
from snake_rl.config import DQNConfig, QLearningConfig
from snake_rl.environment import Action, Direction, SnakeEnvironment
from snake_rl.evaluation import compare_agents
from snake_rl.rendering import render_svg
from snake_rl.session import (
    EvaluationSession,
    TrainingSession,
    list_available_checkpoints,
    safe_save_checkpoint,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Snake AI: Q-Learning to DQN",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for clean typography, dark theme cards, and HUD styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .badge-explore {
        background-color: #ca8a04;
        color: #fef08a;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-exploit {
        background-color: #2563eb;
        color: #bfdbfe;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-banner {
        background-color: #1e1b4b;
        border-left: 4px solid #6366f1;
        padding: 10px 14px;
        border-radius: 4px;
        color: #c7d2fe;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

def init_session_state() -> None:
    """Ensure all required session state variables are safely initialized."""
    if "train_session" not in st.session_state:
        st.session_state.train_session = None
    if "is_training" not in st.session_state:
        st.session_state.is_training = False
    if "is_train_paused" not in st.session_state:
        st.session_state.is_train_paused = False
    if "train_step_requested" not in st.session_state:
        st.session_state.train_step_requested = False

    if "eval_session" not in st.session_state:
        st.session_state.eval_session = None
    if "is_eval_playing" not in st.session_state:
        st.session_state.is_eval_playing = False
    if "eval_step_requested" not in st.session_state:
        st.session_state.eval_step_requested = False

    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None


init_session_state()


# -----------------------------------------------------------------------------
# Sidebar: Global Navigation & Environment Info
# -----------------------------------------------------------------------------

st.sidebar.title("🐍 Snake RL")
st.sidebar.markdown(
    "**Educational Reinforcement Learning**\n"
    "Comparing Tabular Q-Learning and Deep Q-Networks (DQN) from first principles."
)

active_tab = st.sidebar.radio(
    "Navigation",
    ["🎮 Play", "🧠 Train", "📊 Compare", "📖 How It Works"],
    index=1,  # Default to Train tab as requested
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.markdown("- **Platform:** CPU Execution")
st.sidebar.markdown("- **Grid Size:** 20 × 20 cells")
st.sidebar.markdown("- **State Features:** 11 binary bits (2,048 states)")
st.sidebar.markdown("- **Action Space:** 3 relative actions")

checkpoints = list_available_checkpoints("models")
st.sidebar.markdown(f"**Found Checkpoints:** {len(checkpoints)}")
for c in checkpoints:
    icon = "🧠" if c["algorithm"] == "dqn" else "📋"
    st.sidebar.caption(f"{icon} `{c['name']}` ({c['size_kb']} KB)")

st.sidebar.markdown("---")
st.sidebar.caption("Educational Project: Antigravity Snake RL")


# -----------------------------------------------------------------------------
# Tab 1: 🎮 Play (Evaluation Playback)
# -----------------------------------------------------------------------------

def render_play_tab() -> None:
    st.header("🎮 Play: Model Evaluation Playback")
    st.markdown(
        """
        Watch a trained agent or baseline policy play Snake under **pure exploitation**.
        In evaluation mode, the exploration rate $\\epsilon$ is set to $0.0$, and **no weights,
        Q-values, or replay memory are updated**.
        """
    )

    col_setup, col_game = st.columns([1, 2])

    with col_setup:
        st.subheader("1. Select Agent / Model")
        available_models = list_available_checkpoints("models")
        model_options = ["Heuristic Baseline (Rule-Based)", "Random Baseline (Uncontrolled)"]
        model_map: Dict[str, Optional[Dict[str, Any]]] = {
            "Heuristic Baseline (Rule-Based)": None,
            "Random Baseline (Uncontrolled)": None,
        }

        for m in available_models:
            label = f"{m['name']} ({m['algorithm'].upper()}, {m['size_kb']} KB)"
            model_options.append(label)
            model_map[label] = m

        selected_label = st.selectbox("Model to load", model_options)
        selected_meta = model_map.get(selected_label)

        eval_seed = st.number_input("Episode Seed", value=1000, step=1, key="play_seed")
        speed_delay = st.slider("Playback Speed (Delay s)", 0.01, 0.25, 0.05, 0.01, key="play_speed")

        # Load / Instantiate Agent button
        if st.button("Load & Reset Game", use_container_width=True, type="secondary"):
            if selected_meta is None:
                if "Heuristic" in selected_label:
                    agent = HeuristicAgent()
                else:
                    agent = RandomAgent(seed=eval_seed)
            else:
                path = selected_meta["path"]
                if selected_meta["algorithm"] == "dqn":
                    agent = load_dqn_checkpoint(path)
                else:
                    agent = load_q_table(path)

            env = SnakeEnvironment()
            st.session_state.eval_session = EvaluationSession(agent=agent, env=env, base_seed=eval_seed)
            st.session_state.is_eval_playing = False
            st.success(f"Loaded: {selected_label}")

        st.markdown("---")
        st.subheader("2. Playback Controls")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("▶️ Play Episode", use_container_width=True, disabled=st.session_state.eval_session is None):
                st.session_state.is_eval_playing = True
        with col_btn2:
            if st.button("⏭️ Next Step", use_container_width=True, disabled=st.session_state.eval_session is None):
                st.session_state.eval_step_requested = True

        if st.button("🔄 New Episode", use_container_width=True, disabled=st.session_state.eval_session is None):
            if st.session_state.eval_session:
                st.session_state.eval_session.next_episode()
                st.session_state.is_eval_playing = False

    with col_game:
        st.subheader("Visual Board")
        session: Optional[EvaluationSession] = st.session_state.eval_session

        if session is None:
            st.info("👈 Select an agent or checkpoint on the left and click **'Load & Reset Game'** to start.")
        else:
            board_slot = st.empty()
            metrics_slot = st.empty()

            # Handle single manual step
            if st.session_state.eval_step_requested:
                st.session_state.eval_step_requested = False
                session.step()

            # Handle live playback loop
            if st.session_state.is_eval_playing:
                while session.can_step() and st.session_state.is_eval_playing:
                    session.step()
                    # Render updated frame
                    overlay = f"Game Over ({session.last_termination_reason})" if session.is_episode_done else None
                    svg_html = render_svg(session.current_obs, cell_size=23, status_overlay=overlay)
                    board_slot.markdown(
                        f'<div style="display:flex;justify-content:center;">{svg_html}</div>',
                        unsafe_allow_html=True,
                    )
                    render_eval_metrics(metrics_slot, session.get_metrics_summary())
                    time.sleep(speed_delay)

                st.session_state.is_eval_playing = False

            # Display static current state if not actively looping
            overlay = f"Game Over ({session.last_termination_reason})" if session.is_episode_done else None
            svg_html = render_svg(session.current_obs, cell_size=23, status_overlay=overlay)
            board_slot.markdown(
                f'<div style="display:flex;justify-content:center;">{svg_html}</div>',
                unsafe_allow_html=True,
            )
            render_eval_metrics(metrics_slot, session.get_metrics_summary())


def render_eval_metrics(slot: Any, metrics: Dict[str, Any]) -> None:
    """Render live evaluation KPI metrics in the provided Streamlit container."""
    with slot.container():
        st.markdown(
            '<div class="status-banner">🛡️ <b>EVALUATION MODE ACTIVE:</b> Pure Exploitation (ε = 0.0). '
            'Weights, Q-values, and replay memory are completely frozen.</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Score</div>'
                f'<div class="metric-value">{metrics["score"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Steps</div>'
                f'<div class="metric-value">{metrics["steps"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Episode Reward</div>'
                f'<div class="metric-value">{metrics["reward"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Mean Score</div>'
                f'<div class="metric-value">{metrics["mean_score"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Max Score</div>'
                f'<div class="metric-value">{metrics["max_score"]}</div></div>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Tab 2: 🧠 Train (Visual Live Training)
# -----------------------------------------------------------------------------

def render_train_tab() -> None:
    st.header("🧠 Train: Observable Real-Time Training")
    st.markdown(
        """
        Watch the reinforcement learning agent **learn live** as it interacts with the Snake environment.
        Each move updates real Q-values or neural network weights through the Bellman equation.
        """
    )

    col_cfg, col_live = st.columns([1, 2])

    with col_cfg:
        st.subheader("1. Training Setup")
        algo_choice = st.selectbox(
            "Algorithm",
            ["Tabular Q-Learning", "Deep Q-Network (DQN)"],
            key="train_algo",
        )
        init_mode = st.radio(
            "Agent Initialization",
            ["Fresh Untrained Agent (ε = 1.0)", "Continue from Checkpoint"],
            key="train_init_mode",
        )

        selected_ckpt_path = None
        if init_mode == "Continue from Checkpoint":
            expected_algo = "dqn" if "DQN" in algo_choice else "q_learning"
            compatible_ckpts = [
                c for c in list_available_checkpoints("models")
                if c["algorithm"] == expected_algo
            ]
            if compatible_ckpts:
                ckpt_choice = st.selectbox(
                    "Compatible Checkpoint",
                    [f"{c['name']} ({c['size_kb']} KB)" for c in compatible_ckpts],
                )
                chosen_meta = next(c for c in compatible_ckpts if c["name"] in ckpt_choice)
                selected_ckpt_path = chosen_meta["path"]
            else:
                st.warning(f"No existing `{expected_algo}` checkpoints found under `models/`.")

        episodes_input = st.number_input(
            "Episodes to Train",
            min_value=1,
            max_value=200,
            value=10,
            step=5,
            key="train_episodes_input",
        )
        base_seed_input = st.number_input(
            "Base Random Seed",
            value=42,
            step=1,
            key="train_base_seed",
        )
        speed_delay = st.slider(
            "Live Step Delay (seconds)",
            min_value=0.005,
            max_value=0.20,
            value=0.03,
            step=0.005,
            key="train_speed_delay",
        )

        # Initialize / Reset Training Session
        if st.button("Initialize / Reset Session", use_container_width=True, type="secondary"):
            if "DQN" in algo_choice:
                if selected_ckpt_path:
                    agent = load_dqn_checkpoint(selected_ckpt_path)
                else:
                    agent = DQNAgent(config=DQNConfig(), seed=base_seed_input)
            else:
                if selected_ckpt_path:
                    agent = load_q_table(selected_ckpt_path)
                else:
                    agent = QLearningAgent(config=QLearningConfig(), seed=base_seed_input)

            env = SnakeEnvironment()
            st.session_state.train_session = TrainingSession(
                agent=agent,
                env=env,
                total_episodes=episodes_input,
                base_seed=base_seed_input,
            )
            st.session_state.is_training = False
            st.session_state.is_train_paused = False
            st.success(f"Training session initialized ({algo_choice}).")

        st.markdown("---")
        st.subheader("2. Execution Controls")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if not st.session_state.is_training:
                if st.button(
                    "▶️ Start Training",
                    use_container_width=True,
                    type="primary",
                    disabled=st.session_state.train_session is None,
                ):
                    st.session_state.is_training = True
                    st.session_state.is_train_paused = False
            else:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.is_train_paused = not st.session_state.is_train_paused
                    if st.session_state.train_session:
                        if st.session_state.is_train_paused:
                            st.session_state.train_session.pause()
                        else:
                            st.session_state.train_session.resume()

        with col_c2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.is_training):
                st.session_state.is_training = False
                st.session_state.is_train_paused = False
                if st.session_state.train_session:
                    st.session_state.train_session.stop()

        if st.button("⏭️ Step 1 Action", use_container_width=True, disabled=st.session_state.train_session is None):
            st.session_state.train_step_requested = True

        st.markdown("---")
        st.subheader("3. Save Trained Checkpoint")
        session: Optional[TrainingSession] = st.session_state.train_session
        is_dqn = isinstance(getattr(session, "agent", None), DQNAgent)
        default_fn = "models/dqn_trained.pt" if is_dqn else "models/q_table_trained.json"
        dest_filename = st.text_input("Destination Path", value=default_fn, key="save_dest_input")
        overwrite_ok = st.checkbox("Overwrite existing file if present", value=False)

        if st.button("Save Checkpoint", use_container_width=True, disabled=session is None):
            if session:
                success, msg = safe_save_checkpoint(
                    agent=session.agent,
                    filepath=dest_filename,
                    overwrite=overwrite_ok,
                    metadata={"episodes_completed": len(session.history)},
                )
                if success:
                    st.success(msg)
                else:
                    st.warning(msg)

    with col_live:
        st.subheader("Live Training View")
        session: Optional[TrainingSession] = st.session_state.train_session

        if session is None:
            st.info("👈 Set up parameters on the left and click **'Initialize / Reset Session'** to start.")
        else:
            board_slot = st.empty()
            metrics_slot = st.empty()
            chart_slot = st.empty()

            # Handle manual single step
            if st.session_state.train_step_requested:
                st.session_state.train_step_requested = False
                session.step()
                if session.is_episode_done:
                    session.next_episode()

            # Live observable training loop
            if st.session_state.is_training and not st.session_state.is_train_paused:
                while not session.is_finished() and st.session_state.is_training:
                    if st.session_state.is_train_paused:
                        break

                    obs, done, info = session.step()

                    # Live graphical board rendering
                    overlay = f"Episode {session.current_episode} Done: {session.last_termination_reason}" if done else None
                    svg_html = render_svg(obs, cell_size=23, status_overlay=overlay)
                    board_slot.markdown(
                        f'<div style="display:flex;justify-content:center;">{svg_html}</div>',
                        unsafe_allow_html=True,
                    )

                    # Update live HUD telemetry
                    render_train_metrics(metrics_slot, session.get_metrics_summary())

                    # If episode completed, advance and render charts
                    if done:
                        render_training_charts(chart_slot, session.history, is_dqn)
                        if not session.next_episode():
                            st.session_state.is_training = False
                            break

                    time.sleep(speed_delay)

            # Static render when paused or idle
            overlay = f"Episode {session.current_episode} Done: {session.last_termination_reason}" if session.is_episode_done else None
            svg_html = render_svg(session.current_obs, cell_size=23, status_overlay=overlay)
            board_slot.markdown(
                f'<div style="display:flex;justify-content:center;">{svg_html}</div>',
                unsafe_allow_html=True,
            )
            render_train_metrics(metrics_slot, session.get_metrics_summary())
            if session.history.episodes:
                render_training_charts(chart_slot, session.history, is_dqn)


def render_train_metrics(slot: Any, metrics: Dict[str, Any]) -> None:
    """Render real-time training telemetry KPI cards."""
    with slot.container():
        badge_cls = "badge-explore" if metrics["is_exploring"] else "badge-exploit"
        badge_text = "🎲 EXPLORING (Random)" if metrics["is_exploring"] else "🧠 EXPLOITING (Greedy)"

        col_p, col_b = st.columns([3, 1])
        with col_p:
            pct = metrics["episode"] / max(1, metrics["total_episodes"])
            st.progress(pct, text=f"Episode {metrics['episode']} / {metrics['total_episodes']}")
        with col_b:
            st.markdown(f'<div class="{badge_cls}">{badge_text}</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Score</div>'
                f'<div class="metric-value">{metrics["score"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Episode Steps</div>'
                f'<div class="metric-value">{metrics["steps"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Reward</div>'
                f'<div class="metric-value">{metrics["reward"]:+0.1f}</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Epsilon (ε)</div>'
                f'<div class="metric-value">{metrics["epsilon"]:.4f}</div></div>',
                unsafe_allow_html=True,
            )

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Rolling Avg Score</div>'
                f'<div class="metric-value">{metrics["rolling_avg_score"]:.2f}</div></div>',
                unsafe_allow_html=True,
            )
        with m6:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Total Env Steps</div>'
                f'<div class="metric-value">{metrics["total_steps"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m7:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Last TD-Error</div>'
                f'<div class="metric-value">{metrics["last_td_error"]:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with m8:
            if metrics["train_step_count"] > 0:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">DQN Loss (Recent)</div>'
                    f'<div class="metric-value">{metrics["recent_loss_avg"]:.4f}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">Last Action</div>'
                    f'<div class="metric-value">{metrics["last_action_name"]}</div></div>',
                    unsafe_allow_html=True,
                )


def render_training_charts(slot: Any, history: Any, is_dqn: bool) -> None:
    """Render interactive line charts tracking scores, rewards, and loss curves."""
    with slot.container():
        st.markdown("### 📈 Live Training Progress Curves")
        c1, c2 = st.columns(2)
        with c1:
            df_scores = pd.DataFrame(
                {
                    "Episode": range(1, len(history) + 1),
                    "Score": history.scores,
                    "Rolling Avg": history.moving_average_scores(window=5),
                }
            ).set_index("Episode")
            st.line_chart(df_scores, height=220)

        with c2:
            if is_dqn and any(l > 0 for l in history.losses):
                df_loss = pd.DataFrame(
                    {
                        "Episode": range(1, len(history) + 1),
                        "Huber Loss": history.losses,
                        "Rolling Loss": history.moving_average_losses(window=5),
                    }
                ).set_index("Episode")
                st.line_chart(df_loss, height=220)
            else:
                df_rewards = pd.DataFrame(
                    {
                        "Episode": range(1, len(history) + 1),
                        "Reward": history.rewards,
                        "Rolling Reward": history.moving_average_rewards(window=5),
                    }
                ).set_index("Episode")
                st.line_chart(df_rewards, height=220)


# -----------------------------------------------------------------------------
# Tab 3: 📊 Compare (Multi-Agent Benchmark)
# -----------------------------------------------------------------------------

def render_compare_tab() -> None:
    st.header("📊 Compare: Multi-Agent Benchmark Evaluation")
    st.markdown(
        """
        Evaluate and compare multiple agents across **identical, reproducible seeds**.
        Fair evaluation guarantees that every agent encounters the **exact same board layouts
        and food sequences**.
        """
    )

    col_ctrl, col_stats = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Benchmark Settings")
        eval_episodes = st.number_input("Evaluation Episodes per Agent", min_value=5, max_value=50, value=20, step=5)
        eval_base_seed = st.number_input("Base Benchmark Seed", value=1000, step=1)

        st.markdown("#### Agents to Evaluate")
        include_random = st.checkbox("Random Agent (Lower Baseline)", value=True)
        include_heuristic = st.checkbox("Heuristic Agent (Greedy Rule-Based)", value=True)

        q_ckpts = [c for c in list_available_checkpoints("models") if c["algorithm"] == "q_learning"]
        include_q = st.checkbox("Tabular Q-Learning", value=bool(q_ckpts))
        selected_q_path = None
        if include_q and q_ckpts:
            q_choice = st.selectbox("Q-Learning Checkpoint", [c["name"] for c in q_ckpts])
            selected_q_path = next(c["path"] for c in q_ckpts if c["name"] == q_choice)

        dqn_ckpts = [c for c in list_available_checkpoints("models") if c["algorithm"] == "dqn"]
        include_dqn = st.checkbox("Deep Q-Network (DQN)", value=bool(dqn_ckpts))
        selected_dqn_path = None
        if include_dqn and dqn_ckpts:
            dqn_choice = st.selectbox("DQN Checkpoint", [c["name"] for c in dqn_ckpts])
            selected_dqn_path = next(c["path"] for c in dqn_ckpts if c["name"] == dqn_choice)

        if st.button("🚀 Run Benchmark Comparison", type="primary", use_container_width=True):
            agents_to_run: Dict[str, Any] = {}
            if include_random:
                agents_to_run["Random Agent"] = RandomAgent()
            if include_heuristic:
                agents_to_run["Heuristic Agent"] = HeuristicAgent()
            if include_q:
                if selected_q_path:
                    agents_to_run["Q-Learning Agent"] = load_q_table(selected_q_path)
                elif st.session_state.train_session and isinstance(st.session_state.train_session.agent, QLearningAgent):
                    agents_to_run["Q-Learning (Current Session)"] = st.session_state.train_session.agent
            if include_dqn:
                if selected_dqn_path:
                    agents_to_run["DQN Agent"] = load_dqn_checkpoint(selected_dqn_path)
                elif st.session_state.train_session and isinstance(st.session_state.train_session.agent, DQNAgent):
                    agents_to_run["DQN (Current Session)"] = st.session_state.train_session.agent

            if not agents_to_run:
                st.warning("Please select at least one agent to evaluate.")
            else:
                with st.spinner(f"Evaluating {len(agents_to_run)} agents across {eval_episodes} seeds..."):
                    env = SnakeEnvironment()
                    results = compare_agents(
                        env=env,
                        agents=agents_to_run,
                        num_episodes=eval_episodes,
                        base_seed=eval_base_seed,
                    )
                    st.session_state.benchmark_results = results
                st.success("Benchmark evaluation complete!")

    with col_stats:
        st.subheader("Benchmark Results")
        results = st.session_state.benchmark_results

        if results is None:
            st.info("👈 Configure benchmark settings and click **'Run Benchmark Comparison'** to generate results.")
        else:
            records = []
            for name, res in results.items():
                records.append(
                    {
                        "Agent": name,
                        "Mean Score": res.mean_score,
                        "Std Score": res.std_score,
                        "Max Score": res.max_score,
                        "Min Score": res.min_score,
                        "Mean Steps": res.mean_steps,
                        "Mean Reward": res.mean_reward,
                    }
                )

            df_summary = pd.DataFrame(records).set_index("Agent")
            st.dataframe(df_summary, use_container_width=True)

            st.markdown("#### Mean & Max Score Comparison")
            df_chart = df_summary[["Mean Score", "Max Score"]]
            st.bar_chart(df_chart)

            st.markdown("#### Mean Survival Steps per Episode")
            st.bar_chart(df_summary[["Mean Steps"]])


# -----------------------------------------------------------------------------
# Tab 4: 📖 How It Works (Educational Explanations)
# -----------------------------------------------------------------------------

def render_how_it_works_tab() -> None:
    st.header("📖 How It Works: Reinforcement Learning Fundamentals")
    st.markdown(
        """
        This project provides an educational, ground-up implementation of Reinforcement Learning
        algorithms applied to the classic game of Snake.
        """
    )

    t1, t2, t3, t4 = st.tabs(
        ["1. Game & State Encoding", "2. Tabular Q-Learning", "3. Deep Q-Networks (DQN)", "4. Training vs Evaluation"]
    )

    with t1:
        st.markdown(
            """
            ### The Snake Environment & State Representation
            
            - **Board Mechanics:** A discrete $20 \\times 20$ grid. The snake moves forward one square each step.
              Eating food ($+10.0$) increases score and snake length. Colliding with a wall or self ($-10.0$) terminates the game.
            - **Tail-Vacating Mechanics:** When the snake moves, its tail cell vacates on the same step unless food is eaten,
              preventing spurious collisions when chasing its own tail.
            - **Relative Action Space:**
              Rather than absolute directions (`UP`, `DOWN`, `LEFT`, `RIGHT`), the agent chooses relative actions:
              1. `STRAIGHT` (continue current heading)
              2. `TURN_RIGHT` (turn 90° clockwise)
              3. `TURN_LEFT` (turn 90° counter-clockwise)
              
              *Why relative actions?* They guarantee the snake can **never make an illegal 180° instant reversal into itself**.

            #### The 11-Bit State Vector
            Instead of raw pixels, the observation is encoded into a compact 11-bit binary vector:
            """
        )
        st.table(
            pd.DataFrame(
                [
                    {"Index": "0", "Feature": "Danger Ahead", "Meaning": "1 if wall or snake body is 1 cell straight ahead"},
                    {"Index": "1", "Feature": "Danger Right", "Meaning": "1 if wall or snake body is 1 cell to the right"},
                    {"Index": "2", "Feature": "Danger Left", "Meaning": "1 if wall or snake body is 1 cell to the left"},
                    {"Index": "3-6", "Feature": "Heading (One-Hot)", "Meaning": "Direction snake is facing: [Left, Right, Up, Down]"},
                    {"Index": "7-10", "Feature": "Food Direction", "Meaning": "Relative food quadrant: [Food Left, Food Right, Food Up, Food Down]"},
                ]
            )
        )
        st.info("With 11 binary bits, there are exactly $2^{11} = 2,048$ possible encoded states.")

    with t2:
        st.markdown(
            r"""
            ### Tabular Q-Learning
            
            In tabular Q-learning, the agent maintains a lookup table $Q(s, a)$ of shape $(2048, 3)$.
            Each cell represents the expected cumulative discounted future reward for taking action $a$ in state $s$.

            #### The Bellman Equation Update:
            $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

            - $\alpha = 0.1$ (Learning Rate): Controls how quickly new estimates replace old values.
            - $\gamma = 0.9$ (Discount Factor): Balances immediate rewards vs. long-term future gains.
            - $r$: Immediate scalar environment reward ($+10$ for food, $-10$ for collision, $-0.1$ for movement).
            - $\max_{a'} Q(s', a')$: Highest estimated value achievable from the subsequent state $s'$.

            > **Strict Terminal Non-Bootstrapping:** When an episode terminates ($done = \text{True}$), no future states exist.
            > The target simplifies strictly to $y = r$, with no future bootstrapping.
            """
        )

    with t3:
        st.markdown(
            r"""
            ### Deep Q-Networks (DQN)

            In DQN, the tabular lookup is replaced by a neural network function approximator $Q(s, a; \theta)$.

            ```text
            Input (11 binary features)
                     │
                     ▼
            Linear(11 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> 3)
                     │
                     ▼
            Output: 3 scalar Q-values [Q(s, STRAIGHT), Q(s, RIGHT), Q(s, LEFT)]
            ```

            #### Core Pillars of DQN Stability:
            1. **Experience Replay Buffer ($N=10,000$):**
               Consecutive game frames are highly temporally correlated. Storing transitions $(s, a, r, s', done)$ in a FIFO buffer
               and sampling random minibatches ($B=64$) breaks this correlation and stabilizes gradient descent.
            2. **Frozen Target Network:**
               Bootstrapping from the same network being updated causes moving-target instability. DQN maintains a separate
               target network with frozen weights ($\text{requires\_grad}=\text{False}$), synchronized only once every 500 gradient steps.
            3. **Huber Loss (`nn.SmoothL1Loss`):**
               Quadratic for small errors and linear for large errors, preventing gradient explosion caused by extreme TD errors.
            """
        )

    with t4:
        st.markdown(
            r"""
            ### Training vs. Evaluation


            | Property | Training Mode | Evaluation Mode |
            | :--- | :--- | :--- |
            | **Exploration ($\epsilon$)** | Active ($\epsilon > 0$, decays per episode) | Disabled ($\epsilon = 0.0$, pure exploitation) |
            | **Weights / Q-Table** | Updated continuously via Bellman updates | Completely frozen (zero updates) |
            | **Replay Buffer** | Transitions pushed and sampled | Untouched |
            | **Purpose** | Trial-and-error discovery | Objective assessment on fixed seeds |

            #### Metric Definitions:
            - **Epsilon ($\epsilon$):** Probability of selecting a completely random action. Decays from $1.0$ down to $0.01$.
            - **TD-Error:** The magnitude of the difference between the Bellman target and current Q-value estimate.
            - **Huber Loss:** The optimization loss minimized by the Adam optimizer during DQN gradient updates.
            - **Rolling Avg Score:** 10-episode trailing moving average smoothing out random food placement variance.
            """
        )


# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------

def main() -> None:
    if active_tab == "🎮 Play":
        render_play_tab()
    elif active_tab == "🧠 Train":
        render_train_tab()
    elif active_tab == "📊 Compare":
        render_compare_tab()
    elif active_tab == "📖 How It Works":
        render_how_it_works_tab()


if __name__ == "__main__":
    main()
