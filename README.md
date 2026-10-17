# Snake AI: From Q-Learning to DQN

An educational, reproducible Reinforcement Learning project exploring decision-making in the classic Snake game. This project implements RL algorithms from the ground up—from Tabular Q-learning to Deep Q-Networks (DQN)—with clean decoupled architecture, mathematical verifications, and interactive visual playback.

---

## Architecture Overview

```text
Visual Viewer / Streamlit UI (app.py, viewer.py)
        │
        ▼
Application & Orchestration Layer (snake_rl/training.py, evaluation.py, checkpoints.py)
        │
        ▼
Core RL Package (snake_rl/environment.py, state.py, agents/)
        │
        ▼
Headless Game Engine & Utilities (snake_rl/config.py, rendering.py)
```

- **Headless Environment (`snake_rl/environment.py`):** Pure Python game mechanics on a configurable $20 \times 20$ grid. Completely decoupled from UI, graphics, or display servers.
- **11-Bit State Encoding (`snake_rl/state.py`):** Compact feature vector capturing danger ahead/right/left, one-hot heading (4 directions), and food quadrants (4 directions). Maps to $2^{11} = 2,048$ distinct states.
- **Tabular Q-Learning (`snake_rl/agents/q_learning.py`):** $2048 \times 3$ Q-table updated using the Bellman equation, terminal transition handling, and per-episode $\epsilon$-decay.
- **Deep Q-Networks (`snake_rl/agents/dqn.py`, `replay_buffer.py`):** PyTorch MLP (11 → 128 → 128 → 3) with target network freezing, Huber loss (`SmoothL1Loss`), experience replay (FIFO capacity 10,000, batch size 64), and Adam optimizer.
- **Baselines (`snake_rl/agents/baselines.py`):** `RandomAgent` (lower bound) and `HeuristicAgent` (greedy rule-based upper baseline).
- **Checkpoints (`snake_rl/checkpoints.py`):** Safe JSON serialization for tabular Q-tables and PyTorch `.pt` state dict serialization for DQN agents.
- **Visual Playback (`snake_rl/rendering.py`, `snake_rl/viewer.py`):** Terminal-based animated viewer with Heads-Up Display (HUD), supporting both Q-learning and DQN agents.

---

## Quickstart & Verification

### 1. Launch the Streamlit Web Dashboard
Launch the interactive browser dashboard containing Play, Train, Compare, and How It Works tabs:
```bash
.venv/bin/python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 2. Run Unit Tests
Run the complete unit test suite (76 tests covering environment, state encoding, Bellman arithmetic, DQN networks, replay buffer, baselines, session state, safe checkpointing, and rendering):
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Launch the Terminal Visualizer (CLI Alternative)
Launch the interactive terminal viewer:
```bash
.venv/bin/python viewer.py
```

Or run dedicated terminal commands:
- **Watch Live Q-Learning Training (10 episodes):**
  ```bash
  .venv/bin/python viewer.py --mode train --agent q_learning --episodes 10
  ```
- **Watch Live DQN Training (10 episodes):**
  ```bash
  .venv/bin/python viewer.py --mode train --agent dqn --episodes 10
  ```
- **Watch Trained Model Playback (Greedy Evaluation):**
  ```bash
  .venv/bin/python viewer.py --mode eval --episodes 5 --checkpoint models/q_table_500ep.json
  ```

  .venv/bin/python viewer.py --mode eval --episodes 5 --speed 0.08 --seed 42
  ```

---

## Deep Q-Network (DQN) Architecture & Design

```text
State Observation (11 features)
         │
         ▼
Online Network: Linear(11, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 3)
         │
         ├──► Q(s, a) minibatches
         │
Target Network: [Identical architecture, frozen weights, requires_grad = False]
         │
         └──► r + gamma * max_a' Q_target(s', a') * (1 - done)
                   │
                   ▼
         Huber Loss (SmoothL1Loss) + Adam Optimizer (lr=0.001)
                   │
         Synchronize target network every 500 gradient steps
```

### Key Technical Pillars:
1. **Target Network Freezing:** Prevents the moving-target instability inherent to bootstrapping Q-values. Synchronized from online network every 500 training steps.
2. **Experience Replay Buffer:** Fixed capacity (10,000 transitions) with FIFO eviction. Minibatch sampling ($B=64$) breaks temporal autocorrelation between sequential frames.
3. **Huber Loss (`nn.SmoothL1Loss`):** Quadratic for small errors, linear for large errors; provides robustness against outlier gradient spikes.
4. **Strict CPU Compatibility:** Runs efficiently on CPU without requiring GPU or CUDA dependencies.
5. **Exact Terminal Target Formulation:** When an episode terminates (`done = True`), the target is strictly the reward without bootstrapping:
   $$y = r + (1 - d) \cdot \gamma \cdot \max_{a'} Q_{\text{target}}(s', a')$$

---

## Interactive Viewer Controls & Metrics

### In-Flight Controls
- **`Space` or `Enter`:** Pause / Resume playback.
- **`1` / `f`:** Fast speed ($10\text{ ms}$ per step).
- **`2` / `n`:** Normal speed ($40\text{ ms}$ per step).
- **`3` / `s`:** Slow speed ($120\text{ ms}$ per step).
- **`q`:** Stop playback safely.

### HUD Metrics Explained
- **`Episode` & Progress bar:** Current episode number and completion percentage.
- **`Score`:** Total food items eaten in the current episode.
- **`Steps`:** Actions taken before the episode concludes.
- **`Reward`:** Cumulative scalar reward ($+10.0$ for food, $-10.0$ for collision, $-0.1$ for movement).
- **`Rolling Avg Score`:** 10-episode trailing moving average score.
- **`Epsilon`:** Current exploration probability ($1.0 \rightarrow 0.01$).
- **`Decision Badge`:** 
  - `🎲 EXPLORING (Random Action)`: The agent made a random exploratory choice.
  - `🧠 EXPLOITING (Greedy Q-Value)`: The agent chose the highest Q-value action from its network/table.

---

## Training vs. Evaluation

| Feature | Training Mode | Evaluation Mode |
| :--- | :--- | :--- |
| **Exploration ($\epsilon$)** | Active ($\epsilon > 0$, decays per episode) | Disabled ($\epsilon = 0.0$, pure exploitation) |
| **Model Updates** | Active (Bellman updates applied each step) | Frozen (zero weight/value modifications) |
| **Purpose** | Learning through trial and error | Assessing true performance on fixed seeds |

---

## Benchmark Results (20 Fixed Evaluation Seeds)

| Agent | Mean Score | Max Score | Mean Steps | Mean Reward |
| :--- | :--- | :--- | :--- | :--- |
| **Random Agent** | **0.05** | 1 | 74.5 | -16.84 |
| **Q-Learning (500 episodes)** | **20.95 – 22.60** | **49** | 303.5 – 328.7 | +171.34 – +185.49 |
| **Heuristic Agent (Rule-based)** | **30.20** | 65 | 448.1 | +250.31 |

---

## Checkpoint Management

- Tabular Q-learning checkpoints are serialized to `models/q_table_*.json`.
- DQN neural network models are serialized to `models/dqn_*.pt` with architecture metadata.
- `models/` is excluded from Git version control by `.gitignore` to avoid repository bloat.
- When finishing an interactive session in `viewer.py`, you will be prompted whether to save the model.
- The visualizer **never overwrites** existing checkpoints without explicit user confirmation.

---

## Roadmap & Status

- [x] **Step 0:** Workspace & Safety Inspection
- [x] **Step 1:** Headless Snake Environment & Unit Tests
- [x] **Step 2:** 11-Bit State Representation & Action Mapping
- [x] **Step 3:** Tabular Q-Learning Implementation
- [x] **Step 4:** Training Infrastructure, Baselines & Evaluation
- [x] **Step 4.5:** Visual Playback & Checkpoint Verification
- [x] **Step 5:** Deep Q-Networks (DQN) with PyTorch & Replay Buffer
- [x] **Step 6:** Streamlit UI for Web Demos
- [ ] **Step 7:** Deployment Preparation
- [ ] **Step 8:** Controlled Experiments & Comparative Analysis


