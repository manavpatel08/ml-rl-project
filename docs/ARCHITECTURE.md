# Architecture and Responsibilities

## High-level flow

```text
Streamlit UI
  ├── Play / playback
  ├── Training controls
  ├── Metrics and comparison
  └── Educational explanations
          |
          v
Application/service layer
  ├── Training service
  ├── Evaluation service
  └── Checkpoint / metrics handling
          |
          v
Core RL package (must not import Streamlit)
  ├── SnakeEnvironment
  ├── StateEncoder
  ├── QLearningAgent
  ├── DQNAgent
  ├── ReplayBuffer
  ├── Trainer
  └── Evaluator
          |
          v
Utilities
  ├── Config dataclasses
  ├── Checkpoint I/O
  ├── Random seed helpers
  └── Plotting / frame rendering
```

## Recommended repository structure

```text
snake-rl/
├── app.py
├── requirements.txt
├── README.md
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── ARCHITECTURE.md
│   ├── IMPLEMENTATION_ROADMAP.md
│   └── ANTIGRAVITY_WORKING_RULES.md
├── snake_rl/
│   ├── __init__.py
│   ├── config.py
│   ├── environment.py
│   ├── state.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── q_learning.py
│   │   ├── dqn.py
│   │   └── replay_buffer.py
│   ├── training.py
│   ├── evaluation.py
│   ├── checkpoints.py
│   └── rendering.py
├── tests/
│   ├── test_environment.py
│   ├── test_state.py
│   ├── test_q_learning.py
│   ├── test_replay_buffer.py
│   └── test_dqn.py
├── models/              # generated locally; do not commit large checkpoints by default
└── outputs/             # generated metrics; do not commit unless selected results are useful
```

## Dependency direction

- `environment.py` must not import Streamlit, PyTorch, or the agents.
- `state.py` should depend only on the environment data structures and standard Python/NumPy.
- `q_learning.py` may use NumPy but must not depend on Streamlit.
- `dqn.py` may use PyTorch and the replay buffer.
- `training.py` orchestrates environment and agent but must not depend on Streamlit.
- `evaluation.py` evaluates policies and reports metrics without rendering requirements.
- `rendering.py` turns environment state into an image/frame; the core game must still work without it.
- `app.py` is a thin UI layer that calls core APIs.

## Why this separation matters

It lets us test game rules without a browser, train without drawing every frame, compare algorithms in the same environment, and deploy the UI without desktop dependencies. It also prevents a Streamlit rerun from accidentally resetting or retraining the agent.
