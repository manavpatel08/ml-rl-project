# Instructions for Antigravity

You are helping build an educational Snake Reinforcement Learning project. Treat the user as a learner who wants every step and design decision explained clearly.

## Non-negotiable working rules

1. Work in small steps. Implement only the step explicitly requested. Do not jump ahead to DQN or build the entire app in one pass.
2. Before coding each step, briefly state what you will do and why.
3. Inspect the current repository and git status before modifying files. Preserve unrelated user work.
4. Prefer simple, readable, typed Python where useful. Avoid needless abstractions.
5. The core environment and algorithms must be testable without Streamlit or rendering.
6. Implement RL algorithms ourselves. Do not use Stable-Baselines3, RLlib, or another library that provides Q-learning/DQN.
7. Use NumPy for Q-table and numerical utilities. Use PyTorch for DQN layers, autograd, loss, and optimizer. Do not manually implement neural-network backpropagation.
8. Do not fabricate test results, learning results, charts, or performance claims.
9. After code changes, run the relevant tests or clearly state why they could not be run. Never claim tests passed if they were not run.
10. Do not silently change the state representation, action space, reward rules, or hyperparameters. Explain proposed changes and their effects.
11. Avoid unbounded training loops in Streamlit. Training must have explicit bounded episode/step limits and must not run on every app rerun.
12. Keep deployment compatibility in mind. The environment must be headless. Do not require a desktop display server for the hosted app.
13. Do not install packages, delete files, reset git, or make commits/pushes without explaining the action first and obtaining approval when it could affect user data or repository state.
14. Do not proceed to the next roadmap step until the user reviews and approves the current step.

## Response format after each step

- **What we did**
- **Why we did it**
- **How it works**
- **Files changed**
- **How to run**
- **Tests and results**
- **Limitations / next step**

Explain unfamiliar terms in plain language. When introducing a formula, explain each variable and why it is used. The user does not want mathematical derivations for neural-network backpropagation, but does want the RL algorithm logic and the reason for each design decision explained.

## Current task

Start at Step 0 or Step 1 from `IMPLEMENTATION_ROADMAP.md`, depending on the user's instruction. First inspect the existing workspace and report what is already present. Do not implement the full project in one request.
