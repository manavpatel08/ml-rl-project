# Snake RL — Documentation Pack

This pack is the project brief for building a Snake AI with tabular Q-learning and Deep Q-Networks (DQN), with a Streamlit interface for demos and evaluation.

## Read these files in order

1. `PROJECT_SPEC.md` — what we are building and what counts as done.
2. `ARCHITECTURE.md` — components and responsibilities.
3. `IMPLEMENTATION_ROADMAP.md` — build in small, verifiable steps.
4. `ANTIGRAVITY_WORKING_RULES.md` — how the coding agent should work with us.
5. `STREAMLIT_DEPLOYMENT.md` — hosting and deployment constraints.

## Important project decision

“From scratch” means we implement the game environment, state/action/reward logic, tabular Q-learning, DQN algorithm flow, replay buffer, target-network update logic, training/evaluation loops, metrics, and hyperparameter experiments ourselves. We will use NumPy for numerical arrays and PyTorch for neural-network layers, automatic differentiation, loss, and optimizer operations. We will not manually derive or code neural-network backpropagation.

## Working style

Do not generate the whole project in one shot. Work one step at a time. After each step, explain:
- What changed
- Why it is needed
- How it works
- Files created/modified
- Commands to run
- Tests performed and results
- Known limitations / next step

Wait for the user to review the step before proceeding.
