# Step-by-Step Implementation Roadmap

Build only one step at a time. Do not implement the whole roadmap in a single agent request.

## Step 0 — Inspect workspace and prepare project
- Inspect current files and existing git state.
- Do not overwrite unrelated work.
- Create the project folders, initial docs, dependency file, and a minimal app skeleton only if the repository is empty or the user approves.
- Confirm Python environment and give exact setup/run commands.

**Exit criteria:** structure is clear, imports work, no unrelated files modified.

## Step 1 — Implement and test the headless Snake environment
- Add grid, snake body, direction, food, reset, movement, growth, collision, score, step limit.
- No RL agent yet; no Streamlit UI beyond a placeholder.
- Add deterministic tests for game rules and edge cases.
- Keep environment independent of rendering.

**Exit criteria:** all environment tests pass; demonstrate one manually stepped episode.

## Step 2 — Add state encoding and action mapping
- Implement the 11-bit state features specified in `PROJECT_SPEC.md`.
- Implement relative action mapping for straight/right/left.
- Add tests for direction changes, feature order, integer encoding, and bounds.

**Exit criteria:** identical game states encode identically; encoded index is always 0–2047.

## Step 3 — Add tabular Q-learning
- Implement Q-table initialization, epsilon-greedy action selection, valid-action handling, terminal-aware Bellman update, epsilon schedule.
- Unit-test update arithmetic and terminal transitions.
- Do not add DQN yet.

**Exit criteria:** a short training run completes; metrics are logged; no invalid actions or NaNs.

## Step 4 — Add training metrics and baseline policies
- Add random and simple heuristic baselines.
- Log score, reward, episode length, epsilon.
- Plot raw values and moving averages.
- Add greedy evaluation mode and fixed evaluation seeds.

**Exit criteria:** reproducible comparison pipeline works; results are measured, not fabricated.

## Step 5 — Add DQN
- Implement MLP with PyTorch (not manual backpropagation).
- Implement replay buffer, epsilon-greedy action selection, Bellman targets, loss, optimizer step, target-network update.
- Add warm-up and batch-size guards.
- Unit-test shapes, finite loss, buffer sampling, target updates, and checkpoint round-trip.

**Exit criteria:** a bounded CPU training run completes and loss/metrics are recorded.

## Step 6 — Add Streamlit UI
- Add Play, Train, Compare, How it works sections/tabs.
- Use bounded training jobs, explicit user controls, session state, and cached loading where appropriate.
- Render game frames/images without requiring a desktop window.
- Display clear errors and prevent training on every rerun.

**Exit criteria:** `streamlit run app.py` works locally; UI actions do what labels promise.

## Step 7 — Deployment preparation
- Add a clean `requirements.txt` and README instructions.
- Confirm the chosen PyTorch CPU installation works in the deployment environment.
- Avoid Pygame/SDL dependencies unless verified in hosted Linux.
- Run tests and a smoke test using the same Python version planned for deployment.
- Deploy through Streamlit Community Cloud from GitHub.

**Exit criteria:** deployment build succeeds and hosted app launches.

## Step 8 — Controlled experiments and polish
- Tune hyperparameters in small controlled runs.
- Evaluate multiple random seeds and unseen episodes.
- Compare Q-learning and DQN fairly.
- Record limitations and actual results.
- Improve explanatory UI and project README.

**Exit criteria:** project is reproducible and the README describes actual evidence.

## Required response after each step

The coding assistant must explain:
1. What we are building.
2. Why we need it now.
3. Key design choices and trade-offs.
4. Files created/changed.
5. Commands to run.
6. Tests performed and results.
7. What is not done yet.
8. A concise next-step suggestion, then wait for approval.
