# Project Specification: Snake AI — Q-learning to DQN

## 1. Project goal

Build an educational, reproducible Reinforcement Learning project where a Snake-playing agent learns through trial and error. Compare tabular Q-learning against DQN, explain the design decisions, and expose training, playback, and evaluation through a Streamlit web app.

The project is for learning and portfolio demonstration. It is not intended to control an operating-system process or to use a prebuilt RL training framework.

## 2. Scope

### Required
- A deterministic, testable Snake environment.
- A clear state representation and a documented action space.
- Explicit reward and episode-termination rules.
- Tabular Q-learning implemented by us.
- DQN implemented by us at the RL-algorithm level, using PyTorch for neural-network calculations and optimization.
- DQN replay buffer and target-network update logic implemented by us.
- Training loops, checkpoints, evaluation mode, and metrics.
- Streamlit UI for game playback, training controls, metrics, and Q-learning/DQN comparison.
- Unit tests for environment rules and algorithm logic.
- README with setup, commands, architecture, methodology, limitations, and deployment instructions.

### Explicitly out of scope for the initial version
- Manually implementing matrix multiplication or neural-network backpropagation.
- Using Stable-Baselines3, RLlib, or another library that supplies the RL algorithm.
- Training in the browser.
- Connecting to or controlling the real Snake game in another application.
- Claiming DQN is better without measured results.
- Requiring a GPU; CPU training must be supported.

## 3. Technology choices

- Python 3.11 or a compatible Python version supported by the selected package versions.
- NumPy: Q-table, state encoding, arrays, and numerical utilities.
- PyTorch: DQN network, autograd, loss, and optimizer.
- Streamlit: user interface and hosted demo.
- Matplotlib: plots.
- pytest: tests.
- Optional: Pillow for rendering a grid to an image. Prefer a renderer that works on Streamlit Community Cloud without needing a desktop display server.

Important: Do not make Pygame a required dependency for the hosted interface unless there is a clear, tested reason. The core environment should be headless and independent of UI libraries. Streamlit should display frames/images produced by the environment, not depend on a desktop window.

## 4. Game rules

- Use a configurable grid, initially 20 × 20.
- The snake is represented by ordered grid coordinates; the first coordinate is the head.
- Food spawns uniformly in an unoccupied cell.
- A normal move advances the head one cell and shifts the body.
- Eating food increases score and snake length; the tail does not move on that step.
- The episode terminates on wall collision, self-collision, or a configurable step limit to prevent endless loops.
- Handle collision with the current tail correctly: moving into the tail's cell may be legal if that tail cell vacates on the same step; it is not legal when the snake is eating and the tail does not vacate.
- The game logic must be testable without rendering.

## 5. Initial agent design

### Actions
Use three relative actions:
- `STRAIGHT`
- `TURN_RIGHT`
- `TURN_LEFT`

Convert the relative action to an absolute movement direction based on the current direction. The snake must never make an instant 180-degree reversal.

### Initial Q-learning state
Start with a compact, documented 11-bit state:
1. Danger straight
2. Danger right
3. Danger left
4–7. Moving left, right, up, down (one-hot)
8–11. Food left, right, up, down

Encode the 11 binary features as an integer from 0 through 2047, giving 2048 possible indices and 3 actions. Q-table shape: `(2048, 3)`.

State encoding must be deterministic and unit-tested. Document the limitation: this compact state does not encode the complete board and may not distinguish states that require different actions.

### DQN state
For the first DQN version, use the same 11-feature vector as float32 input to keep the comparison controlled. The network outputs three Q-values. This makes it easy to compare methods but does not guarantee DQN outperforms tabular Q-learning. A richer state or board representation can be a later experiment, not a prerequisite.

## 6. Reward and episode rules

Start with these configurable values:
- Food eaten: `+10`
- Collision/game over: `-10`
- Normal move: `-0.1`

The environment returns a structured step result containing at least: next state/game observation, reward, done flag, score, and reason if done. Keep game mechanics separate from agent-specific learning updates.

Do not add many reward terms initially. Record the reward configuration in the README and allow it to be changed through a config object. Add tests to verify reward and termination semantics.

## 7. Initial hyperparameters (starting points, not claimed optimal)

### Q-learning
- learning rate alpha: `0.1`
- discount gamma: `0.9`
- epsilon start: `1.0`
- epsilon minimum: `0.01`
- epsilon decay: configurable; initial candidate `0.995` per episode
- Initial Q-values: zero

Ensure epsilon schedule is implemented consistently and logged. If per-episode decay is too fast or slow, tune it using recorded experiments.

### DQN
- MLP input: 11 features
- Hidden layers: 128, 128
- Hidden activation: ReLU
- Output: 3 Q-values
- Optimizer: Adam
- Learning rate: start at `0.001`, tune if unstable
- Gamma: start at `0.9`
- Replay buffer capacity: `10_000`
- Batch size: `64`
- Target-network hard update: initial candidate every `500` gradient updates
- Epsilon start: `1.0`
- Epsilon minimum: `0.01`
- Use configurable warm-up before gradient updates; don't train until enough replay samples exist.
- Use a fixed random seed option for reproducibility.

These are starting configurations, not promises of good performance. Make them configurable and run controlled experiments.

## 8. Training and evaluation

- Keep training separate from rendering and UI.
- Support saving/loading Q-table and DQN checkpoints.
- Log per-episode score, total reward, steps, epsilon, and training loss (for DQN).
- Provide moving-average metrics with a configurable window.
- Evaluation must use `epsilon=0` (greedy policy), disable weight updates, and use fixed sets of random seeds.
- Compare random or simple heuristic policy, Q-learning, and DQN where available.
- Use the same grid rules and evaluation episodes for all agents.
- Report mean and variability across multiple seeds; do not report only the best game.
- Do not hardcode or fabricate benchmark results. Results must come from actual runs.
- CPU training must work. Keep the default training batch/run sizes small enough for a hosted demo; provide sensible limits in Streamlit.

## 9. Streamlit interface requirements

Suggested pages/tabs:
1. **Play** — watch a trained agent play; select agent/checkpoint; show score and steps; start/reset playback.
2. **Train** — choose Q-learning or DQN, configure safe hyperparameters, train a bounded number of episodes, show progress.
3. **Compare** — compare recorded evaluations using tables and charts.
4. **How it works** — explain state, actions, reward, Q-learning, DQN, replay buffer, and target network in beginner-friendly language.

The interface must not run a long unbounded training job on every Streamlit rerun. Use explicit buttons, bounded episode limits, and appropriate Streamlit session state/caching. Avoid relying on infinite loops that prevent UI interaction. Keep the core environment independent of Streamlit.

For a simple first UI, display the grid as a rendered image updated at controlled intervals or render a final episode/trajectory. Do not promise real-time 60 FPS animation if Streamlit architecture does not support it reliably.

## 10. Acceptance criteria

- Environment tests pass for movement, food, growth, wall collision, self-collision, no-food spawn, and step limit.
- Q-learning update is tested on a small deterministic example.
- DQN forward pass returns shape `(batch_size, 3)` and training produces finite loss.
- Replay buffer stores and samples valid transitions.
- Target network updates only according to configured schedule.
- Checkpoints can be saved and loaded.
- Evaluation does not update model weights and uses greedy actions.
- Streamlit app launches locally with the documented command.
- A small training run completes on CPU.
- No fabricated performance claims.
- Deployment instructions work with a GitHub repository and Streamlit Community Cloud.
