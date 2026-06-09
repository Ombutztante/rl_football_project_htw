# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement Learning project for the Computational Intelligence course at HTW Berlin. The project implements and compares two RL approaches — tabular Q-Learning and DQN (PyTorch) — on a custom 2D gridworld football environment. The goal is didactic: demonstrate RL concepts from the lecture (Q-table, ε-greedy, exploration/exploitation) in a football context, analogous to the maze example from class.

## Environment Setup

Uses Python 3.11 with a local virtualenv at `.venv`:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

```bash
# Run Q-Table training
python src/train_q_table.py

# Run DQN training
python src/train_dqn.py

# Run tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_environment.py

# Launch Jupyter for experiments
jupyter notebook notebooks/experiments.ipynb
```

## Game Design: Three Stages of Complexity

The environment is a 2D grid (default 8×6). Complexity grows across three stages, each adding to the previous.

**Stage 1 — Basic navigation**
- Agent navigates to the ball (intermediate reward), then to the goal (win).
- State: `(agent_x, agent_y, ball_x, ball_y)`
- Actions: up, down, left, right (4 total)
- Ball stays fixed; goal is fixed at the right edge (center row).

**Stage 2 — Ball possession and shooting**
- Agent must pick up ball, carry it toward goal, and shoot.
- Adds `has_ball` to state: `(agent_x, agent_y, ball_x, ball_y, has_ball)`
- Adds `shoot` action (5 total). Shoot sends ball right; goal if agent is aligned with goal row.
- Reward design becomes central: ball pickup, direction-toward-goal, scoring, missed shots.

**Stage 3 — Opponent**
- A rule-based opponent (moves toward the ball) is added.
- Adds opponent position to state: `(agent_x, agent_y, ball_x, ball_y, has_ball, opp_x, opp_y)`
- Penalties for opponent contact and ball loss.

The active stage is set via `config.py` (`STAGE = 1 | 2 | 3`).

## Architecture

The environment API (Gym-style) is the stable interface between all agents:

```python
state = env.reset()
action = agent.choose_action(state)
next_state, reward, done = env.step(action)
agent.learn(state, action, reward, next_state)
```

**Agents are interchangeable** — switching between Q-Table and DQN does not require changing the environment.

- `src/environment.py` — Gridworld: state, action, reward, step logic, ASCII render. No gym dependency.
- `src/q_table_agent.py` — Tabular Q-Learning using a `defaultdict` as Q-table. State tuple is used directly as key.
- `src/dqn_agent.py` — DQN with PyTorch. Neural net takes normalized state vector as input, outputs Q-value per action. Includes replay buffer and target network.
- `src/train_q_table.py` — Episode loop for Q-Table agent; saves logs and model.
- `src/train_dqn.py` — Episode loop for DQN agent; saves logs and model.
- `src/utils.py` — Shared: `ReplayBuffer`, seeding helpers.
- `src/visualize.py` — Reward/steps/epsilon plots; Q-Table vs DQN comparison plot.
- `config.py` — All hyperparameters (grid size, stage, learning rate, gamma, epsilon schedule, batch size, etc.) and output paths.

**Results** written to `results/`:
- `results/models/` — Q-table `.pkl` files or PyTorch `.pt` checkpoints
- `results/logs/` — Per-episode training metrics (JSON)
- `results/plots/` — Matplotlib figures (PNG)

**Documentation** in `docs/`:
- `entscheidungen.md` — Design and algorithm decisions
- `projektplan.md` — Project plan and timeline
- `vortrag_notizen.md` — Notes for the final presentation

## Key Design Decisions

**Agent vs. DQN state representation:** The Q-table uses the raw state tuple as key (no normalization needed). The DQN uses a normalized float array (`env.state_to_array(state)`), which also includes the fixed goal position (as per spec). `env.get_state_size()` returns the DQN input dimension.

**DQN input (per spec):** player x/y, ball x/y, has_ball, goal x/y, opponent x/y (stage 3). All coordinates normalized to [0, 1].

**Algorithm decision (still open):** Q-Table is the baseline (close to lecture content, reliable for small grids). DQN via PyTorch is the extension (justifies PyTorch dependency, shows where Q-tables hit limits). Final weighting between the two is to be determined.

**Stage determines n_actions:** Stage 1 → 4 actions (no shoot). Stage 2+ → 5 actions (adds shoot).
