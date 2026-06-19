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

## Game Design: Five Levels of Complexity

The environment is a 2D grid (default 10×6, configured in `config.py`). Complexity grows across five levels, each adding to the previous. **All levels share the same 5 actions** so agent code stays clean across levels. Levels are trained independently (no progressive transfer).

**Actions (all levels):** 0=up, 1=down, 2=left, 3=right, 4=shoot

---

**Level 1 — Shoot only from a good position**

The agent must learn: *get ball → reach shooting zone → shoot*. The RL challenge is that `shoot` is only useful in certain states — the agent must learn situational action selection, not just navigation.

- State: `(agent_x, agent_y, ball_x, ball_y, has_ball)` — 5 elements
- Shooting zone: columns near the goal (e.g. `agent_x >= width - 2`)
- Rewards:
  - `+30` goal scored
  - `+5` ball picked up
  - `+1` agent/ball moved closer to goal
  - `-1` per step
  - `-5` shoot without ball
  - `-5` shoot from bad position (outside shooting zone)

---

**Level 2 — Dribbling vs. forward pass**

A real decision emerges: safe dribbling (slow) vs. fast but risky forward pass.

- State: `(agent_x, agent_y, ball_x, ball_y, has_ball)` — 5 elements
- **Dribbling:** agent moves 1 field but with ball incurs a time penalty (movement costs 2 steps effectively, e.g. every 2nd move is skipped or costs `-2`)
- **Shoot (forward pass):** ball travels 2–3 fields toward goal, agent loses possession and must chase
- Ball out of bounds: `-5` penalty, ball lost
- Rewards:
  - `+40` goal scored
  - `+5` ball picked up
  - `+2` ball moved closer to goal via forward pass
  - `+1` ball moved closer to goal via dribbling
  - `-1` per step
  - `-5` ball shot out of bounds
  - `-3` unnecessary/bad shot

---

**Level 3 — Opponent moves toward ball**

A dynamic opponent is added that moves 1 field toward the ball every 2nd turn. If the opponent reaches the ball, a penalty is applied.

- State: `(agent_x, agent_y, ball_x, ball_y, has_ball, opp_x, opp_y)` — 7 elements
- Opponent: rule-based, starts at `(goal_x − OPP_START_X_FROM_GOAL, 0)` (top row), moves every 2nd step toward ball
- Reaching ball = penalty; episode may end on ball loss
- Rewards:
  - `+50` goal scored
  - `+5` ball picked up
  - `+2` ball moved closer to goal
  - `-1` per step
  - `-10` opponent reaches ball
  - `-20` ball lost to opponent
  - `-5` bad shot

---

**Level 4 — Static obstacle blocks direct path (extends Level 3)**

A vertical obstacle wall is added at column `OBSTACLE_X=6`, blocking rows 0–3. Rows 4–5 remain free for navigation below. Shots that hit the obstacle are blocked; the agent must navigate around the obstacle or shoot from a different angle.

- State: `(agent_x, agent_y, ball_x, ball_y, has_ball, opp_x, opp_y)` — 7 elements (same as Level 3)
- Obstacle: static, `OBSTACLE_X=6`, rows `OBSTACLE_Y_START=0` to `OBSTACLE_Y_START + OBSTACLE_HEIGHT - 1=3`
- Rewards:
  - `+60` goal scored
  - `+5` ball picked up
  - `+2` ball moved closer to goal via pass
  - `+1` moved closer to goal
  - `-1` per step
  - `-2` agent walks into obstacle
  - `-5` shot blocked by obstacle
  - `-10` opponent reaches ball
  - `-20` ball lost to opponent

---

**Level 5 — Cooperative play with teammate (extends Level 3, no obstacle)**

A rule-based teammate is added. The opponent starts mid-field blocking the direct dribble path. The agent must cooperate: action 4 outside the shooting zone passes the ball diagonally toward the teammate (ball travels `PASS_SPEED_L5=2` cells per step while visible on the field); inside the shooting zone it shoots directly (Level-1 style). The teammate positions itself to receive passes and then dribbles autonomously to score.

- State: `(agent_x, agent_y, ball_x, ball_y, has_ball, opp_x, opp_y, tm_x, tm_y, tm_has_ball)` — 10 elements
- Opponent: starts at `(OPP_START_X_L5=6, OPP_START_Y_L5=3)`, blocking the centre path
- Teammate: starts at `(TM_START_X_L5=5, TM_START_Y_L5=0)`, moves toward ball or receiving position
- Pass mechanic: ball travels diagonally (both x and y per step) — avoids Manhattan path that would go through the opponent
- Rewards:
  - `+70` goal scored (agent shoots OR teammate scores)
  - `+15` teammate picks up the passed ball
  - `+5` ball picked up
  - `+1` moved closer to goal
  - `-1` per step
  - `-5` shoot/pass without ball
  - `-10` opponent reaches ball
  - `-20` ball lost to opponent

---

The active level is set via `config.py` (`LEVEL = 1 | 2 | 3 | 4 | 5`).

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

**DQN input (normalised float array, all coordinates divided by grid size − 1):**
- Level 1+2 (7 values): ax, ay, bx, by, has_ball, gx, gy
- Level 3+4 (9 values): + opp_x, opp_y
- Level 5   (12 values): + opp_x, opp_y, tm_x, tm_y, tm_has_ball

**Algorithm decision:** Q-Table is the baseline (close to lecture content, reliable and stable on small grids). DQN is the extension — it converges comparably on Level 1+2, but shows characteristic instability on Level 3 (Q-value drift after ε reaches minimum), demonstrating where tabular methods are actually more robust.

**n_actions is always 5** across all levels (up, down, left, right, shoot). The meaning of `shoot` expands per level: Level 1 = score if in shooting zone; Level 2/3/4 = forward pass (ball travels `SHOOT_RANGE` cells right, agent chases); Level 5 = pass to teammate (ball travels diagonally, `PASS_SPEED_L5` cells/step) or direct shot if in shooting zone.
