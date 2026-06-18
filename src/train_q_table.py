import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import pickle
import numpy as np
import config
from src.environment import FootballEnv
from src.q_table_agent import QTableAgent


def train(n_snapshots=10):
    env = FootballEnv(level=config.LEVEL)
    agent = QTableAgent(n_actions=env.n_actions)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    snap_every = max(config.N_EPISODES // n_snapshots, 1)
    snapshots = []
    recent_goals = []
    recent_rewards = []
    window = min(snap_every, 50)

    log = []
    print(f"Q-Table training  |  Level {config.LEVEL}  |  {config.N_EPISODES} episodes")
    print(f"Snapshots: alle {snap_every} Episoden  ({n_snapshots} gesamt)")
    print("-" * 50)

    for episode in range(1, config.N_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        goal_scored = False

        for _ in range(env.max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                if reward > 0:
                    goal_scored = True
                break

        agent.decay_epsilon()
        recent_goals.append(float(goal_scored))
        recent_rewards.append(total_reward)
        if len(recent_goals) > window:
            recent_goals.pop(0)
            recent_rewards.pop(0)

        log.append({
            "episode": episode,
            "reward": round(total_reward, 2),
            "steps": env.step_count,
            "epsilon": round(agent.epsilon, 4),
            "goal": goal_scored,
            "q_table_size": len(agent.q_table),
        })

        if episode % snap_every == 0 or episode == 1:
            goal_rate = np.mean(recent_goals) * 100
            avg_reward = np.mean(recent_rewards)
            snapshots.append({
                "episode":    episode,
                "q_table":    copy.deepcopy(dict(agent.q_table)),
                "epsilon":    agent.epsilon,
                "goal_rate":  goal_rate,
                "avg_reward": avg_reward,
            })

        if episode % 500 == 0 or episode == 1:
            recent = log[-min(500, episode):]
            avg_r = sum(e["reward"] for e in recent) / len(recent)
            goal_rate = sum(e["goal"] for e in recent) / len(recent) * 100
            print(
                f"Ep {episode:5d}/{config.N_EPISODES}"
                f"  avg_reward={avg_r:7.1f}"
                f"  goal%={goal_rate:5.1f}"
                f"  eps={agent.epsilon:.3f}"
                f"  states={len(agent.q_table)}"
            )

    stem = f"q_table_level{config.LEVEL}_ep{config.N_EPISODES}"
    model_path = os.path.join(config.MODELS_DIR, f"{stem}.pkl")
    log_path   = os.path.join(config.LOGS_DIR,   f"{stem}.json")
    snap_path  = os.path.join(config.MODELS_DIR, f"{stem}_snapshots.pkl")

    agent.save(model_path)
    with open(log_path, "w") as f:
        json.dump(log, f)
    with open(snap_path, "wb") as f:
        pickle.dump(snapshots, f)

    print("-" * 50)
    print(f"Model saved      →  {model_path}")
    print(f"Log saved        →  {log_path}")
    print(f"Snapshots saved  →  {snap_path}  ({len(snapshots)} Snapshots)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=None,
                    help="Überschreibt config.LEVEL (1, 2 oder 3)")
    ap.add_argument("--n-snapshots", type=int, default=10,
                    help="Anzahl der Zwischenstände für animate_training.py (Standard: 10)")
    args = ap.parse_args()
    if args.level is not None:
        config.LEVEL = args.level
    train(n_snapshots=args.n_snapshots)
