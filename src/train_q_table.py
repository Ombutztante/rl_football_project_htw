import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import config
from src.environment import FootballEnv
from src.q_table_agent import QTableAgent


def train():
    env = FootballEnv(level=config.LEVEL)
    agent = QTableAgent(n_actions=env.n_actions)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    log = []
    print(f"Q-Table training  |  Level {config.LEVEL}  |  {config.N_EPISODES} episodes")
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
                # Goal check: episode ended before max_steps with positive terminal reward
                if reward > 0:
                    goal_scored = True
                break

        agent.decay_epsilon()

        log.append({
            "episode": episode,
            "reward": round(total_reward, 2),
            "steps": env.step_count,
            "epsilon": round(agent.epsilon, 4),
            "goal": goal_scored,
            "q_table_size": len(agent.q_table),
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

    # Save model and log — filename encodes level + episode count for easy comparison
    stem = f"q_table_level{config.LEVEL}_ep{config.N_EPISODES}"
    model_path = os.path.join(config.MODELS_DIR, f"{stem}.pkl")
    log_path = os.path.join(config.LOGS_DIR, f"{stem}.json")

    agent.save(model_path)
    with open(log_path, "w") as f:
        json.dump(log, f)

    print("-" * 50)
    print(f"Model saved  →  {model_path}")
    print(f"Log saved    →  {log_path}")


if __name__ == "__main__":
    train()
