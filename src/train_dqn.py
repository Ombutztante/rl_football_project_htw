import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import config
from src.environment import FootballEnv
from src.dqn_agent import DQNAgent


def train():
    env = FootballEnv(level=config.LEVEL)
    agent = DQNAgent(
        state_size=env.get_state_size(),
        n_actions=env.n_actions,
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    log = []
    print(f"DQN training  |  Level {config.LEVEL}  |  {config.N_EPISODES} episodes  |  device={agent.device}")
    print("-" * 60)

    for episode in range(1, config.N_EPISODES + 1):
        state = env.reset()
        state_arr = env.state_to_array(state)
        total_reward = 0
        episode_losses = []
        goal_scored = False

        for _ in range(env.max_steps):
            action = agent.choose_action(state_arr)
            next_state, reward, done = env.step(action)
            next_arr = env.state_to_array(next_state)

            agent.store(state_arr, action, reward, next_arr, done)
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            state_arr = next_arr
            total_reward += reward

            if done:
                if reward > 0:
                    goal_scored = True
                break

        agent.decay_epsilon()

        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else None
        log.append({
            "episode": episode,
            "reward": round(total_reward, 2),
            "steps": env.step_count,
            "epsilon": round(agent.epsilon, 4),
            "goal": goal_scored,
            "avg_loss": round(avg_loss, 4) if avg_loss is not None else None,
        })

        if episode % 500 == 0 or episode == 1:
            recent = log[-min(500, episode):]
            avg_r = sum(e["reward"] for e in recent) / len(recent)
            goal_rate = sum(e["goal"] for e in recent) / len(recent) * 100
            loss_vals = [e["avg_loss"] for e in recent if e["avg_loss"] is not None]
            avg_l = sum(loss_vals) / len(loss_vals) if loss_vals else 0.0
            print(
                f"Ep {episode:5d}/{config.N_EPISODES}"
                f"  avg_reward={avg_r:7.1f}"
                f"  goal%={goal_rate:5.1f}"
                f"  loss={avg_l:.4f}"
                f"  eps={agent.epsilon:.3f}"
            )

    # Save model and log
    model_path = os.path.join(config.MODELS_DIR, f"dqn_level{config.LEVEL}.pt")
    log_path = os.path.join(config.LOGS_DIR, f"dqn_level{config.LEVEL}.json")

    agent.save(model_path)
    with open(log_path, "w") as f:
        json.dump(log, f)

    print("-" * 60)
    print(f"Model saved  →  {model_path}")
    print(f"Log saved    →  {log_path}")


if __name__ == "__main__":
    train()
