import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import matplotlib.pyplot as plt
import config


def _rolling(values, window=100):
    """Simple rolling mean; pads early episodes with expanding window."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start : i + 1]))
    return np.array(result)


def _load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def plot_training(log_path, title=None, save_path=None, window=100):
    """
    Three-panel training overview for a single agent:
      1. Episode reward + rolling mean
      2. Rolling goal rate (%)
      3. Epsilon decay  (+ avg loss if DQN log)
    """
    log = _load_log(log_path)
    episodes = [e["episode"] for e in log]
    rewards  = [e["reward"]  for e in log]
    goals    = [e["goal"]    for e in log]
    epsilons = [e["epsilon"] for e in log]
    has_loss = log[0].get("avg_loss") is not None

    n_panels = 4 if has_loss else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)
    fig.suptitle(title or os.path.basename(log_path), fontsize=13, fontweight="bold")

    # 1 — Reward
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.25, linewidth=0.8, color="steelblue", label="pro Episode")
    ax.plot(episodes, _rolling(rewards, window), linewidth=2, color="steelblue", label=f"Ø {window} Ep.")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # 2 — Goal rate
    ax = axes[1]
    goal_rate = _rolling([float(g) * 100 for g in goals], window)
    ax.plot(episodes, goal_rate, linewidth=2, color="seagreen")
    ax.set_ylabel(f"Tor-Rate % (Ø {window})")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # 3 — Epsilon
    ax = axes[2]
    ax.plot(episodes, epsilons, linewidth=1.5, color="tomato")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if not has_loss:
        ax.set_xlabel("Episode")

    # 4 (DQN only) — Loss
    if has_loss:
        losses = [e["avg_loss"] if e["avg_loss"] is not None else 0.0 for e in log]
        ax = axes[3]
        ax.plot(episodes, losses, alpha=0.35, linewidth=0.8, color="darkorange", label="pro Episode")
        ax.plot(episodes, _rolling(losses, window), linewidth=2, color="darkorange", label=f"Ø {window} Ep.")
        ax.set_ylabel("MSE Loss")
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"Plot gespeichert → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_comparison(q_log_path, dqn_log_path, level=None, save_path=None, window=100):
    """
    Side-by-side comparison of Q-Table vs DQN:
      1. Rolling average reward
      2. Rolling goal rate (%)
    """
    q_log   = _load_log(q_log_path)
    dqn_log = _load_log(dqn_log_path)

    q_ep  = [e["episode"] for e in q_log]
    q_r   = _rolling([e["reward"] for e in q_log], window)
    q_gr  = _rolling([float(e["goal"]) * 100 for e in q_log], window)

    d_ep  = [e["episode"] for e in dqn_log]
    d_r   = _rolling([e["reward"] for e in dqn_log], window)
    d_gr  = _rolling([float(e["goal"]) * 100 for e in dqn_log], window)

    lv = f"Level {level}" if level else ""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Q-Table vs. DQN  {lv}  (Ø {window} Episoden)", fontsize=13, fontweight="bold")

    axes[0].plot(q_ep, q_r,  linewidth=2, color="steelblue", label="Q-Table")
    axes[0].plot(d_ep, d_r,  linewidth=2, color="darkorange", label="DQN")
    axes[0].axhline(0, color="gray", linewidth=0.6, linestyle="--")
    axes[0].set_ylabel("Ø Reward")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(q_ep, q_gr, linewidth=2, color="steelblue", label="Q-Table")
    axes[1].plot(d_ep, d_gr, linewidth=2, color="darkorange", label="DQN")
    axes[1].set_ylabel(f"Tor-Rate %")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"Plot gespeichert → {save_path}")
    else:
        plt.show()
    plt.close()


def _ep_count(log_path):
    """Read the episode count from the last entry of a log file."""
    log = _load_log(log_path)
    return log[-1]["episode"]


def _plot_stem(log_path):
    """Derive a plot filename stem from the log filename (replaces .json with .png)."""
    return os.path.splitext(os.path.basename(log_path))[0]


def _find_latest_log(logs_dir, pattern):
    """Return the most recently modified log matching glob pattern, or None."""
    import glob as _glob
    matches = _glob.glob(os.path.join(logs_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _find_log(logs_dir, filename):
    """Return exact log file path if it exists, else None."""
    path = os.path.join(logs_dir, filename)
    return path if os.path.exists(path) else None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--run", type=str, default=None,
                    help="Run-Verzeichnis (z.B. 'a_dev_1_2206_1'). Standard: neuester Run.")
    _args = ap.parse_args()

    if _args.run:
        config.set_run_dir(_args.run)
    else:
        run = config.latest_run()
        if run:
            config.set_run_dir(run)
            print(f"Verwende neuesten Run: {run}")

    level = _args.level if _args.level is not None else config.LEVEL
    ep    = _args.episodes if _args.episodes is not None else config.N_EPISODES

    # In a run dir files have no date suffix; fall back to glob if exact not found
    q_log = (_find_log(config.LOGS_DIR, f"q_table_level{level}_ep{ep}.json")
             or _find_latest_log(config.LOGS_DIR, f"q_table_level{level}_ep{ep}_*.json"))
    dqn_log = (_find_log(config.LOGS_DIR, f"dqn_level{level}_ep{ep}.json")
               or _find_latest_log(config.LOGS_DIR, f"dqn_level{level}_ep{ep}_*.json"))

    if q_log:
        stem = _plot_stem(q_log)
        plot_training(
            q_log,
            title=f"Q-Table – Level {level} – {_ep_count(q_log)} Episoden",
            save_path=os.path.join(config.PLOTS_DIR, f"{stem}.png"),
        )
    else:
        print(f"Kein Q-Table-Log gefunden für Level {level}, ep{ep}.")

    if dqn_log:
        stem = _plot_stem(dqn_log)
        plot_training(
            dqn_log,
            title=f"DQN – Level {level} – {_ep_count(dqn_log)} Episoden",
            save_path=os.path.join(config.PLOTS_DIR, f"{stem}.png"),
        )
    else:
        print(f"Kein DQN-Log gefunden für Level {level}, ep{ep}.")

    if q_log and dqn_log:
        import re as _re
        q_stem = _plot_stem(q_log)
        cmp_stem = _re.sub(r"^q_table", "comparison", q_stem)
        plot_comparison(
            q_log, dqn_log,
            level=level,
            save_path=os.path.join(config.PLOTS_DIR, f"{cmp_stem}.png"),
        )
