"""
Animate the training evolution of an existing model.

Reads the snapshot file created by train_q_table.py / train_dqn.py and
renders a GIF showing how the agent's policy evolved over training.

Usage:
    python src/animate_training.py --level 1 --agent qtable
    python src/animate_training.py --level 2 --agent dqn --fps 4

Output: results/animations/training_evolution_{agent}_level{N}.gif
"""

import matplotlib
matplotlib.use("Agg")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import copy
import io
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import config
from src.environment import FootballEnv
from src.q_table_agent import QTableAgent
from src.dqn_agent import DQNAgent

ACTION_NAMES = ["Hoch", "Runter", "Links", "Rechts", "Schuss"]

C = {
    "bg":        "#0F0F1A",
    "cell":      "#1E2A4A",
    "zone":      "#3D3200",
    "goal":      "#0D3018",
    "obstacle":  "#2D2D2D",
    "agent":     "#4FC3F7",
    "pball":     "#CE93D8",
    "ball":      "#FF8A65",
    "opp":       "#EF5350",
    "teammate":  "#81C784",
    "tm_pball":  "#A5D6A7",
    "border":    "#2A3A6A",
    "goal_edge": "#66BB6A",
    "text":      "#ECF0F1",
    "text_dim":  "#78909C",
    "card_bg":   "#0F0F1A",
    "card_text": "#ECF0F1",
    "card_dim":  "#78909C",
    "bar_fill":  "#66BB6A",
    "bar_bg":    "#1E2A4A",
}


# ---------------------------------------------------------------------------
# Agent snapshots
# ---------------------------------------------------------------------------

def snapshot_qtable(agent):
    snap = QTableAgent(n_actions=5)
    snap.q_table = defaultdict(
        lambda: [0.0] * 5, copy.deepcopy(dict(agent.q_table))
    )
    snap.epsilon = 0.0
    return snap


def snapshot_dqn(agent, state_size):
    snap = DQNAgent(state_size=state_size, n_actions=5)
    snap.q_net.load_state_dict(copy.deepcopy(agent.q_net.state_dict()))
    snap.q_net.eval()
    snap.epsilon = 0.0
    return snap


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_title_card(episode, n_episodes, epsilon, goal_rate, avg_reward,
                       level, agent_label, frame_size):
    """Dark title card shown before each snapshot's gameplay sequence."""
    fig, ax = plt.subplots(figsize=(frame_size[0] / 90, frame_size[1] / 90),
                           facecolor=C["card_bg"])
    ax.set_facecolor(C["card_bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Header
    ax.text(5, 6.3, f"{agent_label}  —  Level {level}",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["card_text"])
    ax.text(5, 5.7, "Trainingsevolution", ha="center", va="center",
            fontsize=9, color=C["card_dim"])

    # Divider
    ax.axhline(5.35, xmin=0.05, xmax=0.95, color=C["card_dim"], linewidth=0.7)

    # Episode counter
    ax.text(5, 4.85, f"Episode  {episode} / {n_episodes}",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="#F0B27A")

    # Metrics
    metrics = [
        ("Tor-Rate",   f"{goal_rate:.0f} %"),
        ("Ø Reward",   f"{avg_reward:+.1f}"),
        ("Epsilon",    f"{epsilon:.3f}"),
    ]
    for i, (label, value) in enumerate(metrics):
        x = 2.0 + i * 3.0
        ax.text(x, 3.8, label, ha="center", va="center",
                fontsize=9, color=C["card_dim"])
        ax.text(x, 3.25, value, ha="center", va="center",
                fontsize=13, fontweight="bold", color=C["card_text"])

    # Progress bar
    progress = episode / n_episodes
    bar_x, bar_y, bar_w, bar_h = 1.0, 2.4, 8.0, 0.45
    ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_w, bar_h,
                               facecolor=C["bar_bg"], edgecolor="none", zorder=2))
    ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_w * progress, bar_h,
                               facecolor=C["bar_fill"], edgecolor="none", zorder=3))
    ax.text(5, bar_y + bar_h + 0.25, f"Trainingsfortschritt  {progress*100:.0f} %",
            ha="center", va="bottom", fontsize=8, color=C["card_dim"])

    # Footer
    ax.text(5, 1.35, "Folgende Frames: Greedy-Episode (epsilon = 0)",
            ha="center", va="center", fontsize=8, color=C["card_dim"],
            style="italic")

    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=C["card_bg"])
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _render_game_frame(env, action=None, step_reward=0, total_reward=0,
                       label="", episode=0, n_episodes=0, done=False):
    """Render one gameplay frame (same style as animate.py)."""
    W, H = env.width, env.height
    fig, ax = plt.subplots(figsize=(W * 1.05 + 1.5, H * 1.1 + 1.6),
                           facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, W)
    ax.set_ylim(-0.45, H + 0.85)
    ax.set_aspect("equal")
    ax.axis("off")

    gx, gy = env.goal_pos

    for row in range(H):
        dy = H - 1 - row
        for col in range(W):
            is_goal = (col == gx and row == gy)
            is_zone = (env.level == 1 and env.shoot_zone_x <= col < W - 1)
            is_obstacle = (env.level == 4 and (col, row) in env.obstacle_cells)
            fc = C["goal"] if is_goal else (C["obstacle"] if is_obstacle else (C["zone"] if is_zone else C["cell"]))
            ec = C["goal_edge"] if is_goal else ("#555555" if is_obstacle else C["border"])
            lw = 2.2 if is_goal else (1.5 if is_obstacle else 0.9)
            ax.add_patch(FancyBboxPatch(
                (col + 0.05, dy + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1,
            ))
            if is_obstacle:
                ax.text(col + 0.5, dy + 0.5, "#",
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="#888888", zorder=2)

    ax.text(gx + 0.5, H - 1 - gy + 0.5, "G",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=C["goal_edge"], zorder=3)

    if env.level == 1:
        mid = env.shoot_zone_x + (W - 1 - env.shoot_zone_x) / 2
        ax.text(mid, H + 0.62, "Schusszone",
                ha="center", va="center", fontsize=7, color="#856404",
                bbox=dict(fc=C["zone"], ec="#FFC107", boxstyle="round,pad=0.25", lw=0.8))

    tm_has_ball = getattr(env, 'tm_has_ball', False)
    if not env.has_ball and not tm_has_ball:
        bx, by = env.ball_pos
        if 0 <= bx < W and 0 <= by < H:
            bdy = H - 1 - by
            ax.add_patch(plt.Circle((bx + 0.5, bdy + 0.5), 0.27, color=C["ball"], zorder=4))
            ax.text(bx + 0.5, bdy + 0.5, "B",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white", zorder=5)

    if env.level >= 3 and env.level != 6:
        ox, oy = env.opp_pos
        if 0 <= ox < W and 0 <= oy < H:
            ody = H - 1 - oy
            ax.add_patch(FancyBboxPatch(
                (ox + 0.12, ody + 0.12), 0.76, 0.76,
                boxstyle="round,pad=0.04",
                facecolor=C["opp"], edgecolor="#7B0000", linewidth=1.8, zorder=4,
            ))
            ax.text(ox + 0.5, ody + 0.5, "X",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white", zorder=5)

    if env.level == 6:
        for opos, lbl, fc, edge in [
                (env.opp1_pos, "X", C["opp"],   "#7B0000"),
                (env.opp2_pos, "Y", "#9C27B0", "#4A0070")]:
            ox, oy = opos
            if 0 <= ox < W and 0 <= oy < H:
                ody = H - 1 - oy
                ax.add_patch(FancyBboxPatch(
                    (ox + 0.12, ody + 0.12), 0.76, 0.76,
                    boxstyle="round,pad=0.04",
                    facecolor=fc, edgecolor=edge, linewidth=1.8, zorder=4,
                ))
                ax.text(ox + 0.5, ody + 0.5, lbl,
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white", zorder=5)

    if env.level in (5, 6):
        tx, ty = env.tm_pos
        if 0 <= tx < W and 0 <= ty < H:
            tdy = H - 1 - ty
            fc_tm = C["tm_pball"] if tm_has_ball else C["teammate"]
            ax.add_patch(FancyBboxPatch(
                (tx + 0.12, tdy + 0.12), 0.76, 0.76,
                boxstyle="round,pad=0.04",
                facecolor=fc_tm, edgecolor="#1B5E20", linewidth=1.8, zorder=4,
            ))
            ax.text(tx + 0.5, tdy + 0.5, "M" if tm_has_ball else "T",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white", zorder=5)

    ax_x, ax_y = env.agent_pos
    ady = H - 1 - ax_y
    fc = C["pball"] if env.has_ball else C["agent"]
    ax.add_patch(FancyBboxPatch(
        (ax_x + 0.12, ady + 0.12), 0.76, 0.76,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor="#00001A", linewidth=2.0, zorder=6,
    ))
    ax.text(ax_x + 0.5, ady + 0.5, "P" if env.has_ball else "A",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", zorder=7)

    for col in range(W):
        ax.text(col + 0.5, -0.28, str(col),
                ha="center", va="center", fontsize=6.5, color=C["text_dim"])

    action_str = ACTION_NAMES[action] if action is not None else "—"
    status = "  [FERTIG]" if done else ""
    ep_str = f"Episode {episode}/{n_episodes}"
    ax.text(W / 2, H + 0.72,
            f"{label}  —  Level {env.level}  |  {ep_str}{status}",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C["text"])
    ax.text(W / 2, H + 0.42,
            f"Schritt {env.step_count}  |  Aktion: {action_str}  |  "
            f"Reward: {step_reward:+.0f}  |  Gesamt: {total_reward:+.0f}",
            ha="center", va="center", fontsize=8.5, color=C["text_dim"])

    handles = [
        mpatches.Patch(facecolor=C["agent"],  label="Agent (A)"),
        mpatches.Patch(facecolor=C["pball"],  label="Agent+Ball (P)"),
        mpatches.Patch(facecolor=C["ball"],   label="Ball (B)"),
        mpatches.Patch(facecolor=C["goal"], edgecolor=C["goal_edge"], label="Tor (G)"),
    ]
    if env.level == 1:
        handles.append(mpatches.Patch(facecolor=C["zone"], label="Schusszone"))
    if env.level >= 3 and env.level != 6:
        handles.append(mpatches.Patch(facecolor=C["opp"], label="Gegner (X)"))
    if env.level == 4:
        handles.append(mpatches.Patch(facecolor=C["obstacle"], edgecolor="#555555", label="Hindernis (#)"))
    if env.level in (5, 6):
        handles.append(mpatches.Patch(facecolor=C["teammate"], label="Mitspieler (T)"))
        handles.append(mpatches.Patch(facecolor=C["tm_pball"], label="Mitspieler+Ball (M)"))
    if env.level == 6:
        handles.append(mpatches.Patch(facecolor=C["opp"],  label="Gegner 1 – Ball (X)"))
        handles.append(mpatches.Patch(facecolor="#9C27B0", label="Gegner 2 – Press (Y)"))
    ax.legend(handles=handles, loc="lower center", fontsize=7.5, ncol=len(handles),
              bbox_to_anchor=(0.5, -0.12), framealpha=0.92, edgecolor=C["border"])

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _run_greedy_episode(snap, env, is_dqn, label, episode, n_episodes,
                        max_steps=40):
    """Run one greedy episode with the snapshot agent. Returns list of frames."""
    state = env.reset()
    total_reward = 0
    frames = [_render_game_frame(env, action=None, total_reward=0,
                                 label=label, episode=episode,
                                 n_episodes=n_episodes)]
    for _ in range(max_steps):
        if is_dqn:
            action = snap.choose_action(env.state_to_array(state))
        else:
            action = snap.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        frames.append(_render_game_frame(
            env, action=action, step_reward=reward,
            total_reward=total_reward, label=label,
            episode=episode, n_episodes=n_episodes, done=done,
        ))
        state = next_state
        if done:
            break
    for _ in range(3):
        frames.append(frames[-1].copy())
    return frames


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _train_qtable(level, n_episodes, snapshot_every):
    """Train Q-Table, collect snapshots, return (snapshots, agent, log)."""
    env = FootballEnv(level=level)
    agent = QTableAgent(n_actions=5)
    recent_goals = []
    recent_rewards = []
    window = min(snapshot_every, 50)
    snapshots = []
    log = []

    for episode in range(1, n_episodes + 1):
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
        log.append({"episode": episode, "reward": round(total_reward, 2),
                    "steps": env.step_count, "epsilon": round(agent.epsilon, 4),
                    "goal": goal_scored, "q_table_size": len(agent.q_table)})

        if episode % snapshot_every == 0 or episode == 1:
            goal_rate = np.mean(recent_goals) * 100
            avg_reward = np.mean(recent_rewards)
            snapshots.append((episode, snapshot_qtable(agent), agent.epsilon,
                              goal_rate, avg_reward))
            print(f"  Snapshot @ Ep {episode:5d}  |  Tor-Rate {goal_rate:5.1f}%  "
                  f"|  Ø Reward {avg_reward:7.1f}  |  ε={agent.epsilon:.3f}")

    return snapshots, agent, log


def _train_dqn(level, n_episodes, snapshot_every):
    """Train DQN, collect snapshots, return (snapshots, agent, log)."""
    env = FootballEnv(level=level)
    agent = DQNAgent(state_size=env.get_state_size(), n_actions=5)
    state_size = env.get_state_size()
    recent_goals = []
    recent_rewards = []
    window = min(snapshot_every, 50)
    snapshots = []
    log = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        state_arr = env.state_to_array(state)
        total_reward = 0
        goal_scored = False
        losses = []
        for _ in range(env.max_steps):
            action = agent.choose_action(state_arr)
            next_state, reward, done = env.step(action)
            next_arr = env.state_to_array(next_state)
            agent.store(state_arr, action, reward, next_arr, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state_arr = next_arr
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
        avg_loss = sum(losses) / len(losses) if losses else None
        log.append({"episode": episode, "reward": round(total_reward, 2),
                    "steps": env.step_count, "epsilon": round(agent.epsilon, 4),
                    "goal": goal_scored,
                    "avg_loss": round(avg_loss, 4) if avg_loss is not None else None})

        if episode % snapshot_every == 0 or episode == 1:
            goal_rate = np.mean(recent_goals) * 100
            avg_reward = np.mean(recent_rewards)
            snapshots.append((episode, snapshot_dqn(agent, state_size), agent.epsilon,
                              goal_rate, avg_reward))
            print(f"  Snapshot @ Ep {episode:5d}  |  Tor-Rate {goal_rate:5.1f}%  "
                  f"|  Ø Reward {avg_reward:7.1f}  |  ε={agent.epsilon:.3f}")

    return snapshots, agent, log


# ---------------------------------------------------------------------------
# Public GIF renderer
# ---------------------------------------------------------------------------

def render_evolution_gif(snapshots, agent_type, level, n_episodes,
                         fps=3, out_path=None, max_steps=40):
    """
    Build and save a training-evolution GIF from pre-collected in-memory snapshots.

    snapshots : list of (episode, snap_agent, epsilon, goal_rate, avg_reward)
    agent_type: "qtable" | "dqn"
    """
    import matplotlib
    matplotlib.use("Agg")

    is_dqn = agent_type == "dqn"
    agent_label = "DQN" if is_dqn else "Q-Table"

    if out_path is None:
        out_path = os.path.join(
            config.ANIMATIONS_DIR,
            f"training_evolution_{agent_type}_level{level}.gif",
        )

    env = FootballEnv(level=level)
    probe_frame = _render_game_frame(env, label=agent_label,
                                     episode=1, n_episodes=n_episodes)
    frame_size = probe_frame.size

    all_frames = []
    for episode, snap, epsilon, goal_rate, avg_reward in snapshots:
        card = _render_title_card(
            episode, n_episodes, epsilon, goal_rate, avg_reward,
            level, agent_label, frame_size,
        )
        all_frames.extend([card.resize(frame_size, Image.LANCZOS)] * 3)

        game_frames = _run_greedy_episode(
            snap, env, is_dqn, agent_label, episode, n_episodes,
            max_steps=max_steps,
        )
        all_frames.extend(f.resize(frame_size, Image.LANCZOS) for f in game_frames)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_frames[0].save(
        out_path,
        save_all=True,
        append_images=all_frames[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=True,
    )
    print(f"\nGIF gespeichert: {out_path}")
    print(f"  {len(all_frames)} Frames  @  {fps} fps  "
          f"({len(all_frames) / fps:.0f} s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_snapshots(level, agent_type, ep=None):
    """Load snapshot file created by train_q_table.py / train_dqn.py."""
    import glob
    is_dqn = agent_type == "dqn"
    prefix = "dqn" if is_dqn else "q_table"
    ext    = "_snapshots.pt" if is_dqn else "_snapshots.pkl"
    suffix = f"ep{ep}" if ep else "ep*"
    pattern = os.path.join(config.MODELS_DIR, f"{prefix}_level{level}_{suffix}{ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None
    path = files[-1]
    if is_dqn:
        import torch
        raw = torch.load(path, map_location="cpu", weights_only=False)
    else:
        import pickle
        with open(path, "rb") as f:
            raw = pickle.load(f)

    # Reconstruct agent instances from raw data
    env_tmp = FootballEnv(level=level)
    snapshots = []
    for entry in raw:
        if is_dqn:
            snap = DQNAgent(state_size=env_tmp.get_state_size(), n_actions=5)
            snap.q_net.load_state_dict(entry["state_dict"])
            snap.q_net.eval()
            snap.epsilon = 0.0
        else:
            snap = QTableAgent(n_actions=5)
            snap.q_table = defaultdict(lambda: [0.0] * 5,
                                       copy.deepcopy(entry["q_table"]))
            snap.epsilon = 0.0
        snapshots.append((entry["episode"], snap,
                          entry["epsilon"], entry["goal_rate"], entry["avg_reward"]))

    n_episodes = raw[-1]["episode"]
    return snapshots, n_episodes


def main():
    ap = argparse.ArgumentParser(
        description="Trainingsevolution eines vorhandenen Modells als GIF"
    )
    ap.add_argument("--level",     type=int, choices=[1, 2, 3, 4, 5, 6], default=1)
    ap.add_argument("--agent",     choices=["qtable", "dqn"], default="qtable")
    ap.add_argument("--episodes",  type=int, default=None,
                    help="Episodenzahl des zu ladenden Snapshots (Standard: neuestes)")
    ap.add_argument("--fps",       type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--run",       type=str, default=None,
                    help="Run-Verzeichnis (z.B. 'opt_iter5_2906'). Standard: neuester Run.")
    args = ap.parse_args()

    if args.run:
        config.set_run_dir(args.run)
    else:
        run = config.latest_run()
        if run:
            config.set_run_dir(run)

    agent_label = "DQN" if args.agent == "dqn" else "Q-Table"
    print(f"\nTrainings-Animation: {agent_label}  —  Level {args.level}")
    print("-" * 55)

    snapshots, n_episodes = _load_snapshots(args.level, args.agent, ep=args.episodes)
    if snapshots is None:
        print(f"Fehler: Keine Snapshot-Datei gefunden für {agent_label} Level {args.level}.")
        print(f"Führe zuerst das Training aus: python src/train_{'dqn' if args.agent == 'dqn' else 'q_table'}.py")
        return

    print(f"  {len(snapshots)} Snapshots geladen  |  {n_episodes} Episoden")

    out_path = os.path.join(
        config.ANIMATIONS_DIR,
        f"training_evolution_{args.agent}_level{args.level}_ep{n_episodes}.gif",
    )
    render_evolution_gif(snapshots, args.agent, args.level, n_episodes,
                         fps=args.fps, out_path=out_path,
                         max_steps=args.max_steps)


if __name__ == "__main__":
    main()
