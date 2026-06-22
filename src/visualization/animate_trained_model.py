"""
Generate animated GIFs from trained RL agents.

Usage:
    python src/animate.py                 # all levels, both agents, 3 fps
    python src/animate.py --level 1       # only level 1
    python src/animate.py --agent dqn     # only DQN
    python src/animate.py --fps 4         # 4 frames per second

Output: results/plots/animation_{agent}_{level}.gif
"""

import matplotlib
matplotlib.use("Agg")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import glob
import io
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
    "border":    "#2A3A6A",
    "goal_edge": "#66BB6A",
    "text":      "#ECF0F1",
    "text_dim":  "#78909C",
}


def _find_model(pattern):
    import re as _re
    files = [f for f in glob.glob(pattern) if "_snapshots" not in f]
    if not files:
        return None
    def _key(p):
        m = _re.search(r"_ep(\d+)", os.path.basename(p))
        ep = int(m.group(1)) if m else 0
        return (ep, os.path.getmtime(p))  # newest date wins on equal episode count
    return max(files, key=_key)


def _find_model_for_ep(base_pattern, ep):
    """Find model for a specific episode count, preferring dated files over undated."""
    dated   = _find_model(base_pattern.replace("{ep}", f"ep{ep}_*"))
    undated = _find_model(base_pattern.replace("{ep}", f"ep{ep}"))
    if dated and undated:
        return dated if os.path.getmtime(dated) >= os.path.getmtime(undated) else undated
    return dated or undated


_LEGACY_MODELS = "results/old/models"


def _load_qtable(level, ep=None):
    base = os.path.join(config.MODELS_DIR, f"q_table_level{level}_{{ep}}.pkl")
    path = _find_model_for_ep(base, ep) if ep else _find_model(base.replace("{ep}", "ep*"))
    if not path and config.MODELS_DIR != _LEGACY_MODELS:
        base = os.path.join(_LEGACY_MODELS, f"q_table_level{level}_{{ep}}.pkl")
        path = _find_model_for_ep(base, ep) if ep else _find_model(base.replace("{ep}", "ep*"))
    if not path:
        return None, None, None
    agent = QTableAgent(n_actions=5)
    agent.load(path)
    agent.epsilon = 0.0
    import re
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_ep(\d+)", stem)
    loaded_ep = int(m.group(1)) if m else ep
    print(f"  Q-Table geladen: {os.path.basename(path)}")
    return agent, loaded_ep, stem


def _load_dqn(level, ep=None):
    base = os.path.join(config.MODELS_DIR, f"dqn_level{level}_{{ep}}.pt")
    path = _find_model_for_ep(base, ep) if ep else _find_model(base.replace("{ep}", "ep*"))
    if not path and config.MODELS_DIR != _LEGACY_MODELS:
        base = os.path.join(_LEGACY_MODELS, f"dqn_level{level}_{{ep}}.pt")
        path = _find_model_for_ep(base, ep) if ep else _find_model(base.replace("{ep}", "ep*"))
    if not path:
        return None, None, None
    env_tmp = FootballEnv(level=level)
    agent = DQNAgent(state_size=env_tmp.get_state_size(), n_actions=5)
    agent.load(path)
    agent.epsilon = 0.0
    import re
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_ep(\d+)", stem)
    loaded_ep = int(m.group(1)) if m else ep
    print(f"  DQN geladen:     {os.path.basename(path)}")
    return agent, loaded_ep, stem


def render_frame(env, action=None, step_reward=0, total_reward=0, label="", done=False):
    """Render current env state as a matplotlib figure and return a PIL Image."""
    W, H = env.width, env.height

    fig, ax = plt.subplots(figsize=(W * 1.05 + 1.5, H * 1.1 + 1.6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, W)
    ax.set_ylim(-0.45, H + 0.85)
    ax.set_aspect("equal")
    ax.axis("off")

    gx, gy = env.goal_pos

    # --- Grid cells ---
    for row in range(H):
        dy = H - 1 - row  # flip so row 0 is at top
        for col in range(W):
            is_goal = (col == gx and row == gy)
            is_zone = (env.level == 1 and env.shoot_zone_x <= col < W - 1)
            is_obstacle = (env.level >= 4 and (col, row) in env.obstacle_cells)
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

    # Goal label
    ax.text(gx + 0.5, H - 1 - gy + 0.5, "G",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=C["goal_edge"], zorder=3)

    # Shooting zone label (Level 1)
    if env.level == 1:
        mid = env.shoot_zone_x + (W - 1 - env.shoot_zone_x) / 2
        ax.text(mid, H + 0.6, "Schusszone",
                ha="center", va="center", fontsize=7, color="#856404",
                bbox=dict(fc=C["zone"], ec="#FFC107", boxstyle="round,pad=0.25", lw=0.8))

    # Ball (loose)
    if not env.has_ball:
        bx, by = env.ball_pos
        if 0 <= bx < W and 0 <= by < H:
            bdy = H - 1 - by
            ax.add_patch(plt.Circle((bx + 0.5, bdy + 0.5), 0.27, color=C["ball"], zorder=4))
            ax.text(bx + 0.5, bdy + 0.5, "B",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white", zorder=5)

    # Opponent (Level 3)
    if env.level >= 3:
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

    # Agent
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

    # Column index labels
    for col in range(W):
        ax.text(col + 0.5, -0.28, str(col),
                ha="center", va="center", fontsize=6.5, color=C["text_dim"])

    # --- Title ---
    action_str = ACTION_NAMES[action] if action is not None else "—"
    status = "  [FERTIG]" if done else ""
    ax.text(W / 2, H + 0.72, f"{label}  —  Level {env.level}{status}",
            ha="center", va="center", fontsize=11, fontweight="bold", color=C["text"])
    ax.text(W / 2, H + 0.42,
            f"Schritt {env.step_count}  |  Aktion: {action_str}  |  "
            f"Reward: {step_reward:+.0f}  |  Gesamt: {total_reward:+.0f}",
            ha="center", va="center", fontsize=8.5, color=C["text_dim"])

    # --- Legend ---
    handles = [
        mpatches.Patch(facecolor=C["agent"],  label="Agent (A)"),
        mpatches.Patch(facecolor=C["pball"],  label="Agent+Ball (P)"),
        mpatches.Patch(facecolor=C["ball"],   label="Ball (B)"),
        mpatches.Patch(facecolor=C["goal"], edgecolor=C["goal_edge"], label="Tor (G)"),
    ]
    if env.level == 1:
        handles.append(mpatches.Patch(facecolor=C["zone"], label="Schusszone"))
    if env.level >= 3:
        handles.append(mpatches.Patch(facecolor=C["opp"], label="Gegner (X)"))
    if env.level >= 4:
        handles.append(mpatches.Patch(facecolor=C["obstacle"], edgecolor="#555555", label="Hindernis (#)"))
    ax.legend(handles=handles, loc="lower center", fontsize=7.5, ncol=len(handles),
              bbox_to_anchor=(0.5, -0.12), framealpha=0.92, edgecolor=C["border"])

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def run_episode(level, agent, is_dqn, label, max_steps=80):
    """Play one greedy episode and collect rendered frames."""
    env = FootballEnv(level=level)
    state = env.reset()
    total_reward = 0
    frames = [render_frame(env, action=None, total_reward=0, label=label)]

    for _ in range(max_steps):
        if is_dqn:
            action = agent.choose_action(env.state_to_array(state))
        else:
            action = agent.choose_action(state)

        next_state, reward, done = env.step(action)
        total_reward += reward
        frames.append(render_frame(env, action=action, step_reward=reward,
                                   total_reward=total_reward, label=label, done=done))
        state = next_state
        if done:
            break

    # Hold final frame a moment longer
    for _ in range(4):
        frames.append(frames[-1].copy())
    return frames


def save_gif(frames, path, fps):
    os.makedirs(config.ANIMATIONS_DIR, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=True,
    )
    print(f"  GIF gespeichert: {path}  ({len(frames)} Frames @ {fps} fps)")


def main():
    ap = argparse.ArgumentParser(description="Animierte GIFs von trainierten RL-Agenten")
    ap.add_argument("--level", type=int, choices=[1, 2, 3, 4],
                    help="Level (1/2/3/4); Standard: alle")
    ap.add_argument("--agent", choices=["qtable", "dqn", "both"], default="both",
                    help="Welchen Agenten animieren (Standard: beide)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="Episodenzahl des zu ladenden Modells (Standard: neuestes)")
    ap.add_argument("--fps", type=int, default=3, help="Frames pro Sekunde (Standard: 3)")
    ap.add_argument("--run", type=str, default=None,
                    help="Run-Verzeichnis (z.B. 'a_dev_1_2206_1'). Standard: neuester Run.")
    args = ap.parse_args()

    if args.run:
        config.set_run_dir(args.run)
    else:
        run = config.latest_run()
        if run:
            config.set_run_dir(run)
            print(f"Verwende neuesten Run: {run}")
        else:
            config.setup_run_dir()

    levels = [args.level] if args.level else [1, 2, 3]
    do_qt  = args.agent in ("qtable", "both")
    do_dqn = args.agent in ("dqn", "both")

    for lv in levels:
        print(f"\n=== Level {lv} ===")
        if do_qt:
            agent, ep, model_stem = _load_qtable(lv, ep=args.episodes)
            if agent:
                frames = run_episode(lv, agent, is_dqn=False, label="Q-Table")
                gif_stem = model_stem.replace("q_table_level", "animation_qtable_level", 1)
                save_gif(frames,
                         os.path.join(config.ANIMATIONS_DIR, f"{gif_stem}.gif"),
                         args.fps)
            else:
                print(f"  Kein Q-Table-Modell für Level {lv} gefunden.")

        if do_dqn:
            agent, ep, model_stem = _load_dqn(lv, ep=args.episodes)
            if agent:
                frames = run_episode(lv, agent, is_dqn=True, label="DQN")
                gif_stem = model_stem.replace("dqn_level", "animation_dqn_level", 1)
                save_gif(frames,
                         os.path.join(config.ANIMATIONS_DIR, f"{gif_stem}.gif"),
                         args.fps)
            else:
                print(f"  Kein DQN-Modell für Level {lv} gefunden.")

    print("\nFertig.")


if __name__ == "__main__":
    main()
