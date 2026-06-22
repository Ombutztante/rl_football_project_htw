"""
Side-by-side GIF: Q-Table vs. DQN playing the same level simultaneously.

python src/animate_compare.py --level 3
python src/animate_compare.py --level 1 --fps 2

Output: results/plots/compare_level{N}.gif
"""

import matplotlib
matplotlib.use("Agg")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import glob
import io

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
    "panel":     "#1A1A2E",
    "cell":      "#1E2A4A",
    "zone":      "#3D3200",
    "goal":      "#0D3018",
    "obstacle":  "#2D2D2D",
    "agent_qt":  "#4FC3F7",
    "agent_dqn": "#CE93D8",
    "pball_qt":  "#B2EBF2",
    "pball_dqn": "#E1BEE7",
    "ball":      "#FF8A65",
    "opp":       "#EF5350",
    "border":    "#2A3A6A",
    "goal_edge": "#66BB6A",
    "text":      "#ECF0F1",
    "text_dim":  "#78909C",
    "divider":   "#2D3561",
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


def _load_qtable(level, ep=None):
    base = os.path.join(config.MODELS_DIR, f"q_table_level{level}_{{ep}}.pkl")
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
    return agent, loaded_ep, stem


def _load_dqn(level, ep=None):
    base = os.path.join(config.MODELS_DIR, f"dqn_level{level}_{{ep}}.pt")
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
    return agent, loaded_ep, stem


def _draw_grid(ax, env, label, action, total_reward, done,
               agent_color, pball_color):
    ax.cla()
    W, H = env.width, env.height
    ax.set_xlim(-0.1, W + 0.1)
    ax.set_ylim(-0.4, H + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(C["panel"])

    gx, gy = env.goal_pos

    # Cells
    for row in range(H):
        dy = H - 1 - row
        for col in range(W):
            is_goal = (col == gx and row == gy)
            is_zone = (env.level == 1 and env.shoot_zone_x <= col < W - 1)
            is_obstacle = (env.level >= 4 and (col, row) in env.obstacle_cells)
            fc = C["goal"] if is_goal else (C["obstacle"] if is_obstacle else (C["zone"] if is_zone else C["cell"]))
            ec = C["goal_edge"] if is_goal else ("#555555" if is_obstacle else C["border"])
            lw = 2.0 if is_goal else (1.5 if is_obstacle else 0.7)
            ax.add_patch(FancyBboxPatch(
                (col + 0.05, dy + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1,
            ))
            if is_obstacle:
                ax.text(col + 0.5, dy + 0.5, "#",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="#888888", zorder=2)

    ax.text(gx + 0.5, H - 1 - gy + 0.5, "G",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C["goal_edge"], zorder=3)

    if env.level == 1:
        mid = env.shoot_zone_x + (W - 1 - env.shoot_zone_x) / 2
        ax.text(mid, H + 0.75, "Schusszone",
                ha="center", va="center", fontsize=7, color="#FFD54F",
                bbox=dict(fc=C["zone"], ec="#FFD54F",
                          boxstyle="round,pad=0.2", lw=0.7))

    if not env.has_ball:
        bx, by = env.ball_pos
        if 0 <= bx < W and 0 <= by < H:
            bdy = H - 1 - by
            ax.add_patch(plt.Circle((bx + 0.5, bdy + 0.5), 0.27,
                                    color=C["ball"], zorder=4))
            ax.text(bx + 0.5, bdy + 0.5, "B",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white", zorder=5)

    if env.level >= 3:
        ox, oy = env.opp_pos
        if 0 <= ox < W and 0 <= oy < H:
            ody = H - 1 - oy
            ax.add_patch(FancyBboxPatch(
                (ox + 0.12, ody + 0.12), 0.76, 0.76,
                boxstyle="round,pad=0.04",
                facecolor=C["opp"], edgecolor="#B71C1C", linewidth=1.5, zorder=4,
            ))
            ax.text(ox + 0.5, ody + 0.5, "X",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white", zorder=5)

    ax_x, ax_y = env.agent_pos
    ady = H - 1 - ax_y
    fc = pball_color if env.has_ball else agent_color
    ax.add_patch(FancyBboxPatch(
        (ax_x + 0.12, ady + 0.12), 0.76, 0.76,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor="#000033", linewidth=2.0, zorder=6,
    ))
    ax.text(ax_x + 0.5, ady + 0.5, "P" if env.has_ball else "A",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white", zorder=7)

    # Header
    action_str = ACTION_NAMES[action] if action is not None else "—"
    status = "  ✓" if done else ""
    ax.text(W / 2, H + 0.82, f"{label}{status}",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=agent_color)
    ax.text(W / 2, H + 0.50,
            f"Schritt {env.step_count}  |  {action_str}  |  Reward {total_reward:+.0f}",
            ha="center", va="center", fontsize=8, color=C["text_dim"])


def render_frame(env_qt, env_dqn, action_qt, action_dqn,
                 total_qt, total_dqn, step, level, done_qt, done_dqn):
    W, H = env_qt.width, env_qt.height
    fig = plt.figure(figsize=(W * 2.2 + 2.5, H * 1.15 + 2.2),
                     facecolor=C["bg"])

    ax_l = fig.add_axes([0.03, 0.08, 0.46, 0.82])
    ax_r = fig.add_axes([0.51, 0.08, 0.46, 0.82])

    for ax in (ax_l, ax_r):
        ax.set_facecolor(C["panel"])

    _draw_grid(ax_l, env_qt, "Q-Table", action_qt, total_qt, done_qt,
               C["agent_qt"], C["pball_qt"])
    _draw_grid(ax_r, env_dqn, "DQN", action_dqn, total_dqn, done_dqn,
               C["agent_dqn"], C["pball_dqn"])

    # Divider
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.95],
                              transform=fig.transFigure,
                              color=C["divider"], linewidth=1.5))

    # Title
    fig.text(0.5, 0.97, f"Q-Table  vs.  DQN  —  Level {level}  |  Schritt {step}",
             ha="center", va="top", fontsize=12, fontweight="bold",
             color=C["text"])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    ap = argparse.ArgumentParser(
        description="Vergleichs-GIF: Q-Table vs. DQN nebeneinander"
    )
    ap.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=3)
    ap.add_argument("--fps",   type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=None,
                    help="Episodenzahl des zu ladenden Modells (Standard: neuestes)")
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

    print(f"Lade Modelle für Level {args.level} ...")
    qt_agent,  ep_qt,  stem_qt  = _load_qtable(args.level, ep=args.episodes)
    dqn_agent, ep_dqn, stem_dqn = _load_dqn(args.level,    ep=args.episodes)

    if not qt_agent:
        print("Fehler: Kein Q-Table-Modell gefunden.")
        return
    if not dqn_agent:
        print("Fehler: Kein DQN-Modell gefunden.")
        return
    ep = ep_qt or ep_dqn

    env_qt  = FootballEnv(level=args.level)
    env_dqn = FootballEnv(level=args.level)

    qt_state  = env_qt.reset()
    dqn_state = env_dqn.reset()

    qt_total  = 0
    dqn_total = 0
    done_qt   = False
    done_dqn  = False

    frames = [render_frame(env_qt, env_dqn, None, None,
                           0, 0, 0, args.level, False, False)]

    for step in range(1, args.max_steps + 1):
        if not done_qt:
            action_qt = qt_agent.choose_action(qt_state)
            qt_state, reward_qt, done_qt = env_qt.step(action_qt)
            qt_total += reward_qt
        else:
            action_qt = None

        if not done_dqn:
            action_dqn = dqn_agent.choose_action(env_dqn.state_to_array(dqn_state))
            dqn_state, reward_dqn, done_dqn = env_dqn.step(action_dqn)
            dqn_total += reward_dqn
        else:
            action_dqn = None

        frames.append(render_frame(
            env_qt, env_dqn, action_qt, action_dqn,
            qt_total, dqn_total, step, args.level, done_qt, done_dqn,
        ))

        if done_qt and done_dqn:
            break

    for _ in range(5):
        frames.append(frames[-1].copy())

    target = frames[0].size
    frames = [f.resize(target, Image.LANCZOS) for f in frames]

    out = os.path.join(config.ANIMATIONS_DIR, f"compare_level{args.level}_ep{ep}.gif")
    os.makedirs(config.ANIMATIONS_DIR, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   loop=0, duration=int(1000 / args.fps), optimize=True)
    print(f"GIF gespeichert: {out}  ({len(frames)} Frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
