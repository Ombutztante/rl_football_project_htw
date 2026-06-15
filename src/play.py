"""
Manual control of the football environment via a Matplotlib window.

Controls:
  Arrow keys  — move (up / down / left / right)
  Space or S  — shoot
  R           — reset episode
  Q           — quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

import config
from src.environment import FootballEnv

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Shoot"]

C = {
    "bg":        "#0F0F1A",
    "cell":      "#1E2A4A",
    "zone":      "#3D3200",
    "goal":      "#0D3018",
    "agent":     "#4FC3F7",
    "pball":     "#CE93D8",
    "ball":      "#FF8A65",
    "opp":       "#EF5350",
    "border":    "#2A3A6A",
    "goal_edge": "#66BB6A",
    "text":      "#ECF0F1",
    "text_dim":  "#78909C",
}

KEY_TO_ACTION = {
    "up":    0,
    "down":  1,
    "left":  2,
    "right": 3,
    " ":     4,
    "s":     4,
}


def draw(ax, env, total_reward, last_action, episode, message=""):
    ax.cla()
    W, H = env.width, env.height
    ax.set_xlim(0, W)
    ax.set_ylim(-1.1, H + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(C["bg"])

    gx, gy = env.goal_pos

    for row in range(H):
        dy = H - 1 - row
        for col in range(W):
            is_goal = (col == gx and row == gy)
            is_zone = (env.level == 1 and env.shoot_zone_x <= col < W - 1)
            fc = C["goal"] if is_goal else (C["zone"] if is_zone else C["cell"])
            ec = C["goal_edge"] if is_goal else C["border"]
            lw = 2.2 if is_goal else 0.9
            ax.add_patch(FancyBboxPatch(
                (col + 0.05, dy + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1,
            ))

    ax.text(gx + 0.5, H - 1 - gy + 0.5, "G",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C["goal_edge"], zorder=3)

    if env.level == 1:
        mid = env.shoot_zone_x + (W - 1 - env.shoot_zone_x) / 2
        ax.text(mid, H + 0.9, "Shoot Zone",
                ha="center", va="center", fontsize=8, color="#856404",
                bbox=dict(fc=C["zone"], ec="#FFC107", boxstyle="round,pad=0.25", lw=0.8))

    if not env.has_ball:
        bx, by = env.ball_pos
        if 0 <= bx < W and 0 <= by < H:
            bdy = H - 1 - by
            ax.add_patch(plt.Circle((bx + 0.5, bdy + 0.5), 0.27, color=C["ball"], zorder=4))
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
                facecolor=C["opp"], edgecolor="#7B0000", linewidth=1.8, zorder=4,
            ))
            ax.text(ox + 0.5, ody + 0.5, "X",
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

    action_str = ACTION_NAMES[last_action] if last_action is not None else "—"
    ax.text(W / 2, H + 1.05,
            f"Step {env.step_count}  |  Action: {action_str}  |  Score: {total_reward:+.0f}  |  Episode: {episode}",
            ha="center", va="center", fontsize=9, fontweight="bold", color=C["text"])

    if message:
        ax.text(W / 2, H + 0.55, message,
                ha="center", va="center", fontsize=11, fontweight="bold", color="#FFD700")

    ax.text(W / 2, -0.75,
            "Arrow keys: move  |  Space / S: shoot  |  R: reset  |  Q: quit",
            ha="center", va="center", fontsize=8, color=C["text_dim"])

    handles = [
        mpatches.Patch(facecolor=C["agent"], label="Agent (A)"),
        mpatches.Patch(facecolor=C["pball"], label="Agent + Ball (P)"),
        mpatches.Patch(facecolor=C["ball"],  label="Ball (B)"),
        mpatches.Patch(facecolor=C["goal"],  edgecolor=C["goal_edge"], label="Goal (G)"),
    ]
    if env.level == 1:
        handles.append(mpatches.Patch(facecolor=C["zone"], label="Shoot Zone"))
    if env.level >= 3:
        handles.append(mpatches.Patch(facecolor=C["opp"], label="Opponent (X)"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.28),
              fontsize=7.5, ncol=3, framealpha=0.9, edgecolor=C["border"],
              facecolor=C["cell"])


def main():
    env = FootballEnv()

    state = [env.reset()]
    total_reward = [0]
    last_action = [None]
    episode = [1]
    done = [False]
    message = [""]

    plt.rcParams.update({"text.color": C["text"]})
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle(f"Football Environment  —  Level {env.level}",
                 fontsize=13, fontweight="bold", color=C["text"])

    def refresh(msg=""):
        message[0] = msg
        draw(ax, env, total_reward[0], last_action[0], episode[0], msg)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("q", "Q"):
            plt.close(fig)
            return

        if event.key in ("r", "R"):
            state[0] = env.reset()
            total_reward[0] = 0
            last_action[0] = None
            done[0] = False
            episode[0] += 1
            refresh()
            return

        if done[0]:
            return

        key = event.key.lower() if event.key else ""
        if key not in KEY_TO_ACTION and event.key not in KEY_TO_ACTION:
            return

        action = KEY_TO_ACTION.get(key) or KEY_TO_ACTION.get(event.key)
        if action is None:
            return

        next_state, reward, is_done = env.step(action)
        state[0] = next_state
        total_reward[0] += reward
        last_action[0] = action
        done[0] = is_done

        msg = ""
        if is_done:
            msg = "*** GOAL! ***" if reward >= 10 else "--- Episode ended --- Press R to restart"

        refresh(msg)

    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
