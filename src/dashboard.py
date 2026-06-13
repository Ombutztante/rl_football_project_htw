"""
Interactive matplotlib dashboard for watching trained RL agents play.

Usage:
    python src/dashboard.py                  # Level 1, Q-Table
    python src/dashboard.py --level 2        # Level 2
    python src/dashboard.py --agent dqn      # DQN agent
    python src/dashboard.py --level 3 --agent dqn

Controls:
    Play / Pause  — toggle auto-play
    Step          — advance one step manually
    Reset         — start a new episode
    Geschwindigkeit slider — steps per second
    Level radio   — switch between Level 1 / 2 / 3
    Agent radio   — switch between Q-Table / DQN
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.animation import FuncAnimation

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
    "agent":     "#4FC3F7",
    "pball":     "#CE93D8",
    "ball":      "#FF8A65",
    "opp":       "#EF5350",
    "border":    "#2A3A6A",
    "goal_edge": "#66BB6A",
    "text":      "#ECF0F1",
    "text_dim":  "#78909C",
    "pos":       "#66BB6A",
    "neg":       "#EF5350",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_model(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_agent(level, agent_type):
    """Load a trained greedy agent. Returns (agent, is_dqn) or (None, is_dqn)."""
    if agent_type == "qtable":
        path = _find_model(os.path.join(config.MODELS_DIR, f"q_table_level{level}_ep*.pkl"))
        if not path:
            print(f"[Warnung] Kein Q-Table-Modell für Level {level} gefunden.")
            return None, False
        agent = QTableAgent(n_actions=5)
        agent.load(path)
        agent.epsilon = 0.0
        print(f"[OK] Q-Table geladen: {os.path.basename(path)}")
        return agent, False
    else:
        path = _find_model(os.path.join(config.MODELS_DIR, f"dqn_level{level}_ep*.pt"))
        if not path:
            print(f"[Warnung] Kein DQN-Modell für Level {level} gefunden.")
            return None, True
        env_tmp = FootballEnv(level=level)
        agent = DQNAgent(state_size=env_tmp.get_state_size(), n_actions=5)
        agent.load(path)
        agent.epsilon = 0.0
        print(f"[OK] DQN geladen:     {os.path.basename(path)}")
        return agent, True


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_grid(ax, env, episode_count, total_reward, last_action, done):
    ax.cla()
    W, H = env.width, env.height
    ax.set_xlim(0, W)
    ax.set_ylim(-0.55, H + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(C["bg"])

    gx, gy = env.goal_pos

    # Cells
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

    # Goal label
    ax.text(gx + 0.5, H - 1 - gy + 0.5, "G",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C["goal_edge"], zorder=3)

    # Shooting zone label (Level 1)
    if env.level == 1:
        mid = env.shoot_zone_x + (W - 1 - env.shoot_zone_x) / 2
        ax.text(mid, H + 0.68, "Schusszone",
                ha="center", va="center", fontsize=7.5, color="#856404",
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

    # Column numbers
    for col in range(W):
        ax.text(col + 0.5, -0.38, str(col),
                ha="center", va="center", fontsize=6.5, color=C["text_dim"])

    # Status header
    action_str = ACTION_NAMES[last_action] if last_action is not None else "—"
    status = "  ✓ FERTIG" if done else ""
    ax.text(W / 2, H + 0.75, f"Schritt {env.step_count}{status}",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C["text"])
    ax.text(W / 2, H + 0.47,
            f"Aktion: {action_str}  |  Reward: {total_reward:+.0f}  |  Episoden: {episode_count}",
            ha="center", va="center", fontsize=8.5, color=C["text_dim"])

    # Legend
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
    ax.legend(handles=handles, loc="lower center", fontsize=7.5, ncol=3,
              framealpha=0.92, edgecolor=C["border"], facecolor=C["cell"])


def draw_stats(ax, rewards, episode_rewards, agent_label, level):
    ax.cla()
    ax.set_facecolor("#111827")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])

    # Current episode cumulative reward
    if len(rewards) > 0:
        steps = list(range(len(rewards)))
        cumulative = np.cumsum(rewards)
        ax.plot(steps, cumulative, color=C["agent"], linewidth=2, zorder=3)
        ax.fill_between(steps, 0, cumulative,
                        where=cumulative >= 0, color=C["pos"], alpha=0.12)
        ax.fill_between(steps, 0, cumulative,
                        where=cumulative < 0, color=C["neg"], alpha=0.15)
        ax.axhline(0, color=C["border"], linewidth=0.9, linestyle="--")

    ax.set_xlabel("Schritt", fontsize=9, color=C["text_dim"])
    ax.set_ylabel("Kum. Reward (Episode)", fontsize=9, color=C["text_dim"])
    ax.tick_params(labelsize=8, colors=C["text_dim"])
    ax.set_title(f"Level {level}  —  {agent_label}", fontsize=10, fontweight="bold",
                 color=C["text"], pad=6)

    # Episode summary table
    if episode_rewards:
        last_n = episode_rewards[-8:]
        summary = "\n".join(
            f"Ep {len(episode_rewards) - len(last_n) + i + 1:>3}: {r:+.0f}"
            for i, r in enumerate(last_n)
        )
        ax.text(0.98, 0.98, summary, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, family="monospace",
                color=C["text"],
                bbox=dict(fc="#1E2A4A", ec=C["border"], boxstyle="round,pad=0.4", lw=0.8))


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class DashState:
    def __init__(self, level, agent_type):
        self.level = level
        self.agent_type = agent_type
        self._load(level, agent_type)
        self.episode_rewards = []  # total reward per completed episode
        self.episode_count = 0

    def _load(self, level, agent_type):
        self.env = FootballEnv(level=level)
        self.agent, self.is_dqn = load_agent(level, agent_type)
        self.playing = False
        self.last_step_time = 0.0
        self.speed = 2.0
        self._reset_episode()

    def _reset_episode(self):
        self.state = self.env.reset()
        self.total_reward = 0
        self.last_action = None
        self.rewards = []      # per-step rewards in current episode
        self.done = False

    def reset(self):
        self._reset_episode()
        self.playing = False

    def step_once(self):
        if self.done or self.agent is None:
            return
        if self.is_dqn:
            action = self.agent.choose_action(self.env.state_to_array(self.state))
        else:
            action = self.agent.choose_action(self.state)
        next_state, reward, done = self.env.step(action)
        self.total_reward += reward
        self.last_action = action
        self.rewards.append(reward)
        self.state = next_state
        self.done = done
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.total_reward)
            self.playing = False

    def reload(self, level=None, agent_type=None):
        if level is not None:
            self.level = level
        if agent_type is not None:
            self.agent_type = agent_type
        self._load(self.level, self.agent_type)
        self.episode_rewards = []
        self.episode_count = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Interaktives RL-Football-Dashboard")
    ap.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--agent", choices=["qtable", "dqn"], default="qtable")
    args = ap.parse_args()

    s = DashState(level=args.level, agent_type=args.agent)

    # -----------------------------------------------------------------------
    # Figure & axes layout
    # -----------------------------------------------------------------------
    plt.rcParams.update({
        "text.color":        C["text"],
        "axes.labelcolor":   C["text_dim"],
        "xtick.color":       C["text_dim"],
        "ytick.color":       C["text_dim"],
    })

    fig = plt.figure(figsize=(13, 8), facecolor=C["bg"])
    fig.suptitle("RL Football — Interaktives Demo", fontsize=14,
                 fontweight="bold", color=C["text"], y=0.98)

    # Main panels (grid left, stats right) occupy top 68% of figure
    ax_grid  = fig.add_axes([0.04, 0.26, 0.50, 0.68], facecolor=C["bg"])
    ax_stats = fig.add_axes([0.60, 0.26, 0.38, 0.68], facecolor="#111827")

    # -----------------------------------------------------------------------
    # Control widgets  (bottom 22%)
    # -----------------------------------------------------------------------
    # Buttons
    ax_play  = fig.add_axes([0.04, 0.14, 0.11, 0.07])
    ax_step  = fig.add_axes([0.17, 0.14, 0.11, 0.07])
    ax_reset = fig.add_axes([0.30, 0.14, 0.11, 0.07])

    btn_play  = Button(ax_play,  "Play",  color="#28A745", hovercolor="#218838")
    btn_step  = Button(ax_step,  "Step",  color="#007BFF", hovercolor="#0069D9")
    btn_reset = Button(ax_reset, "Reset", color="#DC3545", hovercolor="#C82333")

    for btn in (btn_play, btn_step, btn_reset):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")
        btn.label.set_color("white")

    # Speed slider
    ax_speed = fig.add_axes([0.50, 0.165, 0.34, 0.035])
    slider_speed = Slider(ax_speed, "Geschw.", 0.5, 8.0,
                          valinit=s.speed, valstep=0.5, color=C["agent"])
    ax_speed.set_xlabel("Schritte / Sekunde", fontsize=7.5, labelpad=1)

    # Level radio buttons
    ax_level = fig.add_axes([0.04, 0.00, 0.22, 0.12])
    ax_level.set_facecolor(C["cell"])
    radio_level = RadioButtons(ax_level, ("Level 1", "Level 2", "Level 3"),
                               active=s.level - 1,
                               activecolor=C["agent"])
    ax_level.set_title("Level auswählen", fontsize=8.5, pad=3, color=C["text"])
    for lbl in radio_level.labels:
        lbl.set_color(C["text"])
        lbl.set_fontsize(9)

    # Agent radio buttons
    ax_agent = fig.add_axes([0.36, 0.00, 0.22, 0.12])
    ax_agent.set_facecolor(C["cell"])
    radio_agent = RadioButtons(ax_agent, ("Q-Table", "DQN"),
                               active=0 if s.agent_type == "qtable" else 1,
                               activecolor=C["agent"])
    ax_agent.set_title("Agent auswählen", fontsize=8.5, pad=3, color=C["text"])
    for lbl in radio_agent.labels:
        lbl.set_color(C["text"])
        lbl.set_fontsize(9)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    def refresh():
        agent_label = "DQN" if s.is_dqn else "Q-Table"
        draw_grid(ax_grid, s.env, s.episode_count, s.total_reward, s.last_action, s.done)
        draw_stats(ax_stats, s.rewards, s.episode_rewards, agent_label, s.level)
        fig.canvas.draw_idle()

    def on_play(event):
        if s.agent is None:
            return
        if s.done:
            s.reset()
        s.playing = not s.playing
        btn_play.label.set_text("Pause" if s.playing else "Play")
        s.last_step_time = time.time()
        refresh()

    def on_step(event):
        s.playing = False
        btn_play.label.set_text("Play")
        s.step_once()
        refresh()

    def on_reset(event):
        s.reset()
        btn_play.label.set_text("Play")
        refresh()

    def on_speed(val):
        s.speed = slider_speed.val

    def on_level(label):
        lv = int(label.split()[1])
        s.reload(level=lv)
        btn_play.label.set_text("Play")
        refresh()

    def on_agent(label):
        at = "qtable" if label == "Q-Table" else "dqn"
        s.reload(agent_type=at)
        btn_play.label.set_text("Play")
        refresh()

    btn_play.on_clicked(on_play)
    btn_step.on_clicked(on_step)
    btn_reset.on_clicked(on_reset)
    slider_speed.on_changed(on_speed)
    radio_level.on_clicked(on_level)
    radio_agent.on_clicked(on_agent)

    # -----------------------------------------------------------------------
    # Animation loop
    # -----------------------------------------------------------------------
    def animate(_frame):
        if s.playing and not s.done:
            now = time.time()
            if now - s.last_step_time >= 1.0 / s.speed:
                s.step_once()
                s.last_step_time = now
                if s.done:
                    btn_play.label.set_text("Play")
                refresh()
        return []

    anim = FuncAnimation(fig, animate, interval=80, blit=False, cache_frame_data=False)

    refresh()
    plt.show()
    _ = anim  # keep reference alive


if __name__ == "__main__":
    main()
