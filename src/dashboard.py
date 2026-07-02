"""
Interactive RL Football Dashboard

Modi (Tabs oben):
  Live Play  — Agent spielt live Schritt für Schritt
  Agent GIF  — Animierter Spielzug des trainierten Agenten
  Vergleich  — Q-Table vs DQN Compare-GIF / Plot
  Plots      — Trainingsplots anzeigen

python src/dashboard.py [--level 1] [--agent qtable]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, glob, time, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.animation import FuncAnimation
from PIL import Image

import config
from src.environment import FootballEnv
from src.q_table_agent import QTableAgent
from src.dqn_agent import DQNAgent

# ── Constants ────────────────────────────────────────────────────────────────

ACTION_NAMES = ["Hoch", "Runter", "Links", "Rechts", "Schuss"]
MODE_LIVE    = "live"
MODE_TRAIN   = "train"
MODE_COMPARE = "compare"
MODE_PLOTS   = "plots"

LEVEL_DESC = {
    1: "Ball holen → Schusszone → Tor\nAgent lernt situative Aktionswahl",
    2: "Dribbling vs. Vorwärtspass\nEchte Entscheidung: sicher oder schnell?",
    3: "Gegner bewegt sich alle 2 Züge\nDynamisches Hindernis, Zeitdruck",
    4: "Statisches Hindernis (Spalte 6)\nUmweg durch freien Korridor (y≥4)",
    5: "Mitspieler + Gegner auf Torreihe\nPass-Strategie oder Dribbling?",
    6: "Zwei Gegner + Mitspieler\nGroßer Zustandsraum → DQN-Vorteil",
}

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
    "opp2":      "#9C27B0",
    "teammate":  "#81C784",
    "tm_pball":  "#A5D6A7",
    "border":    "#2A3A6A",
    "goal_edge": "#66BB6A",
    "text":      "#ECF0F1",
    "text_dim":  "#78909C",
    "pos":       "#66BB6A",
    "neg":       "#EF5350",
    "tab_on":    "#1D4ED8",
    "tab_off":   "#1E2A4A",
    "btn_sel":   "#1D4ED8",
    "btn_unsel": "#1E2A4A",
}

CTX = {
    MODE_LIVE:    {"title": "Agent",    "opts": ["Q-Table", "DQN", "", ""]},
    MODE_TRAIN:   {"title": "Agent",    "opts": ["Q-Table", "DQN", "", ""]},
    MODE_COMPARE: {"title": "",         "opts": ["", "", "", ""]},
    MODE_PLOTS:   {"title": "Plot-Typ", "opts": ["Q-Table", "DQN", "Vergleich", "Übersicht"]},
}

# ── File helpers ─────────────────────────────────────────────────────────────

def _find_file(pattern):
    files = [f for f in glob.glob(pattern) if "_snapshots" not in f]
    if not files:
        return None
    def _key(p):
        m = re.search(r"_ep(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return max(files, key=_key)


_EXCLUDED_DIRS = {"old", "opt_iter6_2906", "a_dev_4_2206_1"}

def _find_across_runs(filename_pattern, ext=None):
    """Search for a file matching filename_pattern across all results subdirs.
    Excludes legacy and discarded experiment dirs.
    Returns the path with the highest episode count, preferring newer run dirs.
    If ext is given (e.g. '.png', '.gif'), only files with that extension are returned."""
    base = config.RESULTS_BASE
    candidates = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]
        for f in files:
            if ext and not f.endswith(ext):
                continue
            full = os.path.join(root, f)
            if re.search(filename_pattern + r"[^/]*$", f) and "_snapshots" not in f:
                m = re.search(r"_ep(\d+)", f)
                ep = int(m.group(1)) if m else 0
                candidates.append((ep, os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _find_animation(level, agent):
    """Find animation GIF for given level and agent across all run dirs."""
    pattern = rf"animation_{agent}_level{level}_ep\d+"
    return _find_across_runs(pattern)


def _find_training_evo(level, agent):
    """Find training-evolution GIF for given level and agent across all run dirs."""
    pattern = rf"training_evolution_{agent}_level{level}_ep\d+"
    return _find_across_runs(pattern)


def _find_compare_gif(level):
    """Find comparison GIF for given level across all run dirs."""
    pattern = rf"compare_level{level}_ep\d+"
    return _find_across_runs(pattern, ext=".gif")


def _find_plot(level, plot_type, prefer_ep=3000):
    """Find the best matching training plot PNG across all run dirs."""
    if plot_type == "summary":
        p = os.path.join(config.RESULTS_BASE, "plots", "summary.png")
        return p if os.path.exists(p) else None

    prefix = {"qtable": "q_table", "dqn": "dqn", "comparison": "comparison"}.get(plot_type, plot_type)
    pattern = rf"{prefix}_level{level}_ep\d+"
    path = _find_across_runs(pattern, ext=".png")
    return path


# ── Agent loader ─────────────────────────────────────────────────────────────

def load_agent(level, agent_type):
    if agent_type == "qtable":
        path = _find_animation_model(level, "q_table")
        if not path:
            print(f"[!] Kein Q-Table-Modell für Level {level}")
            return None, False
        agent = QTableAgent(n_actions=5)
        agent.load(path)
        agent.epsilon = 0.0
        print(f"[OK] Q-Table: {os.path.basename(path)}")
        return agent, False
    else:
        path = _find_animation_model(level, "dqn")
        if not path:
            print(f"[!] Kein DQN-Modell für Level {level}")
            return None, True
        env_tmp = FootballEnv(level=level)
        agent = DQNAgent(state_size=env_tmp.get_state_size(), n_actions=5)
        agent.load(path)
        agent.epsilon = 0.0
        print(f"[OK] DQN: {os.path.basename(path)}")
        return agent, True


def _find_animation_model(level, prefix):
    """Find model file across all run dirs (prefer highest ep count); excludes legacy dirs."""
    pattern = rf"{prefix}_level{level}_ep\d+\.(?:pkl|pt)$"
    candidates = []
    for root, dirs, files in os.walk(config.RESULTS_BASE):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]
        for f in files:
            if re.search(pattern, f) and "_snapshots" not in f:
                full = os.path.join(root, f)
                m = re.search(r"_ep(\d+)", f)
                ep = int(m.group(1)) if m else 0
                candidates.append((ep, os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def load_gif(path):
    if not path or not os.path.exists(path):
        return [], 333
    gif = Image.open(path)
    frames, dur = [], gif.info.get("duration", 333)
    try:
        while True:
            frames.append(np.array(gif.convert("RGB")))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames, int(dur)


def load_png(path):
    if not path or not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("RGB"))


# ── Grid drawing (Live Play) ─────────────────────────────────────────────────

def draw_grid(ax, env, episode_count, total_reward, last_action, done):
    ax.cla()
    W, H = env.width, env.height
    ax.set_xlim(0, W)
    ax.set_ylim(-0.55, H + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    gx, gy = env.goal_pos

    for row in range(H):
        dy = H - 1 - row
        for col in range(W):
            is_goal     = col == gx and row == gy
            is_zone     = env.level == 1 and env.shoot_zone_x <= col < W - 1
            is_obstacle = env.level == 4 and (col, row) in env.obstacle_cells
            fc = (C["goal"] if is_goal else
                  C["obstacle"] if is_obstacle else
                  C["zone"] if is_zone else C["cell"])
            ec = C["goal_edge"] if is_goal else ("#555555" if is_obstacle else C["border"])
            lw = 2.2 if is_goal else 1.5 if is_obstacle else 0.9
            ax.add_patch(FancyBboxPatch(
                (col + 0.05, dy + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1))
            if is_obstacle:
                ax.text(col + 0.5, dy + 0.5, "#", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="#888888", zorder=2)

    ax.text(gx + 0.5, H - 1 - gy + 0.5, "G", ha="center", va="center",
            fontsize=16, fontweight="bold", color=C["goal_edge"], zorder=3)

    if env.level == 1:
        mid = env.shoot_zone_x + (W - 1 - env.shoot_zone_x) / 2
        ax.text(mid, H + 0.68, "Schusszone", ha="center", va="center",
                fontsize=7.5, color="#856404",
                bbox=dict(fc=C["zone"], ec="#FFC107", boxstyle="round,pad=0.25", lw=0.8))

    tm_has_ball = getattr(env, 'tm_has_ball', False)
    if not env.has_ball and not tm_has_ball:
        bx, by = env.ball_pos
        if 0 <= bx < W and 0 <= by < H:
            bdy = H - 1 - by
            ax.add_patch(plt.Circle((bx + 0.5, bdy + 0.5), 0.27, color=C["ball"], zorder=4))
            ax.text(bx + 0.5, bdy + 0.5, "B", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=5)

    if env.level >= 3 and env.level != 6:
        ox, oy = env.opp_pos
        if 0 <= ox < W and 0 <= oy < H:
            ody = H - 1 - oy
            ax.add_patch(FancyBboxPatch(
                (ox + 0.12, ody + 0.12), 0.76, 0.76,
                boxstyle="round,pad=0.04",
                facecolor=C["opp"], edgecolor="#7B0000", linewidth=1.8, zorder=4))
            ax.text(ox + 0.5, ody + 0.5, "X", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white", zorder=5)

    if env.level == 6:
        for opos, lbl, fc, edge in [
                (env.opp1_pos, "X", C["opp"],  "#7B0000"),
                (env.opp2_pos, "Y", C["opp2"], "#4A0070")]:
            ox, oy = opos
            if 0 <= ox < W and 0 <= oy < H:
                ody = H - 1 - oy
                ax.add_patch(FancyBboxPatch(
                    (ox + 0.12, ody + 0.12), 0.76, 0.76,
                    boxstyle="round,pad=0.04",
                    facecolor=fc, edgecolor=edge, linewidth=1.8, zorder=4))
                ax.text(ox + 0.5, ody + 0.5, lbl, ha="center", va="center",
                        fontsize=13, fontweight="bold", color="white", zorder=5)

    if env.level in (5, 6):
        tx, ty = env.tm_pos
        if 0 <= tx < W and 0 <= ty < H:
            tdy = H - 1 - ty
            fc_tm = C["tm_pball"] if tm_has_ball else C["teammate"]
            ax.add_patch(FancyBboxPatch(
                (tx + 0.12, tdy + 0.12), 0.76, 0.76,
                boxstyle="round,pad=0.04",
                facecolor=fc_tm, edgecolor="#1B5E20", linewidth=1.8, zorder=4))
            ax.text(tx + 0.5, tdy + 0.5, "M" if tm_has_ball else "T",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white", zorder=5)

    ax_x, ax_y = env.agent_pos
    ady = H - 1 - ax_y
    ax.add_patch(FancyBboxPatch(
        (ax_x + 0.12, ady + 0.12), 0.76, 0.76,
        boxstyle="round,pad=0.04",
        facecolor=C["pball"] if env.has_ball else C["agent"],
        edgecolor="#00001A", linewidth=2.0, zorder=6))
    ax.text(ax_x + 0.5, ady + 0.5, "P" if env.has_ball else "A",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", zorder=7)

    for col in range(W):
        ax.text(col + 0.5, -0.38, str(col), ha="center", va="center",
                fontsize=6.5, color=C["text_dim"])

    act_str = ACTION_NAMES[last_action] if last_action is not None else "—"
    ax.text(W / 2, H + 0.75,
            f"Schritt {env.step_count}" + ("  ✓ FERTIG" if done else ""),
            ha="center", va="center", fontsize=10, fontweight="bold", color=C["text"])
    ax.text(W / 2, H + 0.47,
            f"Aktion: {act_str}  |  Reward: {total_reward:+.0f}  |  Episoden: {episode_count}",
            ha="center", va="center", fontsize=8.5, color=C["text_dim"])

    handles = [
        mpatches.Patch(facecolor=C["agent"], label="Agent (A)"),
        mpatches.Patch(facecolor=C["pball"], label="Agent+Ball (P)"),
        mpatches.Patch(facecolor=C["ball"],  label="Ball (B)"),
        mpatches.Patch(facecolor=C["goal"],  edgecolor=C["goal_edge"], label="Tor (G)"),
    ]
    if env.level == 1:
        handles.append(mpatches.Patch(facecolor=C["zone"], label="Schusszone"))
    if env.level >= 3 and env.level != 6:
        handles.append(mpatches.Patch(facecolor=C["opp"], label="Gegner (X)"))
    if env.level == 4:
        handles.append(mpatches.Patch(facecolor=C["obstacle"], edgecolor="#555555",
                                      label="Hindernis (#)"))
    if env.level in (5, 6):
        handles.append(mpatches.Patch(facecolor=C["teammate"], label="Mitspieler (T)"))
        handles.append(mpatches.Patch(facecolor=C["tm_pball"], label="Mitspieler+Ball (M)"))
    if env.level == 6:
        handles.append(mpatches.Patch(facecolor=C["opp"],  label="Gegner 1 (X)"))
        handles.append(mpatches.Patch(facecolor=C["opp2"], label="Gegner 2 (Y)"))
    ax.legend(handles=handles, loc="lower center", fontsize=7.5, ncol=3,
              framealpha=0.92, edgecolor=C["border"], facecolor=C["cell"])


def draw_stats(ax, step_rewards, episode_rewards, agent_label, level):
    ax.cla()
    ax.set_facecolor("#111827")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])

    if step_rewards:
        steps = list(range(len(step_rewards)))
        cum = np.cumsum(step_rewards)
        ax.plot(steps, cum, color=C["agent"], linewidth=2, zorder=3)
        ax.fill_between(steps, 0, cum, where=cum >= 0, color=C["pos"], alpha=0.12)
        ax.fill_between(steps, 0, cum, where=cum <  0, color=C["neg"], alpha=0.15)
        ax.axhline(0, color=C["border"], linewidth=0.9, linestyle="--")

    ax.set_xlabel("Schritt", fontsize=9, color=C["text_dim"])
    ax.set_ylabel("Kum. Reward", fontsize=9, color=C["text_dim"])
    ax.tick_params(labelsize=8, colors=C["text_dim"])
    ax.set_title(f"Level {level}  —  {agent_label}",
                 fontsize=10, fontweight="bold", color=C["text"], pad=6)

    if episode_rewards:
        last_n = episode_rewards[-8:]
        txt = "\n".join(
            f"Ep {len(episode_rewards) - len(last_n) + i + 1:>3}: {r:+.0f}"
            for i, r in enumerate(last_n))
        ax.text(0.98, 0.98, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, family="monospace",
                color=C["text"],
                bbox=dict(fc="#1E2A4A", ec=C["border"], boxstyle="round,pad=0.4", lw=0.8))


# ── Application state ────────────────────────────────────────────────────────

class State:
    def __init__(self, level, agent_type):
        self.mode       = MODE_LIVE
        self.level      = level
        self.agent_type = agent_type
        self.plot_type  = "qtable"

        self.agent_ctx_idx = 0 if agent_type == "qtable" else 1
        self.plot_ctx_idx  = 0

        self.env             = FootballEnv(level=level)
        self.agent, self.is_dqn = load_agent(level, agent_type)
        self.playing         = False
        self.speed           = 2.0
        self.last_step_t     = 0.0
        self.episode_rewards = []
        self.episode_count   = 0
        self._reset_ep()

        self.gif_frames = []
        self.gif_dur_ms = 333
        self.gif_idx    = 0
        self.gif_t      = 0.0
        self.gif_play   = True
        self.gif_path   = None

    def _reset_ep(self):
        self.state        = self.env.reset()
        self.total_reward = 0.0
        self.last_action  = None
        self.step_rewards = []
        self.done         = False

    def reset(self):
        self._reset_ep()
        self.playing = False

    def step(self):
        if self.done or self.agent is None:
            return
        if self.is_dqn:
            action = self.agent.choose_action(self.env.state_to_array(self.state))
        else:
            action = self.agent.choose_action(self.state)
        nxt, r, done = self.env.step(action)
        self.total_reward += r
        self.last_action   = action
        self.step_rewards.append(r)
        self.state = nxt
        self.done  = done
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.total_reward)
            self.playing = False

    def reload_agent(self, level=None, agent_type=None):
        if level      is not None: self.level      = level
        if agent_type is not None: self.agent_type = agent_type
        self.env             = FootballEnv(level=self.level)
        self.agent, self.is_dqn = load_agent(self.level, self.agent_type)
        self.playing         = False
        self.episode_rewards = []
        self.episode_count   = 0
        self._reset_ep()

    def reload_gif(self):
        if self.mode == MODE_TRAIN:
            ag   = self.agent_type
            path = _find_training_evo(self.level, ag)
        elif self.mode == MODE_COMPARE:
            path = _find_compare_gif(self.level)
            if not path:
                # fall back to comparison PNG shown as static image
                path = _find_plot(self.level, "comparison")
                if path and path.endswith(".png"):
                    self.gif_frames = [load_png(path)] if load_png(path) is not None else []
                    self.gif_dur_ms = 5000
                    self.gif_idx    = 0
                    self.gif_t      = time.time()
                    self.gif_play   = False
                    self.gif_path   = path
                    print(f"[Plot] Vergleich-PNG als Fallback: {os.path.basename(path)}")
                    return
        else:
            self.gif_frames = []
            return

        if path:
            self.gif_frames, self.gif_dur_ms = load_gif(path)
            self.gif_path = path
            print(f"[GIF] {os.path.basename(path)}  ({len(self.gif_frames)} frames)")
        else:
            self.gif_frames, self.gif_dur_ms = [], 333
            self.gif_path = None
            ag = self.agent_type if self.mode == MODE_TRAIN else "—"
            print(f"[!] Kein GIF für Level {self.level} / {ag}")
        self.gif_idx  = 0
        self.gif_t    = time.time()
        self.gif_play = True

    def get_plot_image(self):
        lv, pt = self.level, self.plot_type
        path = _find_plot(lv, pt)
        return load_png(path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=[1, 2, 3, 4, 5, 6], default=1)
    ap.add_argument("--agent", choices=["qtable", "dqn"], default="qtable")
    args = ap.parse_args()

    s = State(args.level, args.agent)

    plt.rcParams.update({"text.color": C["text"], "axes.labelcolor": C["text_dim"],
                         "xtick.color": C["text_dim"], "ytick.color": C["text_dim"]})

    fig = plt.figure(figsize=(15, 9), facecolor=C["bg"])
    fig.suptitle("RL Football Dashboard", fontsize=13, fontweight="bold",
                 color=C["text"], y=0.995)

    # ── Sidebar: Level selector ───────────────────────────────────────────
    ax_lv = fig.add_axes([0.01, 0.54, 0.12, 0.39], facecolor=C["cell"])
    ax_lv.set_title("Level", fontsize=9, pad=4, color=C["text"])
    radio_lv = RadioButtons(
        ax_lv,
        ("Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6"),
        active=s.level - 1, activecolor=C["agent"])
    for lb in radio_lv.labels:
        lb.set_color(C["text"]); lb.set_fontsize(9)

    # ── Sidebar: Level description ────────────────────────────────────────
    ax_desc = fig.add_axes([0.01, 0.40, 0.12, 0.12], facecolor="#0D1117")
    ax_desc.axis("off")
    desc_text = ax_desc.text(
        0.5, 0.5, LEVEL_DESC.get(s.level, ""),
        ha="center", va="center", fontsize=7.0, color=C["text_dim"],
        transform=ax_desc.transAxes, multialignment="center")

    def _update_desc(level):
        desc_text.set_text(LEVEL_DESC.get(level, ""))
        fig.canvas.draw_idle()

    # ── Sidebar: Kontext-Titel ────────────────────────────────────────────
    ax_ctx_lbl = fig.add_axes([0.01, 0.36, 0.12, 0.04])
    ax_ctx_lbl.axis("off")
    ax_ctx_lbl.set_facecolor(C["bg"])
    ctx_title = ax_ctx_lbl.text(0.5, 0.5, "Agent", ha="center", va="center",
                                fontsize=9, fontweight="bold", color=C["text"],
                                transform=ax_ctx_lbl.transAxes)

    # ── Sidebar: 4 Kontext-Buttons ────────────────────────────────────────
    CTX_Y = [0.26, 0.17, 0.08, -0.01]
    ctx_axes = []
    ctx_btns = []
    for y in CTX_Y:
        axt = fig.add_axes([0.015, y, 0.11, 0.075])
        btn = Button(axt, "", color=C["btn_unsel"], hovercolor="#2A3A6A")
        btn.label.set_fontsize(9)
        btn.label.set_color(C["text"])
        ctx_axes.append(axt)
        ctx_btns.append(btn)

    def _update_ctx(mode):
        opts  = CTX[mode]["opts"]
        title = CTX[mode]["title"]
        ctx_title.set_text(title)
        ax_ctx_lbl.set_visible(bool(title))
        sel = s.agent_ctx_idx if mode in (MODE_LIVE, MODE_TRAIN) else \
              s.plot_ctx_idx  if mode == MODE_PLOTS else -1
        for i, (axt, btn) in enumerate(zip(ctx_axes, ctx_btns)):
            label = opts[i]
            if label:
                axt.set_visible(True)
                btn.label.set_text(label)
                col = C["btn_sel"] if i == sel else C["btn_unsel"]
                axt.set_facecolor(col)
                btn.color = col
            else:
                axt.set_visible(False)

    # ── Mode tabs ─────────────────────────────────────────────────────────
    TAB_DEFS = [
        ("Live Play",    MODE_LIVE),
        ("Training Evo", MODE_TRAIN),
        ("Vergleich",    MODE_COMPARE),
        ("Plots",        MODE_PLOTS),
    ]
    tab_axes = []
    tab_btns = []
    for i, (label, _) in enumerate(TAB_DEFS):
        axt = fig.add_axes([0.15 + i * 0.214, 0.92, 0.20, 0.06])
        col = C["tab_on"] if i == 0 else C["tab_off"]
        btn = Button(axt, label, color=col, hovercolor="#2563EB")
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")
        btn.label.set_color(C["text"])
        tab_axes.append(axt)
        tab_btns.append(btn)

    def _update_tabs():
        order = [MODE_LIVE, MODE_TRAIN, MODE_COMPARE, MODE_PLOTS]
        for i, m in enumerate(order):
            col = C["tab_on"] if m == s.mode else C["tab_off"]
            tab_axes[i].set_facecolor(col)
            tab_btns[i].color = col

    # ── Live Play content ─────────────────────────────────────────────────
    ax_grid  = fig.add_axes([0.16, 0.14, 0.43, 0.76], facecolor=C["bg"])
    ax_stats = fig.add_axes([0.62, 0.14, 0.36, 0.76], facecolor="#111827")

    ax_bp  = fig.add_axes([0.16, 0.03, 0.09, 0.07])
    ax_bs  = fig.add_axes([0.27, 0.03, 0.09, 0.07])
    ax_br  = fig.add_axes([0.38, 0.03, 0.09, 0.07])
    ax_spd = fig.add_axes([0.54, 0.055, 0.30, 0.035])

    btn_play  = Button(ax_bp, "Play",  color="#166534", hovercolor="#14532D")
    btn_step  = Button(ax_bs, "Step",  color="#1E3A5F", hovercolor="#1E40AF")
    btn_reset = Button(ax_br, "Reset", color="#7F1D1D", hovercolor="#991B1B")
    slider_spd = Slider(ax_spd, "Geschw.", 0.25, 10.0, valinit=s.speed,
                        valstep=0.25, color=C["agent"])
    ax_spd.set_xlabel("Schritte / Sek.", fontsize=7.5, labelpad=1)
    for b in (btn_play, btn_step, btn_reset):
        b.label.set_color(C["text"])
        b.label.set_fontsize(10)
        b.label.set_fontweight("bold")

    LIVE_AXES = [ax_grid, ax_stats, ax_bp, ax_bs, ax_br]

    # ── GIF / Image content ───────────────────────────────────────────────
    ax_img = fig.add_axes([0.16, 0.10, 0.82, 0.80], facecolor=C["bg"])
    ax_gp  = fig.add_axes([0.16, 0.02, 0.13, 0.06])
    ax_gr  = fig.add_axes([0.31, 0.02, 0.13, 0.06])
    btn_gp = Button(ax_gp, "|| Pause",   color="#166534", hovercolor="#14532D")
    btn_gr = Button(ax_gr, "<< Neustart", color="#7F1D1D", hovercolor="#991B1B")
    for b in (btn_gp, btn_gr):
        b.label.set_color(C["text"])
        b.label.set_fontsize(10)
        b.label.set_fontweight("bold")

    GIF_CTRL = [ax_gp, ax_gr]

    for ax in [ax_img] + GIF_CTRL:
        ax.set_visible(False)
    ax_spd.set_visible(True)

    # ── Content draw helpers ──────────────────────────────────────────────
    def _draw_live():
        label = "DQN" if s.is_dqn else "Q-Table"
        draw_grid(ax_grid, s.env, s.episode_count, s.total_reward, s.last_action, s.done)
        draw_stats(ax_stats, s.step_rewards, s.episode_rewards, label, s.level)

    def _draw_gif():
        ax_img.cla()
        ax_img.set_facecolor(C["bg"])
        ax_img.axis("off")
        if s.gif_frames:
            ax_img.imshow(s.gif_frames[s.gif_idx])
            total = len(s.gif_frames)
            pct   = int((s.gif_idx + 1) / total * 100)
            ag    = "Q-Table" if s.agent_type == "qtable" else "DQN"
            if s.mode == MODE_TRAIN:
                mode_label = f"Training Evo — {ag}"
            else:
                mode_label = "Vergleich"
            fname = os.path.basename(s.gif_path) if s.gif_path else ""
            ax_img.set_title(
                f"{mode_label}  —  Level {s.level}  |  "
                f"Frame {s.gif_idx + 1}/{total}  ({pct}%)  |  {fname}",
                fontsize=8.5, color=C["text_dim"], pad=4)
        else:
            ag = "Q-Table" if s.agent_type == "qtable" else "DQN"
            if s.mode == MODE_TRAIN:
                hint = f"training_evolution_{s.agent_type}_level{s.level}_ep*.gif"
                gen_cmd = f"python src/visualization/animate_model_training.py --level {s.level} --agent {s.agent_type}"
            else:
                hint = f"compare_level{s.level}_ep*.gif"
                gen_cmd = f"python src/visualization/animate_trained_model.py --level {s.level}"
            ax_img.text(0.5, 0.5,
                        f"Kein GIF gefunden für Level {s.level} / {ag}\n\n"
                        f"Erwarteter Dateiname: {hint}\n"
                        f"Generieren mit: {gen_cmd}",
                        ha="center", va="center", transform=ax_img.transAxes,
                        fontsize=10, color=C["text_dim"])

    def _draw_plots():
        ax_img.cla()
        ax_img.set_facecolor(C["bg"])
        ax_img.axis("off")
        img = s.get_plot_image()
        if img is not None:
            ax_img.imshow(img)
        else:
            pt_label = {"qtable": "Q-Table", "dqn": "DQN",
                        "comparison": "Vergleich", "summary": "Übersicht"}.get(s.plot_type, "")
            ax_img.text(0.5, 0.5, f"Plot nicht gefunden\n({pt_label} / Level {s.level})",
                        ha="center", va="center", transform=ax_img.transAxes,
                        fontsize=12, color=C["text_dim"])

    def refresh():
        if   s.mode == MODE_LIVE:  _draw_live()
        elif s.mode == MODE_PLOTS: _draw_plots()
        else:                      _draw_gif()
        fig.canvas.draw_idle()

    # ── Mode switching ────────────────────────────────────────────────────
    def set_mode(mode):
        s.mode = mode
        _update_tabs()
        _update_ctx(mode)

        if mode == MODE_LIVE:
            for ax in LIVE_AXES: ax.set_visible(True)
            ax_spd.set_visible(True)
            ax_spd.set_xlabel("Schritte / Sek.", fontsize=7.5, labelpad=1)
            ax_img.set_visible(False)
            for ax in GIF_CTRL: ax.set_visible(False)
            s.playing = False
            btn_play.label.set_text("Play")

        elif mode in (MODE_TRAIN, MODE_COMPARE):
            for ax in LIVE_AXES: ax.set_visible(False)
            ax_spd.set_visible(True)
            ax_spd.set_xlabel("Abspielgeschw. (×)", fontsize=7.5, labelpad=1)
            ax_img.set_visible(True)
            for ax in GIF_CTRL: ax.set_visible(True)
            s.reload_gif()
            btn_gp.label.set_text("|| Pause")

        elif mode == MODE_PLOTS:
            for ax in LIVE_AXES: ax.set_visible(False)
            ax_spd.set_visible(False)
            ax_img.set_visible(True)
            for ax in GIF_CTRL: ax.set_visible(False)

        refresh()

    # ── Tab callbacks ─────────────────────────────────────────────────────
    tab_btns[0].on_clicked(lambda e: set_mode(MODE_LIVE))
    tab_btns[1].on_clicked(lambda e: set_mode(MODE_TRAIN))
    tab_btns[2].on_clicked(lambda e: set_mode(MODE_COMPARE))
    tab_btns[3].on_clicked(lambda e: set_mode(MODE_PLOTS))

    # ── Level callback ────────────────────────────────────────────────────
    def on_level(label):
        lv = int(label.split()[1])
        s.level = lv
        _update_desc(lv)
        if s.mode == MODE_LIVE:
            s.reload_agent(level=lv)
            btn_play.label.set_text("Play")
        elif s.mode in (MODE_TRAIN, MODE_COMPARE):
            s.reload_gif()
        refresh()

    radio_lv.on_clicked(on_level)

    # ── Kontext-Button callbacks ──────────────────────────────────────────
    def _make_ctx_cb(i):
        def cb(e):
            opts = CTX[s.mode]["opts"]
            if not opts[i]:
                return
            if s.mode in (MODE_LIVE, MODE_TRAIN):
                s.agent_ctx_idx = i
                at = "qtable" if i == 0 else "dqn"
                s.agent_type = at
                if s.mode == MODE_LIVE:
                    s.reload_agent(agent_type=at)
                    btn_play.label.set_text("Play")
                else:
                    s.reload_gif()
            elif s.mode == MODE_PLOTS:
                s.plot_ctx_idx = i
                s.plot_type = ["qtable", "dqn", "comparison", "summary"][i]
            _update_ctx(s.mode)
            refresh()
        return cb

    for i, btn in enumerate(ctx_btns):
        btn.on_clicked(_make_ctx_cb(i))

    # ── Live Play button callbacks ─────────────────────────────────────────
    def on_play(e):
        if s.mode != MODE_LIVE or s.agent is None:
            return
        if s.done:
            s.reset()
        s.playing = not s.playing
        btn_play.label.set_text("Pause" if s.playing else "Play")
        s.last_step_t = time.time()
        refresh()

    def on_step(e):
        if s.mode != MODE_LIVE:
            return
        s.playing = False
        btn_play.label.set_text("Play")
        s.step()
        refresh()

    def on_reset(e):
        if s.mode != MODE_LIVE:
            return
        s.reset()
        btn_play.label.set_text("Play")
        refresh()

    def on_speed(val):
        s.speed = slider_spd.val

    btn_play.on_clicked(on_play)
    btn_step.on_clicked(on_step)
    btn_reset.on_clicked(on_reset)
    slider_spd.on_changed(on_speed)

    # ── GIF button callbacks ───────────────────────────────────────────────
    def on_gif_play(e):
        s.gif_play = not s.gif_play
        btn_gp.label.set_text("|| Pause" if s.gif_play else "> Play")

    def on_gif_reset(e):
        s.gif_idx  = 0
        s.gif_t    = time.time()
        s.gif_play = True
        btn_gp.label.set_text("|| Pause")
        refresh()

    btn_gp.on_clicked(on_gif_play)
    btn_gr.on_clicked(on_gif_reset)

    # ── Animation loop ─────────────────────────────────────────────────────
    def animate(_):
        if s.mode == MODE_LIVE:
            if s.playing and not s.done:
                now = time.time()
                if now - s.last_step_t >= 1.0 / s.speed:
                    s.step()
                    s.last_step_t = now
                    if s.done:
                        btn_play.label.set_text("Play")
                    _draw_live()
                    fig.canvas.draw_idle()

        elif s.mode in (MODE_TRAIN, MODE_COMPARE):
            if s.gif_play and s.gif_frames:
                now = time.time()
                delay = (s.gif_dur_ms / 1000) / max(s.speed, 0.25)
                if now - s.gif_t >= delay:
                    s.gif_idx = (s.gif_idx + 1) % len(s.gif_frames)
                    s.gif_t   = now
                    _draw_gif()
                    fig.canvas.draw_idle()
        return []

    anim = FuncAnimation(fig, animate, interval=40, blit=False, cache_frame_data=False)

    _update_ctx(MODE_LIVE)
    refresh()
    plt.show()
    _ = anim


if __name__ == "__main__":
    main()
