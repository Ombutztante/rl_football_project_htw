"""
Generate a presentation-ready summary figure from all training logs.

python src/summary_plot.py

Output: results/plots/summary.png
"""

import matplotlib
matplotlib.use("Agg")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

import config

WINDOW = 50

C_QT  = "#4FC3F7"   # Q-Table: light blue
C_DQN = "#CE93D8"   # DQN: light purple
C_BG  = "#0F0F1A"   # dark background
C_PAN = "#1A1A2E"   # panel background
C_TXT = "#ECF0F1"   # main text
C_DIM = "#78909C"   # dim text
C_GRD = "#2A3A6A"   # grid lines


def _load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _rolling(values, w):
    out = []
    for i in range(len(values)):
        s = max(0, i - w + 1)
        out.append(np.mean(values[s: i + 1]))
    return np.array(out)


def _final_stats(log, last_n=100):
    """Return (avg_reward, goal_rate%) averaged over the last N episodes."""
    recent = log[-last_n:]
    avg_r = np.mean([e["reward"] for e in recent])
    goal  = np.mean([float(e["goal"]) for e in recent]) * 100
    return avg_r, goal


def main():
    levels = [1, 2, 3, 4]

    # Load all logs
    qt_logs  = {lv: _load(os.path.join(config.LOGS_DIR, f"q_table_level{lv}_ep*.json"))
                for lv in levels}
    dqn_logs = {lv: _load(os.path.join(config.LOGS_DIR, f"dqn_level{lv}_ep*.json"))
                for lv in levels}

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 9), facecolor=C_BG)
    fig.text(0.5, 0.965,
             "Q-Table  vs.  DQN  —  Ergebnisübersicht",
             ha="center", va="top", fontsize=16, fontweight="bold", color=C_TXT)
    fig.text(0.5, 0.935,
             "Reinforcement Learning  |  Computational Intelligence  |  HTW Berlin",
             ha="center", va="top", fontsize=10, color=C_DIM)

    n_levels = len(levels)
    # Top row: goal-rate curves per level  (n_levels panels)
    # Bottom:  final performance bars      (1 wide panel)
    gs = gridspec.GridSpec(
        2, n_levels,
        figure=fig,
        top=0.90, bottom=0.10,
        left=0.06, right=0.97,
        hspace=0.52, wspace=0.30,
        height_ratios=[1.15, 0.85],
    )

    # -----------------------------------------------------------------------
    # Top row — learning curves (goal rate) per level
    # -----------------------------------------------------------------------
    for col, lv in enumerate(levels):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(C_PAN)
        ax.tick_params(colors=C_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(C_GRD)
        ax.grid(True, color=C_GRD, linewidth=0.5, alpha=0.5)

        qt  = qt_logs[lv]
        dqn = dqn_logs[lv]
        n   = max(
            (qt[-1]["episode"]  if qt  else 0),
            (dqn[-1]["episode"] if dqn else 0),
        )

        if qt:
            ep  = [e["episode"] for e in qt]
            gr  = _rolling([float(e["goal"]) * 100 for e in qt], WINDOW)
            ax.plot(ep, gr, color=C_QT, linewidth=2.0, label="Q-Table")
            ax.fill_between(ep, 0, gr, color=C_QT, alpha=0.12)

        if dqn:
            ep  = [e["episode"] for e in dqn]
            gr  = _rolling([float(e["goal"]) * 100 for e in dqn], WINDOW)
            ax.plot(ep, gr, color=C_DQN, linewidth=2.0, label="DQN",
                    linestyle="--")
            ax.fill_between(ep, 0, gr, color=C_DQN, alpha=0.10)

        ax.set_xlim(1, n)
        ax.set_ylim(0, 108)
        ax.set_title(f"Level {lv}", fontsize=11, fontweight="bold",
                     color=C_TXT, pad=6)
        ax.set_xlabel("Episode", fontsize=8, color=C_DIM)
        if col == 0:
            ax.set_ylabel(f"Tor-Rate % (Ø {WINDOW} Ep.)", fontsize=8, color=C_DIM)
        ax.legend(fontsize=8, loc="lower right",
                  facecolor=C_PAN, edgecolor=C_GRD, labelcolor=C_TXT)

    # -----------------------------------------------------------------------
    # Bottom row — final performance bar chart (spans all 3 columns)
    # -----------------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor(C_PAN)
    ax_bar.tick_params(colors=C_DIM, labelsize=9)
    for spine in ax_bar.spines.values():
        spine.set_color(C_GRD)
    ax_bar.grid(True, axis="y", color=C_GRD, linewidth=0.5, alpha=0.5)

    bar_w  = 0.32
    gap    = 1.0
    n_lv   = len(levels)
    x_base = np.arange(n_lv) * gap

    for i, lv in enumerate(levels):
        qt_r,  qt_g  = _final_stats(qt_logs[lv])  if qt_logs[lv]  else (0, 0)
        dqn_r, dqn_g = _final_stats(dqn_logs[lv]) if dqn_logs[lv] else (0, 0)

        x_qt  = x_base[i] - bar_w / 2 - 0.02
        x_dqn = x_base[i] + bar_w / 2 + 0.02

        b1 = ax_bar.bar(x_qt,  qt_g,  bar_w, color=C_QT,  alpha=0.9,
                        label="Q-Table" if i == 0 else "")
        b2 = ax_bar.bar(x_dqn, dqn_g, bar_w, color=C_DQN, alpha=0.9,
                        label="DQN" if i == 0 else "")

        # Value labels on bars
        for bar, val in [(b1, qt_g), (b2, dqn_g)]:
            ax_bar.text(bar[0].get_x() + bar[0].get_width() / 2,
                        val + 1.5, f"{val:.0f}%",
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=C_TXT)

    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels([f"Level {lv}" for lv in levels],
                           fontsize=10, color=C_TXT)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_ylabel("Finale Tor-Rate %\n(Ø letzte 100 Ep.)",
                      fontsize=9, color=C_DIM)
    ax_bar.set_title("Finale Leistung im Vergleich",
                     fontsize=11, fontweight="bold", color=C_TXT, pad=8)
    ax_bar.legend(fontsize=9, loc="upper left",
                  facecolor=C_PAN, edgecolor=C_GRD, labelcolor=C_TXT)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    out = os.path.join(config.PLOTS_DIR, "summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Zusammenfassung gespeichert: {out}")


if __name__ == "__main__":
    main()
