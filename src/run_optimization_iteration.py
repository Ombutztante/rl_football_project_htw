#!/usr/bin/env python3
"""
Optimization loop iteration runner.

Trains Q-Table and DQN on all levels (1-5) for 1000 and 3000 episodes.
All results land in a single run directory: results/opt_iter{N}_{DDMM}/

Usage:
    python src/run_optimization_iteration.py --iteration 0
    python src/run_optimization_iteration.py --iteration 1 --levels 1 2 3
    python src/run_optimization_iteration.py --iteration 2 --episodes 1000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import date

import config
from src.train_q_table import train as _train_qtable
from src.train_dqn import train as _train_dqn
from src.visualization.plot_training import plot_training, plot_comparison


LEVELS_DEFAULT   = [1, 2, 3, 4, 5]
EPISODES_DEFAULT = [1000, 3000]
METRICS_WINDOW   = 100   # last N episodes for goal-rate / avg-reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_metrics(logs_dir, stem):
    path = os.path.join(logs_dir, f"{stem}.json")
    if not os.path.exists(path):
        return {"goal_rate_pct": None, "avg_reward": None}
    with open(path) as f:
        log = json.load(f)
    last = log[-min(METRICS_WINDOW, len(log)):]
    return {
        "goal_rate_pct": round(float(np.mean([e["goal"]   for e in last])) * 100, 1),
        "avg_reward":    round(float(np.mean([e["reward"] for e in last])),       1),
    }


def _plot_pair(logs_dir, plots_dir, qt_stem, dqn_stem, level, ep_count):
    """Generate individual and comparison plots for one level/episode pair."""
    qt_log  = os.path.join(logs_dir, f"{qt_stem}.json")
    dqn_log = os.path.join(logs_dir, f"{dqn_stem}.json")

    if os.path.exists(qt_log):
        plot_training(
            qt_log,
            title=f"Q-Table – Level {level} – {ep_count} Ep.",
            save_path=os.path.join(plots_dir, f"{qt_stem}.png"),
        )
    if os.path.exists(dqn_log):
        plot_training(
            dqn_log,
            title=f"DQN – Level {level} – {ep_count} Ep.",
            save_path=os.path.join(plots_dir, f"{dqn_stem}.png"),
        )
    if os.path.exists(qt_log) and os.path.exists(dqn_log):
        plot_comparison(
            qt_log, dqn_log,
            level=level,
            save_path=os.path.join(plots_dir,
                                   f"comparison_level{level}_ep{ep_count}.png"),
        )


def _print_table(summary, iteration):
    print(f"\n{'='*72}")
    print(f"ERGEBNISSE  —  Iteration {iteration}  (letzte {METRICS_WINDOW} Episoden)")
    print(f"{'='*72}")
    hdr = f"{'':22s}  {'Goal% (QT)':>10}  {'AvgR (QT)':>9}  {'Goal% (DQN)':>11}  {'AvgR (DQN)':>10}"
    print(hdr)
    print("─" * 72)
    for key in sorted(summary.keys()):
        v   = summary[key]
        qt  = v["qtable"]
        dqn = v["dqn"]
        g_qt  = f"{qt['goal_rate_pct']:.1f}%"  if qt['goal_rate_pct']  is not None else "–"
        r_qt  = f"{qt['avg_reward']:.1f}"       if qt['avg_reward']     is not None else "–"
        g_dqn = f"{dqn['goal_rate_pct']:.1f}%"  if dqn['goal_rate_pct'] is not None else "–"
        r_dqn = f"{dqn['avg_reward']:.1f}"       if dqn['avg_reward']    is not None else "–"
        print(f"{key:22s}  {g_qt:>10}  {r_qt:>9}  {g_dqn:>11}  {r_dqn:>10}")
    print("─" * 72)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_iteration(iteration, levels=None, episodes=None):
    levels   = levels   or LEVELS_DEFAULT
    episodes = episodes or EPISODES_DEFAULT

    date_str = date.today().strftime("%d%m")
    run_name = f"opt_iter{iteration}_{date_str}"
    config.set_run_dir(run_name)
    os.makedirs(config.MODELS_DIR,     exist_ok=True)
    os.makedirs(config.LOGS_DIR,       exist_ok=True)
    os.makedirs(config.PLOTS_DIR,      exist_ok=True)
    os.makedirs(config.ANIMATIONS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION ITERATION {iteration}")
    print(f"Run-Verzeichnis : results/{run_name}")
    print(f"Level           : {levels}")
    print(f"Episoden        : {episodes}")
    print(f"{'='*60}")

    summary = {}

    for ep_count in episodes:
        for level in levels:
            config.LEVEL      = level
            config.N_EPISODES = ep_count

            print(f"\n{'─'*55}")
            print(f"  Q-Table  |  Level {level}  |  {ep_count} Episoden")
            print(f"{'─'*55}")
            _train_qtable(n_snapshots=5)
            qt_stem = f"q_table_level{level}_ep{ep_count}"

            print(f"\n{'─'*55}")
            print(f"  DQN      |  Level {level}  |  {ep_count} Episoden")
            print(f"{'─'*55}")
            _train_dqn(n_snapshots=5)
            dqn_stem = f"dqn_level{level}_ep{ep_count}"

            _plot_pair(config.LOGS_DIR, config.PLOTS_DIR,
                       qt_stem, dqn_stem, level, ep_count)

            key = f"L{level}_ep{ep_count}"
            summary[key] = {
                "qtable": _extract_metrics(config.LOGS_DIR, qt_stem),
                "dqn":    _extract_metrics(config.LOGS_DIR, dqn_stem),
            }

    _print_table(summary, iteration)

    summary_path = os.path.join(config.LOGS_DIR, "iteration_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"iteration": iteration, "run": run_name, "results": summary}, f, indent=2)
    print(f"\nSummary gespeichert → {summary_path}")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Führt eine vollständige Optimierungsiteration durch.")
    ap.add_argument("--iteration", type=int, required=True,
                    help="Iterationsnummer (0 = Baseline)")
    ap.add_argument("--levels", nargs="+", type=int, default=None,
                    help="Level-Liste  (Standard: 1 2 3 4 5)")
    ap.add_argument("--episodes", nargs="+", type=int, default=None,
                    help="Episodenzahlen  (Standard: 1000 3000)")
    args = ap.parse_args()

    run_iteration(
        iteration=args.iteration,
        levels=args.levels,
        episodes=args.episodes,
    )
