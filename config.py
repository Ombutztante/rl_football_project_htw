# Grid
GRID_WIDTH = 10
GRID_HEIGHT = 6

# Active level (1, 2 or 3)
LEVEL = 1

# Episode settings
MAX_STEPS = 300
N_EPISODES = 1000

# Level 1 shooting zone: 2D box around the goal.
# Valid shoot position: agent_x >= SHOOT_ZONE_X  AND  |agent_y - goal_y| <= SHOOT_ZONE_Y_RADIUS
# On a 10×6 grid with goal at (9,3): zone cells are (8,2),(8,3),(8,4),(9,2),(9,4)
SHOOT_ZONE_X = GRID_WIDTH - 2  # = 8
SHOOT_ZONE_Y_RADIUS = 1        # rows around goal row that count as shooting zone

# Ball starting x-column per level (y is always height // 2)
# Level 1: centre — agent must navigate to ball, then reach shooting zone
# Level 2+: column 1 (near agent) — full field to cross, dribble-vs-pass decision has real weight
# None → falls back to width // 2
BALL_START_X_L1 = None
BALL_START_X_L2 = 1
BALL_START_X_L3 = 1
BALL_START_X_L4 = 1

# Rewards — shared across all levels
REWARD_STEP = -1           # every step
REWARD_BALL_PICKUP = 5     # picking up the ball
REWARD_CLOSER = 1          # moved closer to goal (shaping, movement actions)

# Rewards — Level 1
REWARD_GOAL = 30             # goal scored via shoot from zone (aligned with goal row)
REWARD_SHOOT_NO_BALL = -5    # shoot without ball
REWARD_SHOOT_BAD_POS = -5    # shoot from outside shooting zone
REWARD_SHOOT_ZONE_MISS = -3  # shoot in zone but wrong row (off by 1 row)
REWARD_GOAL_ROW_ALIGN = 1    # agent has ball and is on goal row (alignment shaping)

# Rewards — Level 2
SHOOT_RANGE = 3            # cells the ball travels on a forward pass
REWARD_GOAL_L2 = 40        # goal scored (dribble to goal or pass landing on goal)
REWARD_PASS_CLOSER = 2     # ball advanced toward goal via forward pass
REWARD_BALL_OUT = -5       # ball exits right wall without scoring
REWARD_SHOOT_WASTED = -3   # shoot without ball

# Rewards — Level 3
REWARD_GOAL_L3 = 50        # goal scored in Level 3
REWARD_BAD_SHOT_L3 = -5    # shoot without ball in Level 3
REWARD_OPP_REACHES_BALL = -10  # opponent reaches loose ball → episode ends
REWARD_BALL_LOST = -20     # opponent tackles agent who has ball → episode ends

# Level 4 — static obstacle (vertical wall, 1 column wide, 4 rows tall)
# Rows 0–3 at column 6 are blocked; rows 4–5 are free to pass around below.
OBSTACLE_X       = 6  # column of the obstacle
OBSTACLE_Y_START = 0  # first blocked row (0 = top)
OBSTACLE_HEIGHT  = 4  # number of blocked rows

# Rewards — Level 4 (extends Level 3 + obstacle)
REWARD_GOAL_L4         = 60  # goal scored (harder than L3, so higher reward)
REWARD_HIT_OBSTACLE    = -2  # agent walks into obstacle wall
REWARD_SHOT_BLOCKED    = -5  # shot intercepted by obstacle
REWARD_BYPASS_OBSTACLE =  2  # agent carries ball through free corridor (y >= OBSTACLE_HEIGHT) past obstacle

# DQN Level 4 — slower epsilon decay so the agent explores long enough to find the detour
# With decay=0.995 epsilon reaches 0.05 at ~ep600, too early for a multi-step obstacle bypass.
# With decay=0.998 epsilon reaches 0.05 at ~ep1800, giving enough exploration on 2000 episodes.
DQN_EPSILON_DECAY_L4 = 0.998

# Level 3 opponent
# Opponent starts at x = (GRID_WIDTH - 1) - OPP_START_X_FROM_GOAL, y = 0 (top row)
# With defaults (10×6 grid): x = 9 - 1 = 8, y = 0
# Tune OPP_START_X_FROM_GOAL and OPP_MOVE_EVERY to adjust difficulty.
# On larger grids the agent has more room to manoeuvre before the opponent closes in.
OPP_START_X_FROM_GOAL = 1  # columns left of goal where opponent starts
OPP_MOVE_EVERY = 2         # opponent moves 1 cell every N agent steps (1 = every step)

# Q-Learning
Q_LR = 0.1
Q_GAMMA = 0.99
Q_EPSILON_START = 1.0
Q_EPSILON_MIN = 0.05
Q_EPSILON_DECAY = 0.995

# DQN (PyTorch)
DQN_LR = 1e-4              # 1e-3 caused Q-value explosion; 1e-4 is more stable
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_MIN = 0.05
DQN_EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 15000        # slightly larger buffer keeps diverse experiences longer
TAU = 0.005                # soft target update factor: target = τ·online + (1-τ)·target
REPLAY_WARMUP = 500        # minimum buffer size before learning starts
GRAD_CLIP_NORM = 1.0       # max gradient norm for clipping
HIDDEN_SIZE = 128

# Output paths — updated by setup_run_dir() or set_run_dir() at runtime
RESULTS_BASE   = "results"
MODELS_DIR     = "results/models"
LOGS_DIR       = "results/logs"
PLOTS_DIR      = "results/plots"
ANIMATIONS_DIR = "results/animations"


# ---------------------------------------------------------------------------
# Run-directory helpers
# ---------------------------------------------------------------------------
import os as _os
import re as _re
import glob as _glob


def _current_branch():
    try:
        import subprocess
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return branch.replace("/", "-") or "unknown"
    except Exception:
        return "unknown"


def _apply_run_dir(run_dir, create_dirs=False):
    """Update module-level path constants to point inside run_dir."""
    global MODELS_DIR, LOGS_DIR, PLOTS_DIR, ANIMATIONS_DIR
    MODELS_DIR     = _os.path.join(run_dir, "models")
    LOGS_DIR       = _os.path.join(run_dir, "logs")
    PLOTS_DIR      = _os.path.join(run_dir, "plots")
    ANIMATIONS_DIR = _os.path.join(run_dir, "animations")
    if create_dirs:
        for d in [MODELS_DIR, LOGS_DIR, PLOTS_DIR, ANIMATIONS_DIR]:
            _os.makedirs(d, exist_ok=True)


def setup_run_dir():
    """
    Create a new numbered run directory and update all path constants.
    Call once at the start of every training script.

    Naming: results/{branch}_{DDMM}_{counter}/
    Example: results/a_dev_4_2206_1/
    """
    from datetime import date as _date
    branch   = _current_branch()
    date_str = _date.today().strftime("%d%m")
    prefix   = f"{branch}_{date_str}_"
    existing = _glob.glob(_os.path.join(RESULTS_BASE, f"{prefix}*"))
    nums = []
    for p in existing:
        m = _re.search(r"(\d+)$", _os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    counter  = max(nums) + 1 if nums else 1
    run_dir  = _os.path.join(RESULTS_BASE, f"{prefix}{counter}")
    _apply_run_dir(run_dir, create_dirs=True)
    print(f"Run-Verzeichnis: {run_dir}")
    return run_dir


def set_run_dir(name_or_path):
    """
    Point all path constants to an existing run directory (for visualization).
    Accepts a run name ('a_dev_4_2206_1') or a full path.
    """
    if _os.path.isabs(name_or_path):
        run_dir = name_or_path
    else:
        run_dir = _os.path.join(RESULTS_BASE, name_or_path)
    _apply_run_dir(run_dir, create_dirs=True)
    return run_dir


def latest_run():
    """Return the name of the most recently modified run directory, or None."""
    dirs = [
        p for p in _glob.glob(_os.path.join(RESULTS_BASE, "*_*_*"))
        if _os.path.isdir(p)
    ]
    if not dirs:
        return None
    return _os.path.basename(max(dirs, key=_os.path.getmtime))


def list_runs():
    """Print all available run directories."""
    dirs = sorted([
        _os.path.basename(p)
        for p in _glob.glob(_os.path.join(RESULTS_BASE, "*_*_*"))
        if _os.path.isdir(p)
    ])
    if dirs:
        print("Verfügbare Runs:")
        for d in dirs:
            print(f"  {d}")
    else:
        print("Keine Runs gefunden.")
    return dirs
