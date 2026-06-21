# Grid
GRID_WIDTH = 10
GRID_HEIGHT = 6

# Active level (1, 2 or 3)
LEVEL = 1

# Episode settings
MAX_STEPS = 300
N_EPISODES = 1000

# Shooting zone: columns from which a shoot is considered a "good position"
# (agent_x >= SHOOT_ZONE_X). For a 10-wide grid this means x=8 or x=9.
SHOOT_ZONE_X = GRID_WIDTH - 2  # = 8

# Ball starting x-column per level (y is always height // 2)
# Level 1: centre — agent must navigate to ball, then reach shooting zone
# Level 2+: column 1 (near agent) — full field to cross, dribble-vs-pass decision has real weight
# None → falls back to width // 2
BALL_START_X_L1 = None
BALL_START_X_L2 = 1
BALL_START_X_L3 = 1
BALL_START_X_L4 = 1
BALL_START_X_L5 = 1
BALL_START_X_LX = 1

# Rewards — shared across all levels
REWARD_STEP = -1           # every step
REWARD_BALL_PICKUP = 5     # picking up the ball
REWARD_CLOSER = 1          # moved closer to goal (shaping, movement actions)

# Rewards — Level 1
REWARD_GOAL = 30           # goal scored via shoot from zone
REWARD_SHOOT_NO_BALL = -5  # shoot without ball
REWARD_SHOOT_BAD_POS = -5  # shoot from outside shooting zone

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
REWARD_GOAL_L4      = 60  # goal scored (harder than L3, so higher reward)
REWARD_HIT_OBSTACLE = -2  # agent walks into obstacle wall
REWARD_SHOT_BLOCKED = -5  # shot intercepted by obstacle

# Level 3/4 opponent
# Opponent starts at x = (GRID_WIDTH - 1) - OPP_START_X_FROM_GOAL, y = 0 (top row)
# With defaults (10×6 grid): x = 9 - 1 = 8, y = 0
OPP_START_X_FROM_GOAL = 1  # columns left of goal where opponent starts
OPP_MOVE_EVERY = 2         # opponent moves 1 cell every N agent steps (1 = every step)

# Level 5 — cooperative play with teammate
# Opponent starts mid-field on the goal row, blocking the direct dribble path.
# Teammate starts top-left and positions itself to receive passes.
OPP_START_X_L5 = 6         # column — blocks goal row mid-field
OPP_START_Y_L5 = 3         # row   — same as goal row (GRID_HEIGHT // 2)
TM_START_X_L5  = 5         # teammate column
TM_START_Y_L5  = 0         # teammate row (top)
PASS_SPEED_L5  = 2          # cells the ball travels per step while in flight

# Rewards — Level 5
REWARD_GOAL_L5      = 70   # goal scored (agent shoots OR teammate scores)
REWARD_PASS_SUCCESS = 15   # teammate picks up the passed ball
REWARD_BAD_SHOT_L5  = -5   # shoot/pass without ball

# Level X (6) — two opponents + teammate (state space explosion vs. Q-Table)
# Opp1 starts near goal (ball-chaser like L3).
# Opp2 starts mid-field (agent-presser — tackles when agent carries ball).
# Teammate starts top mid-field, positions to receive passes.
OPP1_START_X_LX = 8
OPP1_START_Y_LX = 0
OPP2_START_X_LX = 4
OPP2_START_Y_LX = 5
TM_START_X_LX   = 5
TM_START_Y_LX   = 0
PASS_SPEED_LX   = 2

# Rewards — Level X
REWARD_GOAL_LX         = 80
REWARD_BAD_SHOT_LX     = -5
REWARD_PASS_SUCCESS_LX = 15

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

# Output paths
MODELS_DIR      = "results/models"
LOGS_DIR        = "results/logs"
PLOTS_DIR       = "results/plots"       # static PNG plots
ANIMATIONS_DIR  = "results/animations"  # animated GIFs
