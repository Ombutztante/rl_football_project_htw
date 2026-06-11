# Grid
GRID_WIDTH = 6
GRID_HEIGHT = 4

# Active level (1, 2 or 3)
LEVEL = 1

# Episode settings
MAX_STEPS = 200
N_EPISODES = 2000

# Shooting zone: columns from which a shoot is considered a "good position"
# (agent_x >= SHOOT_ZONE_X). For a 6-wide grid this means x=4 or x=5.
SHOOT_ZONE_X = GRID_WIDTH - 2  # = 4

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

# Level 3 opponent
# Opponent starts at x = (GRID_WIDTH - 1) - OPP_START_X_FROM_GOAL, y = GRID_HEIGHT // 2
# With defaults (6×4 grid): x = 5 - 1 = 4, y = 2
# Tune OPP_START_X_FROM_GOAL and OPP_MOVE_EVERY to adjust difficulty.
# On small grids (6×4) the agent learns to shoot immediately; larger grids allow richer tactics.
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
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100   # episodes between target network syncs (controlled in training script)
REPLAY_WARMUP = 500        # minimum buffer size before learning starts
GRAD_CLIP_NORM = 1.0       # max gradient norm for clipping
HIDDEN_SIZE = 128

# Output paths
MODELS_DIR = "results/models"
LOGS_DIR = "results/logs"
PLOTS_DIR = "results/plots"
