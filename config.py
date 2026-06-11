# Grid
GRID_WIDTH = 6
GRID_HEIGHT = 4

# Active level (1, 2 or 3)
LEVEL = 1

# Episode settings
MAX_STEPS = 200
N_EPISODES = 5000

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

# Q-Learning
Q_LR = 0.1
Q_GAMMA = 0.99
Q_EPSILON_START = 1.0
Q_EPSILON_MIN = 0.05
Q_EPSILON_DECAY = 0.995

# DQN (PyTorch)
DQN_LR = 1e-3
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_MIN = 0.05
DQN_EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10
HIDDEN_SIZE = 128

# Output paths
MODELS_DIR = "results/models"
LOGS_DIR = "results/logs"
PLOTS_DIR = "results/plots"
