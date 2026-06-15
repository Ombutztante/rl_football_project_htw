# Grid dimensions
GRID_WIDTH  = 6
GRID_HEIGHT = 4

# Active level (1, 2 or 3)
LEVEL = 1

# Maximum steps per episode
MAX_STEPS = 200

# Shooting zone: agent must be at x >= SHOOT_ZONE_X to shoot (Level 1)
SHOOT_ZONE_X = GRID_WIDTH - 2  # = 4

# Rewards — shared
REWARD_STEP        = -1   # every step
REWARD_BALL_PICKUP =  5   # picking up the ball
REWARD_CLOSER      =  1   # moved closer to goal (shaping)

# Rewards — Level 1
REWARD_GOAL         = 30  # goal scored via shoot from zone
REWARD_SHOOT_NO_BALL = -5  # shoot without ball
REWARD_SHOOT_BAD_POS = -5  # shoot from outside shooting zone

# Rewards — Level 2
SHOOT_RANGE       =  3   # cells the ball travels on a forward pass
REWARD_GOAL_L2    = 40   # goal scored
REWARD_PASS_CLOSER =  2   # ball advanced toward goal via forward pass
REWARD_BALL_OUT   = -5   # ball exits right wall without scoring
REWARD_SHOOT_WASTED = -3  # shoot without ball

# Rewards — Level 3
REWARD_GOAL_L3           = 50
REWARD_BAD_SHOT_L3       = -5
REWARD_OPP_REACHES_BALL  = -10  # opponent reaches loose ball
REWARD_BALL_LOST         = -20  # opponent tackles agent with ball

# Level 3 opponent settings
OPP_START_X_FROM_GOAL = 1  # columns left of goal where opponent starts
OPP_MOVE_EVERY        = 2  # opponent moves every N agent steps
