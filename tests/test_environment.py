import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import FootballEnv


# ===========================================================================
# Level 1 — Shoot only from good position
# ===========================================================================

def test_l1_reset_state_has_5_elements():
    env = FootballEnv(level=1)
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 5  # (agent_x, agent_y, ball_x, ball_y, has_ball)


def test_l1_reset_agent_starts_left():
    env = FootballEnv(level=1)
    state = env.reset()
    assert state[0] == 0


def test_l1_reset_has_ball_is_false():
    env = FootballEnv(level=1)
    state = env.reset()
    assert state[4] == 0


def test_l1_step_moves_agent_right():
    env = FootballEnv(level=1)
    state = env.reset()
    next_state, _, _ = env.step(3)  # right
    assert next_state[0] == state[0] + 1


def test_l1_boundary_left_wall():
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [0, 1]
    state, _, _ = env.step(2)  # left — blocked
    assert state[0] == 0


def test_l1_boundary_top_wall():
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [1, 0]
    state, _, _ = env.step(0)  # up — blocked
    assert state[1] == 0


def test_l1_step_penalty_when_moving_away():
    # (4,3) → down → (4,4): dist to goal (9,3) increases, no shaping, no pickup
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [4, 3]
    _, reward, _ = env.step(1)  # down
    assert reward == -1


def test_l1_shaping_reward_moving_toward_goal():
    # (0,2) → right → (1,2): dist decreases → -1 + 1 = 0
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [0, 2]
    _, reward, _ = env.step(3)
    assert reward == 0


def test_l1_ball_pickup_sets_has_ball():
    env = FootballEnv(level=1)
    env.reset()
    bx, by = env.ball_pos
    env.agent_pos = [bx - 1, by]
    state, _, _ = env.step(3)
    assert state[4] == 1


def test_l1_ball_pickup_reward():
    # Agent at (bx-1, by), ball at (5,3), goal at (9,3)
    # step(-1) + pickup(+5) + closer(+1) = 5
    env = FootballEnv(level=1)
    env.reset()
    bx, by = env.ball_pos  # (5,3)
    env.agent_pos = [bx - 1, by]
    _, reward, _ = env.step(3)
    assert reward == 5


def test_l1_ball_follows_agent_when_carried():
    env = FootballEnv(level=1)
    env.reset()
    bx, by = env.ball_pos
    env.agent_pos = [bx - 1, by]
    env.step(3)   # pick up
    env.step(3)   # move right with ball
    assert env.ball_pos == env.agent_pos


def test_l1_shoot_without_ball_gives_penalty():
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [4, 2]
    _, reward, done = env.step(4)
    assert reward == -6   # -1 (step) + -5 (no ball)
    assert not done


def test_l1_shoot_bad_position_gives_penalty():
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [1, 2]
    env.has_ball = True
    env.ball_pos = [1, 2]
    _, reward, done = env.step(4)
    assert reward == -6   # -1 (step) + -5 (bad pos)
    assert not done
    assert not env.has_ball


def test_l1_shoot_in_zone_aligned_scores():
    # SHOOT_ZONE_X=8, goal row=3 → agent at (8,3)
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [8, 3]
    env.has_ball = True
    env.ball_pos = [8, 3]
    _, reward, done = env.step(4)
    assert done
    assert reward == 29   # -1 (step) + 30 (goal)


def test_l1_shoot_in_zone_wrong_row_is_miss():
    # SHOOT_ZONE_X=8, row 0 ≠ goal row 3 → miss, ball to right wall
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [8, 0]
    env.has_ball = True
    env.ball_pos = [8, 0]
    _, reward, done = env.step(4)
    assert not done
    assert reward == -1   # only step penalty
    assert not env.has_ball
    assert env.ball_pos == [env.width - 1, 0]


def test_l1_walking_into_goal_does_not_score():
    env = FootballEnv(level=1)
    env.reset()
    gx, gy = env.goal_pos
    env.agent_pos = [gx - 1, gy]
    env.has_ball = True
    env.ball_pos = [gx - 1, gy]
    _, _, done = env.step(3)
    assert not done


def test_l1_max_steps_ends_episode():
    env = FootballEnv(level=1)
    env.reset()
    done = False
    for _ in range(env.max_steps + 5):
        if done:
            break
        _, _, done = env.step(0)
    assert done


# ===========================================================================
# Level 2 — Dribbling vs. forward pass
# ===========================================================================

def test_l2_state_still_5_elements():
    env = FootballEnv(level=2)
    state = env.reset()
    assert len(state) == 5


def test_l2_ball_pickup_works():
    env = FootballEnv(level=2)
    env.reset()
    bx, by = env.ball_pos
    env.agent_pos = [bx - 1, by]
    state, _, _ = env.step(3)
    assert state[4] == 1


def test_l2_dribble_to_goal_scores():
    # Agent carries ball into goal cell → goal, +40
    env = FootballEnv(level=2)
    env.reset()
    gx, gy = env.goal_pos
    env.agent_pos = [gx - 1, gy]
    env.has_ball = True
    env.ball_pos = [gx - 1, gy]
    _, reward, done = env.step(3)  # right → into goal
    assert done
    assert reward == 40   # -1 (step) + 1 (closer, dist 1→0) + 40 (goal)


def test_l2_forward_pass_advances_ball():
    # Ball at x=0, SHOOT_RANGE=3 → ball ends at x=3, agent loses possession
    import config as cfg
    env = FootballEnv(level=2)
    env.reset()
    env.agent_pos = [0, 0]
    env.has_ball = True
    env.ball_pos = [0, 0]
    env.step(4)  # shoot
    assert env.ball_pos[0] == cfg.SHOOT_RANGE  # 0 + 3 = 3
    assert not env.has_ball


def test_l2_forward_pass_stays_in_field():
    import config as cfg
    env = FootballEnv(level=2)
    env.reset()
    start_bx = 1
    env.agent_pos = [start_bx, 0]
    env.has_ball = True
    env.ball_pos = [start_bx, 0]
    env.step(4)
    assert env.ball_pos[0] == start_bx + cfg.SHOOT_RANGE  # = 4, within grid
    assert not env.has_ball


def test_l2_forward_pass_exits_wrong_row_gives_penalty():
    # Ball exits right wall at wrong row → -5
    env = FootballEnv(level=2)
    env.reset()
    # Agent at x=7, SHOOT_RANGE=3 → raw_new=10 → exits (width=10)
    env.agent_pos = [7, 0]   # row 0 ≠ goal_row 3
    env.has_ball = True
    env.ball_pos = [7, 0]
    _, reward, done = env.step(4)
    assert not done
    assert reward == -6      # -1 (step) + -5 (out)
    assert env.ball_pos == [env.width - 1, 0]
    assert not env.has_ball


def test_l2_forward_pass_exits_goal_row_scores():
    # Ball exits at goal row → +40
    env = FootballEnv(level=2)
    env.reset()
    _, gy = env.goal_pos
    env.agent_pos = [7, gy]   # row == goal row, raw_new = 10 → exits → GOAL
    env.has_ball = True
    env.ball_pos = [7, gy]
    _, reward, done = env.step(4)
    assert done
    assert reward == 39      # -1 (step) + 40 (goal)


def test_l2_forward_pass_lands_on_goal():
    # Ball lands exactly on goal cell without exiting → +40
    import config as cfg
    env = FootballEnv(level=2)
    env.reset()
    gx, gy = env.goal_pos    # (5, 2)
    start_bx = gx - cfg.SHOOT_RANGE  # = 5 - 3 = 2
    env.agent_pos = [start_bx, gy]
    env.has_ball = True
    env.ball_pos = [start_bx, gy]
    _, reward, done = env.step(4)
    assert done
    assert reward == 39      # -1 (step) + 40 (goal)


def test_l2_forward_pass_closer_gives_shaping():
    # Ball moves closer but doesn't exit or score → +2
    env = FootballEnv(level=2)
    env.reset()
    env.agent_pos = [0, 0]   # far from goal, wrong row
    env.has_ball = True
    env.ball_pos = [0, 0]
    _, reward, done = env.step(4)
    # raw_new = 0+3=3 (in field), dist before > dist after → +2
    assert not done
    assert reward == 1       # -1 (step) + 2 (pass closer)


def test_l2_shoot_without_ball_gives_penalty():
    env = FootballEnv(level=2)
    env.reset()
    env.agent_pos = [2, 2]
    _, reward, done = env.step(4)
    assert reward == -4      # -1 (step) + -3 (wasted)
    assert not done


# ===========================================================================
# Level 3 — Opponent moves toward ball
# ===========================================================================

def test_l3_state_has_7_elements():
    env = FootballEnv(level=3)
    state = env.reset()
    assert len(state) == 7  # (ax, ay, bx, by, has_ball, opp_x, opp_y)


def test_l3_opponent_initial_position():
    import config as cfg
    env = FootballEnv(level=3)
    state = env.reset()
    expected_opp_x = env.goal_pos[0] - cfg.OPP_START_X_FROM_GOAL
    assert state[5] == expected_opp_x
    assert state[6] == 0


def test_l3_opponent_does_not_move_on_first_step():
    # With OPP_MOVE_EVERY >= 2 (default), opponent stays put on step 1 (1 % 2 != 0)
    import config as cfg
    env = FootballEnv(level=3)
    env.reset()
    env.agent_pos = [0, 0]
    env.ball_pos = [0, 0]
    env.has_ball = False
    init_opp = env.opp_pos.copy()
    env.step(0)  # step 1 — no opponent move expected
    assert env.opp_pos == init_opp


def test_l3_opponent_moves_after_n_steps():
    # After OPP_MOVE_EVERY steps the opponent must have moved at least once.
    import config as cfg
    env = FootballEnv(level=3)
    env.reset()
    # Keep ball far from opponent so the episode does not end prematurely
    env.ball_pos = [0, 0]
    env.agent_pos = [0, 0]
    env.has_ball = False
    init_opp = env.opp_pos.copy()
    for _ in range(cfg.OPP_MOVE_EVERY):
        if not env.done:
            env.step(0)
    assert env.opp_pos != init_opp


def test_l3_opponent_reaches_loose_ball_ends_episode():
    import config as cfg
    env = FootballEnv(level=3)
    env.reset()
    # Place opponent one step to the right of ball; agent well away
    env.ball_pos = [3, 0]
    env.agent_pos = [0, 0]
    env.has_ball = False
    env.opp_pos = [4, 0]
    # Force step_count so the next step triggers an opponent move
    env.step_count = cfg.OPP_MOVE_EVERY - 1
    _, reward, done = env.step(0)  # agent moves (clamped at top), then opp moves to ball
    assert done
    assert reward == -1 + cfg.REWARD_OPP_REACHES_BALL  # -11


def test_l3_opponent_tackles_agent_with_ball_ends_episode():
    import config as cfg
    env = FootballEnv(level=3)
    env.reset()
    # Agent with ball at (3, 0); opponent one step right — agent tries to go up (clamped)
    env.agent_pos = [3, 0]
    env.has_ball = True
    env.ball_pos = [3, 0]
    env.opp_pos = [4, 0]  # opponent to the right
    env.step_count = cfg.OPP_MOVE_EVERY - 1  # next step triggers opp move
    _, reward, done = env.step(0)  # up: clamped at y=0, agent stays, opp moves left to (3,0)
    assert done
    assert reward == -1 + cfg.REWARD_BALL_LOST  # -21
    assert not env.has_ball


def test_l3_shoot_without_ball_penalty():
    import config as cfg
    env = FootballEnv(level=3)
    env.reset()
    env.has_ball = False
    env.agent_pos = [0, 0]
    env.ball_pos = [3, 2]
    env.opp_pos = [0, 3]   # far from ball; step 1, no opponent move
    env.step_count = 0
    _, reward, done = env.step(4)
    assert not done
    assert reward == -1 + cfg.REWARD_BAD_SHOT_L3  # -6


def test_l3_forward_pass_goal_scores_50():
    import config as cfg
    env = FootballEnv(level=3)
    env.reset()
    _, gy = env.goal_pos
    env.agent_pos = [7, gy]   # raw_new = 10 → exits at goal row → GOAL
    env.has_ball = True
    env.ball_pos = [7, gy]
    env.opp_pos = [0, 0]   # far away; step 1, no opponent move
    env.step_count = 0
    _, reward, done = env.step(4)  # pass exits at goal row → goal
    assert done
    assert reward == -1 + cfg.REWARD_GOAL_L3  # 49


def test_l3_dribble_to_goal_scores_50():
    import config as cfg
    env = FootballEnv(level=3)
    env.reset()
    gx, gy = env.goal_pos
    env.agent_pos = [gx - 1, gy]
    env.has_ball = True
    env.ball_pos = [gx - 1, gy]
    env.opp_pos = [0, 0]   # far away; step 1, no opponent move
    env.step_count = 0
    _, reward, done = env.step(3)  # right → into goal
    assert done
    # -1 (step) + 1 (closer, dist 1→0) + 50 (goal L3) = 50
    assert reward == 50


# ===========================================================================
# Level 4 — static obstacle blocks direct path (extends Level 3)
# ===========================================================================

def test_l4_state_has_7_elements():
    env = FootballEnv(level=4)
    state = env.reset()
    assert len(state) == 7  # same as Level 3


def test_l4_obstacle_blocks_agent():
    import config as cfg
    env = FootballEnv(level=4)
    env.reset()
    # OBSTACLE_X=6, OBSTACLE_Y_START=0, OBSTACLE_HEIGHT=4 → cells (6,0)–(6,3) blocked
    env.agent_pos = [5, 1]   # directly left of obstacle row 1
    env.opp_pos = [0, 0]
    env.step_count = 0       # step → 1, 1%2≠0 → no opponent move
    state, reward, _ = env.step(3)   # right → blocked
    assert state[0] == 5             # agent did not move
    assert reward == -1 + cfg.REWARD_HIT_OBSTACLE   # -3


def test_l4_agent_can_navigate_below_obstacle():
    env = FootballEnv(level=4)
    env.reset()
    # Rows 0–3 at x=6 are blocked; row 4 is free
    env.agent_pos = [5, 4]
    env.opp_pos = [0, 0]
    env.step_count = 0
    state, _, _ = env.step(3)   # right → free cell at (6, 4)
    assert state[0] == 6


def test_l4_shot_blocked_by_obstacle():
    import config as cfg
    env = FootballEnv(level=4)
    env.reset()
    # Agent at (4,1) shoots — ball travels through (5,1) then hits obstacle at (6,1)
    env.agent_pos = [4, 1]
    env.has_ball = True
    env.ball_pos = [4, 1]
    env.opp_pos = [0, 5]
    env.step_count = 0
    _, reward, done = env.step(4)
    assert not done
    assert reward == -1 + cfg.REWARD_SHOT_BLOCKED   # -6
    assert env.ball_pos == [5, 1]   # stopped one cell before obstacle


def test_l4_dribble_to_goal_scores_60():
    import config as cfg
    env = FootballEnv(level=4)
    env.reset()
    gx, gy = env.goal_pos
    env.agent_pos = [gx - 1, gy]
    env.has_ball = True
    env.ball_pos = [gx - 1, gy]
    env.opp_pos = [0, 0]
    env.step_count = 0
    _, reward, done = env.step(3)   # right → into goal
    assert done
    # -1 (step) + 1 (closer) + 60 (goal L4) = 60
    assert reward == 60


# ===========================================================================
# Level 5 — cooperative play with teammate
# ===========================================================================

def test_l5_state_has_10_elements():
    env = FootballEnv(level=5)
    state = env.reset()
    assert len(state) == 10   # (ax,ay,bx,by,has_ball,opp_x,opp_y,tm_x,tm_y,tm_has_ball)


def test_l5_teammate_initial_position():
    import config as cfg
    env = FootballEnv(level=5)
    state = env.reset()
    assert state[7] == cfg.TM_START_X_L5
    assert state[8] == cfg.TM_START_Y_L5
    assert state[9] == 0   # tm_has_ball False


def test_l5_opponent_initial_position():
    import config as cfg
    env = FootballEnv(level=5)
    state = env.reset()
    assert state[5] == cfg.OPP_START_X_L5
    assert state[6] == cfg.OPP_START_Y_L5


def test_l5_shoot_without_ball_penalty():
    import config as cfg
    env = FootballEnv(level=5)
    env.reset()
    env.has_ball = False
    env.agent_pos = [3, 3]
    env.ball_pos = [0, 0]
    env.opp_pos = [0, 5]
    env.tm_pos = [5, 0]
    env.step_count = 0
    _, reward, done = env.step(4)
    assert not done
    assert reward == -1 + cfg.REWARD_BAD_SHOT_L5   # -6


def test_l5_ball_travels_diagonally_toward_teammate():
    env = FootballEnv(level=5)
    env.reset()
    env.agent_pos = [3, 3]
    env.has_ball = True
    env.ball_pos = [3, 3]
    env.opp_pos = [0, 5]
    env.tm_pos = [9, 5]   # far teammate so ball doesn't arrive in one step
    env.step_count = 0
    env.step(4)   # pass → ball travels 2 diagonal steps (3,3)→(4,4)→(5,5)
    # _move_teammate moves tm (9,5) toward ball (5,5) → tm at (8,5), no pickup yet
    assert env.ball_pos == [5, 5]
    assert not env.has_ball
    assert not env.tm_has_ball


def test_l5_teammate_picks_up_ball_gives_reward():
    import config as cfg
    env = FootballEnv(level=5)
    env.reset()
    # Teammate at (5,0), ball at (3,3) — after pass ball travels to (5,1),
    # teammate moves from (5,0) to (5,1) and collects it in the same step.
    env.agent_pos = [3, 3]
    env.has_ball = True
    env.ball_pos = [3, 3]
    env.opp_pos = [0, 5]
    env.tm_pos = [5, 0]
    env.step_count = 0
    _, reward, done = env.step(4)
    assert env.tm_has_ball
    assert reward == -1 + cfg.REWARD_PASS_SUCCESS   # 14


def test_l5_shoot_in_zone_aligned_scores():
    import config as cfg
    env = FootballEnv(level=5)
    env.reset()
    gx, gy = env.goal_pos   # (9, 3)
    env.agent_pos = [env.shoot_zone_x, gy]   # SHOOT_ZONE_X=8, goal row
    env.has_ball = True
    env.ball_pos = [env.shoot_zone_x, gy]
    env.opp_pos = [0, 5]
    env.tm_pos = [5, 0]
    env.step_count = 0
    _, reward, done = env.step(4)
    assert done
    assert reward == -1 + cfg.REWARD_GOAL_L5   # 69


# ===========================================================================
# Level 6 — Two opponents + teammate (state space explosion)
# ===========================================================================

def test_l6_state_has_12_elements():
    env = FootballEnv(level=6)
    state = env.reset()
    assert len(state) == 12


def test_l6_state_array_has_14_elements():
    env = FootballEnv(level=6)
    state = env.reset()
    arr = env.state_to_array(state)
    assert arr.shape == (14,)
    assert env.get_state_size() == 14


def test_l6_initial_positions():
    import config as cfg
    env = FootballEnv(level=6)
    state = env.reset()
    # opp1 at indices 5,6 — opp2 at 7,8 — tm at 9,10 — tm_has_ball at 11
    assert state[5] == cfg.OPP1_START_X_LX
    assert state[6] == cfg.OPP1_START_Y_LX
    assert state[7] == cfg.OPP2_START_X_LX
    assert state[8] == cfg.OPP2_START_Y_LX
    assert state[9] == cfg.TM_START_X_LX
    assert state[10] == cfg.TM_START_Y_LX
    assert state[11] == 0   # tm_has_ball False


def test_l6_opp1_reaches_loose_ball_ends_episode():
    import config as cfg
    env = FootballEnv(level=6)
    env.reset()
    env.ball_pos = [3, 0]
    env.agent_pos = [0, 0]
    env.has_ball = False
    env.opp1_pos = [4, 0]          # one step right of ball
    env.opp2_pos = [0, 5]          # far away
    env.step_count = cfg.OPP_MOVE_EVERY - 1
    _, reward, done = env.step(0)  # triggers opp1 move → reaches ball
    assert done
    assert reward == -1 + cfg.REWARD_OPP_REACHES_BALL


def test_l6_opp2_tackles_agent_with_ball():
    import config as cfg
    env = FootballEnv(level=6)
    env.reset()
    env.agent_pos = [3, 0]
    env.has_ball = True
    env.ball_pos = [3, 0]
    env.opp1_pos = [0, 5]          # far away
    env.opp2_pos = [4, 0]          # one step right of agent
    env.step_count = cfg.OPP_MOVE_EVERY - 1
    _, reward, done = env.step(0)  # agent clamped at y=0; opp2 moves to (3,0)
    assert done
    assert reward == -1 + cfg.REWARD_BALL_LOST
    assert not env.has_ball


def test_l6_shoot_without_ball_penalty():
    import config as cfg
    env = FootballEnv(level=6)
    env.reset()
    env.has_ball = False
    env.agent_pos = [3, 3]
    env.ball_pos = [0, 0]
    env.opp1_pos = [0, 5]
    env.opp2_pos = [9, 5]
    env.tm_pos = [5, 0]
    env.step_count = 0
    _, reward, done = env.step(4)
    assert not done
    assert reward == -1 + cfg.REWARD_BAD_SHOT_LX   # -6


def test_l6_shoot_in_zone_aligned_scores():
    import config as cfg
    env = FootballEnv(level=6)
    env.reset()
    _, gy = env.goal_pos
    env.agent_pos = [env.shoot_zone_x, gy]
    env.has_ball = True
    env.ball_pos = [env.shoot_zone_x, gy]
    env.opp1_pos = [0, 5]
    env.opp2_pos = [0, 4]
    env.tm_pos = [5, 0]
    env.step_count = 0
    _, reward, done = env.step(4)
    assert done
    assert reward == -1 + cfg.REWARD_GOAL_LX   # 79


def test_l6_pass_outside_zone_puts_ball_in_flight():
    env = FootballEnv(level=6)
    env.reset()
    env.agent_pos = [3, 3]
    env.has_ball = True
    env.ball_pos = [3, 3]
    env.opp1_pos = [0, 5]
    env.opp2_pos = [9, 5]
    env.tm_pos = [9, 0]
    env.step_count = 0
    env.step(4)   # pass → ball in flight
    assert not env.has_ball
    assert env._ball_in_flight


def test_l6_teammate_picks_up_ball_gives_reward():
    import config as cfg
    env = FootballEnv(level=6)
    env.reset()
    env.agent_pos = [3, 3]
    env.has_ball = True
    env.ball_pos = [3, 3]
    env.opp1_pos = [0, 5]
    env.opp2_pos = [9, 5]
    env.tm_pos = [5, 0]   # same target as L5 test
    env.step_count = 0
    _, reward, done = env.step(4)
    assert env.tm_has_ball
    assert reward == -1 + cfg.REWARD_PASS_SUCCESS_LX   # 14
