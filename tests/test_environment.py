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
    # step(-1) + pickup(+5) + closer(+1) + align(+1, on goal row with ball) = 6
    env = FootballEnv(level=1)
    env.reset()
    bx, by = env.ball_pos  # (5,3)
    env.agent_pos = [bx - 1, by]
    _, reward, _ = env.step(3)
    assert reward == 6


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
    # Zone: x>=8 AND |y-3|<=1 → (8,2) is in zone but wrong row → penalised miss
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [8, 2]
    env.has_ball = True
    env.ball_pos = [8, 2]
    _, reward, done = env.step(4)
    assert not done
    assert reward == -4   # -1 (step) + -3 (zone miss)
    assert not env.has_ball
    assert env.ball_pos == [env.width - 1, 2]


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
