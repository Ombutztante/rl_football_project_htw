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
    # (4,2) → down → (4,3): dist to goal increases, no shaping, no pickup
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [4, 2]
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
    # Agent at (2,2), ball at (3,2), goal at (5,2)
    # step(-1) + pickup(+5) + closer(+1) = 5
    env = FootballEnv(level=1)
    env.reset()
    bx, by = env.ball_pos  # (3,2)
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
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [4, 2]
    env.has_ball = True
    env.ball_pos = [4, 2]
    _, reward, done = env.step(4)
    assert done
    assert reward == 29   # -1 (step) + 30 (goal)


def test_l1_shoot_in_zone_wrong_row_is_miss():
    env = FootballEnv(level=1)
    env.reset()
    env.agent_pos = [4, 0]
    env.has_ball = True
    env.ball_pos = [4, 0]
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
    # Agent at x=3, SHOOT_RANGE=3 → raw_new=6 → exits
    env.agent_pos = [3, 0]   # row 0 ≠ goal_row 2
    env.has_ball = True
    env.ball_pos = [3, 0]
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
    env.agent_pos = [3, gy]   # row == goal row, raw_new = 6 → exits → GOAL
    env.has_ball = True
    env.ball_pos = [3, gy]
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
