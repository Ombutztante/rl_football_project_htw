import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import FootballEnv


# ---------------------------------------------------------------------------
# State & reset
# ---------------------------------------------------------------------------

def test_reset_state_has_5_elements():
    env = FootballEnv()
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 5  # (agent_x, agent_y, ball_x, ball_y, has_ball)


def test_reset_agent_starts_left():
    env = FootballEnv()
    state = env.reset()
    assert state[0] == 0  # agent_x = 0


def test_reset_has_ball_is_false():
    env = FootballEnv()
    state = env.reset()
    assert state[4] == 0  # has_ball = False


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def test_step_moves_agent_right():
    env = FootballEnv()
    state = env.reset()
    ax_before = state[0]
    next_state, _, _ = env.step(3)  # right
    assert next_state[0] == ax_before + 1


def test_boundary_left_wall():
    env = FootballEnv()
    env.reset()
    env.agent_pos = [0, 1]
    state, _, _ = env.step(2)  # left — blocked
    assert state[0] == 0


def test_boundary_top_wall():
    env = FootballEnv()
    env.reset()
    env.agent_pos = [1, 0]
    state, _, _ = env.step(0)  # up — blocked
    assert state[1] == 0


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def test_step_penalty_when_moving_away():
    # Agent at (4,2), goal at (5,2). Moving down → dist increases → only step penalty.
    env = FootballEnv()
    env.reset()
    env.agent_pos = [4, 2]   # goal is at (5,2)
    _, reward, _ = env.step(1)  # down → (4,3), dist 1→2
    assert reward == -1


def test_shaping_reward_moving_toward_goal():
    # Agent at (0,2), goal at (5,2), ball at (3,2) — not in the way.
    # Moving right → dist 5→4 → step(-1) + closer(+1) = 0.
    env = FootballEnv()
    env.reset()
    env.agent_pos = [0, 2]
    _, reward, _ = env.step(3)  # right → (1,2)
    assert reward == 0  # -1 + 1


# ---------------------------------------------------------------------------
# Ball pickup
# ---------------------------------------------------------------------------

def test_ball_pickup_sets_has_ball():
    env = FootballEnv()
    env.reset()
    bx, by = env.ball_pos
    env.agent_pos = [bx - 1, by]
    state, _, _ = env.step(3)  # right → onto ball
    assert state[4] == 1       # has_ball = True


def test_ball_pickup_reward():
    # pickup(+5) + closer(+1) + step(-1) = 5
    env = FootballEnv()
    env.reset()
    bx, by = env.ball_pos      # (3, 2)
    env.agent_pos = [bx - 1, by]   # (2, 2), dist to goal = |2-5|+0 = 3
    _, reward, _ = env.step(3)     # → (3,2), dist = 2, pickup
    assert reward == 5             # -1 + 5 + 1


def test_ball_follows_agent_when_carried():
    env = FootballEnv()
    env.reset()
    bx, by = env.ball_pos
    env.agent_pos = [bx - 1, by]
    env.step(3)                    # pick up ball
    env.step(3)                    # move right with ball
    assert env.ball_pos == env.agent_pos


# ---------------------------------------------------------------------------
# Shoot — no ball / bad position
# ---------------------------------------------------------------------------

def test_shoot_without_ball_gives_penalty():
    env = FootballEnv()
    env.reset()
    env.agent_pos = [4, 2]
    # has_ball = False by default
    _, reward, done = env.step(4)  # shoot
    assert reward == -6            # -1 (step) + -5 (no ball)
    assert not done


def test_shoot_bad_position_gives_penalty():
    # agent_x=1 < shoot_zone_x=4 → bad position
    env = FootballEnv()
    env.reset()
    env.agent_pos = [1, 2]
    env.has_ball = True
    env.ball_pos = [1, 2]
    _, reward, done = env.step(4)  # shoot
    assert reward == -6            # -1 (step) + -5 (bad pos)
    assert not done
    assert not env.has_ball        # ball dropped


# ---------------------------------------------------------------------------
# Shoot — in zone
# ---------------------------------------------------------------------------

def test_shoot_in_zone_aligned_scores():
    # agent in zone (x=4 >= 4) and aligned with goal row (y=2) → GOAL
    env = FootballEnv()
    env.reset()
    env.agent_pos = [4, 2]
    env.has_ball = True
    env.ball_pos = [4, 2]
    _, reward, done = env.step(4)
    assert done
    assert reward == 29            # -1 (step) + 30 (goal)


def test_shoot_in_zone_wrong_row_is_miss():
    # agent in zone (x=4) but wrong row (y=0 != goal_y=2) → miss, no penalty
    env = FootballEnv()
    env.reset()
    env.agent_pos = [4, 0]
    env.has_ball = True
    env.ball_pos = [4, 0]
    _, reward, done = env.step(4)
    assert not done
    assert reward == -1            # only step penalty
    assert not env.has_ball        # ball lost after miss
    assert env.ball_pos == [env.width - 1, 0]  # ball at right wall, same row


def test_walking_into_goal_does_not_score():
    # Scoring is ONLY via shoot — walking into goal cell must not end episode.
    env = FootballEnv()
    env.reset()
    gx, gy = env.goal_pos
    env.agent_pos = [gx - 1, gy]
    env.has_ball = True
    env.ball_pos = [gx - 1, gy]
    _, _, done = env.step(3)  # walk into goal cell
    assert not done


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

def test_max_steps_ends_episode():
    env = FootballEnv()
    env.reset()
    done = False
    for _ in range(env.max_steps + 5):
        if done:
            break
        _, _, done = env.step(0)  # keep bouncing off top wall
    assert done
