import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import FootballEnv


def test_reset_returns_state_tuple():
    env = FootballEnv()
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 4  # (agent_x, agent_y, ball_x, ball_y)


def test_reset_agent_starts_left():
    env = FootballEnv()
    state = env.reset()
    assert state[0] == 0  # agent_x = 0 (left column)


def test_step_moves_agent_right():
    env = FootballEnv()
    state = env.reset()
    ax_before = state[0]
    next_state, _, _ = env.step(3)  # right
    assert next_state[0] == ax_before + 1


def test_step_returns_step_penalty():
    env = FootballEnv()
    env.reset()
    # Move away from ball and goal — should only get the step penalty
    env.agent_pos = [0, 0]  # top-left, far from ball/goal row
    _, reward, _ = env.step(3)  # right
    assert abs(reward - (-0.01)) < 1e-6


def test_boundary_left_wall():
    env = FootballEnv()
    env.reset()
    env.agent_pos = [0, 1]
    state, _, _ = env.step(2)  # left — should be blocked
    assert state[0] == 0


def test_boundary_top_wall():
    env = FootballEnv()
    env.reset()
    env.agent_pos = [1, 0]
    state, _, _ = env.step(0)  # up — should be blocked
    assert state[1] == 0


def test_ball_waypoint_reward():
    env = FootballEnv()
    env.reset()
    # Place agent one step left of ball
    bx, by = env.ball_pos
    env.agent_pos = [bx - 1, by]
    _, reward, _ = env.step(3)  # right → lands on ball
    assert env._ball_reached
    assert reward > 0  # step penalty + ball reward → net positive


def test_ball_waypoint_only_once():
    env = FootballEnv()
    env.reset()
    bx, by = env.ball_pos
    env.agent_pos = [bx - 1, by]
    env.step(3)   # reach ball → rewarded
    env.step(2)   # step left (back off ball)
    env.agent_pos = [bx - 1, by]
    _, reward, _ = env.step(3)  # reach ball again → no extra reward
    assert abs(reward - (-0.01)) < 1e-6


def test_goal_gives_large_reward_and_ends_episode():
    env = FootballEnv()
    env.reset()
    gx, gy = env.goal_pos
    env.agent_pos = [gx - 1, gy]
    _, reward, done = env.step(3)  # right → into goal
    assert done
    assert reward > 5.0


def test_max_steps_ends_episode():
    env = FootballEnv()
    env.reset()
    done = False
    for _ in range(env.max_steps + 5):
        if done:
            break
        _, _, done = env.step(0)  # keep walking up/bouncing off wall
    assert done
