import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config


class FootballEnv:
    """
    2D Gridworld football environment — Stage 1.

    Grid layout (6x4 default, width x height):

        . . . . . .
        . . . . . .
        A . . B . G
        . . . . . .

    A = Agent, B = Ball, G = Goal

    The agent navigates to the ball (intermediate reward) and then to the
    goal (episode win). Ball and goal are fixed throughout the episode.

    State:  tuple (agent_x, agent_y, ball_x, ball_y)  — hashable, used as
            Q-table key directly.
    Actions: 0=up  1=down  2=left  3=right
    Returns: reset() -> state
             step(action) -> (next_state, reward, done)
    """

    ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}

    # Δ(x, y) per action
    _DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    def __init__(self):
        self.width = config.GRID_WIDTH
        self.height = config.GRID_HEIGHT
        self.max_steps = config.MAX_STEPS

        # Fixed positions
        self.goal_pos = (self.width - 1, self.height // 2)
        self._ball_start = (self.width // 2, self.height // 2)
        self._agent_start = (0, self.height // 2)

        self.n_actions = 4

        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset episode. Returns initial state tuple."""
        self.agent_pos = list(self._agent_start)
        self.ball_pos = list(self._ball_start)
        self._ball_reached = False
        self.done = False
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        """
        Apply action and return (next_state, reward, done).

        Reward structure:
            -0.01  every step (encourages efficiency)
            +1.0   first time agent reaches ball position
            +10.0  agent reaches goal position → episode ends
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self.step_count += 1
        reward = config.REWARD_STEP

        # Move agent, clamped to grid boundaries
        dx, dy = self._DELTA[action]
        self.agent_pos[0] = int(np.clip(self.agent_pos[0] + dx, 0, self.width - 1))
        self.agent_pos[1] = int(np.clip(self.agent_pos[1] + dy, 0, self.height - 1))

        # Ball waypoint — rewarded once per episode
        if not self._ball_reached and self.agent_pos == self.ball_pos:
            self._ball_reached = True
            reward += config.REWARD_BALL

        # Goal check
        if self.agent_pos == list(self.goal_pos):
            reward += config.REWARD_GOAL
            self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done

    def render(self):
        """Print ASCII grid to stdout."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        gx, gy = self.goal_pos
        grid[gy][gx] = "G"

        bx, by = self.ball_pos
        grid[by][bx] = "B"

        ax, ay = self.agent_pos
        grid[ay][ax] = "A"  # agent overwrites ball symbol if on same cell

        print(f"\nStep {self.step_count}")
        for row in grid:
            print(" ".join(row))
        print(f"Agent={tuple(self.agent_pos)}  Ball={tuple(self.ball_pos)}  Goal={self.goal_pos}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_state(self):
        """Return state as a hashable tuple of ints."""
        return (
            self.agent_pos[0],
            self.agent_pos[1],
            self.ball_pos[0],
            self.ball_pos[1],
        )
