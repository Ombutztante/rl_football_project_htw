import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config


class FootballEnv:
    """
    2D Gridworld football environment — Level 1.

    Grid layout (6x4 default, width x height):

        . . . . . .
        . . . . . .
        A . . B . G
        . . . . . .

    A = Agent without ball, P = Agent with ball, B = Ball, G = Goal

    The agent must: pick up ball → reach shooting zone → shoot.
    Scoring is ONLY possible via the shoot action.
    Walking into the goal cell does not score.

    Shooting zone: agent_x >= SHOOT_ZONE_X (last 2 columns for a 6-wide grid).
    A shot in the zone is attempted: scores if agent row == goal row, misses otherwise.
    A shot outside the zone is penalised and the ball is dropped.

    State:   tuple (agent_x, agent_y, ball_x, ball_y, has_ball) — hashable,
             used directly as Q-table key.
    Actions: 0=up  1=down  2=left  3=right  4=shoot  (always 5)
    Returns: reset() -> state
             step(action) -> (next_state, reward, done)
    """

    ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right", 4: "shoot"}

    # Δ(x, y) per movement action
    _DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    def __init__(self):
        self.width = config.GRID_WIDTH
        self.height = config.GRID_HEIGHT
        self.max_steps = config.MAX_STEPS
        self.shoot_zone_x = config.SHOOT_ZONE_X

        # Fixed positions — goal: right edge, centre row
        self.goal_pos = (self.width - 1, self.height // 2)
        self._ball_start = (self.width // 2, self.height // 2)
        self._agent_start = (0, self.height // 2)

        self.n_actions = 5

        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset episode. Returns initial state tuple."""
        self.agent_pos = list(self._agent_start)
        self.ball_pos = list(self._ball_start)
        self.has_ball = False
        self.done = False
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        """
        Apply action and return (next_state, reward, done).

        Reward structure (Level 1):
            -1    every step
            +5    picking up the ball
            +1    moving closer to goal (Manhattan distance, movement only)
            +30   goal scored via shoot
            -5    shoot without ball
            -5    shoot from outside shooting zone
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self.step_count += 1
        reward = config.REWARD_STEP  # -1 every step

        if action == 4:  # shoot
            reward += self._handle_shoot()
        else:
            prev_dist = self._dist_to_goal()

            dx, dy = self._DELTA[action]
            self.agent_pos[0] = int(np.clip(self.agent_pos[0] + dx, 0, self.width - 1))
            self.agent_pos[1] = int(np.clip(self.agent_pos[1] + dy, 0, self.height - 1))

            # Ball follows agent when carried
            if self.has_ball:
                self.ball_pos = self.agent_pos.copy()

            # Ball pickup
            if not self.has_ball and self.agent_pos == self.ball_pos:
                self.has_ball = True
                reward += config.REWARD_BALL_PICKUP  # +5

            # Shaping: reward progress toward goal
            if self._dist_to_goal() < prev_dist:
                reward += config.REWARD_CLOSER  # +1

        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done

    def render(self):
        """Print ASCII grid to stdout."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        # Mark shooting zone columns
        for y in range(self.height):
            for x in range(self.shoot_zone_x, self.width - 1):
                grid[y][x] = ","  # visual hint for shooting zone

        gx, gy = self.goal_pos
        grid[gy][gx] = "G"

        if not self.has_ball:
            bx, by = self.ball_pos
            if 0 <= bx < self.width and 0 <= by < self.height:
                grid[by][bx] = "B"

        ax, ay = self.agent_pos
        grid[ay][ax] = "P" if self.has_ball else "A"

        has_str = "YES" if self.has_ball else "NO"
        zone_str = f"x>={self.shoot_zone_x}"
        print(f"\nStep {self.step_count} | Ball: {has_str} | ShootZone: {zone_str}")
        for row in grid:
            print(" ".join(row))
        print(f"Agent={tuple(self.agent_pos)}  Ball={tuple(self.ball_pos)}  Goal={self.goal_pos}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_shoot(self):
        """
        Resolve a shoot action. Returns the reward delta (step penalty excluded).

        - No ball:           -5, no state change
        - Outside zone:      -5, ball dropped at agent position
        - In zone, goal row: +30, episode ends (goal)
        - In zone, off row:  no penalty, ball travels to right wall (miss)
        """
        if not self.has_ball:
            return config.REWARD_SHOOT_NO_BALL  # -5

        ax, ay = self.agent_pos
        _, gy = self.goal_pos

        if ax < self.shoot_zone_x:
            # Bad position — penalise and drop ball
            self.has_ball = False
            return config.REWARD_SHOOT_BAD_POS  # -5

        # In shooting zone — attempt shot
        if ay == gy:
            # Aligned with goal row → GOAL
            self.ball_pos = list(self.goal_pos)
            self.has_ball = False
            self.done = True
            return config.REWARD_GOAL  # +30
        else:
            # In zone but off row → miss, ball goes to right wall
            self.ball_pos = [self.width - 1, ay]
            self.has_ball = False
            return 0

    def _dist_to_goal(self):
        """Manhattan distance from agent to goal."""
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        return abs(ax - gx) + abs(ay - gy)

    def _get_state(self):
        """Return state as a hashable tuple of ints."""
        return (
            self.agent_pos[0],
            self.agent_pos[1],
            self.ball_pos[0],
            self.ball_pos[1],
            int(self.has_ball),
        )
