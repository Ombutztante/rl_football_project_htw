import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config


class FootballEnv:
    """
    2D Gridworld football environment — supports Level 1 and Level 2.
    The active level is read from config.LEVEL (or passed as argument).

    Grid layout (6x4 default, width x height):

        . . . . . .
        . . . . . .
        A . . B . G
        . . . . . .

    A = Agent without ball, P = Agent with ball, B = Ball, G = Goal

    Level 1 — shoot only from good position
        Agent picks up ball, reaches shooting zone, shoots.
        Scoring: shoot in zone AND aligned with goal row.

    Level 2 — dribbling vs. forward pass
        Shoot sends ball SHOOT_RANGE cells to the right (forward pass).
        Agent loses possession and must chase.
        Scoring: ball lands on goal cell OR agent dribbles ball to goal cell.

    State:   tuple (agent_x, agent_y, ball_x, ball_y, has_ball) — hashable,
             used directly as Q-table key. Same structure for both levels.
    Actions: 0=up  1=down  2=left  3=right  4=shoot  (always 5)
    Returns: reset() -> state
             step(action) -> (next_state, reward, done)
    """

    ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right", 4: "shoot"}
    _DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    def __init__(self, level=None):
        self.level = level if level is not None else config.LEVEL
        self.width = config.GRID_WIDTH
        self.height = config.GRID_HEIGHT
        self.max_steps = config.MAX_STEPS
        self.shoot_zone_x = config.SHOOT_ZONE_X  # used by Level 1

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

        Level 1 reward structure:
            -1   every step
            +5   ball picked up
            +1   moved closer to goal (shaping)
            +30  goal scored via shoot from zone
            -5   shoot without ball
            -5   shoot from outside shooting zone

        Level 2 reward structure:
            -1   every step
            +5   ball picked up
            +1   moved closer to goal (shaping, movement only)
            +40  goal scored (dribble to goal OR pass landing on goal)
            +2   ball advanced toward goal via forward pass
            -5   ball exits right wall without scoring
            -3   shoot without ball
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

            # Level 2: agent dribbles ball into goal cell → score
            if self.level >= 2 and self.has_ball and self.agent_pos == list(self.goal_pos):
                reward += config.REWARD_GOAL_L2  # +40
                self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done

    def render(self):
        """Print ASCII grid to stdout."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        if self.level == 1:
            # Mark shooting zone with a visual hint
            for y in range(self.height):
                for x in range(self.shoot_zone_x, self.width - 1):
                    grid[y][x] = ","

        gx, gy = self.goal_pos
        grid[gy][gx] = "G"

        if not self.has_ball:
            bx, by = self.ball_pos
            if 0 <= bx < self.width and 0 <= by < self.height:
                grid[by][bx] = "B"

        ax, ay = self.agent_pos
        grid[ay][ax] = "P" if self.has_ball else "A"

        has_str = "YES" if self.has_ball else "NO "
        print(f"\nLevel {self.level} | Step {self.step_count} | Ball: {has_str}")
        for row in grid:
            print(" ".join(row))
        print(f"Agent={tuple(self.agent_pos)}  Ball={tuple(self.ball_pos)}  Goal={self.goal_pos}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_shoot(self):
        """
        Resolve a shoot action. Returns the reward delta (step penalty excluded).
        Dispatches to the level-specific implementation.
        """
        if self.level == 1:
            return self._shoot_l1()
        else:
            return self._shoot_l2()

    def _shoot_l1(self):
        """
        Level 1 shoot: zone-based goal attempt.
            No ball       → -5
            Outside zone  → -5, ball dropped
            In zone, aligned with goal row  → +30, episode ends
            In zone, wrong row              → miss, ball to right wall, no penalty
        """
        if not self.has_ball:
            return config.REWARD_SHOOT_NO_BALL  # -5

        ax, ay = self.agent_pos
        _, gy = self.goal_pos

        if ax < self.shoot_zone_x:
            self.has_ball = False
            return config.REWARD_SHOOT_BAD_POS  # -5

        # In shooting zone
        if ay == gy:
            self.ball_pos = list(self.goal_pos)
            self.has_ball = False
            self.done = True
            return config.REWARD_GOAL  # +30
        else:
            # Miss — ball rolls to right wall, same row
            self.ball_pos = [self.width - 1, ay]
            self.has_ball = False
            return 0

    def _shoot_l2(self):
        """
        Level 2 shoot: forward pass — ball travels SHOOT_RANGE cells to the right.
            No ball                              → -3
            Ball exits right wall, goal row      → +40, episode ends
            Ball exits right wall, wrong row     → -5, ball at right wall
            Ball stays in field, lands on goal   → +40, episode ends
            Ball stays in field, moves closer    → +2
            Ball stays in field, no progress     → 0
        """
        if not self.has_ball:
            return config.REWARD_SHOOT_WASTED  # -3

        bx, by = self.ball_pos
        gx, gy = self.goal_pos
        raw_new_bx = bx + config.SHOOT_RANGE

        self.has_ball = False

        if raw_new_bx >= self.width:
            # Ball would exit the right side
            self.ball_pos = [self.width - 1, by]
            if by == gy:
                self.done = True
                return config.REWARD_GOAL_L2  # +40
            else:
                return config.REWARD_BALL_OUT  # -5

        # Ball stays in field
        self.ball_pos = [raw_new_bx, by]
        if self.ball_pos == list(self.goal_pos):
            self.done = True
            return config.REWARD_GOAL_L2  # +40

        old_dist = abs(bx - gx) + abs(by - gy)
        new_dist = abs(raw_new_bx - gx) + abs(by - gy)
        if new_dist < old_dist:
            return config.REWARD_PASS_CLOSER  # +2
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
