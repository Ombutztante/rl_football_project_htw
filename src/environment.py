import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config


class FootballEnv:
    """
    2D Gridworld football environment — supports Level 1, 2, 3, and 4.
    The active level is read from config.LEVEL (or passed as argument).

    Grid layout (10x6 default, width x height):

        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . .
        A . B . . . . . . G
        . . . . . . . . . .
        . . . . . . . . . .

    A = Agent without ball, P = Agent with ball, B = Ball, G = Goal, X = Opponent, # = Obstacle

    Level 1 — shoot only from good position
        Agent picks up ball, reaches shooting zone, shoots.
        Scoring: shoot in zone AND aligned with goal row.

    Level 2 — dribbling vs. forward pass
        Shoot sends ball SHOOT_RANGE cells to the right (forward pass).
        Agent loses possession and must chase.
        Scoring: ball lands on goal cell OR agent dribbles ball to goal cell.

    Level 3 — opponent moves toward ball
        Same mechanics as Level 2 plus a rule-based opponent.
        Opponent starts near the goal and moves one cell toward the ball
        every OPP_MOVE_EVERY agent steps.
        Episode ends with a penalty when opponent reaches the ball.

    Level 4 — obstacle blocks direct shots (extends Level 3)
        Same mechanics as Level 3 plus a static vertical wall at column OBSTACLE_X.
        The obstacle blocks both agent movement and forward passes.
        Rows 4–5 are free so the agent can navigate around below the obstacle.

    State (Level 1+2):    tuple (agent_x, agent_y, ball_x, ball_y, has_ball)               — 5 elements
    State (Level 3+4):    tuple (agent_x, agent_y, ball_x, ball_y, has_ball, opp_x, opp_y) — 7 elements
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
        self._agent_start = (0, self.height // 2)

        _ball_x_cfg = {
            1: config.BALL_START_X_L1,
            2: config.BALL_START_X_L2,
            3: config.BALL_START_X_L3,
            4: config.BALL_START_X_L4,
        }
        raw_ball_x = _ball_x_cfg.get(self.level)
        ball_x = raw_ball_x if raw_ball_x is not None else self.width // 2
        self._ball_start = (ball_x, self.height // 2)

        self.n_actions = 5

        self.obstacle_cells = frozenset(
            (config.OBSTACLE_X, y)
            for y in range(config.OBSTACLE_Y_START, config.OBSTACLE_Y_START + config.OBSTACLE_HEIGHT)
        ) if self.level >= 4 else frozenset()

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

        if self.level >= 3:
            opp_x = self.goal_pos[0] - config.OPP_START_X_FROM_GOAL
            self.opp_pos = [int(np.clip(opp_x, 0, self.width - 1)), 0]

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

        Level 3 reward structure (same mechanics as Level 2 + opponent):
            -1   every step
            +5   ball picked up
            +1   moved closer to goal (shaping, movement only)
            +50  goal scored (dribble to goal OR pass landing on goal)
            +2   ball advanced toward goal via forward pass
            -5   ball exits right wall without scoring
            -5   shoot without ball
            -10  opponent reaches loose ball (episode ends)
            -20  opponent tackles agent carrying ball (episode ends)
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
            new_x = int(np.clip(self.agent_pos[0] + dx, 0, self.width - 1))
            new_y = int(np.clip(self.agent_pos[1] + dy, 0, self.height - 1))

            if self.level >= 4 and (new_x, new_y) in self.obstacle_cells:
                reward += config.REWARD_HIT_OBSTACLE  # blocked, agent stays in place
            else:
                self.agent_pos[0] = new_x
                self.agent_pos[1] = new_y

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

            # Level 2+: agent dribbles ball into goal cell → score
            if self.level >= 2 and self.has_ball and self.agent_pos == list(self.goal_pos):
                if self.level >= 4:
                    goal_reward = config.REWARD_GOAL_L4
                elif self.level >= 3:
                    goal_reward = config.REWARD_GOAL_L3
                else:
                    goal_reward = config.REWARD_GOAL_L2
                reward += goal_reward
                self.done = True

        # Level 3: opponent moves and may end the episode
        if self.level >= 3 and not self.done:
            reward += self._move_and_check_opponent()

        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done

    def state_to_array(self, state):
        """
        Convert a state tuple to a normalised float numpy array for DQN input.

        Layout (coordinates divided by grid dimensions − 1 to land in [0, 1]):
            Level 1+2   (7 values):  ax, ay, bx, by, has_ball, gx, gy
            Level 3+4   (9 values):  ax, ay, bx, by, has_ball, gx, gy, opp_x, opp_y
        """
        W, H = self.width - 1, self.height - 1
        gx, gy = self.goal_pos
        ax, ay, bx, by, has_ball = state[0], state[1], state[2], state[3], state[4]
        arr = [ax / W, ay / H, bx / W, by / H, float(has_ball), gx / W, gy / H]
        if self.level >= 3:
            arr += [state[5] / W, state[6] / H]
        return np.array(arr, dtype=np.float32)

    def get_state_size(self):
        """Return the length of the array produced by state_to_array()."""
        return 9 if self.level >= 3 else 7

    def render(self):
        """Print ASCII grid to stdout."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        if self.level == 1:
            for y in range(self.height):
                for x in range(self.shoot_zone_x, self.width - 1):
                    grid[y][x] = ","

        if self.level >= 4:
            for (ox, oy) in sorted(self.obstacle_cells):
                if 0 <= ox < self.width and 0 <= oy < self.height:
                    grid[oy][ox] = "#"

        gx, gy = self.goal_pos
        grid[gy][gx] = "G"

        if not self.has_ball:
            bx, by = self.ball_pos
            if 0 <= bx < self.width and 0 <= by < self.height:
                grid[by][bx] = "B"

        if self.level >= 3:
            ox, oy = self.opp_pos
            if 0 <= ox < self.width and 0 <= oy < self.height:
                grid[oy][ox] = "X"

        ax, ay = self.agent_pos
        grid[ay][ax] = "P" if self.has_ball else "A"

        has_str = "YES" if self.has_ball else "NO "
        print(f"\nLevel {self.level} | Step {self.step_count} | Ball: {has_str}")
        for row in grid:
            print(" ".join(row))
        info = f"Agent={tuple(self.agent_pos)}  Ball={tuple(self.ball_pos)}  Goal={self.goal_pos}"
        if self.level >= 3:
            info += f"  Opp={tuple(self.opp_pos)}"
        print(info)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_shoot(self):
        """Resolve a shoot action. Returns the reward delta (step penalty excluded)."""
        if self.level == 1:
            return self._shoot_l1()
        else:
            return self._shoot_forward_pass()

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

    def _shoot_forward_pass(self):
        """
        Level 2+: ball travels SHOOT_RANGE cells to the right.
        Level 2: no-ball = -3, goal = +40
        Level 3: no-ball = -5, goal = +50
        Level 4: no-ball = -5, goal = +60, obstacle blocks shot
        """
        if not self.has_ball:
            return config.REWARD_SHOOT_WASTED if self.level == 2 else config.REWARD_BAD_SHOT_L3

        if self.level == 2:
            goal_reward = config.REWARD_GOAL_L2
        elif self.level == 3:
            goal_reward = config.REWARD_GOAL_L3
        else:
            goal_reward = config.REWARD_GOAL_L4

        bx, by = self.ball_pos
        gx, gy = self.goal_pos
        self.has_ball = False

        # Level 4: step through each cell to detect obstacle collision
        if self.level >= 4:
            for step in range(1, config.SHOOT_RANGE + 1):
                check_x = bx + step
                if check_x >= self.width:
                    self.ball_pos = [self.width - 1, by]
                    if by == gy:
                        self.done = True
                        return goal_reward
                    return config.REWARD_BALL_OUT
                if [check_x, by] == list(self.goal_pos):
                    self.ball_pos = [check_x, by]
                    self.done = True
                    return goal_reward
                if (check_x, by) in self.obstacle_cells:
                    self.ball_pos = [check_x - 1, by]
                    return config.REWARD_SHOT_BLOCKED
            self.ball_pos = [bx + config.SHOOT_RANGE, by]
            old_dist = abs(bx - gx) + abs(by - gy)
            new_dist = abs(self.ball_pos[0] - gx) + abs(by - gy)
            return config.REWARD_PASS_CLOSER if new_dist < old_dist else 0

        # Level 2/3: direct calculation
        raw_new_bx = bx + config.SHOOT_RANGE
        if raw_new_bx >= self.width:
            self.ball_pos = [self.width - 1, by]
            if by == gy:
                self.done = True
                return goal_reward
            return config.REWARD_BALL_OUT
        self.ball_pos = [raw_new_bx, by]
        if self.ball_pos == list(self.goal_pos):
            self.done = True
            return goal_reward
        old_dist = abs(bx - gx) + abs(by - gy)
        new_dist = abs(raw_new_bx - gx) + abs(by - gy)
        return config.REWARD_PASS_CLOSER if new_dist < old_dist else 0

    def _move_and_check_opponent(self):
        """
        Move opponent toward ball (every OPP_MOVE_EVERY steps), then check
        if opponent has reached the ball. Returns reward delta.
        """
        if self.step_count % config.OPP_MOVE_EVERY == 0:
            self._move_opponent()

        if self.opp_pos == self.ball_pos:
            self.done = True
            if self.has_ball:
                self.has_ball = False
                return config.REWARD_BALL_LOST       # -20 — opponent tackles agent
            return config.REWARD_OPP_REACHES_BALL    # -10 — opponent reaches loose ball
        return 0

    def _move_opponent(self):
        """Move opponent one cell toward the ball (Manhattan greedy, prefer x-axis)."""
        ox, oy = self.opp_pos
        bx, by = self.ball_pos
        dx = bx - ox
        dy = by - oy
        if dx == 0 and dy == 0:
            return
        if abs(dx) >= abs(dy):
            ox += int(np.sign(dx))
        else:
            oy += int(np.sign(dy))
        self.opp_pos = [int(np.clip(ox, 0, self.width - 1)), int(np.clip(oy, 0, self.height - 1))]

    def _dist_to_goal(self):
        """Manhattan distance from agent to goal."""
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        return abs(ax - gx) + abs(ay - gy)

    def _get_state(self):
        """Return state as a hashable tuple of ints."""
        state = (
            self.agent_pos[0],
            self.agent_pos[1],
            self.ball_pos[0],
            self.ball_pos[1],
            int(self.has_ball),
        )
        if self.level >= 3:
            state += (self.opp_pos[0], self.opp_pos[1])
        return state
