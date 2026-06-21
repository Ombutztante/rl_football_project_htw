import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config


class FootballEnv:
    """
    2D Gridworld football environment — supports Level 1, 2, 3, 4, and 5.
    The active level is read from config.LEVEL (or passed as argument).

    Grid layout (10x6 default, width x height):

        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . .
        A . B . . . . . . G
        . . . . . . . . . .
        . . . . . . . . . .

    A = Agent without ball, P = Agent with ball, B = Ball, G = Goal,
    X = Opponent, # = Obstacle, T = Teammate, M = Teammate with ball

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

    Level 5 — cooperative play with teammate (extends Level 3, no obstacle)
        A rule-based teammate is added. The opponent starts mid-field on the goal
        row, blocking the direct dribble path. The agent must cooperate:
        action 4 outside the shooting zone passes to the teammate; inside the
        shooting zone it shoots directly (Level-1 style).
        The teammate positions itself to receive passes and scores autonomously.

    State (Level 1+2):  tuple (ax, ay, bx, by, has_ball)                           — 5 elements
    State (Level 3+4):  tuple (ax, ay, bx, by, has_ball, opp_x, opp_y)             — 7 elements
    State (Level 5):    tuple (ax, ay, bx, by, has_ball, opp_x, opp_y,
                               tm_x, tm_y, tm_has_ball)                             — 10 elements
    State (Level 6/X):  tuple (ax, ay, bx, by, has_ball,
                               opp1_x, opp1_y, opp2_x, opp2_y,
                               tm_x, tm_y, tm_has_ball)                            — 12 elements
    Actions: 0=up  1=down  2=left  3=right  4=shoot/pass  (always 5)
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
            5: config.BALL_START_X_L5,
            6: config.BALL_START_X_LX,
        }
        raw_ball_x = _ball_x_cfg.get(self.level)
        ball_x = raw_ball_x if raw_ball_x is not None else self.width // 2
        self._ball_start = (ball_x, self.height // 2)

        self.n_actions = 5

        self.obstacle_cells = frozenset(
            (config.OBSTACLE_X, y)
            for y in range(config.OBSTACLE_Y_START, config.OBSTACLE_Y_START + config.OBSTACLE_HEIGHT)
        ) if self.level == 4 else frozenset()

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

        if self.level >= 3 and self.level != 5:
            opp_x = self.goal_pos[0] - config.OPP_START_X_FROM_GOAL
            self.opp_pos = [int(np.clip(opp_x, 0, self.width - 1)), 0]

        if self.level == 5:
            self.opp_pos = [config.OPP_START_X_L5, config.OPP_START_Y_L5]
            self.tm_pos = [config.TM_START_X_L5, config.TM_START_Y_L5]
            self.tm_has_ball = False
            self._ball_in_flight = False
            self._pass_target = None

        if self.level == 6:
            self.opp1_pos = [config.OPP1_START_X_LX, config.OPP1_START_Y_LX]
            self.opp2_pos = [config.OPP2_START_X_LX, config.OPP2_START_Y_LX]
            self.tm_pos   = [config.TM_START_X_LX,   config.TM_START_Y_LX]
            self.tm_has_ball = False
            self._ball_in_flight = False
            self._pass_target    = None

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

        if action == 4:  # shoot / pass
            reward += self._handle_shoot()
        else:
            prev_dist = self._dist_to_goal()

            dx, dy = self._DELTA[action]
            new_x = int(np.clip(self.agent_pos[0] + dx, 0, self.width - 1))
            new_y = int(np.clip(self.agent_pos[1] + dy, 0, self.height - 1))

            if self.level == 4 and (new_x, new_y) in self.obstacle_cells:
                reward += config.REWARD_HIT_OBSTACLE  # blocked, agent stays in place
            else:
                self.agent_pos[0] = new_x
                self.agent_pos[1] = new_y

            # Ball follows agent when carried
            if self.has_ball:
                self.ball_pos = self.agent_pos.copy()

            # Ball pickup — blocked while agent's own pass is still in flight
            if not self.has_ball and not getattr(self, 'tm_has_ball', False) \
                    and not getattr(self, '_ball_in_flight', False) \
                    and self.agent_pos == self.ball_pos:
                self.has_ball = True
                reward += config.REWARD_BALL_PICKUP  # +5

            # Shaping: reward progress toward goal
            if self._dist_to_goal() < prev_dist:
                reward += config.REWARD_CLOSER  # +1

            # Level 2–4: agent dribbles ball into goal cell → score
            if 2 <= self.level <= 4 and self.has_ball and self.agent_pos == list(self.goal_pos):
                if self.level >= 4:
                    goal_reward = config.REWARD_GOAL_L4
                elif self.level >= 3:
                    goal_reward = config.REWARD_GOAL_L3
                else:
                    goal_reward = config.REWARD_GOAL_L2
                reward += goal_reward
                self.done = True

        # Level 5: advance ball in flight, then teammate moves
        if self.level == 5 and not self.done:
            self._advance_ball()
        if self.level == 5 and not self.done:
            reward += self._move_teammate()

        # Level 6: advance ball in flight, then teammate moves, then both opponents
        if self.level == 6 and not self.done:
            self._advance_ball()
        if self.level == 6 and not self.done:
            reward += self._move_teammate_lx()
        if self.level == 6 and not self.done:
            reward += self._move_and_check_opponents_lx()

        # Level 3–5: single opponent moves and may end the episode
        if self.level >= 3 and self.level != 6 and not self.done:
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
            Level 5    (12 values):  ax, ay, bx, by, has_ball, gx, gy,
                                     opp_x, opp_y, tm_x, tm_y, tm_has_ball
        """
        W, H = self.width - 1, self.height - 1
        gx, gy = self.goal_pos
        ax, ay, bx, by, has_ball = state[0], state[1], state[2], state[3], state[4]
        arr = [ax / W, ay / H, bx / W, by / H, float(has_ball), gx / W, gy / H]
        if self.level >= 3:
            arr += [state[5] / W, state[6] / H]  # opp_x, opp_y (or opp1 for L6)
        if self.level == 5:
            arr += [state[7] / W, state[8] / H, float(state[9])]
        if self.level == 6:
            # state[5,6]=opp1 already added above; add opp2 + teammate
            arr += [state[7] / W, state[8] / H,
                    state[9] / W, state[10] / H, float(state[11])]
        return np.array(arr, dtype=np.float32)

    def get_state_size(self):
        """Return the length of the array produced by state_to_array()."""
        if self.level == 6:
            return 14  # 5 + 2(goal) + 2(opp1) + 2(opp2) + 3(tm)
        if self.level == 5:
            return 12
        return 9 if self.level >= 3 else 7

    def render(self):
        """Print ASCII grid to stdout."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        if self.level == 1:
            for y in range(self.height):
                for x in range(self.shoot_zone_x, self.width - 1):
                    grid[y][x] = ","

        if self.level == 4:
            for (ox, oy) in sorted(self.obstacle_cells):
                if 0 <= ox < self.width and 0 <= oy < self.height:
                    grid[oy][ox] = "#"

        gx, gy = self.goal_pos
        grid[gy][gx] = "G"

        if not self.has_ball and not getattr(self, 'tm_has_ball', False):
            bx, by = self.ball_pos
            if 0 <= bx < self.width and 0 <= by < self.height:
                grid[by][bx] = "B"

        if self.level >= 3 and self.level != 6:
            ox, oy = self.opp_pos
            if 0 <= ox < self.width and 0 <= oy < self.height:
                grid[oy][ox] = "X"

        if self.level == 6:
            for opos, label in [(self.opp1_pos, "X"), (self.opp2_pos, "Y")]:
                ox, oy = opos
                if 0 <= ox < self.width and 0 <= oy < self.height:
                    grid[oy][ox] = label

        if self.level in (5, 6):
            tx, ty = self.tm_pos
            if 0 <= tx < self.width and 0 <= ty < self.height:
                grid[ty][tx] = "M" if self.tm_has_ball else "T"

        ax, ay = self.agent_pos
        grid[ay][ax] = "P" if self.has_ball else "A"

        has_str = "YES" if self.has_ball else "NO "
        print(f"\nLevel {self.level} | Step {self.step_count} | Ball: {has_str}")
        for row in grid:
            print(" ".join(row))
        info = f"Agent={tuple(self.agent_pos)}  Ball={tuple(self.ball_pos)}  Goal={self.goal_pos}"
        if self.level >= 3 and self.level != 6:
            info += f"  Opp={tuple(self.opp_pos)}"
        if self.level == 6:
            info += f"  Opp1={tuple(self.opp1_pos)}  Opp2={tuple(self.opp2_pos)}"
        if self.level in (5, 6):
            info += f"  TM={tuple(self.tm_pos)}  TM_ball={self.tm_has_ball}"
        print(info)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_shoot(self):
        """Resolve a shoot action. Returns the reward delta (step penalty excluded)."""
        if self.level == 1:
            return self._shoot_l1()
        elif self.level == 5:
            return self._shoot_l5()
        elif self.level == 6:
            return self._shoot_lx()
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

    def _shoot_l5(self):
        """
        Level 5 shoot/pass action.
            No ball                             → -5
            Has ball, in shooting zone, aligned → +70, score, episode ends
            Has ball, in shooting zone, bad row → miss, ball to right wall
            Has ball, outside shooting zone     → kick ball toward teammate;
                                                  ball travels PASS_SPEED cells per step,
                                                  remains visible on the field until tm picks up.
                                                  Reward of +15 given when tm collects the ball.
        """
        if not self.has_ball:
            return config.REWARD_BAD_SHOT_L5

        ax, ay = self.agent_pos
        _, gy = self.goal_pos

        if ax >= self.shoot_zone_x:
            self.has_ball = False
            if ay == gy:
                self.ball_pos = list(self.goal_pos)
                self.done = True
                return config.REWARD_GOAL_L5
            else:
                self.ball_pos = [self.width - 1, ay]
                return 0

        # Outside shooting zone: kick ball toward teammate's current position.
        # Ball travels PASS_SPEED cells per step via _advance_ball() each subsequent step.
        self.has_ball = False
        self._ball_in_flight = True
        self._pass_target = list(self.tm_pos)
        return 0

    def _advance_ball(self):
        """
        Move the in-flight ball PASS_SPEED steps toward _pass_target per env step.
        Both x and y are updated simultaneously each mini-step (diagonal travel),
        so the ball can fly over obstacles/opponents rather than hugging one axis.
        """
        if not self._ball_in_flight:
            return

        bx, by = self.ball_pos
        tx, ty = self._pass_target

        for _ in range(config.PASS_SPEED_L5):
            dx, dy = tx - bx, ty - by
            if dx == 0 and dy == 0:
                self._ball_in_flight = False
                self._pass_target = None
                break
            # Diagonal: move in both axes simultaneously (like a kicked ball)
            if dx != 0:
                bx += int(np.sign(dx))
            if dy != 0:
                by += int(np.sign(dy))
            bx = int(np.clip(bx, 0, self.width - 1))
            by = int(np.clip(by, 0, self.height - 1))
            if [bx, by] == [tx, ty]:
                self._ball_in_flight = False
                self._pass_target = None
                break

        self.ball_pos = [bx, by]

    def _move_teammate(self):
        """
        Move the rule-based teammate one cell per step and check for a goal.
        When the teammate has the ball: moves toward the goal and scores on arrival.
        When agent has the ball: moves toward the optimal receiving position (shoot zone).
        When ball is loose or in flight: runs toward the ball to collect it.
        Returns reward delta.
        """
        reward = 0
        tx, ty = self.tm_pos
        gx, gy = self.goal_pos

        if self.tm_has_ball:
            target_x, target_y = gx, gy
        elif self.has_ball:
            # Position teammate to receive a pass in the shooting zone
            target_x, target_y = self.shoot_zone_x, gy
        else:
            # Ball loose or in flight: run toward ball
            target_x, target_y = self.ball_pos[0], self.ball_pos[1]

        dx = target_x - tx
        dy = target_y - ty
        if dx != 0 or dy != 0:
            if abs(dx) >= abs(dy):
                tx += int(np.sign(dx))
            else:
                ty += int(np.sign(dy))
            self.tm_pos = [int(np.clip(tx, 0, self.width - 1)),
                           int(np.clip(ty, 0, self.height - 1))]

        if self.tm_has_ball:
            self.ball_pos = self.tm_pos.copy()
            if self.tm_pos == list(self.goal_pos):
                self.tm_has_ball = False
                self.done = True
                reward += config.REWARD_GOAL_L5
        elif not self.has_ball and self.tm_pos == self.ball_pos:
            # Teammate collects ball (loose or arriving pass)
            self.tm_has_ball = True
            self._ball_in_flight = False
            self._pass_target = None
            self.ball_pos = self.tm_pos.copy()
            reward += config.REWARD_PASS_SUCCESS  # +15 whenever tm gets the ball

        return reward

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
                return config.REWARD_BALL_LOST           # -20 — opponent tackles agent
            if getattr(self, 'tm_has_ball', False):
                self.tm_has_ball = False
                return config.REWARD_OPP_REACHES_BALL    # -10 — opponent tackles teammate
            return config.REWARD_OPP_REACHES_BALL        # -10 — opponent reaches loose ball
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

    def _shoot_lx(self):
        """
        Level 6 shoot/pass — same mechanic as Level 5.
        Outside shooting zone: diagonal pass toward teammate.
        Inside shooting zone, aligned: direct goal (+80).
        """
        if not self.has_ball:
            return config.REWARD_BAD_SHOT_LX

        ax, ay = self.agent_pos
        _, gy = self.goal_pos

        if ax >= self.shoot_zone_x:
            self.has_ball = False
            if ay == gy:
                self.ball_pos = list(self.goal_pos)
                self.done = True
                return config.REWARD_GOAL_LX
            else:
                self.ball_pos = [self.width - 1, ay]
                return 0

        self.has_ball = False
        self._ball_in_flight = True
        self._pass_target = list(self.tm_pos)
        return 0

    def _move_teammate_lx(self):
        """
        Rule-based teammate for Level 6 — identical movement logic to Level 5
        but uses Level X reward constants.
        """
        reward = 0
        tx, ty = self.tm_pos
        gx, gy = self.goal_pos

        if self.tm_has_ball:
            target_x, target_y = gx, gy
        elif self.has_ball:
            target_x, target_y = self.shoot_zone_x, gy
        else:
            target_x, target_y = self.ball_pos[0], self.ball_pos[1]

        dx = target_x - tx
        dy = target_y - ty
        if dx != 0 or dy != 0:
            if abs(dx) >= abs(dy):
                tx += int(np.sign(dx))
            else:
                ty += int(np.sign(dy))
            self.tm_pos = [int(np.clip(tx, 0, self.width - 1)),
                           int(np.clip(ty, 0, self.height - 1))]

        if self.tm_has_ball:
            self.ball_pos = self.tm_pos.copy()
            if self.tm_pos == list(self.goal_pos):
                self.tm_has_ball = False
                self.done = True
                reward += config.REWARD_GOAL_LX
        elif not self.has_ball and self.tm_pos == self.ball_pos:
            self.tm_has_ball = True
            self._ball_in_flight = False
            self._pass_target = None
            self.ball_pos = self.tm_pos.copy()
            reward += config.REWARD_PASS_SUCCESS_LX

        return reward

    def _move_and_check_opponents_lx(self):
        """
        Level 6: move opp1 (ball-chaser) and opp2 (agent-presser).
        Opp1 reaching the ball ends the episode (like Level 3).
        Opp2 reaching the agent while agent carries ball → tackle, episode ends.
        """
        reward = 0
        if self.step_count % config.OPP_MOVE_EVERY == 0:
            self._move_entity_toward(self.opp1_pos, self.ball_pos)
            self._move_entity_toward(self.opp2_pos, self.agent_pos)

        # Opp1 reaches ball (not while ball is in flight)
        if not self._ball_in_flight and self.opp1_pos == self.ball_pos:
            self.done = True
            if self.has_ball:
                self.has_ball = False
                return config.REWARD_BALL_LOST
            if self.tm_has_ball:
                self.tm_has_ball = False
                return config.REWARD_OPP_REACHES_BALL
            return config.REWARD_OPP_REACHES_BALL

        # Opp2 tackles agent carrying ball
        if self.opp2_pos == self.agent_pos and self.has_ball:
            self.has_ball = False
            self.ball_pos = self.agent_pos.copy()
            self.done = True
            return config.REWARD_BALL_LOST

        return reward

    def _move_entity_toward(self, entity_pos, target_pos):
        """Move a list [x, y] one cell toward target (Manhattan greedy, prefer x-axis)."""
        ex, ey = entity_pos
        tx, ty = target_pos
        dx, dy = tx - ex, ty - ey
        if dx == 0 and dy == 0:
            return
        if abs(dx) >= abs(dy):
            ex += int(np.sign(dx))
        else:
            ey += int(np.sign(dy))
        entity_pos[0] = int(np.clip(ex, 0, self.width - 1))
        entity_pos[1] = int(np.clip(ey, 0, self.height - 1))

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
        if self.level >= 3 and self.level != 6:
            state += (self.opp_pos[0], self.opp_pos[1])
        if self.level == 5:
            state += (self.tm_pos[0], self.tm_pos[1], int(self.tm_has_ball))
        if self.level == 6:
            state += (self.opp1_pos[0], self.opp1_pos[1],
                      self.opp2_pos[0], self.opp2_pos[1],
                      self.tm_pos[0],   self.tm_pos[1],
                      int(self.tm_has_ball))
        return state
