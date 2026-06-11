import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import random
from collections import defaultdict
import config


class QTableAgent:
    """
    Tabular Q-Learning agent.

    Q-table is a defaultdict keyed by the raw state tuple returned by the
    environment. Each entry is a list of Q-values, one per action.

    Update rule (off-policy):
        Q(s,a) ← Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))

    Exploration: ε-greedy. Call decay_epsilon() once per episode.
    """

    def __init__(
        self,
        n_actions=5,
        lr=None,
        gamma=None,
        epsilon_start=None,
        epsilon_min=None,
        epsilon_decay=None,
    ):
        self.n_actions = n_actions
        self.lr = lr if lr is not None else config.Q_LR
        self.gamma = gamma if gamma is not None else config.Q_GAMMA
        self.epsilon = epsilon_start if epsilon_start is not None else config.Q_EPSILON_START
        self.epsilon_min = epsilon_min if epsilon_min is not None else config.Q_EPSILON_MIN
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else config.Q_EPSILON_DECAY

        # Q[state] = list of Q-values indexed by action
        self.q_table = defaultdict(lambda: [0.0] * self.n_actions)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def choose_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self._greedy_action(state)

    def learn(self, state, action, reward, next_state, done):
        """Apply one Q-learning update step."""
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.lr * (target - current_q)

    def decay_epsilon(self):
        """Multiply epsilon by decay factor, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Pickle the Q-table to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        """Load a previously saved Q-table."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: [0.0] * self.n_actions, data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _greedy_action(self, state):
        """Return the action with the highest Q-value; break ties randomly."""
        q_values = self.q_table[state]
        max_q = max(q_values)
        best = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best)
