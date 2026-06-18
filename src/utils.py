import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer for DQN experience replay.

    Stores (state, action, reward, next_state, done) tuples.
    States are expected to be 1-D float numpy arrays.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Return a random batch as five numpy arrays ready for torch conversion."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def set_seed(seed):
    """Set Python, NumPy, and (if available) PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
