import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from src.utils import ReplayBuffer


class QNetwork(nn.Module):
    """
    Two-hidden-layer MLP: state_size → hidden → hidden → n_actions.
    Input is a normalised float vector; output is one Q-value per action.
    """

    def __init__(self, state_size, n_actions, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network agent (Mnih et al., 2015).

    Stabilisation measures applied:
      - Huber loss (SmoothL1) instead of MSE — less sensitive to large Q-value errors
      - Gradient clipping (GRAD_CLIP_NORM) — prevents gradient explosion
      - Replay warm-up (REPLAY_WARMUP) — learning only starts once the buffer
        holds enough diverse transitions
      - Soft target update (TAU) after every gradient step:
            target = τ·online + (1-τ)·target
        No hard copy, no abrupt anchor loss → eliminates the periodic dip.

    The agent works on normalised float state arrays (env.state_to_array()).
    The training script is responsible for the conversion:

        state_arr = env.state_to_array(state)
        action = agent.choose_action(state_arr)
        next_state_arr = env.state_to_array(next_state)
        agent.store(state_arr, action, reward, next_state_arr, done)
        agent.learn()

    Call decay_epsilon() once per episode.
    """

    def __init__(
        self,
        state_size,
        n_actions=5,
        lr=None,
        gamma=None,
        epsilon_start=None,
        epsilon_min=None,
        epsilon_decay=None,
        batch_size=None,
        memory_size=None,
        hidden_size=None,
        replay_warmup=None,
        grad_clip_norm=None,
    ):
        self.state_size = state_size
        self.n_actions = n_actions
        self.lr = lr if lr is not None else config.DQN_LR
        self.gamma = gamma if gamma is not None else config.DQN_GAMMA
        self.epsilon = epsilon_start if epsilon_start is not None else config.DQN_EPSILON_START
        self.epsilon_min = epsilon_min if epsilon_min is not None else config.DQN_EPSILON_MIN
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else config.DQN_EPSILON_DECAY
        self.batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm is not None else config.GRAD_CLIP_NORM
        self.tau = config.TAU
        # Warm-up must be at least batch_size so the first sample call always succeeds
        warmup = replay_warmup if replay_warmup is not None else config.REPLAY_WARMUP
        self.replay_warmup = max(warmup, self.batch_size)
        hidden_size = hidden_size if hidden_size is not None else config.HIDDEN_SIZE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_size, n_actions, hidden_size).to(self.device)
        self.target_net = QNetwork(state_size, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(memory_size if memory_size is not None else config.MEMORY_SIZE)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def choose_action(self, state_array):
        """ε-greedy action selection. Expects a normalised numpy float array."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        """Add a transition to the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample a batch and perform one gradient update.
        Returns the loss value, or None if the buffer has not reached replay_warmup yet.
        """
        if len(self.memory) < self.replay_warmup:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for the taken actions
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: r  +  γ · max_a' Q_target(s', a')  (zero for terminal states)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        # Huber loss — less sensitive to large Q-value errors than MSE
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents the explosion seen with MSE + high LR
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # Soft target update: θ_target = τ·θ_online + (1-τ)·θ_target
        for target_p, online_p in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

        return loss.item()

    def decay_epsilon(self):
        """Multiply epsilon by decay factor, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save network weights and current epsilon to a .pt file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        """Load network weights and epsilon from a .pt file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint["epsilon"]
        self.target_net.eval()

