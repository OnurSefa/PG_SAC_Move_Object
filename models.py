import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """Critic network for SAC, estimates Q-values."""

    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256):
        super().__init__()

        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._initialize_weights(self.q1_network[-1], 1e-3)
        self._initialize_weights(self.q2_network[-1], 1e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)

        q1 = self.q1_network(x)
        q2 = self.q2_network(x)

        return q1, q2

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_network(x)

    def _initialize_weights(self, layer, init_w):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=init_w)
            nn.init.constant_(layer.bias, 0)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256,
                 log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self._initialize_weights(self.mean_head, 1e-3)
        self._initialize_weights(self.log_std_head, 1e-3)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        self.action_dim = action_dim

    def forward(self, state):
        features = self.policy_net(state)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Correction for tanh squashing
        # log_prob = log_prob - torch.log(1 - y_t.pow(2) + self.epsilon)
        log_prob = log_prob - 2 * (np.log(2) - x_t - F.softplus(-2 * x_t))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)

        return y_t, log_prob, mean_action

    def _initialize_weights(self, layer, init_w):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=init_w)
            nn.init.constant_(layer.bias, 0)


class ActorHigh(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_handler = nn.Sequential(
            nn.Linear(33, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # x_mean, y_mean, x_var, y_var
        )

    def forward(self, x):
        x = self.state_handler(x)
        return x

