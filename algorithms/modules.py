import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )
        self.logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        action_mean = self.net(x)
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return action_mean, action_std


class SACActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.mean_layer = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        self.log_std_layer = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)

    def forward(self, x):
        feat = self.net(x)
        mean = self.mean_layer(feat)
        log_std = self.log_std_layer(feat)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size, num_agents=None, num_obstacles=None):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward(self, x, a):
        cat_inputs = torch.cat([x, a], dim=1)
        return self.net(cat_inputs)


class Critic(nn.Module):
    def __init__(self, obs_dim, num_agents=None, hidden_size=64, num_obs=None):
        super().__init__()
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def forward(self, x):
        return self.value_head(x)