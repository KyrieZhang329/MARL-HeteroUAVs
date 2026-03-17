import copy

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.buffer import ReplayBuffer
from algorithms.modules import QNetwork, SACActor


class MASACAgent:
    def __init__(self, obs_dim, action_dim, num_agents, num_obs, args, device):
        self.device = device
        self.args = args
        self.tau = 0.005
        self.gamma = 0.99

        self.actor = SACActor(obs_dim, action_dim, args.hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.q1 = QNetwork(obs_dim, action_dim, args.hidden_size, num_agents, num_obs).to(device)
        self.q2 = QNetwork(obs_dim, action_dim, args.hidden_size, num_agents, num_obs).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=args.learning_rate,
        )

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.learning_rate)
        self.alpha = self.log_alpha.exp().item()
        self.buffer = ReplayBuffer(args.buffer_size, obs_dim, action_dim, device)

    def get_action(self, obs, evaluate=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, mean_action = self.actor.get_action(obs)
        if evaluate:
            return mean_action.cpu().numpy()[0]
        return action.cpu().numpy()[0]

    def update(self, batch_size):
        if self.buffer.size < batch_size:
            return 0, 0

        obs, actions, rewards, next_obs, dones = self.buffer.sample(batch_size)
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action(next_obs)
            target_q1 = self.q1_target(next_obs, next_actions)
            target_q2 = self.q2_target(next_obs, next_actions)
            min_target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * (min_target_q - self.alpha * next_log_probs)

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        q_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False

        new_actions, log_probs, _ = self.actor.get_action(obs)
        q1_new = self.q1(obs, new_actions)
        q2_new = self.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item(), actor_loss.item()