#!/home/a/anaconda3/envs/torch/bin/python3
import torch
from collections import deque
from random import sample
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class Actor_Linear(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_Linear, self).__init__()
        self.h1 = nn.Linear(state_dim, 512)
        self.h2 = nn.Linear(512, 256)
        self.h3 = nn.Linear(256, 64)

        self.mean = nn.Linear(64, action_dim)
        self.std = nn.Linear(64, action_dim)

        self.apply(weights_init_)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, x):
        # x_y_z = torch.cat((x, y), 1)
        out = F.relu(self.h1(x))
        out = F.relu(self.h2(out))
        out = F.relu(self.h3(out))

        mean = self.mean(out)
        log_std = self.std(out)

        std = F.softplus(log_std)
        dist = Normal(mean, std)

        normal_sample = dist.rsample()  # 在标准化正态分布上采样
        log_prob = dist.log_prob(normal_sample)  # 计算该值的标准正太分布上的概率

        # action = torch.tanh(normal_sample)  # 对数值进行tanh

        action = torch.zeros(normal_sample.shape[0], 2).to(self.device)
        action[:, 0] = torch.tanh(mean[:, 0])
        # action[:, 1] = torch.tanh(mean[:, 1])
        action[:, 1] = torch.sigmoid(mean[:, 1])

        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)  # 为了提升目标对应的概率值
        # action = action * 2  # 对action求取范围

        action[:, 0] = action[:, 0] * 2.5
        action[:, 1] = action[:, 1] * 0.22

        return action, log_prob

class Actor_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_net, self).__init__()
        # self.cnn = cnn()
        self.Linear = Actor_Linear(state_dim=state_dim, action_dim=action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state: tuple):
        # out_cnn = self.cnn(state[0])
        # mean, log_std = self.Linear(state[1], out_cnn)
        action, log_std = self.Linear(state)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return action, log_std