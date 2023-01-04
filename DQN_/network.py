#!/home/a/anaconda3/envs/torch/bin/python3
import torch
from collections import deque
from random import sample
import torch.nn as nn
import torch.nn.functional as F


class DDQN(nn.Module):
    def __init__(self, state_dim):
        super(DDQN, self).__init__()
        self.h1 = nn.Linear(state_dim, 512)
        self.h1.weight.data.normal_(0, 0.1)
        self.h2 = nn.Linear(512, 256)
        self.h2.weight.data.normal_(0, 0.1)
        self.h3 = nn.Linear(256, 64)
        self.h3.weight.data.normal_(0, 0.1)
        self.h4 = nn.Linear(64, 5)
        self.h4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x_y=torch.cat((x, y), -1)
        out = F.relu(self.h1(x))
        out = F.relu(self.h2(out))
        out = F.relu(self.h3(out))
        out = self.h4(out)
        return out


class Replay_Buffers():
    def __init__(self):
        self.buffer_size = 1000000
        self.buffer = deque([], maxlen=self.buffer_size)
        self.batch = 64

    def write_Buffers(self, state, next_state, reward, action, done):
        once = {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done, }
        self.buffer.append(once)
        if len(self.buffer) > self.batch:
            return sample(self.buffer, self.batch)
