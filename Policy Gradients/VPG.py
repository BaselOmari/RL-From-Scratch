import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn, optim


class DQN(nn.Module):
    def __init__(self, state_dim, action_size):
        super(DQN, self).__init__()

        hidden_size = 16
        self.inp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.PReLU(),
        )
        self.h1 = torch.nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
        )
        self.h2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        h = self.inp(x)
        h = self.h1(h)
        h = self.h2(h)
        Q = self.out(h)
        return Q

    def action(self, x):
        A = self.forward(x)
        return int(torch.multinomial(A, 1))


env = gym.make("CartPole-v1", render_mode="human")

lr = 0.001  # learning rate
bs = 64  # batch size
eps = 0.2  # for epsilon-greedy selection
gamma = 0.9  # discount factor
an = 2  # number of actions
sf = 4  # number of state features
I = 25
M = 100  # episode count
T = 2000  # max timesteps


net = DQN(sf, an)
optimizer = optim.Adam(net.parameters(), lr)


for iter in range(I):
    # Step 1: Collect step of trajectories by running policy
    D = []
    for e in range(M):
        ep = []
        st, _ = env.reset()
        st = torch.from_numpy(st)
        for t in range(T):
            with torch.no_grad():
                at = net.action(st)
            st1, rt, ter, _, _ = env.step(at)
            st1 = torch.from_numpy(st1)
            ep.append([st, at, rt])
            if ter:
                break
            else:
                st = st1
        g = 0
        for t in range(len(ep) - 1, -1, -1):
            ep[t][2] += g
            g = ep[t][2] * gamma

        D.extend(ep)
