import gymnasium as gym
import torch
from numpy import log
from torch import nn, optim


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()

        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
        )
        self.h1 = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
        )
        self.h2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        h = self.inp(x)
        h = self.h1(h)
        h = self.h2(h)
        Z = self.out(h)
        return Z


class DQN(Net):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQN, self).__init__(state_dim, hidden_dim, action_dim)

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
epch = 100
I = 25 # iteration count
M = 100  # episode count
T = 2000  # max timesteps

value = Net(sf, 16, 1)
v_optim = optim.Adam(value.parameters(), lr)

policy = DQN(sf, 16, an)
p_optim = optim.Adam(policy.parameters(), lr)


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
