import gymnasium as gym
import torch
from torch import log, nn, optim
from torch.distributions.categorical import Categorical


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
        A = Categorical(self.forward(x))
        act = A.sample()
        log_act = A.log_prob(act)
        return act, log_act


env = gym.make("CartPole-v1", render_mode="human")

lr = 0.001  # learning rate
bs = 64  # batch size
eps = 0.2  # for epsilon-greedy selection
gamma = 0.9  # discount factor
an = 2  # number of actions
sf = 4  # number of state features
epch = 100
I = 32  # iteration count
M = 32  # episode count
T = 2000  # max timesteps

value = Net(sf, 64, 1)
v_optim = optim.Adam(value.parameters(), lr)

policy = DQN(sf, 64, an)
p_optim = optim.Adam(policy.parameters(), lr)


for k in range(I):
    # Step 1: Collect step of trajectories by running policy
    D = []
    for e in range(M):
        ep = []
        st, _ = env.reset()
        st = torch.from_numpy(st)
        for t in range(T):
            with torch.no_grad():
                at, log_at = policy.action(st)
            st1, rt, ter, _, _ = env.step(at)
            st1 = torch.from_numpy(st1)
            ep.append([st, at, rt, log_at])
            if ter:
                break
            else:
                st = st1

        # Step 2+3: Compute rewards to go and advantage estimates
        G = 0
        for t in range(len(ep) - 1, -1, -1):
            # Compute rewards to go
            ep[t][2] += G
            G = ep[t][2] * gamma

            # Compute advantage estimate
            with torch.no_grad():
                A = G - value(ep[t][0])
            ep[t].append(A)

        D.extend(ep)

    # Step 4: Estimate policy gradient
    p_optim.zero_grad()
    gk = 0
    for s, act, r, logp, e in D:
        gk += adv * log(policy(s)[act])
    gk /= len(D)
    gk.backward()
    p_optim.step()

    # Step 5: Fit value function on mean-squared error
    for e in range(epch):
        v_optim.zero_grad()
        loss = 0
        for s, act, r, adv in D:
            loss += (value(s) - r) ** 2
        loss.backward()
        v_optim.step()
