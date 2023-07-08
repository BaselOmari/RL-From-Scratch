import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import log, nn, optim
from torch.distributions.categorical import Categorical


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()

        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.h1 = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        h = self.inp(x)
        h = self.h1(h)
        Z = self.out(h)
        return Z


class PI(Net):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PI, self).__init__(state_dim, hidden_dim, action_dim)

    def action(self, x):
        logits = self.forward(x)
        slogits = F.softmax(logits, dim=-1)
        A = Categorical(slogits)
        act = A.sample()
        log_act = A.log_prob(act)
        return act.item(), log_act.item()

    def log_prob(self, st, at):
        logits = self.forward(st)
        slogits = F.softmax(logits, dim=-1)
        A = Categorical(slogits)
        log_act = A.log_prob(at)
        return log_act


envNonGUI = gym.make("CartPole-v1")
envGUI = gym.make("CartPole-v1", render_mode="human")

lr = 0.005  #  learning rate
bs = 64  # batch size
eps = 0.2  # for clipping threshold
gamma = 0.9  # discount factor
an = 2  # number of actions
sf = 4  # number of state features
epch = 100
I = 32  # iteration count
M = 32  # episode count
T = 2000  # max timesteps


policy = PI(sf, 64, an)
p_optim = optim.Adam(policy.parameters(), lr)

value = Net(sf, 64, 1)
v_optim = optim.Adam(value.parameters(), lr)


for k in range(I):
    print("Iteration", k)
    env = envNonGUI
    if k % 4 == 0:
        env = envGUI

    # Step 1: Collect step of trajectories by running policy
    D_st = []
    D_at = []
    D_log_prob = []
    D_rt = []

    for e in range(3):
        ep_st = []  # episode states
        ep_at = []  # episode actions
        ep_log_prob = []  # episode action log probabilities
        ep_rt = []  # episode rewards

        st, _ = env.reset()
        st = torch.from_numpy(st)
        for t in range(T):
            with torch.no_grad():
                at, log_at = policy.action(st)
            st1, rt, ter, _, _ = env.step(at)
            st1 = torch.from_numpy(st1)

            ep_st.append(st)
            ep_at.append(at)
            ep_log_prob.append(log_at)
            ep_rt.append(rt)
            if ter:
                break
            else:
                st = st1

        # Step 2+3: Compute rewards to go and advantage estimates
        G = 0
        for t in range(len(ep_rt) - 1, -1, -1):
            # Compute rewards to go
            ep_rt[t] += G
            G = ep_rt[t] * gamma

        D_st.extend(ep_st)
        D_at.extend(ep_at)
        D_log_prob.extend(ep_log_prob)
        D_rt.extend(ep_rt)

    D_st = torch.stack(D_st)
    D_at = torch.Tensor(D_at)
    D_log_prob = torch.Tensor(D_log_prob)
    D_rt = torch.Tensor(D_rt)
    D_adv = D_rt - value(D_st)
