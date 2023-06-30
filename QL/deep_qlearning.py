import gymnasium as gym
import numpy as np
import random
import torch
from torch import nn, optim


class DQN(nn.Module):
    def __init__(self, state_dim, action_size):
        super(DQN, self).__init__()

        hidden_size = 32
        self.inp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        h = self.inp(x)
        Q = self.out(h)
        return Q


def max_Q(state, net):
    with torch.no_grad():
        Qs = net(state)
        max_a = int(np.argmax(Qs))
        max_val = Qs[max_a]
    return max_val, max_a


env = gym.make("CartPole-v1", render_mode="human")

lr = 0.001  # learning rate
bs = 64  # batch size
eps = 0.2  # for epsilon-greedy selection
gamma = 0.9  # discount factor
an = 2  # number of actions
sf = 4  # number of state features
M = 1000  # episode count
T = 20000  # max timesteps


D = []
net = DQN(sf, an)
optimizer = optim.Adam(net.parameters(), lr)


for episode in range(M):
    st, _ = env.reset()
    st = torch.from_numpy(st)

    for t in range(T):
        # Step 1: Select action at
        p = random.random()
        at = 0
        if p < eps:
            at = random.randint(0, 1)
        else:
            _, at = max_Q(st, net)

        # Step 2: Execute action at in emulator and observe reward rt and state st
        st1, rt, ter, _, _ = env.step(at)
        st1 = torch.from_numpy(st1)

        # Step 3: Store transition in D
        D.append((st, at, rt, st1, ter))

        # Step 4: Sample random minibatch of transitions from D
        B = random.choices(D, k=min(bs, len(D)))

        # Step 5: Gradient descent
        for s, a, r, s1, tm in B:
            optimizer.zero_grad()
            # Set y
            y = r
            if not tm:
                with torch.no_grad():
                    qm, _ = max_Q(s1, net)
                    y += gamma * qm

            # Perform gradient descent step
            loss = (y - net(s)[a]) ** 2
            loss.backward()
            optimizer.step()

        # Step 6: Terminate if necessary, other set next state
        if ter:
            break
        else:
            st = st1

    print(f"Episode {episode} complete")
