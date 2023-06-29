import gymnasium as gym
import numpy as np
import random
from math import inf


def max_q(state, W, action_count, state_dim):
    max_val = 0
    max_a = 0
    for a in range(action_count):
        val = np.dot(state, W[a * state_dim : (a + 1) * state_dim])
        if val > max_val:
            max_val = val
            max_a = a
    return max_val, max_a


env = gym.make("CartPole-v1")

bs = 32  # batch size
eps = 0.2  # for epsilon-greedy selection
gamma = 0.9  # discount factor
an = 2  # number of actions
sf = 4  # number of state features
M = 1000  # episode count
T = 1000  # max timesteps


D = []
W = np.random.normal(0, 0.1, (sf * an,))  # State features stacked twice for two actions

for episode in range(5):
    st, _ = env.reset()
    for t in range(1000):
        # Step 1: Select action at
        p = random.random()
        at = 0
        if p < eps:
            at = random.randint(0, 1)
        else:
            _, at = max_q(st, W, an, sf)

        # Step 2: Execute action at in emulator and observe reward rt and state st
        st1, rt, ter, _, _ = env.step(at)

        # Step 3: Store transition in D
        D.append((st, at, rt, st1, ter))

        # Step 4: Sample random minibatch of transitions from D
        B = random.choices(D, k=min(bs, len(D)))

        # Step 5: Gradient descent
        for s, a, r, s1, tm in B:
            # Set y
            y = r
            if not tm:
                qm, _ = max_q(s1, W, an, sf)
                y += gamma * qm

            # TODO: Perform gradient descent step

        # Step 6: Terminate if necessary, other set next state
        if ter:
            break
        else:
            st = st1
