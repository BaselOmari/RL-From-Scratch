import gymnasium as gym
import numpy as np
import random
from math import inf


def Q(state, action, W, state_dim):
    return np.dot(state, W[action * state_dim : (action + 1) * state_dim])


def max_Q(state, W, action_count, state_dim):
    max_val = 0
    max_a = 0
    for a in range(action_count):
        val = Q(state, a, W, state_dim)
        if val > max_val:
            max_val = val
            max_a = a
    return max_val, max_a


env = gym.make("CartPole-v1")

lr = 0.1  # learning rate
bs = 32  # batch size
eps = 0.2  # for epsilon-greedy selection
gamma = 0.9  # discount factor
an = 2  # number of actions
sf = 4  # number of state features
M = 1000  # episode count
T = 20000  # max timesteps


D = []
W = np.random.normal(0, 0.1, (sf * an,))  # State features stacked twice for two actions

for episode in range(M):
    st, _ = env.reset()
    for t in range(T):
        # Step 1: Select action at
        p = random.random()
        at = 0
        if p < eps:
            at = random.randint(0, 1)
        else:
            _, at = max_Q(st, W, an, sf)

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
                qm, _ = max_Q(s1, W, an, sf)
                y += gamma * qm

            # Perform gradient descent step
            dW = -2 * (y - Q(s, a, W, sf)) * s
            W[a * sf : (a + 1) * sf] -= lr * dW

        # Step 6: Terminate if necessary, other set next state
        if ter:
            break
        else:
            st = st1

    print(f"Episode {episode} complete")

print(f"Weights {W}")
