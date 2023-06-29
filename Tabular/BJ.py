import gymnasium as gym
import math
import numpy as np
import random


class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n

    def get_action(self, state, Q, eps):
        assert 0 <= eps <= 1
        p = random.random()
        if p > eps:
            max_val = max(Q[state])
            all_idx = [i for i, x in enumerate(Q[state]) if math.isclose(x, max_val)]
            return random.choice(all_idx)
        else:
            return random.randint(0, self.action_size - 1)


class Environment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.state_dims = [s.n for s in self.env.observation_space]
        self.Q = None
        self.N = None
        self.reset_values()

    def reset_values(self):
        self.Q = np.zeros((*self.state_dims, self.action_size))
        self.G = np.zeros(self.Q.shape)
        self.N = np.zeros(self.Q.shape)


def expand_state(state):
    row = state // 12
    col = state % 12
    return row, col
