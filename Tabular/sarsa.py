# %%
import gymnasium as gym
import numpy as np
import sys

from math import log10
from BJ import Environment, Agent, expand_state


# %%
def sarsa_episode_updates(space, agent, eps, gamma, lr):
    state, _ = space.env.reset()
    action = agent.get_action(state, space.Q, eps)
    terminate = False
    while not terminate:
        curr, dealer, ace = state
        nextState, reward, terminate, _, _ = space.env.step(action)
        nextCurr, nextDealer, nextAce = nextState
        nextAction = agent.get_action((nextCurr, nextDealer, nextAce), space.Q, eps)
        space.Q[curr, dealer, ace, action] += lr * (
            (
                reward
                + gamma * space.Q[nextCurr, nextDealer, nextAce, nextAction]
                - space.Q[curr, dealer, ace, action]
            )
        )
        state = nextState
        action = nextAction


# %%
space = Environment("Blackjack-v1")
agent = Agent(space.env)

eps = 0.5
lr = 0.5
gamma = 0.5
num_episodes = 100000

# %%
for i in range(10, num_episodes + 1):
    print("episode", i)
    sarsa_episode_updates(space, agent, eps, gamma, lr)

# %%
np.set_printoptions(threshold=sys.maxsize)
print(space.Q)

# %%
print(space.Q[20])
