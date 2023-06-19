# %%
import gymnasium as gym
import numpy as np
import sys

from math import log10
from BJ import Environment, Agent, expand_state


# %%
def sample_episode(space, agent, eps):
    episode = []
    state, _ = space.env.reset()
    terminate = False
    i = 0
    while not terminate:
        curr, dealer, ace = state
        action = agent.get_action(state, space.Q, eps)
        space.N[curr, dealer, ace, action] += 1
        nextState, reward, terminate, _, _ = space.env.step(action)
        episode.append((state, action, reward))
        state = nextState
        i += 1
        if i % 10000 == 0:
            print("step", i)
    return episode


# %%
def monte_carlo_update_value(space, episode, gamma):
    states, actions, rewards = zip(*(episode))

    discounts = np.array([gamma**i for i in range(len(rewards))])
    total_reward = sum(rewards * discounts)

    for i in range(len(episode)):
        curr, dealer, ace = states[i]
        space.N[curr, dealer, ace, actions[i]] += 1.0
        space.Q[curr, dealer, ace, actions[i]] += (
            total_reward - space.Q[curr, dealer, ace, actions[i]]
        ) / space.N[curr, dealer, ace, actions[i]]
        total_reward -= rewards[i]
        total_reward /= gamma


# %%
space = Environment("Blackjack-v1")
agent = Agent(space.env)

gamma = 0.5
num_iterations = 100000

# %%
for iter in range(10, num_iterations + 1):
    print("iteration", iter)
    episode = sample_episode(space, agent, eps=1 / log10(iter))
    monte_carlo_update_value(space, episode, gamma)

# %%
np.set_printoptions(threshold=sys.maxsize)
print(space.Q)

# %%
print(space.Q[20])

# %%
