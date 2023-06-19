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
def td_0_update_value(space, episode, gamma, lr):
    states, actions, rewards = zip(*(episode))

    for i in range(len(episode)):
        curr, dealer, ace = states[i]

        next_value = 0
        if i < len(episode) - 1:
            next_curr, next_dealer, next_ace = states[i + 1]
            next_value = space.Q[next_curr, next_dealer, next_ace, actions[i + 1]]

        space.Q[curr, dealer, ace, actions[i]] += lr * (
            (rewards[i] + gamma * next_value) - space.Q[curr, dealer, ace, actions[i]]
        )


# %%
space = Environment("Blackjack-v1")
agent = Agent(space.env)

lr = 0.5
gamma = 0.9
num_iterations = 100000

# %%
for iter in range(10, num_iterations + 1):
    print("iteration", iter)
    episode = sample_episode(space, agent, eps=1 / log10(iter))
    # td_0_update_value(space, episode, gamma, lr)
    monte_carlo_update_value(space, episode, gamma)

# %%
np.set_printoptions(threshold=sys.maxsize)
print(space.Q)

# %%
print(space.Q[20])

# %%
