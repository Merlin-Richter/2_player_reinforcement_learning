from environments.env_base import Env_base
import random
import numpy as np

'''
The Multi-armed Bandit is the simplest of the Reinforcement Learning problems.
It refers to a slot machine with different Levers which each have a different reward distribution
for example 'Lever 1' may have a reward distribution of (25% -> +2 | 25% -> -3 | 50% -> -4)
    while   'Lever 2' may have a reward distribution of (25% -> -2 | 25% -> +3 | 50% -> +4)
One of the levers has a better average rewards than the other (Lever 2) however it is not
sufficient to test each Lever once and then choose the one where the agent observed the 
highest reward

It is considered the simplest of the RL problems because it doesn't have a "state"
The agent can consider the finite Actions in a vacuum. No past actions or 'board positions'
have an effect on the reward distribution of the Actions 

In Practice it is most common to define the reward distribution as a normal distribution
around a median which is different for each "arm".

'''


class Environment(Env_base):

    def __init__(self, amount_of_arms):
        super().__init__()
        # How many different Levers with different reward distributions exist
        self.amount_of_arms = amount_of_arms
        # Generates an array of medians for the reward distribution of the different arms/actions randomly between
        #  -1 and +1
        self.arms = [2 * (random.random()-0.5) for _ in range(self.amount_of_arms)]

    def inner_workings(self, action):
        # Defines the reward as a normal distribution around the underlying value of the arm/action
        self.reward = np.random.normal(self.arms[action])

    def get_actions(self):
        return [x for x in range(self.amount_of_arms)]
