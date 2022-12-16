import numpy as np


class Agent:
    def __init__(self, env, max_episode_length):
        self.policy = np.array([])
        self.episode = np.array([0] * max_episode_length)
        self.Q = np.array([])
        self.changes = np.array([])
        self.env = env


    def learn(self):
        self.episode