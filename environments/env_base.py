import numpy as np


class Env_base:
    def __init__(self):
        self.state = None
        self.reward = 0
        self.done = None
        self.new_state = None
        self.info = None

    def step(self, action):
        self.state = np.copy(self.new_state)
        self.reward = 0
        self.done = False
        self.info = None
        self.inner_workings(action)
        return self.new_state, self.reward, self.done

    def get_actions(self):
        return None

    def step_with_state(self, state, action, counter_action = None):
        self.state = np.copy(state)
        self.new_state = np.copy(state)
        self.reward = 0
        self.done = False
        self.info = None
        self.inner_workings(action)
        return self.state, self.reward, self.new_state, self.done, self.info

    def inner_workings(self, action):
        pass

    def render(self):
        print(self.state)

    def random_action(self):
        self.step(np.random.choice(self.get_actions()))

