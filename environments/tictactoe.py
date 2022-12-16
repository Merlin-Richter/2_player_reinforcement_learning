from environments.env_base import Env_base
import numpy as np
import math

'''





'''


class Environment(Env_base):

    def __init__(self):
        super().__init__()
        self.state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.new_state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.reward = 0
        self.who_turn = 1
        self.win_player = 0
        self.dict = {-1:"X", 0:"-", 1:"O"}

    def reset(self):
        self.__init__()

    def inner_workings(self, action):
        if self.state[math.floor(action/3)][action%3] != 0:
            print('k')
            assert IndexError
        else:
            self.new_state[math.floor(action/3)][action % 3] = self.who_turn
            self.who_turn *= -1
            if self.detect_win():
                return

    def get_actions(self):
        return np.where(self.new_state == 0)[0]*3 + np.where(self.new_state == 0)[1]

    def hash(self, state = None):
        if state == None:
            state = self.new_state
        return sum((state.flatten()+1) * np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561]))

    def unhash(self, hash_):
        a = []
        for x in range(9):
            a.append(math.floor(hash_ / math.pow(3, 8-x)))
            hash_ = hash_ % math.pow(3, 8-x)
        return np.flip(a).reshape((3, 3)) - 1

    def detect_win(self):
        for row in range(3):

            uniques = np.unique(self.new_state[row])
            if len(uniques) == 1 and uniques[0] != 0:
                self.win_player = uniques[0]
                break

            uniques = np.unique(self.new_state[0:3, row])
            if len(uniques) == 1 and uniques[0] != 0:
                self.win_player = uniques[0]
                break
            elif row == 2:
                self.win_player = 0
                break

        if (self.new_state[0, 0] == self.new_state[1, 1] == self.new_state[2, 2]) and self.new_state[1,1] != 0:
            self.win_player = self.new_state[1, 1]

        if (self.new_state[0, 2] == self.new_state[1, 1] == self.new_state[2, 0]) and self.new_state[1,1] != 0:
            self.win_player = self.new_state[1, 1]

        if self.win_player != 0:
            # self.new_state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.reward = self.win_player
            self.done = True
            # self.who_turn = 1
            return True

        if 0 not in np.unique(self.new_state):
            # self.new_state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.reward = 0.5
            self.done = True
            return True

        else:
            self.reward = self.win_player
            self.done = False
            return False

    def render(self):
        for x in range(3):
            print(f"{self.dict[self.new_state[x][0]]} {self.dict[self.new_state[x][1]]} {self.dict[self.new_state[x][2]]}")

    def render_a_state(self, state):

        for x in range(3):
            print(f"{self.dict[state[3*x+0]]} {self.dict[state[3*x+1]]} {self.dict[state[3*x+2]]}")


if __name__ == "__main__":
    T = Env()
    while True:
        x = int(input("what to play"))
        T.step(np.random.choice(T.get_actions()))
        print(T.get_actions())
        print(T.hash())
        print(T.unhash(T.hash()))
        print(T.new_state)
        T.render()

