import numpy as np
import environments.env_base as env
from scipy.signal import convolve2d
import random

class Environment(env.Env_base):

    def __init__(self):
        super().__init__()
        self.state = np.zeros((2, 7, 6))
        self.new_state = np.zeros((2, 7, 6))
        self.board_height = [0 for x in range(7)]
        self.whos_turn = 0
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        self.detection_kernels = [horizontal_kernel,vertical_kernel, diag1_kernel, diag2_kernel]

    def reset(self):
        self.state = np.zeros((2, 7, 6))
        self.new_state = np.zeros((2, 7, 6))
        self.board_height = [0, 0, 0, 0, 0, 0, 0]
        self.whos_turn = 0
        self.done = False
        self.reward = 0

    def check_win(self, board):
        for kernel in self.detection_kernels:
            if (convolve2d(board, kernel, mode="valid") == 4).any():
                return True
        return False

    def inner_workings(self, action, counter_action=None):
        if self.board_height[action] >= 6:
            self.reward = self.whos_turn
            self.done = True
            return
        self.new_state[self.whos_turn, action, self.board_height[action]] = 1
        self.board_height[action] += 1
        self.reward = 0
        if self.whos_turn == 0:
            self.whos_turn = 1
        else:
            self.whos_turn = 0
        if self.check_win(self.new_state[0]):
            self.done = True
            self.reward = 1
        elif self.check_win(self.new_state[1]):
            self.done = True
            self.reward = 0
        elif np.sum(self.new_state[1]) == 21:
            self.done = True
            self.reward = 0.5

    def get_actions(self):
        list = np.where(np.array(self.board_height) < 6)
        return list[0]

    def human_move(self):
        done = False
        while not done:
            line = input("What line do you want to play in? ")
            try:
                _line = int(line)-1
            except:
                print("That is not possible")
                continue
            if _line not in self.get_actions():
                print("That is not possible")
                continue
            self.step(_line)
            done = True

    def two_dimensional_representation(self):
        if self.whos_turn == 0:
            return np.append(self.new_state, np.zeros((1, 7, 6)), axis=0)
        else:
            return np.append(self.new_state, np.ones((1, 7, 6)), axis=0)


    def restore_state(self, state):
        state = np.copy(np.array(state))
        self.new_state = state
        self.state = state
        for x in range(7):
            # TODO: try except: pass ?
            try:
                height = np.where(self.new_state[0, x] == 1)[0][-1] + 1
            except:
                height = 0
            try:
                height = max(height, np.where(self.new_state[1, x] == 1)[0][-1] + 1)
            except:
                pass
            self.board_height[x] = height

        if np.sum(self.new_state[0]) > np.sum(self.new_state[1]):
            self.whos_turn = 1
        else:
            self.whos_turn = 0


    def render(self):
        for x in range(6):
            string = ''
            for a in range(7):
                if self.new_state[0, a, 5-x] == 1:
                    string += 'O '
                elif self.new_state[1, a, 5-x] == 1:
                    string += 'X '
                else:
                    string += '  '
            print(string)
        print("1 2 3 4 5 6 7")


if __name__ == "__main__":
    Env = Environment()
    Env.step(random.choice(Env.get_actions()))
    Env.render()
    Env.step(random.choice(Env.get_actions()))
    Env.step(random.choice(Env.get_actions()))
    print(Env.two_dimensional_representation())
    Env.step(random.choice(Env.get_actions()))
    print(Env.two_dimensional_representation())
    save_state = np.copy(Env.new_state)
    for x in range(4002):
        Env.step(np.random.choice(Env.get_actions()))
        Env.render()
        print(Env.get_actions())
        print(Env.done)
        if Env.done:
            Env.reset()



