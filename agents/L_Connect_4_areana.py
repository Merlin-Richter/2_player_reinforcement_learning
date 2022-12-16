import environments.connect_4 as Connect_4
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


'''

'''
class Conv_Net(nn.Module):

    def __init__(self):
        super(Conv_Net, self).__init__()
        self.Conv_layer = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))
        self.fully_connected1 = nn.Linear(in_features=12*12, out_features=128)
        self.fully_connected2 = nn.Linear(in_features=128, out_features=128)
        self.fully_connected3 = nn.Linear(in_features=128, out_features=7)
        self.Relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Conv_layer(x)
        x = x.flatten(start_dim=1)
        x = self.Relu(x)
        x = self.fully_connected1(x)
        x = self.Relu(x)
        x = self.fully_connected2(x)
        x = self.Relu(x)
        x = self.fully_connected3(x)
        x = self.sigmoid(x)

        return x
Env = Connect_4.Environment()

# The state how many steps in the future is used to update the value of the current state. λ=0 means the next state




def depth_search(depth, first=True, alpha=-10, beta=10):
    save_state = np.copy(Env.new_state)
    player = Env.whos_turn
    best_value = 100 * Env.whos_turn - 50
    best_action = None
    list_actions = Env.get_actions()
    if first:
        print("Thinking...\n[", end='')
        bars = 0
        len_actions = len(list_actions)
    for z, action in enumerate(list_actions):
        Env.step(action)
        if Env.done:
            value = torch.tensor(Env.reward)
        elif depth > 1:
            value = depth_search(depth-1, False, alpha, beta)[1]
        else:
            prediction = Conv_net(torch.from_numpy(Env.two_dimensional_representation().reshape((1, 3, 7, 6))))
            if player == 0:
                value = torch.min(prediction)
            else:
                value = torch.max(prediction)

        if player == 0 and value > best_value:
            best_action = action
            best_value = value
            alpha = max(alpha, value)
        elif value < best_value and player == 1:
            best_action = action
            best_value = value
            beta = min(beta, value)
        if beta <= alpha:
            break
        Env.restore_state(np.copy(save_state))
        while first and (z + 1) / len_actions > bars / 40:
            bars += 1
            print("-", end="")
    if first:
        print("]")
    return best_action, best_value


def human_move():
    Inp = None
    while Inp not in Env.get_actions():
        try:
            Inp = int(input("where do you want to play?  "))-1
            if not Inp in Env.get_actions():
                print("incorrect input")
        except:
            print("incorrect input")
    Env.step(Inp)

if __name__ == "__main__":

    Conv_net = Conv_Net()
    Conv_net.load_state_dict(torch.load("../models/Connect_4/RL_model_1.pth"))
    Conv_net.to(torch.double)
    wins = 0
    n = 0
    Human = False
    self_play = False
    for game in range(5000):
        while not Env.done:
            observation = torch.from_numpy(Env.two_dimensional_representation())

            # result = depth_search(6)
            # best_action = result[0]
            # print(round(100*result[1].item(), 2))

            if Human:
                print(Conv_net(observation.unsqueeze(0)))
                if self_play:
                    input("waiting...")
            best_action = torch.argmax(Conv_net(observation.unsqueeze(0)))

            Env.step(best_action)
            observation = torch.from_numpy(Env.two_dimensional_representation())
            if Human:
                Env.render()

            if Env.done:
                wins += Env.reward
                break

            if Human:
                if self_play:
                    print(Conv_net(observation.unsqueeze(0)))
                    input("waiting...")
                    best_action = torch.argmin(Conv_net(observation.unsqueeze(0)))
                    Env.step(best_action)
                else:
                    human_move()
                Env.render()
            else:
                if self_play:
                    best_action = torch.argmin(Conv_net(observation.unsqueeze(0)))
                    Env.step(best_action)
                else:
                    Env.step(random.choice(Env.get_actions()))

        Env.reset()
        print(round(100*wins/(game+1), 2))

