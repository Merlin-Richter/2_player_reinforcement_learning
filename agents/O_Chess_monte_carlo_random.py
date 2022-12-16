import chess
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


'''

'''


Env = Connect_4.Environment()

# The state how many steps in the future is used to update the value of the current state. λ=0 means the next state
Lambda = 2

alpha = 0.0001

epsilon = 0.5




class Conv_Net(nn.Module):

    def __init__(self):
        super(Conv_Net, self).__init__()
        self.Conv_layer = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
        self.fully_connected = nn.Linear(in_features=12*12, out_features=1)
        self.Relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Conv_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Relu(x)
        x = self.fully_connected(x)
        x = self.sigmoid(x)

        return x


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
            value = Env.reward
        elif depth > 1:
            value = depth_search(depth-1, False, alpha, beta)[1]
        else:
            value = Conv_net(torch.from_numpy(Env.two_dimensional_representation().reshape((1, 3, 7, 6)))).item()

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


def generate_data_through_self_play(games):
    global Env, y, states, epsilon
    print("generating new data...")
    print("[", end="")
    bars = 0
    for game in range(games):
        state_indexes = []

        while not Env.done:
            if random.random() < epsilon:
                Env.step(random.choice(Env.get_actions()))
            else:
                prediction = depth_search(0, False)
                best_action = prediction[0]
                value = prediction[1]

                Env.step(best_action)

            states.append(Env.two_dimensional_representation())
            if Env.done:
                break

            if random.random() < epsilon:
                Env.step(random.choice(Env.get_actions()))
            else:
                prediction = depth_search(0, False)
                best_action = prediction[0]
                value = prediction[1]

                Env.step(best_action)

            states.append(Env.two_dimensional_representation())

        for _ in range(len(states) - len(y)):
            y.append(Env.reward)

        while (game + 1) / games > bars/40:
            bars += 1
            print("-", end="")
        Env.reset()
    print("]")


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == "__main__":

    print("\n[----------------------------------------]\n\n")

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    Conv_net = Conv_Net().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(Conv_net.parameters(), lr=alpha)

    Conv_net.double()

    iterations = 5
    epochs = 5
    batch_size = 32
    for iterr in range(iterations):
        states = []
        y = []

        generate_data_through_self_play(5_000)

        states = np.array(states[:(len(states) - (len(states) % 32))])  # to make states have a length of L * 32
        y = np.array(y[:(len(states) - (len(states) % 32))])

        states, y = unison_shuffled_copies(states, y)

        Data = []

        for x in range(int(len(y)/32)):
            Data.append((torch.from_numpy(np.array(states[32*x:32*(x+1)])).double(), torch.from_numpy(np.array(y[32*x:32*(x+1)]).reshape((32, 1))).double()))

        while input("Continue? (Y/N)") == "Y":
            for epoch in range(epochs):
                total_loss = 0
                for x_train, y_train in Data:

                    x_train = x_train.to(device)
                    y_train = y_train.to(device)

                    predictions = Conv_net(x_train)
                    loss = criterion(predictions, y_train)
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                print(total_loss)
        torch.save(Conv_net, "../models/Connect_4/monte_carlo_self_play_iteration_" + str(iterr))
    Env.reset()

    wins = 0
    n = 0
    for game in range(1000):
        while not Env.done:
            '''
            save_state = np.copy(Env.new_state)
            max_value = -100
            best_action = -1
            for action in Env.get_actions():
                Env.step(action)
                prediction = Conv_net(torch.from_numpy(Env.two_dimensional_representation().reshape((1, 3, 7, 6))))[0]
                if prediction > max_value:
                    best_action = action
                    max_value = prediction
                Env.restore_state(np.copy(save_state))
            '''
            Env.step(random.choice(Env.get_actions()))
            prediction = depth_search(0, False)
            best_action = prediction[0]
            value = prediction[1]

            Env.step(best_action)

        wins += 1 - Env.reward
        n += 1
        Env.reset()
        print(str(round(100*wins/n, 1)) + "%")
