import environments.connect_4 as Connect_4
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import replay_memory as RM


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
        # x = self.sigmoid(x)

        return x


def train(replay_memory, model_, target_model_, other_target_model, optimizer_, maximizing):
    # train based on Replay Memory
    states, actions, next_states, rewards, dones = replay_memory.draw_minibatch(batch_size)
    next_states = next_states.to(device)

    next_predictions = other_target_model(next_states).detach()
    if maximizing:
        next_actions = torch.argmin(next_predictions, dim=1)  # not a mistake!!
    else:
        next_actions = torch.argmax(next_predictions, dim=1)

    next_predictions = target_model_(next_states).detach()

    next_values = torch.stack([next_predictions[i, next_actions[i]] for i in range(batch_size)])

    Y_train = (rewards.to(device) + gamma * next_values.to(device) * (1 - dones)).to(device)

    outputs = model_(states.to(device))
    # only calculate gradient of the outputs of the action taken
    outputs = torch.stack([outputs[i, actions[i]] for i in range(batch_size)])
    # calculate loss
    loss = loss_fn(outputs, Y_train.detach())
    returny = loss.item()
    optimizer_.zero_grad()
    # backwards pass
    loss.backward()
    # update weights
    optimizer_.step()
    return returny


if __name__ == "__main__":

    model = [Conv_Net(), Conv_Net()]
    target_model = [copy.deepcopy(model[0]), copy.deepcopy(model[1])]
    runs = 100_001
    env = Connect_4.Environment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = 0.7
    ReplayMemory = [RM.ReplayMemory(2000, (3, 7, 6)), RM.ReplayMemory(2000, (3, 7, 6))]
    batch_size = 128
    loss_fn = torch.nn.MSELoss()
    optimizer = [torch.optim.SGD(model[0].parameters(), lr=0.05, momentum=0.0), torch.optim.SGD(model[1].parameters(), lr=0.05, momentum=0.0)]

    gamma = 0.99
    total_reward = 0
    moving_average = 0.5

    for id in range(2):
        model[id].to(device)
        model[id].to(torch.float)
        target_model[id].to(device)
        target_model[id].to(torch.float)

    for run in range(runs):
        if run % 150 == 0 and run != 0:
            epsilon = epsilon * 0.95 + 0.005
            for id in range(2):
                target_model[id] = copy.deepcopy(model[id])
                target_model[id].to(device)
                target_model[id].to(torch.float)
            print("###COPIED###  epsilion: " + str(round(epsilon, 3)))
        total_reward = 0
        lossy = 0
        env.reset()
        observation = torch.from_numpy(env.two_dimensional_representation())
        done = False
        step = 0
        while not done:
            step += 1
            player = env.whos_turn
            if torch.rand(1) < epsilon:
                # with probability of epsilon take a random action
                action = torch.randint(0, 7, (1,))[0]
            else:
                # otherwise act according to the pertained policy model.
                # Maybe check if my math here for indexing us correct
                id = random.randint(0, 1)
                prediction = model[id](observation.to(device).unsqueeze(0).to(torch.float)).detach()
                # print(prediction)
                if player == 0:
                    action = torch.argmax(prediction)
                else:
                    action = torch.argmin(prediction)

            # ACT
            _, reward, done = env.step(action)
            new_observation = torch.from_numpy(env.two_dimensional_representation())
            total_reward += reward
            # make a new entry in the Replay Memory
            ReplayMemory[player].new_entry(state=observation, action=action, reward=reward, next_state=new_observation,
                                           done=int(done))
            observation = new_observation

            if ReplayMemory[player].can_sample(batch_size):
                id = random.randint(0, 1)
                lossy += train(replay_memory=ReplayMemory[player],
                               model_=model[id],
                               target_model_=target_model[id],
                               other_target_model=target_model[(id+1) % 2],
                               optimizer_=optimizer[id],
                               maximizing=(player == 0))

        moving_average += 0.005 * (total_reward - moving_average)
        if run % 100 == 0:
            print("Run: " + str(run) + "        Loss: " + str(lossy / step) + "       Reward: " + str(
                total_reward) + "        Average Reward: " + str(round(100*moving_average, 2)) + "%")
        if run % 50000 == 0 and run != 0:
            # save model every 500 runs
            torch.save(model[0].state_dict(), f"RL_model_{int(run/5000)}_1.pth")
            torch.save(model[1].state_dict(), f"RL_model_{int(run/5000)}_2.pth")
            for _ in range(5):
                print("######################################SAVED######################################")

