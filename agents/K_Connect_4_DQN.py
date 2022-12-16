import environments.connect_4 as Connect_4
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import ZZ_tool_replay_memory as RM
import torch.nn.functional as F
import time

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


if __name__ == "__main__":

    model = Conv_Net()
    # model.load_state_dict(torch.load("../models/Connect_4/RL_model_0.pth"))
    target_model = copy.deepcopy(model)
    runs = 100_001
    env = Connect_4.Environment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = 0.7
    maximizing_buffer = RM.ReplayBuffer(2500)
    minimizing_buffer = RM.ReplayBuffer(2500)
    batch_size = 64
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    gamma = 1.
    total_reward = 0
    moving_average = 0.5
    average_q = 0
    average_loss = 0

    model.to(device)
    target_model.to(device)
    model.to(torch.double)
    target_model.to(torch.double)

    step = 0
    last_step = 0
    for run in range(runs):
        if run % 200 == 1 and run != 1:
            epsilon = epsilon * 0.95 + 0.005
            target_model = copy.deepcopy(model)
            target_model.to(device)
            target_model.to(torch.double)
            print("###COPIED###  epsilion: " + str(round(epsilon, 3)))
        total_reward = 0
        env.reset()
        observation = env.two_dimensional_representation()
        done = False
        while not done:
            step += 1
            player = env.whos_turn
            if torch.rand(1) < epsilon:
                # with probability of epsilon take a random action
                action = torch.randint(0, 7, (1,))[0]
            else:
                # otherwise act according to the pertained policy model.
                # Maybe check if my math here for indexing us correct
                prediction = model(torch.from_numpy(observation).unsqueeze(0).to(device)).detach()
                # print(prediction)
                if env.whos_turn == 0:
                    action = torch.argmax(prediction)
                else:
                    action = torch.argmin(prediction)

            # ACT
            _, reward, done = env.step(action)
            new_observation = env.two_dimensional_representation()
            total_reward += reward
            # make a new entry in the Replay Memory
            if player == 0:
                minimizing_buffer.add(obs_t=observation, action=action.item(), reward=reward, obs_tp1=new_observation, done=int(done))
            elif player == 1:
                maximizing_buffer.add(obs_t=observation, action=action.item(), reward=reward, obs_tp1=new_observation, done=int(done))
            observation = new_observation

            if len(maximizing_buffer) > 2 * batch_size:
                # 1. Sample transitions from replay_buffer
                if random.random() < 0.5:
                    replay_buffer_tuple = maximizing_buffer.sample(batch_size=batch_size)
                    maximizing = True
                else:
                    replay_buffer_tuple = minimizing_buffer.sample(batch_size=batch_size)
                    maximizing = False

                (current_state_batch, action_batch, reward_batch, next_state_batch, done_batch) = tuple(
                    torch.from_numpy(x).to(device) for x in replay_buffer_tuple)

                # 2. Compute Q(s_t, a)
                predictions = model(current_state_batch)
                predictions = predictions.gather(1, action_batch.unsqueeze(1).to(device)).squeeze()

                # 3. Compute \max_a Q(s_{t+1}, a) for all next states
                hold = target_model(next_state_batch)
                if maximizing:
                    next_q_values = torch.max(hold, dim=1).values.to(device).squeeze()
                else:
                    next_q_values = torch.min(hold, dim=1).values.to(device).squeeze()
                average_q += torch.sum(next_q_values).item()

                # 4) Mask next state values where episodes have terminated
                mask = (1 - done_batch).float().to(device)
                next_q_values = mask * next_q_values

                # 5) Compute the target
                target = reward_batch.to(device) + gamma * next_q_values

                # 6) Compute the loss -> L2 loss
                loss = F.mse_loss(predictions, target)
                average_loss += loss.item()

                # 7) Calculate the gradients
                optimizer.zero_grad()
                loss.backward()

                # 8) Clip the gradients
                for param in model.parameters():
                    param.grad.data.clamp_(-1, 1)

                # 9) Optimize the model
                optimizer.step()

        moving_average += 0.005 * (total_reward - moving_average)
        if run % 100 == 0:

            print("Run: " + str(run) +
                  "     Loss: " + str(round(average_loss / (step - last_step), 3)) +
                  "     Average q: " + str(round((average_q / (step - last_step)) / batch_size, 3)) +
                  "     Average Reward: " + str(round(moving_average, 3)))
            last_step = step
            average_q = 0
            average_loss = 0
        if run % 5000 == 0 and run != 0:
            # save model every 500 runs
            torch.save(model.state_dict(), f"../models/Connect_4/RL_model_{int(run/5000)}.pth")
            for _ in range(5):
                print("######################################SAVED######################################")
