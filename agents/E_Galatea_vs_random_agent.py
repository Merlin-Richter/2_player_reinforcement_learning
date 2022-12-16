import environments.Galatea as Galatea
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import sys

'''
This is an implementation of the same self play algorithm that was used before on TicTacToe.
Here however, this algorithm gets to its limits. 
It is too many states and too many actions to record all State-Action-values and update them enough to get good results.

This script is just used to show that this approach doesn't work anymore. In the next script we will explore new options
to use Reinforcement learning. Try running this script and try to see the improvement over time have diminishing returns

The chart, which is being updated in real time, shows the average performance against the random policy over time in 
wining percentage.
'''

Env = Galatea.Environment()

# Q_values is a dictionary
Q_values = {}

epsilon = 0.5

# The step size alpha
alpha = 0.1

# amount of time the program will run in seconds
practice_time = 10 * 60  # 10 minutes

'''You may play around with the values above this comment but not the ones below'''

average_rewards = 0
total_games = 0

start_time = time.time()

game = -1

Average_performance = 0.5

Average_performance_history = []


def evaluate():
    global Average_performance_history
    global Average_performance
    wins = 0
    total_positions = 0
    known_positions = [0, 0]
    for x in range(1000):
        while not Env.done:
            total_positions += 1
            if Env.hash() in Q_values:
                Action_index = np.argmax(Q_values[Env.hash()])
                Action = Env.get_actions()[Action_index]
                Env.step(Action)
                known_positions[0] += 2
            else:
                Env.step(random.choice(Env.get_actions()))
            if Env.done:
                break
            Env.step(random.choice(Env.get_actions()))
        wins += (Env.reward == 1)
        Env.reset()

        while not Env.done:
            total_positions += 1
            Env.step(random.choice(Env.get_actions()))
            if Env.done:
                break
            if Env.hash() in Q_values:
                known_positions[1] += 2
                Action_index = np.argmax(Q_values[Env.hash()])
                Action = Env.get_actions()[Action_index]
                Env.step(Action)
            else:
                Env.step(random.choice(Env.get_actions()))

        wins += (Env.reward == 0)
        Env.reset()

    Average_performance += 0.25 * (wins/2000 - Average_performance)
    Average_performance_history.append(100*Average_performance)

    print("\nThe agent, when performing with epsilon = 0, wins " + str(
        wins / 20) + "% of its games against a random agent.\nThe Agents knows " + str(round(100*known_positions[0]/total_positions, 3)) + "% of encountered positions when playing white and " + str(round(100*known_positions[1]/total_positions, 3)) + "% when playing black")




while time.time() - start_time < practice_time:
    game += 1
    state_history = []
    action_history = []

    while not Env.done:
        if Env.hash() not in Q_values:
            # whenever a new state is encountered, a new entry is made with an array for the value of each possible
            #   action in said state. It is initiated with a +0.4 to encourage initial exploration.
            # This trick is called "optimistic initial values"(Chapter 2.6 in "Reinforcement learning, An Introduction")
            Q_values[Env.hash()] = [0.4 for x in range(len(Env.get_actions()))]

        Action_index = -1

        if np.random.random() < epsilon:
            Action_index = np.random.choice(range(len(Env.get_actions())))
        else:
            Action_index = np.argmax(Q_values[Env.hash()])

        Action = Env.get_actions()[Action_index]

        state_history.append(Env.hash())
        action_history.append(Action_index)

        Env.step(Action)
        if Env.done:
            break
        if Env.hash() not in Q_values:
            # whenever a new state is encountered, a new entry is made with an array for the value of each possible
            #   action in said state. It is initiated with a +0.4 to encourage initial exploration.
            # This trick is called "optimistic initial values"(Chapter 2.6 in "Reinforcement learning, An Introduction")
            Q_values[Env.hash()] = [-0.4 for x in range(len(Env.get_actions()))]

        Action_index = -1

        if np.random.random() < epsilon:
            Action_index = np.random.choice(range(len(Env.get_actions())))
        else:
            Action_index = np.argmin(Q_values[Env.hash()])

        Action = Env.get_actions()[Action_index]

        state_history.append(Env.hash())
        action_history.append(Action_index)

        Env.step(Action)

    # The game is now over and we know the reward which means we can now update the Q_values
    for x in range(len(action_history)):

        Q_values[state_history[x]][action_history[x]] += alpha * (Env.reward - Q_values[state_history[x]][action_history[x]])

    average_rewards += 0.0001 * (Env.reward - average_rewards)
    total_games += 1
    if game % 5000 == 0 and game != 0:
        epsilon = 0.45 * (1 - (time.time() - start_time)/practice_time) + 0.05
        print("Epsilon: " + str(round(epsilon, 3)) + "  || Progress: " + str(round(100*(time.time() - start_time)/practice_time, 3)) + "%")
        evaluate()
        print("The length of the Q_values dictionary is already " + str(len(Q_values)) + " Entries. (" + str(round(sys.getsizeof(Q_values)/1000000, 1)) + "MB in memory)\n")
        plt.plot(Average_performance_history)
        plt.pause(0.1)

    Env.reset()

np.save("Galatea_agent_Q_values", Q_values)
np.save("Galatea_Average_performance_history", Average_performance_history)

plt.plot(Average_performance_history)
plt.show()


