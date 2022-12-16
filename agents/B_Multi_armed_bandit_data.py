from environments.multi_armed_bandits import Environment
import numpy as np
import matplotlib.pyplot as plt

'''
Run this script to see an average learning and performance progression of different epsilons. It may take a few minutes.

The graph is averaged over 500 runs for each epsilon

Epsilons and why they perform good or bad:

Epsilon = 0       performs quite bad because it will get stuck at a suboptimal strategy espacialy with more "arms"
Epsilon = 0.001   performs great, reaches almost optimal (0.1% random) after a LOT of iterations
Epsilon = 0.05    performs good, reaches quite optimal (5% random) after a few iterations
Epsilon = 0.3     performs poorly because it will always take 30% of its actions Randomly

A way to partialy negate this problem of having to choose an epsilon is to start out with a big epsilon and over time 
reduce epsilon until it is 0
'''

amount_of_arms = 100

MAB_Environment = Environment(amount_of_arms)

# reward_history records the rewards for progress updates
reward_history = [[0] * 700, [0] * 700, [0] * 700, [0] * 700]

runs = 500

# epsilon is a hyperparameter which determines the probability of a random action being taken
epsilons = [0, 0.001, 0.05, 0.3]

for i, epsilon in enumerate(epsilons):

    for run in range(runs):

        if run % 50 == 0:
            print(str(100 * (i*runs+run)/(runs*len(epsilons))) + "%")

        # Q_values, N and the environment get reset every run
        Q_values = [0 for _ in range(amount_of_arms)]
        N = [0 for _ in range(amount_of_arms)]
        MAB_Environment = Environment(amount_of_arms)

        hold = []

        for _ in range(700):

            if np.random.random() < epsilon:
                # With a probability of epsilon a random Action is chosen
                Action = np.random.choice(range(amount_of_arms))
            else:
                # With a probability of 1 - epsilon the Action which is believed to be best is chosen
                Action = np.argmax(Q_values)

            # Agent interacts with the Environment by taking an action and receiving feedback
            (state, reward, new_state, done) = MAB_Environment.step(Action)

            # Updating the Q_values to account for the newly observed reward
            N[Action] += 1
            Q_values[Action] += 1/N[Action] * (reward - Q_values[Action])

            hold.append(reward)
            if len(hold) > 9:
                hold.pop(0)
                reward_history[i][_] += np.average(hold)


reward_history = np.array(reward_history)/runs
for i in range(4):
    plt.plot(reward_history[i])
plt.legend([("epsilon = " + str(e)) for e in epsilons])
plt.title("Average reward")
plt.show()
