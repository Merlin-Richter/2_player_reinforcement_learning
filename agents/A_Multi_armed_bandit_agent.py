from environments import multi_armed_bandits
import numpy as np
import random

'''
Welcome to this project, in which we explore the different Reinforcement learning algorithms, for their limitations and
effectiveness in mastering two player strategy games.

A) Multi Armed Bandit problem

The Environment is explained under environments/multi_armed_bandits

The following algorithm is the simplest agent in Reinforcement Learning.
The agent simply tries different Actions and records the average of the encountered rewards
in an array. 

Which Action is taken is a delicate balance between exploration and exploitation.
If the agent would chose at random every time then it would never improve to have greater
average rewards. However always doing the Action that is believed to be best is also
not an option, since the agent would get stuck at a suboptimal policy which always takes the
same Action which it believes to be best, only because it hasn't encountered enough of the other
superior options

That's why it is common to at times explore and at times exploit.
The probability of exploration vs exploitation is called epsilon.
epsilon can be anything between 0 and 1. The choice of epsilon greatly effects
the learning outcomes of the agent.

Parameters which effect learning and need to be set by the programmer are called Hyperparameters

To see the learning in action run this script and read the console output

A more visual graph comparing different epsilons can be found in B_multi_armed_bandit_data
'''

amount_of_arms = 10

MAB_Environment = multi_armed_bandits.Environment(amount_of_arms)

# Q_values will store the average observed rewards
Q_values = [0 for _ in range(amount_of_arms)]

# N records the number of times an Action has been taken which is used to update the Q_values
N = [0 for _ in range(amount_of_arms)]

# epsilon is a hyperparameter which determines the probability of a random action being taken
epsilon = 0.3

# reward_history records the rewards for progress updates
reward_history = []


print("\nThe best Action in this Environment is Action " + str(np.argmax(MAB_Environment.arms)) + " with a median reward of " + str(np.max(MAB_Environment.arms).round(3)))
print("Actual Medians:")
print(str(np.array(MAB_Environment.arms).round(2)) + "\n\n")


for _ in range(20_000):

    if random.random() < epsilon:
        # With a probability of epsilon a random Action is chosen
        Action = np.random.choice(MAB_Environment.get_actions())
    else:
        # With a probability of (1 - epsilon) the Action which is believed to be best is chosen
        Action = np.argmax(Q_values)

    # Agent interacts with the Environment by taking an action and receiving feedback
    (state, reward, new_state, done) = MAB_Environment.step(Action)

    # Updating the Q_values to account for the newly observed reward
    N[Action] += 1
    Q_values[Action] += 1/N[Action] * (reward - Q_values[Action])
    # This just calculates the average.
    #  Proof:    R(n) = the n-th observed reward of this action
    #            n = the number of times this action accord
    # I  Q(n)   = (R(0) + R(1) + ... + R(n  )) / n            <=  Formula for averages
    # II Q(n+1) = (R(0) + R(1) + ... + R(n+1)) / n+1
    #
    # I in II   = Q(n) * n / (n+1) + R(n+1) / (n+1)
    #           = Q(n) + (R(n+1) - Q(n)) / (n+1)
    # Therefore Q += 1/n * (R - Q) if n is updated before Q, otherwise 1/(n+1) * (R - Q)

    reward_history.append(reward)

    if _ in [10, 100, 1000]:
        print("\n\nAfter "+str(_)+" time steps:")

        print("Observed averages (Q_values array):")
        print(np.array(Q_values).round(2))
        print("\nThe agent belives Action " + str(np.argmax(Q_values)) + " to be best, which is " + str(
            np.argmax(Q_values) == np.argmax(MAB_Environment.arms)) + "\n")
        input("Press Enter to continue (sometimes twice)")


print("\n\nAfter finishing training")
print("Observed averages (Q_values array):")
print(np.array(Q_values).round(2))
print("\nThe agent belives Action " + str(np.argmax(Q_values)) + " to be best, which is "+ str(np.argmax(Q_values) == np.argmax(MAB_Environment.arms)))
