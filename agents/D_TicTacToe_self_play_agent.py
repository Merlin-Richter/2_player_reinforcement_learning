import environments.tictactoe as TicTacToe
import numpy as np
import random
'''
The next step is to make the Agent play against itself instead of against a random policy

This way it will challenge itself more and more, the better it gets. The problem with practicing against a 
random policy is that sometimes the best way to play against a random policy is not optimal against the optimal policy.
For example in chess the agent might learn that attacking the opponents queen with an unprotected bishop is a good move,
because the likelihood of the random policy actually capturing the unprotected Bishop is so slip. But a good player
would of course always take the unprotected bishop with the queen. That's why it is necessary to iterate over policies
to converge on the optimal policy over time.

We start with epsilon = 0.5 which means the policy is still half random, but slowly decrease epsilon over time so that
the agent learns what is best against the optimal policy.



'''


Env = TicTacToe.Environment()

# Q_values is a dictionary
Q_values = {}

#probability of the agent taking a random action
epsilon = 0.5

# The step size alpha
alpha = 0.3

amount_of_draws = 0
total_games = 0

# 50_000 games are enough to achieve mastery at beating the random policy
amount_of_practice_games = 50_000

for game in range(amount_of_practice_games):

    state_history = []
    action_history = []
    epsilon = 0.5 - 0.5 * game / amount_of_practice_games
    alpha = 0.4 - 0.35 * game / amount_of_practice_games

    while not Env.done:
        if Env.hash() not in Q_values:
            Q_values[Env.hash()] = [0.8 for x in range(len(Env.get_actions()))]

        Action_index = -1

        if random.random() < epsilon:
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
            Q_values[Env.hash()] = [0.2 for x in range(len(Env.get_actions()))]

        Action_index = -1

        if random.random() < epsilon:
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

    amount_of_draws += 0.001 * (Env.reward - amount_of_draws)

    if game % 1000 == 0 and game != 0:
        print(str(round(100*game/amount_of_practice_games, 2)) + "%")
        print("Weighted Average of the past rewards: " + str(round(amount_of_draws, 3)) + "")

    Env.reset()


wins = 0
total_games = 0

for x in range(1000):
    while not Env.done:
        Action_index = np.argmax(Q_values[Env.hash()])
        Action = Env.get_actions()[Action_index]
        Env.step(Action)
        if Env.done:
            break

        Env.step(np.random.choice(Env.get_actions()))
    wins += (Env.reward == 1)
    Env.reset()

print("\nThe agent, when performing with epsilon = 0, wins " + str(wins/10) + "% of its games against a random agent\n")

input("press enter to start a game against the Agent ")

print("\n\n\nThe agent plays first and with the circles in this example")

print("\nTo play a move, send a single digit number to the console (by pressing ENTER) according to this mapping:")
print("1 2 3")
print("4 5 6")
print("7 8 9\n")

print("Enter 0 for the Agent to play for you!!!\n")

input("press enter to continue ")
print()

while True:

    while not Env.done:
        Env.render()
        Action_index = np.argmax(Q_values[Env.hash()])
        Action = Env.get_actions()[Action_index]
        print("\nThe agent believes Action " + str(Action+1) + " to be best.")
        print("The Reward estimate is " + str(np.max(Q_values[Env.hash()]).round(3)) + "\n")
        Env.step(Action)
        if Env.done:
            break
        print("1 2 3")
        print("4 5 6")
        print("7 8 9\n")
        Env.render()
        x = -1
        while x not in Env.get_actions():
            try:
                x = int(input("where do you want to play? "))-1
                if x == -1:
                    Action_index = np.argmin(Q_values[Env.hash()])
                    x = Env.get_actions()[Action_index]

            except ValueError:
                print("invalid input")
        Env.step(x)

    Env.render()
    Env.reset()
    print("\nGAME OVER\n")
    input("press enter to play again")


