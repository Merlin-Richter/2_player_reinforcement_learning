import environments.tictactoe as TicTacToe
import numpy as np

'''
This Agent plays TicTacToe against a "random agent" also known as the random policy.

Definitely read the comment in environments.tictactoe to fully understand how the agent interacts with the Environment.

In this agent we will explore new concepts which are needed for mastering strategy games.
Firstly, with TicTacToe there is now a "state", which has a profound impact on what the best action is.
We previously had Action-values, which now need to be State-Action-values.

Next is the problem that we don't get a reward after we perform an action. The Reward comes at the terminal state, when
the game is over, which may be up to 9 moves away. And then we don't really know which Action lead us to the Reward.
If we lose were all Actions taken during that game bad?
Maybe, maybe not. We don't know.

So how and when do we update the State-Action-Values?

The first approach is to wait until the episode is over and simply change the Q_value for each State-Action taken.
We play a game and if we lose we simply assume all State-Actions we took during that game were bad.
The idea of this is that the not so bad ones that got mixed into the bad moves will occure more often in games that are
won and will ballance out.

This way of updating the State-Action-values is called monte-carlo.

A different approach will be covered later on, in "I) I_Galatea_generalizing_TD_implementation", where the Q values
are updated without waiting until the episode is over and we actually know the reward.


Something else that is very important is that the true State-Action-value changes together with the Action taking policy

Starting by playing in the center is a lot better if the follow-up moves, that the agent plays are better.
The better the agent gets the higher the true State-Action-values are.
That means, that an average of all encountered State-Action-values no longer suffices. More recent observations have to
be weighted with more importance. This is achieved by changing the step size which previously was 1/N, where N is how 
often this State-Action-value has been encountered. 1/N converges to the average of all encountered rewards in this
State-Action pair. To achieve a weighted average one simply has to use a constant step-size parameter such as 'alpha'. 
For more on this, read Chapter 2.5 in "Reinforcement Learning, An Introduction"

The Algorithm employed here is first-visit Monte Carlo.

Run the script and play a game against the Agent! Good Luck, Have Fun
'''


Env = TicTacToe.Environment()

# Q_values is now a dictionary
Q_values = {}

epsilon = 0.5

# The newly introduced step size alpha.
alpha = 0.1

average_rewards = 0
total_games = 0

# 50_000 games are enough to achieve mastery at beating the random policy
amount_of_practice_games = 50_000

for game in range(amount_of_practice_games):
    epsilon = 0.5 - 0.5 * game/amount_of_practice_games
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
        Env.step(np.random.choice(Env.get_actions()))
        after_state = Env.hash()

    # The game is now over and we know the reward which means we can now update the Q_values
    for x in range(len(action_history)):

        Q_values[state_history[x]][action_history[x]] += alpha * (Env.reward - Q_values[state_history[x]][action_history[x]])

    average_rewards += Env.reward
    total_games += 1
    if game % 1000 == 0 and game != 0:
        print(str(round(100*game/amount_of_practice_games, 1)) + "%")
        print("Average reward: " + str(round(average_rewards/total_games, 3)))

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

input("press enter to continue ")
print()

while True:

    while not Env.done:
        Env.render()
        if Env.hash() not in Q_values:
            assert IndexError
            Q_values[Env.hash()] = [0.4 for x in range(len(Env.get_actions()))]
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
            except ValueError:
                print("invalid input")
        Env.step(x)

    Env.render()
    Env.reset()
    print("\nGAME OVER\n")
    input("press enter to play again")


