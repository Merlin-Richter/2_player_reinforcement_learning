import environments.Galatea as Galatea
import numpy as np
import random
import time

'''
Here you can play against the agents that you have trained

when prompted to name a square on the board, input the row and column according to the mapping.
For example '6a' for the square on the bottom left all the way to '1f' for the square on the top right.
Press Enter to confirm your choice once for which piece you want to play and then again for the destination

If you want to check the agents performance before beginning a game against the agent set the following variable to True
'''

Env = Galatea.Environment()

epsilon = 0.2

# loads the weights learned by other python files. The percentage is the likelihood of the agent winning against random
weights = [None for x in range(10)]

weights[0] = np.load("../weights/Very_strong_monte_carlo_weights.npy")
weights[0][[0, 1]] = weights[0][[1, 0]]

weights[1] = np.load("../weights/Very_strong_TD_weights.npy")

weights[2] = [np.random.random((667, 1)) - 0.5, np.random.random((667, 1)) - 0.5]

weights[3] = np.load("../weights/Ultimate_TD_weights.npy")


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def predict(x, w, player_id):
    global weights
    global Env
    return sigmoid(np.dot(x, w[player_id]))


def player_move():
    global Env

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
    Player_action = (0, 0)
    while Player_action not in Env.get_actions():
        start = input("which piece do you want to play?  ")
        end = input('where to?  ')
        try:
            start_position = (int(start[0]) - 1, alphabet.index(start[1]))
            end_position = (int(end[0]) - 1, alphabet.index(end[1]))
            Player_action = (Env.index_to_position.index(start_position), Env.index_to_position.index(end_position))
        except:
            print("input incorrect")
            continue

        if Player_action not in Env.get_actions():
            print("\nThat move is not a legal move\n")
    Env.step(Player_action)


def performence_evaluation(games=500):
    global weights, Env
    print("performance evaluation...")
    print("[", end="")
    wins = 0
    bars = 0
    for z in range(games):
        while not Env.done:
            save_state = np.copy(Env.new_state)
            save_turn = Env.player_turn
            max_value = -100
            best_action = -1
            for action in Env.get_actions():
                Env.step(action)
                prediction = predict(Env.vector_representation(True), 0)
                if prediction > max_value:
                    best_action = action
                    max_value = prediction
                Env.new_state = np.copy(save_state)
                Env.player_turn = save_turn

            Env.step(best_action)
            if Env.done:
                break
            Env.step(random.choice(Env.get_actions()))

        wins += Env.reward
        Env.reset()
        while (z + 1) / games > bars/40:
            bars += 1
            print("-", end="")
    print("]")
    return wins/games


def depth_search(depth, w, first=True, alpha=-10, beta=10):
    save_state = np.copy(Env.new_state)
    player = Env.player_turn
    best_value = (player - 1.5) * 10
    best_action = -1
    list_actions = Env.get_actions()
    list_actions.reverse()
    if first:
        print("Thinking...\n[", end='')
        bars = 0
        len_actions = len(list_actions)
    for z, action in enumerate(list_actions):
        Env.step(action)
        if Env.done:
            value = Env.reward
        elif depth > 1:
            value = depth_search(depth-1, w, False, alpha, beta)[1]
        else:

            value = predict(Env.vector_representation(True), w, Env.get_turn()) + random.random() * epsilon

        if player == 1 and value > best_value:
            best_action = action
            best_value = value
            alpha = max(alpha, value)
        elif value < best_value and player == 2:
            best_action = action
            best_value = value
            beta = min(beta, value)
        if beta <= alpha:
            break

        Env.new_state = np.copy(save_state)
        Env.player_turn = player
        while first and (z + 1) / len_actions > bars / 40:
            bars += 1
            print("-", end="")
    if first:
        print("]")
    return best_action, best_value


def game_against_human(depth=6, agent=0, agent_starts=True):
    global epsilon
    if not agent_starts:
        Env.render()
        player_move()
    while not Env.done:
        Env.render()
        Tuple = depth_search(depth, weights[agent])
        Env.step(Tuple[0])
        print(str(round(float(Tuple[1]*100), 1))+"% estimated win probability for white")
        if Env.done:
            break
        Env.render()
        player_move()
    Env.render()
    print("\nGAME OVER")


def agent_vs_agent(agent1=0, agent2=0, depth=0, render=True):
    if render:
        Env.render()
    while not Env.done:
        Tuple = depth_search(depth, weights[agent1], render)
        Env.step(Tuple[0])
        if render:
            print(str(round(float(Tuple[1]*100), 1)) + "% for white")
            Env.render()
            input("Press Enter to continue")
        if Env.done:
            break
        Tuple = depth_search(depth, weights[agent2], render)
        Env.step(Tuple[0])
        if render:
            print(str(round(float(Tuple[1] * 100), 1)) + "% for white")
            Env.render()
            input("Press Enter to continue")
    if render:
        input("\nGAME OVER")
    R = Env.reward
    Env.reset()
    return R


if __name__ == "__main__":
    epsilon = 0
    game_against_human(7, 3, agent_starts=True)
    agent_vs_agent(3, 3, depth=7)
    wins = [0, 0]
    epsilon = 0.2
    for x in range(100):
        wins[0] += agent_vs_agent(3, 4, 2, False)
        wins[1] += agent_vs_agent(4, 3, 2, False)
        print(x)
    print(wins)




