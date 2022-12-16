import environments.Galatea as Galatea
import numpy as np
import random

'''
This is the first implementation of a generalizing algorithm. Instead of remembering the a value for each state this
algorithm tries to extrapolate what it learned from one state to all states by trying to find features that
correlations to a position being won or lost for white. Every square on the board is a feature. 
(2 actually, (0,0) for no piece, (1,0) for a white piece and (0,1) for a black piece at that position.

To give the model more features to extract correlations from, we extend the feature set with all polynomial of degree 2.
So insted of     x1, x2, x3, ..., xn                                        || Length = 36     + 1 bias
We have          x1, x2, x3, ..., xn, x1*x1, x1*x2, x1*x3, ..., xn*xn       || Length = 666    + 1 bias
Without duplicates of course

The model will, as always, try to predict accumulated upcoming Rewards until reaching a terminal state.

We actually have two models here.
A different one is used if it is whites turn than if it is blacks turn, because the evaluation is very different
depending on who's turn it is to play.
Normally the information of who plays next is encoded into the feature set, but that is more than a simple linear
model, such as this one, could learn. So we use two different ones.

'''

np.set_printoptions(suppress=True)

Env = Galatea.Environment()

# 666 polynomials + 1 bias
weights = [np.random.random((667, 1))-0.5, np.random.random((667, 1))-0.5]

# step_size


epsilon = 0.5

player1_state = []
player2_state = []
y = []


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def predict(x, player_id):
    global weights
    global Env
    if x is None:
        x = Env.vector_representation(polynomials=True)
    return sigmoid(np.dot(x, weights[player_id]))


def update_weights(x, y, player_id):
    global weights, alpha
    h = predict(x, player_id)
    weights[player_id] += (-alpha * (h - y) * h * (1 - h) * x).reshape((667, 1))


def performence_evaluation(games=500):
    global weights, Env
    print("evaluating performance...")
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
                prediction = predict(Env.vector_representation(True), Env.get_turn())
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


def generate_data_through_self_play(games):
    global weights, Env, y, player2_state, player1_state, epsilon
    print("generating new data...")
    print("[", end="")
    bars = 0
    for z in range(games):
        while not Env.done:
            if random.random() < epsilon:
                Env.step(random.choice(Env.get_actions()))
            else:

                save_state = np.copy(Env.new_state)
                save_turn = Env.player_turn
                max_value = -100
                best_action = -1
                for action in Env.get_actions():
                    Env.step(action)
                    prediction = predict(Env.vector_representation(True), Env.get_turn())
                    if prediction > max_value:
                        best_action = action
                        max_value = prediction
                    Env.new_state = np.copy(save_state)
                    Env.player_turn = save_turn

                Env.step(best_action)
            player1_state.append(Env.vector_representation(True))
            if Env.done:
                player2_state.append(Env.vector_representation(True))
                break

            if random.random() < epsilon:
                Env.step(random.choice(Env.get_actions()))
            else:
                save_state = np.copy(Env.new_state)
                save_turn = Env.player_turn
                max_value = 100
                best_action = -1
                for action in Env.get_actions():
                    Env.step(action)
                    prediction = predict(Env.vector_representation(True), Env.get_turn())
                    if prediction < max_value:
                        best_action = action
                        max_value = prediction
                    Env.new_state = np.copy(save_state)
                    Env.player_turn = save_turn

                Env.step(best_action)
            player2_state.append(Env.vector_representation(True))

        for _ in range(len(player1_state) - len(y)):
            y.append(Env.reward)

        while (z + 1) / games > bars/40:
            bars += 1
            print("-", end="")


        Env.reset()
    print("]")


def learn(iterations=20):
    global player1_state, player2_state, y
    print("learning...")
    print("[", end="")
    bars = 0
    for g in range(iterations):
        for index in range(len(player1_state)):
            update_weights(player1_state[index], y[index], 1)
            update_weights(player2_state[index], y[index], 0)
        while (g + 1) / iterations > bars/40:
            bars += 1
            print("-", end="")
    print("]")



iterations = 20
practice_games = 5000
alpha = 0.0006

print("\n[----------------------------------------]\n\n")

for x in range(iterations):
    print("Iteration " + str(x) + " of " + str(iterations))
    epsilon = 0.5 - x * 0.5 / iterations
    alpha -= 3 * alpha / (iterations + 3)
    player1_state = []
    player2_state = []
    y = []

    generate_data_through_self_play(practice_games)
    print(np.sum(y) / len(y))
    learn()

    print(str(performence_evaluation()*100) + "% wins")

    np.save(str(round(performence_evaluation(1500)*100, 1)) + "%_weights_for_galatea_iteration_" + str(x) , weights)
