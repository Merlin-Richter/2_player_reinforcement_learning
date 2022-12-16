import environments.Galatea as Galatea
import numpy as np
import random
import matplotlib.pyplot as plt

'''
This is an implementation of the TD(λ) algorithms with a linear approximation algorithm 

The idea of the TD(0) algorithm is to update the State-Values of one State with the State-Values of the next state.
So that we don't have to wait for the episode to finish and we are a lot better at differentiating rather a position 
was bad, just because it was one position in a game we ended up loosing. 
Imagine we have an environment with two states A and B and no action taking. We observe the following:

A -> Terminal R=1,
A -> Terminal R=0,
A -> Terminal R=1,
B -> A -> Terminal R=0

What is the value of state B?
The monte carlo algorithm would say 0, because whenever the agent was at state B it ended up with R=0.
TD(0) on the other hand learns that A has a value of 0.5 and defines the value of B as the value of A, so 0.5

This is an example of the difference between the two algorithms.

λ is the amount of steps that the algorithm waits before updating the values of a state to better represent the value
of the state, which comes λ time steps later. For example TD(0), so λ = 0 means that the algorithm will update the value
of a state with the predicted value of the next state. If TD(1) / λ = 1 then the value of the next but one state is used
to update a state

If λ is an even number like with TD(0) or TD(2) then we just so happen to have Double Q learning, which means that
predictions of one function are used to update the predictions of the other function.
Normally there is only one function, which is using its own predictions to update its predictions.
We have two here, because the evaluation is very different depending on who's turn it is to play. More different than
such a simple linear model could learn when having the players turn as a feature.
Having two functions relying on each other  makes this a lot more robust.

Test TD(0), TD(1), TD(2) and TD(3) to see the performance difference. (λ uneven ~98% to λ even ~99.3%)
'''

Env = Galatea.Environment()

# 666 polynomials + 1 bias
weights = [np.random.random((667, 1)) - 0.5, np.random.random((667, 1)) - 0.5]
weights = np.load("Very_strong_TD_weights.npy")
weights_of_good_agent = np.load("Very_strong_monte_carlo_weights.npy")
weights_of_good_agent[[0, 1]] = weights_of_good_agent[[1, 0]]
# step_size for weight updating
alpha = 0.03

# The state how many steps in the future is used to update the value of the current state. λ=0 means the next state
Lambda = 2

epsilon = 1


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def predict(x, id):
    global weights
    global Env
    if x is None:
        x = Env.vector_representation(polynomials=True)
    return sigmoid(np.dot(x, weights[id]))[0]


def update_weights(x, y, id):
    global weights, alpha
    h = predict(x, id)
    weights[id] -= (alpha * (h - y) * h * (1 - h) * x).reshape((667, 1))


def learn_while_playing(games):
    bars = 0
    w = 0
    print("Learning through playing...\n[", end="")
    for game in range(games):
        last_states = []
        while not Env.done:
            last_states.append(Env.vector_representation(polynomials=True))
            if len(last_states) > Lambda+1:
                last_states.pop(0)
            actions = Env.get_actions()
            if random.random() < epsilon:
                Env.step(random.choice(actions))
            else:
                best_a, best_v = -1, -100
                save_state, save_turn = np.copy(Env.new_state), Env.player_turn
                for action in actions:
                    Env.step(action)
                    prediction = (predict(Env.vector_representation(True), Env.get_turn())-0.5) * (Env.get_turn()-0.5) * 2 + 0.5 + random.random() * 2 * epsilon
                    if prediction > best_v:
                        best_v = prediction
                        best_a = action
                    Env.new_state = np.copy(save_state)
                    Env.player_turn = save_turn
                Env.step(best_a)
            if len(last_states) == Lambda+1:
                update_weights(last_states[0], predict(Env.vector_representation(polynomials=True), Env.get_turn()), (Env.get_turn()+Lambda+1)%2)
            if Env.done:
                for i, state in enumerate(last_states):
                    update_weights(state, Env.reward, (Env.get_turn()+Lambda+i+1) % 2)
                update_weights(Env.vector_representation(polynomials=True), Env.reward, 0)
                update_weights(Env.vector_representation(polynomials=True), Env.reward, 1)
        w += Env.reward

        Env.reset()
        while (game + 1) / games > bars/40:
            bars += 1
            print("-", end="")
    print("]")
    print("win percentage of white in training is "+str(round(100*w/games, 2)) + "%")


def performance_evaluation(games=500):
    bars = 0
    print("Evaluating performance...\n[", end="")
    wins = 0
    for game in range(games):
        while not Env.done:
            actions = Env.get_actions()
            best_v = -100
            best_a = -1
            save_state = np.copy(Env.new_state)
            save_player = Env.player_turn
            for action in actions:
                Env.step(action)
                prediction = sigmoid(np.dot(Env.vector_representation(polynomials=True), weights_of_good_agent[Env.get_turn()]))[0] + random.random()/5
                if prediction > best_v:
                    best_v = prediction
                    best_a = action
                Env.new_state = np.copy(save_state)
                Env.player_turn = save_player
            Env.step(best_a)
            if Env.done:
                break
            actions = Env.get_actions()
            best_v = 100
            best_a = -1
            save_state = np.copy(Env.new_state)
            save_player = Env.player_turn
            for action in actions:
                Env.step(action)
                prediction = predict(Env.vector_representation(polynomials=True), Env.get_turn()) + random.random()/5
                if prediction < best_v:
                    best_v = prediction
                    best_a = action
                Env.new_state = np.copy(save_state)
                Env.player_turn = save_player
            Env.step(best_a)
        wins += 1-Env.reward
        Env.reset()
        while (game + 1) / games > bars / 40:
            bars += 1
            print("-", end="")
    print("]")

    print(str(round(100*wins/games, 2))+"% of games were won as black against a very good previously trained agent")
    return round(100*wins/games, 2)


if __name__ == "__main__":
    print("\n[----------------------------------------]\n\n")
    iterations = 100
    performance = []
    for x in range(iterations):
        print("Iteration " + str(x+1) + " out of " + str(iterations))
        epsilon = 0.2 - 0.15*x/iterations
        alpha = 0.01 - 0.01*x/iterations
        learn_while_playing(games=3_000)
        performance.append(performance_evaluation(games=100))
        plt.plot(performance)
        plt.pause(0.1)

    np.save(str(performance_evaluation(games=3000))+"%TD_self_play_weights.npy", weights)
    plt.show()


