import environments.Galatea as Galatea
import numpy as np
import random
import matplotlib.pyplot as plt

'''
This is the first implementation of a generalizing algorithm. Instead of remembering the a value for each state, this
algorithm tries to extrapolate what it learned from one state to all states by trying to find features that
correlate to a position being won or lost for white. Every square on the board is a feature. 
(Two features actually, (0,0) for no piece, (1,0) for a white piece and (0,1) for a black piece at that position.)

To give the model more features to extract correlations from, we extend the feature set with all polynomial of degree 2.
So instead of    x1, x2, x3, ..., xn                                        || Length = 36     + 1 bias
We use           x1, x2, x3, ..., xn, x1*x1, x1*x2, x1*x3, ..., xn*xn       || Length = 666    + 1 bias
Without duplicates of course

The model will, as always, try to predict accumulated upcoming Rewards until reaching a terminal state, which in this
case with the Reward of 1 on a win and 0 on a loss for white just so happens to be the expected probability of white 
wining

We actually have two models here.
A different one is used if it is whites turn than if it is blacks turn, because the evaluation is very different
depending on who's turn it is to play.
Normally the information of who plays next is encoded into the feature set, but that is more than a simple linear
model, such as this one, could learn. So we use two different ones.

'''

np.set_printoptions(suppress=True)

Env = Galatea.Environment()

# 666 polynomials + 1 bias
weights = np.random.random((667, 1))-0.5

# step_size
alpha = 0.005

boards = []
y = []


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def predict(x=None):
    global weights
    global Env
    if x is None:
        x = Env.vector_representation(polynomials=True)
    return sigmoid(np.dot(x, weights))


def get_loss(x, y):
    count = 0
    for xx in range(len(x)):
        count += (predict(x[xx]) - y[xx])**2

    return count/len(y)


def update_weights(x, y):
    global weights
    h = predict(x)
    # We want to update the weights proportional to the effect of the loss-function, which in this case is the
    # mean squared error. For that we need the partial derivative of the loss function relative to w (the weights).
    #
    # h(x,w) is the prediction, which should ideally be equal to y.
    # L(h(x,w), y) = (h(x,w)-y)^2    =>    d L(h(x,w), y) / d h(x,w) = 2 * (h(x,w) - y) * 1
    #
    # h(x, w) = σ(xw),  x is the feature vector, w the weights, xw the dot product of the two, σ is the sigmoid function
    # h(x, w) = σ(xw)                =>    dh(x,w)/dw   =   dσ(xw)/dw  =  σ(xw) * (1-σ(xw)) * x
    #
    # L(x,w,y) = (σ(xw)-y)^2         =>    dL(x,w,y)/dw =   dL(x,w,y)/dh(x,w)  *  dh(x,w)/dw
    #                     =   2 * (σ(xw) - y) * 1        *      σ(xw) * (1-σ(xw)) * x
    #
    # The 2 is removed because we have an arbitrary alpha which makes any multiplication by a number irrelevant.
    #
    # Minus because we want to change the weights in a direction that has a negative slope/effect on the Loss
    weights -= (alpha * (h - y) * h * (1 - h) * x).reshape((667, 1))


def generate_data_through_random_play(games):
    bars = 0
    print("Generating data...\n[", end="")
    for game in range(games):
        while not Env.done:
            Env.step(random.choice(Env.get_actions()))
            boards.append(Env.vector_representation(True))
            if Env.done:
                break
            Env.step(random.choice(Env.get_actions()))
        for _ in range(len(boards) - len(y)):
            y.append(Env.reward)
        Env.reset()

        while (game + 1) / games > bars/40:
            bars += 1
            print("-", end="")
    print("]")
    print("\nThe training data consists of " + str(boards.__len__()) + " Board positions")
    print(str(round(100 * np.sum(y) / len(y), 2)) + "% of which resulted in a win for Player 1\n")


def learn(iterations):
    bars = 0
    print("Learning from Data...\n[", end="")
    losses = []
    for g in range(iterations):
        losses.append(get_loss(boards, y))
        for index in range(len(boards)):
            update_weights(boards[index], y[index])
        while (g + 1) / iterations > bars / 40:
            bars += 1
            print("-", end="")
    print("]")

    plt.plot(losses)
    plt.pause(0.1)


def performance_evaluation(games=500):
    wins = 0
    bars = 0
    print("Evaluating performance...\n[", end="")
    for game in range(games):
        while not Env.done:
            save_state = np.copy(Env.new_state)
            save_turn = Env.player_turn
            max_value = -100
            best_action = -1
            for action in Env.get_actions():
                Env.step(action)
                prediction = predict(Env.vector_representation(True))
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
        while (game + 1) / games > bars / 40:
            bars += 1
            print("-", end="")
    print("]")

    print("The agent wins " + str(round(100*wins/games, 2)) + "% of its games against the random policy ")


print("\n[----------------------------------------]\n\n")

generate_data_through_random_play(games=10_000)
learn(iterations=15)
performance_evaluation(games=1_000)
plt.show()
