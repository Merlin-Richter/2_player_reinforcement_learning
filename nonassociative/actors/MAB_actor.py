import numpy as np
import matplotlib.pyplot as plt
import nonassociative.env.MAB_env

class MAB_actor():

    def __init__(self, env, size, greed=0.1):
        self.q = np.array([0.]*size)
        self.h = [0.] * size
        self.env = env
        self.greed = greed

    def take_action(self):
        if self.greed > np.random.random():
            arm = np.random.choice(np.array(range(30)))
            r = self.env.step(arm)
        else:
            arm = np.argmax(self.q)
            r = self.env.step(arm)
        self.h[arm] += 1.
        self.q[arm] += (r - self.q[arm]) / self.h[arm]
        return r


if __name__ == "__main__":
    size = 30
    env = nonassociative.env.MAB_env.MAB_env(size)
    tries = 3000
    t = np.zeros(tries)
    greeds = [0.1, 0.01, 1., 0., 0.5]

    for greed in greeds:
        t = np.zeros(tries)
        for i in range(100):
            env = nonassociative.env.MAB_env.MAB_env(size)
            actor = MAB_actor(env, size, greed)
            for x in range(tries):
                hold = actor.take_action()
                t[x] += hold/100
        plt.plot(t, label=f"greed={greed}")

    plt.legend()
    plt.show()



