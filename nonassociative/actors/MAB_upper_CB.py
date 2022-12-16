import numpy as np
import matplotlib.pyplot as plt
import nonassociative.env.MAB_env
import nonassociative.actors.MAB_actor

class MAB_actor_UCB():

    def __init__(self, env, size, c=1.):
        self.q = np.array([0.] * size)
        self.h = [0.] * size
        self.env = env
        self.c = c
        self.t = 0.
        self.size = size

    def take_action(self):
        self.t += 1.
        hold = np.zeros(self.size)
        for x in range(self.size):
            hold[x] = self.q[x] + c * np.sqrt(np.log(self.t)/(self.h[x]+1.))
        arm = np.argmax(hold)
        r = self.env.step(arm)
        self.h[arm] += 1.
        self.q[arm] += (r - self.q[arm]) / self.h[arm]
        return r


if __name__ == "__main__":
    size = 30
    env = nonassociative.env.MAB_env.MAB_env(size)
    tries = 3000
    t = np.zeros(tries)
    cs = [0.1, 0.5, 1., 2., 5.]

    for c in cs:
        t = np.zeros(tries)
        for i in range(30):
            env = nonassociative.env.MAB_env.MAB_env(size)
            actor = MAB_actor_UCB(env, size, c)
            for x in range(tries):
                hold = actor.take_action()
                t[x] += hold / 30
        plt.plot(t, label=f"c={c}")

    plt.legend()
    plt.show()



