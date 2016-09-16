import numpy as np
from numpy.random import normal, beta, binomial
from math import exp, log, sqrt
import matplotlib.pyplot as plt
import operator
import sys
import seaborn as sns


def get_reward(reward_params):
    r = normal(reward_params[0], reward_params[1])
    return 1.0 / (1.0 + exp(-r))


def get_action_value(s, f):
    return beta(s + 1, f + 1)


def plot_true_rewards(mean, sd, i):
    data = [get_reward([mean, sd]) for j in range(0, 5000)]
    sns.kdeplot(np.array(data), bw=0.05, label="arm-" + str(i))


class Environment:

    def __init__(self, k):
        self.k = k
        self.reward_params = self.__setup_reward_params(k)

    def __setup_reward_params(self, k):
        reward_params = []
        means = [-1.51167799, -2.16126981, 0.0835869, -0.41663176,
                 -2.84016476, 0.02216087, -0.29321112, 0.27555877,
                 1.57168051, 2.99220041]
        for i in range(0, k):
            reward_params.append((means[i], 1))
            # plot_true_rewards(means[i], 1, i)
        # plt.show()
        return reward_params

    def modify_reward_dist(self):
        for i in range(0, self.k):
            self.reward_params[i] = (normal(0, 2), 1)


class Thompson:
    # Ref: Algorithm 2 in "Analysis of Thompson Sampling for the Multi-armed
    # Bandit Problem"
    def __init__(self, e, T, s=0, f=0, stationary=True):
        self.env = e
        self.T = T
        self.reward = []
        self.s = s
        self.f = f
        self.stationary = stationary

    def play(self):
        success = [self.s] * self.env.k
        failure = [self.f] * self.env.k

        for t in range(0, self.T):
            if (not self.stationary) and t % 500 == 0:
                self.env.modify_reward_dist()

            estimated_values = [get_action_value(success[i], failure[i])
                                for i in range(0, self.env.k)]
            arm, _ = max(enumerate(estimated_values),
                         key=operator.itemgetter(1))

            r = get_reward(self.env.reward_params[arm])
            self.reward.append(r)
            b = binomial(1, r)
            if b == 1:
                success[arm] += 1
            else:
                failure[arm] += 1


class UCB:
    def __init__(self, e, T, c, stationary=True):
        self.env = e
        self.T = T
        self.c = c
        self.stationary = stationary
        self.reward = []
        self._estimated_values = [0] * self.env.k
        self._num_times = [0] * self.env.k

    def __get_ucb_value(self, arm, t):
        if self._num_times[arm] == 0:
            return sys.maxsize
        else:
            return self._estimated_values[arm] + \
                self.c * sqrt(log(t) / self._num_times[arm])

    def __update(self, arm, r):
        self._num_times[arm] += 1
        old = self._estimated_values[arm]
        n = self._num_times[arm]
        self._estimated_values[arm] = old + (1.0 / n) * (r - old)

    def play(self):
        for t in range(0, self.T):
            if (not self.stationary) and t % 500 == 0:
                self.env.modify_reward_dist()

            temp = [self.__get_ucb_value(i, t) for i in range(0, self.env.k)]
            arm, _ = max(enumerate(temp), key=operator.itemgetter(1))
            r = get_reward(self.env.reward_params[arm])
            self.reward.append(r)
            self.__update(arm, r)


def show_plot():
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.show()


def expt_1(T, n, e, c):
    """Thompson vs UCB (stationary)
    """
    print("Experiment 1: Thompson vs UCB (stationary)")
    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Thompson Sampling')

    ucb_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        u = UCB(e, T, c)
        u.play()
        ucb_average_reward += np.array(u.reward)

    plt.plot(ucb_average_reward / n, label='UCB')
    show_plot()


def expt_2(T, n, e, c):
    """Thompson with different priors (+ UCB) (stationary)
    """
    print("Experiment 2: Thompson with different priors (+ UCB) (stationary)")
    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, 0, 0)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Beta(1,1)')

    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, 199, 0)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Beta(20,1)')

    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, 0, 19)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Beta(1,20)')

    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, 0, 199)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Beta(1,200)')

    ucb_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        u = UCB(e, T, c)
        u.play()
        ucb_average_reward += np.array(u.reward)

    plt.plot(ucb_average_reward / n, label='UCB')
    show_plot()


def expt_3(T, n, e, c):
    """Thompson vs UCB (nonstationary and stationary)
    """
    print("Experiment 3: Thompson vs UCB (nonstationary and stationary)")
    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Thompson Stationary')

    ucb_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        u = UCB(e, T, c)
        u.play()
        ucb_average_reward += np.array(u.reward)

    plt.plot(ucb_average_reward / n, label='UCB Stationary')

    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, stationary=False)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Thompson non-stationary')

    ucb_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        u = UCB(e, T, c, stationary=False)
        u.play()
        ucb_average_reward += np.array(u.reward)

    plt.plot(ucb_average_reward / n, label='UCB non-stationary')
    show_plot()


def expt_4(T, n, e, c):
    """Thompson with diff. priors (nonstationary)
    """
    print("Experiment 4: Thompson with different priors (nonstationary)")
    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, 0, 0, False)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Beta(1,1)')

    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, 19, 0, False)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Beta(20,1)')

    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, 0, 19, False)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Beta(1,20)')

    thompson_average_reward = np.array([0.0] * T)
    for i in range(0, n):
        t = Thompson(e, T, 0, 199, False)
        t.play()
        thompson_average_reward += np.array(t.reward)

    plt.plot(thompson_average_reward / n, label='Beta(1,200)')
    show_plot()


def main():
    T = 2000
    k = 10
    n = 100
    c = 2
    e = Environment(k)
    expt_1(T, n, e, c)
    expt_2(T, n, e, c)
    expt_3(T, n, e, c)
    expt_4(T, n, e, c)


if __name__ == '__main__':
    main()
