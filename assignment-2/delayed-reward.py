import numpy as np
from numpy.random import binomial
import matplotlib.pyplot as plt
from random import randint
from pylab import savefig
import argparse


class Environment:
    def __init__(self, p):
        self.n_states = 9
        self.curr_state = (0, 0)
        self.prev_reward = 0
        self.new_reward = 0
        self.p = p

    def reset(self):
        self.curr_state = (0, 0)
        self.prev_reward = 0

    def get_state_reward(self):
        # print("Curr state", self.curr_state)
        self.new_reward = self.__make_transition()
        # print("Next state", self.curr_state)
        # print("True reward=", self.new_reward)
        x = binomial(1, self.p)
        prev_reward = self.prev_reward
        self.prev_reward = self.new_reward
        if x == 1 or self.curr_state == (2, 2):
            return (self.curr_state, self.new_reward)
        else:
            return (self.curr_state, prev_reward)

    def get_true_reward(self):
        return self.new_reward

    def __make_transition(self):
        action = self.__get_action()
        # print("Action=", action)
        new_state = self.curr_state
        if(action == 0):
            new_state = (max(new_state[0] - 1, 0), new_state[1])
        if(action == 1):
            new_state = (new_state[0], min(new_state[1] + 1, 2))
        if(action == 2):
            new_state = (min(new_state[0] + 1, 2), new_state[1])

        true_reward = self.__get_reward(self.curr_state, new_state)
        self.curr_state = new_state
        return true_reward

    def __get_action(self):
        return randint(0, 2)

    def __get_reward(self, c, n):
        if c == (1, 0) and n == (2, 0):
            return 1
        elif c == (2, 0) and n == (1, 0):
            return -1
        elif c == (1, 0) and n == (0, 0):
            return 50
        elif c == (0, 0) and n == (1, 0):
            return -50
        elif c == (1, 2) and n == (2, 2):
            return 10
        elif c == (2, 1) and n == (2, 2):
            return 10
        else:
            return 0


class ErrorPlotter:
    def __init__(self, n_states, n_episodes, n_experiments, p):
        self.n_states = n_states
        self.n_episodes = n_episodes
        self.n_experiments = n_experiments
        self.p = p
        self.beta = []
        # Maps Beta values to corresponding index in error_delay_new list for
        # each state
        self.beta_index_map = {}
        # Next available index in error_delay_new list for each state
        self.index = 0
        self.error_delay_new = [[] for _ in range(0, n_states)]

    def add_plotter_data(self, plotter):
        # if this beta is seen for the first time, add it to the dict, and
        # add a new element in error_delay_new list for each state
        if plotter.beta not in self.beta_index_map:
            self.beta_index_map[plotter.beta] = self.index
            self.index += 1
            self.beta.append(plotter.beta)
            for s in range(0, self.n_states):
                self.error_delay_new[s].append(0)

        idx = self.beta_index_map[plotter.beta]
        for s in range(0, self.n_states):
            self.error_delay_new[s][idx] += plotter.error_delay_new[s][-1]

        # print(self.error_delay_new)
        # input()
    def plot_error_curve(self):
        for s in [0, 3, 6]:
            plt.plot(self.beta,
                     np.array(self.error_delay_new[s]) / self.n_experiments,
                     marker='o', label="v_" + str(s))
        plt.title("p = {}".format(self.p))
        plt.xlabel("beta")
        plt.ylabel("Absolute Error averaged over {} experiments".format(
            self.n_experiments))
        plt.legend(loc='best', framealpha=0.5)
        savefig('error-plotter/average-error-{}.png'.format(self.p))
        plt.close()


class Plotter:
    def __init__(self, n_states, n_episodes, p, beta):
        self.n_episodes = n_episodes
        self.p = p
        self.beta = beta
        self.n_states = n_states
        # V estimates in delay case with new reward sharing method
        self.v_delay_new = [[] for _ in range(0, n_states)]
        # V estimates in delay case with traditional method
        self.v_delay_trad = [[] for _ in range(0, n_states)]
        # V estimates without delay
        self.v_no_delay = [[] for _ in range(0, n_states)]
        # true values
        self.v_true = [-8.625, 10, 10, 22.75,
                       10, 10, 15.875, 10, 0]

        self.error_delay_new = [[] for _ in range(0, n_states)]
        self.error_delay_trad = [[] for _ in range(0, n_states)]

    def plot_v_curve(self):
        # for i in range(0, len(self.v_delay_new)):
        for i in [0, 3, 6]:
            n = len(self.v_no_delay[i])
            plt.plot(self.v_no_delay[i],
                     label="No delay TD(0)")
            plt.plot(self.v_delay_trad[i],
                     label="Delay TD(0)")
            plt.plot(self.v_delay_new[i],
                     label="Delay TD(0)-S")
            plt.hlines(self.v_true[i], 0, n, label="True value")
            plt.title("v_{} with p = {}, beta = {}".format(
                i, self.p, self.beta))
            plt.xlabel("Steps")
            plt.ylabel("State Value")
            plt.xlim(0, n)
            plt.legend(loc='best', framealpha=0.5)
            savefig('error-plotter/value-estimate-v-{}.png'.format(i))
            plt.close()

    def plot_error_curve(self):
        # for i in range(0, self.n_states):
        for i in [0, 3, 6]:
            plt.plot(self.error_delay_trad[i], label="TD(0)")
            plt.plot(self.error_delay_new[i], label="TD(0)-S")
            plt.title("v_{} with p = {}, beta = {}".format(
                i, self.p, self.beta))
            plt.xlabel("Episodes")
            plt.ylabel("Absolute error in V estimates")
            plt.xlim(0, self.n_episodes)
            plt.legend(loc='best', framealpha=0.5)
            savefig('error-plotter/error-estimate-v-{}.png'.format(i))
            plt.close()

    def calc_abs_error(self, v_estimate_trad, v_estimate_new):
        error_trad = abs(np.array(self.v_true) - np.array(v_estimate_trad))
        error_new = abs(np.array(self.v_true) - np.array(v_estimate_new))

        for i in range(0, self.n_states):
            self.error_delay_trad[i].append(error_trad[i])
            self.error_delay_new[i].append(error_new[i])


class RL:
    def __init__(self, e, plot, n_episodes, gamma, alpha, beta):
        self.e = e
        self.plot = plot
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.state_index = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3,
                            (1, 1): 4, (1, 2): 5, (2, 0): 6, (2, 1): 7,
                            (2, 2): 8}

    def td_prediction(self):
        v = [0] * self.e.n_states
        v_no_delay = [0] * self.e.n_states
        v_delay_trad = [0] * self.e.n_states

        for t in range(0, self.n_episodes):
            n = 0
            c = 0
            p = 0
            flag = False
            self.e.reset()

            while(c != 8):
                state_tup, r = self.e.get_state_reward()
                n = self.state_index[state_tup]

                # update v for current state
                # when the next state is the terminal state, in which case
                # the reward is available instantaneously
                if n == 8:
                    beta = 0
                else:
                    beta = self.beta

                v[c] = v[c] + self.alpha * \
                    ((1 - beta) * r + self.gamma * v[n] - v[c])

                # update v for previous state
                # except when the next state is terminal state in which case
                # the reward received is the current reward.
                # Don't update previous state for first timestep because there
                # is no previous state!
                if flag:
                    v[p] = v[p] + self.alpha * beta * r

                # get true reward, and calc corresponding v (without delay)
                # for plotting purposes
                true_reward = self.e.get_true_reward()
                v_no_delay[c] = v_no_delay[c] + self.alpha * \
                    (true_reward + self.gamma * v_no_delay[n] - v_no_delay[c])

                # calc v for delayed reward with traditional technique without
                # sharing reward with past states using beta
                v_delay_trad[c] = v_delay_trad[c] + self.alpha * \
                    (r + self.gamma * v_delay_trad[n] - v_delay_trad[c])

                # add estimates to plot arrays
                for i in range(0, self.e.n_states):
                    self.plot.v_delay_new[i].append(v[i])
                    self.plot.v_delay_trad[i].append(v_delay_trad[i])
                    self.plot.v_no_delay[i].append(v_no_delay[i])
                p = c
                c = n
                flag = True

            self.plot.calc_abs_error(v_delay_trad, v)
            self.alpha = self.alpha / 1.001


def expt_1_v_values():
    n_episodes = 10000
    prob = 0.5
    beta = 0.75

    e = Environment(p=prob)
    plot = Plotter(e.n_states, n_episodes, prob, beta)
    rl = RL(e=e, plot=plot, n_episodes=n_episodes,
            gamma=1, alpha=0.5, beta=beta)
    rl.td_prediction()
    plot.plot_v_curve()
    plot.plot_error_curve()


def expt_2_p_beta_variation():
    n_episodes = 8000
    n_states = 9
    n_experiments = 25
    prob = np.linspace(0, 1, 5)
    beta = np.linspace(0, 1, 10)

    for p in prob:
        print("Prob = {}".format(p))
        error_plotter = ErrorPlotter(n_states, n_episodes,
                                     n_experiments, p)
        for expt in range(0, n_experiments):
            print("Experiment # {}".format(expt))
            for b in beta:
                print("beta = {}".format(b))
                e = Environment(p=p)
                plot = Plotter(e.n_states, n_episodes, p, b)
                rl = RL(e=e, plot=plot, n_episodes=n_episodes,
                        gamma=1, alpha=0.5, beta=b)
                rl.td_prediction()
                error_plotter.add_plotter_data(plot)
        error_plotter.plot_error_curve()


def main():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('expt_num', type=int,
                        help='Enter experiment number to run (1 or 2). e.g.' +
                        ' python program.py 1')
    args = parser.parse_args()
    if args.expt_num == 1:
        expt_1_v_values()
    elif args.expt_num == 2:
        expt_2_p_beta_variation()
    else:
        parser.error("Experiment number can either be 1 or 2.")


if __name__ == '__main__':
    main()
