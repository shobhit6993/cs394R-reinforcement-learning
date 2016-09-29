# Based on DynaQ and its variants discussed in Section 8.2 of Introduction to
# Reinforcement Learning book by Sutton and Barto (2nd edition).
# The experiments perfomed are related to Exercise 8.4


import numpy as np
from numpy.random import multinomial
import random
from math import sqrt
import matplotlib.pyplot as plt
from pylab import savefig


R = -0.05
CHANGE_POINT = 1500
DYNA_Q = 'DYNA_Q'
DYNA_Q_PLUS = 'DYNA_Q_PLUS'
DYNA_Q_PLUS_VARIANT = 'DYNA_Q_PLUS_VARIANT'


def save_plot(self):
    savefig('error-plotter/cumulative-reward.png')
    plt.close()


class Plotter:
    def __init__(self, n_experiments):
        self.n_experiments = n_experiments
        self.arr = []
        self.steps_per_episode = []
        self.label_map = {DYNA_Q: "DynaQ",
                          DYNA_Q_PLUS: "DynaQ+",
                          DYNA_Q_PLUS_VARIANT: "DynaQ+ Variant"}

    def plot_cumulative_reward(self, mode):
        plt.plot(self.arr, label=self.label_map[mode])
        plt.xlabel("Time Steps")
        plt.ylabel("Cumulative Reward")
        plt.legend(loc='best', framealpha=0.5)

    def plot_steps_per_episode(self, mode):
        m = np.array(self.steps_per_episode).reshape(self.n_experiments, -1)
        # plt.plot(m.mean(axis=0)[1:], label=self.label_map[mode])
        plt.plot(np.mean(m, axis=0)[1:], label=self.label_map[mode])
        # plt.plot(np.median(m, axis=0)[1:], label=self.label_map[mode])
        plt.xlabel("Episode")
        plt.ylabel("Steps per episode")
        plt.legend(loc='best', framealpha=0.5)

    def plot_change_point(self):
        plt.vlines(CHANGE_POINT, min(self.arr), max(self.arr))


class Environment:
    def __init__(self, grid, start):
        self.grid = grid

    def reset(self):
        self.grid[3][0] = R
        self.grid[3][8] = None

    def change_grid(self):
        self.grid[3][0] = R
        self.grid[3][8] = R
        return [(3, 0), [3, 8]]

    def next_state_and_reward(self, state, action):
        next_state = self.__make_transition(state, action)
        if next_state == state:
            reward = R
        else:
            reward = self.__get_reward(next_state)
        return (next_state, reward)

    def __make_transition(self, state, action):
        if(action == 0):
            next_state = (max(state[0] - 1, 0), state[1])
        if(action == 1):
            next_state = (state[0], min(state[1] + 1, len(self.grid[0]) - 1))
        if(action == 2):
            next_state = (min(state[0] + 1, len(self.grid) - 1), state[1])
        if(action == 3):
            next_state = (state[0], max(state[1] - 1, 0))

        # If the next state is an obstacle, agent remains in the current state
        if self.grid[next_state[0]][next_state[1]] is None:
            next_state = state
        return next_state

    def __get_reward(self, state):
        x = state[0]
        y = state[1]
        return self.grid[x][y]


class DynaQ:
    def __init__(self, e, plotter, mode, start, goal, epsilon, n_episodes,
                 n_plan, gamma, alpha, kappa=0):
        self.e = e
        self.plotter = plotter
        self.mode = mode
        self.start_state = start
        self.goal_state = goal
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.n_plan = n_plan
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.Q = {}
        self.model = {}
        self.last_tried_on = {}

        self.__initialize()

    def __initialize(self):
        for i in range(0, len(self.e.grid)):
            for j in range(0, len(self.e.grid[0])):
                self.Q[(i, j)] = {a: 0 for a in range(0, 4)}
                self.last_tried_on[(i, j)] = {a: 0 for a in range(0, 4)}

    def __reset_last_tried_on(self):
        for i in range(0, len(self.e.grid)):
            for j in range(0, len(self.e.grid[0])):
                self.last_tried_on[(i, j)] = {a: 0 for a in range(0, 4)}

    def __update_model(self, state, action, reward, next_state):
        # For every state, action pair the model is updated with the most
        # recent reward and next_state pair experienced for it. The previous
        # values of the next_state and rewards for the same tuple are
        # overwritten with the new values
        if state in self.model:
            self.model[state][action] = {'next_state': next_state,
                                         'reward': reward}
        else:
            # If this state is being visited for the first time
            # add all possible actions in the model with unexplored actions
            # taking the agent back to the same state with reward 0
            # The action actually taken is added to the model with the actual
            # next state and experienced reward. This is for DynaQplus mode
            # If mode is DynaQ, then only the action actually taken is added
            if self.mode == DYNA_Q:
                self.model[state] = {action: {}}
            else:
                self.model[state] = {a: {'next_state': state, 'reward': 0}
                                     for a in range(0, 4)}
            self.model[state][action]['next_state'] = next_state
            self.model[state][action]['reward'] = reward

    def __next_action(self, state, episode, timestep):
        # Returns next action using an epsilon greedy policy derived from
        # current Q estimates
        # If we are using DynaQ+ variant, then the action selection is based
        # on Q values plus exploration rewards. Note that in this algorithm,
        # exploration rewards are not added to Q values during planning
        if self.mode == DYNA_Q_PLUS_VARIANT and episode != 0:
            q_values = [val + self.__exploration_reward(state, a, timestep)
                        for a, val in self.Q[state].items()]
        else:
            q_values = [val for _, val in self.Q[state].items()]

        max_val = max(q_values)

        # if there are multiple entries with maximal value, then break ties
        # randomly. max operator above always returns the first maximal element
        temp = [i for i, val in enumerate(q_values) if val == max_val]
        if episode == 0:
            random.seed(timestep)
        action = random.sample(temp, 1)[0]

        probabilities = [self.epsilon] * 4    # epsilon-soft policy
        probabilities[action] += 1 - 4 * self.epsilon
        if episode == 0:
            np.random.seed(0)
        outcome = multinomial(1, probabilities)
        for i, o in enumerate(outcome):
            if o == 1:
                return i

    def __exploration_reward(self, state, action, current_timestep):
        # Gives bonus reward for transitions not tried in reality for a while.
        tau = current_timestep - self.last_tried_on[state][action]
        if tau > 0:
            return self.kappa * sqrt(tau)
        else:
            return 0

    def __planning_reward(self, state, action, episode, timestep):
        if self.mode == DYNA_Q_PLUS and episode != 0:
            model_reward = self.model[state][action]['reward']
            exploration_reward = \
                self.__exploration_reward(state, action, timestep)
            return model_reward + exploration_reward
        else:
            return self.model[state][action]['reward']

    def plan_and_learn(self, change_grid=False):
        cumulative_reward = 0
        x = 0
        self.plotter.steps_per_episode.append([0] * self.n_episodes)
        for episode in range(0, self.n_episodes):
            print("episode num={}".format(episode))
            timestep = 0
            s = self.start_state
            self.__reset_last_tried_on()
            while s != self.goal_state:
                timestep += 1
                x += 1

                if change_grid and x == CHANGE_POINT:
                    self.e.change_grid()

                a = self.__next_action(s, episode, timestep)
                s_dash, r = self.e.next_state_and_reward(s, a)
                cumulative_reward += r
                self.plotter.arr.append(cumulative_reward)

                # Update last tried timestep for this (s,a) transition
                self.last_tried_on[s][a] = timestep

                # Update Q value estimate
                maximal_q_val = max([self.Q[s_dash][i] for i in range(0, 4)])
                self.Q[s][a] = self.Q[s][a] + self.alpha * \
                    (r + self.gamma * maximal_q_val - self.Q[s][a])

                self.__update_model(s, a, r, s_dash)    # Model-learning
                next_state = s_dash

                # Planning phase. Updates here do not change the actual
                # state of the agent. Q values might change.
                for _ in range(0, self.n_plan):
                    # Randomly select a previously seen state
                    s = random.sample(list(self.model), 1)[0]
                    a = random.sample(list(self.model[s]), 1)[0]
                    s_dash = self.model[s][a]['next_state']
                    r = self.__planning_reward(s, a, episode, timestep)
                    maximal_q_val = max([self.Q[s_dash][i]
                                         for i in range(0, 4)])
                    self.Q[s][a] = self.Q[s][a] + self.alpha * \
                        (r + self.gamma * maximal_q_val - self.Q[s][a])

                s = next_state
            self.plotter.steps_per_episode[-1][episode] = timestep
            self.alpha = self.alpha / 1.001


def dyna_q(start, goal, e, n_experiments, change_grid):
    mode = DYNA_Q
    plotter = Plotter(n_experiments=n_experiments)
    for _ in range(0, n_experiments):
        rl = DynaQ(e=e, mode=mode, plotter=plotter, start=start, goal=goal,
                   epsilon=0.1, n_episodes=150, n_plan=50, gamma=0.95,
                   alpha=0.1)
        rl.plan_and_learn(change_grid)
        plotter.plot_cumulative_reward(mode)
        # plotter.plot_steps_per_episode(mode)
    plotter.plot_change_point()


def dyna_q_plus(start, goal, e, n_experiments, change_grid):
    mode = DYNA_Q_PLUS
    plotter = Plotter(n_experiments=n_experiments)
    for _ in range(0, n_experiments):
        rl = DynaQ(e=e, mode=mode, plotter=plotter, start=start, goal=goal,
                   epsilon=0.1, n_episodes=150, n_plan=50, gamma=0.95,
                   alpha=0.1, kappa=0.001)
        rl.plan_and_learn(change_grid)
        plotter.plot_cumulative_reward(mode)
        # plotter.plot_steps_per_episode(mode)


def dyna_q_plus_varriant(start, goal, e, n_experiments, change_grid):
    mode = DYNA_Q_PLUS_VARIANT
    plotter = Plotter(n_experiments=n_experiments)
    for _ in range(0, n_experiments):
        rl = DynaQ(e=e, mode=mode, plotter=plotter, start=start, goal=goal,
                   epsilon=0.1, n_episodes=150, n_plan=50, gamma=0.95,
                   alpha=0.1, kappa=0.001)
        rl.plan_and_learn(change_grid)
        plotter.plot_cumulative_reward(mode)
        # plotter.plot_steps_per_episode(mode)


def main():
    grid = [
        [R, R, R, R, R, R, R, R, 1],
        [R, R, R, R, R, R, R, R, R],
        [R, R, R, R, R, R, R, R, R],
        [R, None, None, None, None, None, None, None, None],
        [R, R, R, R, R, R, R, R, R],
        [R, R, R, R, R, R, R, R, R]]

    start = (5, 7)
    goal = (0, 8)
    n_experiments = 1
    change_grid = True
    e = Environment(grid, start)
    dyna_q(start, goal, e, n_experiments, change_grid)
    e.reset()
    dyna_q_plus(start, goal, e, n_experiments, change_grid)
    e.reset()
    dyna_q_plus_varriant(start, goal, e, n_experiments, change_grid)
    e.reset()
    plt.show()


if __name__ == '__main__':
    main()
