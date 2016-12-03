""" The LSTD algorithm for value-estimation using linear function
approximation for states."""

import numpy as np

import utils


class LSTD(object):
    """Implements the LSTD algorithm for value-estimation.

    The algorithm and the terminology of Section 9.7 on "Least-Square TD" in
    the 2nd edition of "Reinforcement Learning: An Introduction" by Sutton and
    Barto is used.

    A linear function approximation is used for value-function.

    Attributes:
        msve (list of float): List of Mean-Square Value Error of the algorithm
            after each episode.
        theta (numpy.ndarray): The TD-fixpoint that LSTD estimates.
    """

    def __init__(self, policy, features, true_values, epsilon, gamma):
        """
        Args:
            policy (:obj:`Policy`): The policy whose value-function needs to
                be estimated.
            features (:obj:`StateFeatures`): Feature function for states.
            true_values (numpy.ndarray): The true state-values calculated
                externally. These are used only for MSVE calculations,
                and are not provided to LSTD.
            epsilon (float): The epsilon parameter.
            gamma (float): Discount factor.
        """
        self._policy = policy
        # The RL environment; in this case, the ChainWorld
        self._env = policy.env
        self._features = features
        # The matrix A initiliazed as epsilon * I
        self._A = np.eye(self._features.dimensions) * epsilon
        # The vector b initialized as an all-zero vector.
        self._b = np.zeros(self._features.dimensions)
        self._gamma = gamma
        self._true_values = true_values
        self.theta = np.zeros(self._features.dimensions)
        self.msve = []

    def run(self, num_episodes):
        """Runs the LSTD algorithm for a specified number of episodes.

        Args:
            num_episodes (int): Number of episodes to run.
        """
        for _ in xrange(num_episodes):
            self._env.reset()
            curr_state = self._env.state
            while not self._env.is_terminal(curr_state):
                reward = self._policy.take_action_and_get_reward()
                next_state = self._env.state
                self._update_parameters(curr_state, reward, next_state)
                curr_state = next_state
            # Estimate the TD-fixpoint.
            self.theta = np.dot(np.linalg.pinv(self._A), self._b)
            # Calculate current MSVE.
            self._calc_msve()

    def _calc_msve(self):
        """Calculates the MSVE between the true state-values and the current
        value-estimates based on the current estimate of TD-fixpoint. The
        calculates MSVE is added to the `msve` list.
        """
        v = []
        for state in self._env.state_iterator():
            feature_vector = self._features.vector(state)
            v.append(utils.state_value(feature_vector, self.theta))

        self.msve.append(utils.rmse(v, self._true_values))

    def _update_parameters(self, curr_state, reward, next_state):
        """Performs one step of paramter updates in accordance with the O(n^3)
        version of LSTD algorithm using naive matrix inversion.

        Args:
            curr_state (int): Current state of the agent.
            reward (float): Reward experienced by the current transition.
            next_state (int): State the agent lands in following the
                transition.
        """
        phi = self._features.vector(curr_state)
        phi_dash = self._features.vector(next_state)

        self._A += np.outer(phi, (phi - self._gamma * phi_dash))
        self._b += reward * phi
